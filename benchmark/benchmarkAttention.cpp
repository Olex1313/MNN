#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cfloat>
#include <chrono>
#include <vector>
#include <random>

#include "MNN_generated.h"
#include "core/Backend.hpp"
#include <MNN/MNNForwardType.h>
#include <MNN/MNNDefine.h>
#define MNN_USER_SET_DEVICE
#include <MNN/MNNSharedContext.h>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>

#include "backend/opencl/core/OpenCLBackend.hpp"

using namespace MNN;
using namespace MNN::Express;

struct BenchCase {
    int batch;
    int seq_len;
    int num_head;
    int head_dim;
    bool use_flash;
    int impl_type; // 0=AUTO, 1=SIMPLE, 2=COOP_MAT for vulkan
};

static void fillRandom(float* ptr, int size) {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < size; i++) {
        ptr[i] = dist(rng);
    }
}

static void flushGPU(const Backend* backend, MNNForwardType forward) {
    if (forward == MNN_FORWARD_OPENCL) {
        auto* ocl = const_cast<OpenCL::OpenCLBackend*>(static_cast<const OpenCL::OpenCLBackend*>(backend));
        ocl->getOpenCLRuntime()->commandQueue().finish();
    }
    // Vulkan: onExecuteEnd() already submits and waits via VulkanFence — no extra sync needed.
}

static void benchAttention(const BenchCase& c, const ScheduleConfig& config,
                           int warmup, int loop) {
    const char* tag = !c.use_flash ? "native" : (c.impl_type == 2 ? "coopmat" : "flash");
    MNN_PRINT("%-7s  B=%d  seq=%4d  H=%2d  D=%3d  \n", tag, c.batch, c.seq_len, c.num_head, c.head_dim);

    std::shared_ptr<OpT> attention(new OpT);
    attention->type = OpType_Attention;
    attention->main.type = OpParameter_AttentionParam;
    attention->main.value = new AttentionParamT;
    attention->main.AsAttentionParam()->kv_cache = false;
    attention->main.AsAttentionParam()->flash_attn_kernel = c.use_flash;
    attention->main.AsAttentionParam()->impl_type = c.impl_type;

    auto Q = _Input({c.batch, c.seq_len, c.num_head, c.head_dim}, NCHW);
    auto K = _Input({c.batch, c.seq_len, c.num_head, c.head_dim}, NCHW);
    auto V = _Input({c.batch, c.seq_len, c.num_head, c.head_dim}, NCHW);

    int elems = c.batch * c.seq_len * c.num_head * c.head_dim;
    fillRandom(Q->writeMap<float>(), elems);
    fillRandom(K->writeMap<float>(), elems);
    fillRandom(V->writeMap<float>(), elems);

    auto Output = Variable::create(Expr::create(attention.get(), {Q, K, V}));
    auto buffer = Variable::save({Output});

    std::unique_ptr<Interpreter> net(Interpreter::createFromBuffer(
        (const void*)buffer.data(), buffer.size()));
    auto session = net->createSession(config);

    auto inputs = net->getSessionInputAll(session);
    std::vector<std::shared_ptr<Tensor>> inputHosts;
    for (auto& kv : inputs) {
        auto host = std::shared_ptr<Tensor>(
            Tensor::createHostTensorFromDevice(kv.second, false));
        fillRandom(host->host<float>(), host->elementSize());
        inputHosts.push_back(host);
    }

    auto outputTensor = net->getSessionOutput(session, NULL);
    auto outputHost = std::shared_ptr<Tensor>(
        Tensor::createHostTensorFromDevice(outputTensor, false));
    const Backend* backend = net->getBackend(session, outputTensor);

    auto runInputs = [&]() {
        for (size_t j = 0; j < inputHosts.size(); j++) {
            auto it = inputs.begin();
            std::advance(it, j);
            it->second->copyFromHostTensor(inputHosts[j].get());
        }
    };

    // Warmup — copy output to ensure GPU is fully done before timing starts.
    for (int i = 0; i < warmup; i++) {
        runInputs();
        net->runSession(session);
        outputTensor->copyToHostTensor(outputHost.get());
    }

    using steady_clock = std::chrono::steady_clock;
    float minUs = FLT_MAX, maxUs = 0, sumUs = 0;
    for (int i = 0; i < loop; i++) {
        runInputs();

        auto t0 = steady_clock::now();
        net->runSession(session);
        flushGPU(backend, config.type);
        auto t1 = steady_clock::now();

        float us = std::chrono::duration<float, std::micro>(t1 - t0).count();
        minUs = std::min(minUs, us);
        maxUs = std::max(maxUs, us);
        sumUs += us;
    }
    float avgUs = sumUs / loop;

    double flops = 4.0 * c.batch * c.num_head * c.seq_len * c.seq_len * c.head_dim;
    double tflops = flops / (minUs / 1e6) / 1e9;

    MNN_PRINT("tag=%s B=%d  seq=%4d  H=%2d  D=%3d min=%8.0f  avg=%8.0f  max=%8.0f us  GFLOPS=%.4f\n\n",
           tag, c.batch, c.seq_len, c.num_head, c.head_dim, minUs, avgUs, maxUs, tflops);
}

static void printUsage(const char* prog) {
    MNN_PRINT("Usage: %s [forward_type] [loop] [warmup] [device_id]\n", prog);
    MNN_PRINT("  forward_type: 0=CPU, 3=OpenCL, 7=Vulkan (default: 3)\n");
    MNN_PRINT("  loop:         number of iterations (default: 10)\n");
    MNN_PRINT("  warmup:       warmup iterations (default: 3)\n");
    MNN_PRINT("  device_id:    OpenCL device index (default: 0)\n");
}

int main(int argc, const char* argv[]) {
    if (argc > 1 && strcmp(argv[1], "help") == 0) {
        printUsage(argv[0]);
        return 0;
    }

    MNNForwardType forward = MNN_FORWARD_OPENCL;
    int loop = 10, warmup = 3, deviceId = 0;
    if (argc >= 2) forward = static_cast<MNNForwardType>(atoi(argv[1]));
    if (argc >= 3) loop = atoi(argv[2]);
    if (argc >= 4) warmup = atoi(argv[3]);
    if (argc >= 5) deviceId = atoi(argv[4]);

    MNNDeviceContext devCtx;
    devCtx.deviceId = deviceId;
    devCtx.platformId = 0;
    devCtx.platformSize = 0;
    devCtx.contextPtr = nullptr;

    ScheduleConfig config;
    config.type = forward;
    if (forward == MNN_FORWARD_VULKAN) {
        config.mode = MNN_GPU_TUNING_WIDE | MNN_GPU_RECORD_BATCH;
    } else {
        config.mode = MNN_GPU_TUNING_NONE | MNN_GPU_MEMORY_BUFFER;
    }

    BackendConfig bnConfig;
    bnConfig.precision = MNN::BackendConfig::Precision_Normal;
    bnConfig.power     = MNN::BackendConfig::Power_Normal;
    bnConfig.memory    = MNN::BackendConfig::Memory_Normal;
    bnConfig.sharedContext = (forward != MNN_FORWARD_VULKAN) ? &devCtx : nullptr;
    config.backendConfig = &bnConfig;

    MNN_PRINT("MNN Attention Benchmark  forward=%d  loop=%d  warmup=%d  device=%d\n\n",
              forward, loop, warmup, deviceId);

    std::vector<BenchCase> cases;
    struct Cfg { int B, H, S, D; };
    std::vector<Cfg> cfgs = {
        // Increasing seqlen — main scaling axis (reference frontend: B=4,H=8)
        // {4, 8, 1024,  64},
        // {4, 8, 2048,  64},
        // {4, 8, 4096,  64},
        // {4, 8, 8192,  64},
        // {4, 8, 16384, 64},
        // {4, 8, 32768, 64},
        // D=128
        {4, 8, 1024, 128},
        {4, 8, 2048, 128},
        {4, 8, 4096, 128},
        // {4, 8, 8192,  128},
        // {4, 8, 16384,  128},
        // {4, 8, 32768,  128},
        // More heads
        // {4, 16, 2048, 64},
        // {4, 32, 2048, 64},
        // // More batch
        // {8,  8, 2048, 64},
        // {16, 8, 2048, 64},
    };
    for (auto& g : cfgs) {
        BenchCase flash{g.B, g.S, g.H, g.D, true, 1};
        cases.push_back(flash);
        // OpenCL native (no flash) OOMs above 8192 skip for now
        if (forward == MNN_FORWARD_OPENCL && g.S <= 8192) {
            BenchCase native = flash;
            native.use_flash = false;
            cases.push_back(native);
        }
        if (forward == MNN_FORWARD_VULKAN) {
            BenchCase coopmat = flash;
            coopmat.impl_type = 2;
            cases.push_back(coopmat);
        }
    }
    
    for (auto& c : cases) {
        benchAttention(c, config, warmup, loop);
    }
}
