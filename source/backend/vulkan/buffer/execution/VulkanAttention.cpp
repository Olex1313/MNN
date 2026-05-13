//
//  VulkanAttention.cpp
//  MNN
//
//  Created by MNN on 2025/04/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN/MNNDefine.h"
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "VulkanBasicExecution.hpp"
#include "VulkanBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include <cmath>

namespace MNN {

// Must match the uniform block in source/backend/vulkan/buffer/execution/glsl/sdpa_flash_mnn.comp
struct AttentionParams {
    int   batch;
    int   seq_len;
    int   kv_seq_len;
    int   num_heads;
    int   kv_num_heads;
    int   head_dim;
    float scale;
};

// Keep block_m * threads_per_row <= 256 and
// (block_m + block_n) * head_dim * 4 bytes <= 32 KB shared memory.
static constexpr int BLOCK_M = 32;
static constexpr int BLOCK_N = 32;
static constexpr int THREADS_PER_ROW = 1;  // splits head_dim across threads, tuned in 1-2-4, 1 is better with vk
static constexpr int LOCAL_SIZE_X = BLOCK_M * THREADS_PER_ROW;

// Cooperative matrix shader uses 32 threads (one full subgroup).
// The simple fallback shader uses LOCAL_SIZE_X (also 32 with THREADS_PER_ROW=1).
static constexpr int COOP_MAT_LOCAL_SIZE_X = 32;

// impl_type values — must match the MNN.fbs schema field.
enum AttentionImplType {
    ATTENTION_IMPL_AUTO     = 0,  // coopmat when device supports it, else simple
    ATTENTION_IMPL_SIMPLE   = 1,  // always use sdpa_flash_mnn
    ATTENTION_IMPL_COOP_MAT = 2,  // always use sdpa_flash_coopmat_mnn
};

class VulkanAttention : public VulkanBasicExecution {
public:
    VulkanAttention(Backend* bn, bool kvCache, int implType)
        : VulkanBasicExecution(bn), mKVCache(kvCache), mImplType(implType) {
    }

    bool init(int headDim) {
        auto vkBn = static_cast<VulkanBackend*>(backend());
        mParam = vkBn->allocUniform();

        bool useCoopMat = false;
        switch (mImplType) {
            case ATTENTION_IMPL_COOP_MAT:
                useCoopMat = true;
                break;
            case ATTENTION_IMPL_SIMPLE:
                useCoopMat = false;
                break;
            case ATTENTION_IMPL_AUTO:
            default:
                useCoopMat = vkBn->device().hasCooperativeMatrix();
                break;
        }

        if (useCoopMat && headDim != 64 && headDim != 128) {
            if (mImplType == ATTENTION_IMPL_COOP_MAT) {
                return false;
            }
            useCoopMat = false;
        }

        std::string shaderName;
        uint32_t localSizeX;
        int blockM;
        if (useCoopMat) {
            // Use default variants (Vr=2,Vc=2,Ur=1,Br=32) — OPT variants (Ur=4) have a
            // correctness issue under investigation (wrong output at q_block boundaries).
            shaderName = (headDim == 64)
                ? "glsl_sdpa_flash_coopmat_mnn_D_HEAD_64_comp"
                : "glsl_sdpa_flash_coopmat_mnn_D_HEAD_128_comp";
            localSizeX = COOP_MAT_LOCAL_SIZE_X;
            blockM     = BLOCK_M;
        } else {
            if (headDim == 64) {
                shaderName = "glsl_sdpa_flash_mnn_D_HEAD_64_comp";
            } else if (headDim == 128) {
                shaderName = "glsl_sdpa_flash_mnn_D_HEAD_128_comp";
            } else {
                return false;
            }
            localSizeX = LOCAL_SIZE_X;
            blockM     = BLOCK_M;
        }
        mBlockM = blockM;

        mPipeline = vkBn->getPipeline(shaderName, {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        }, {localSizeX});
        if (!mPipeline) return false;
        mDescriptorSet.reset(mPipeline->createSet());
        return true;
    }

    virtual ~VulkanAttention() {
        auto vkBn = static_cast<VulkanBackend*>(backend());
        vkBn->recycleUniform(mParam);
    }

    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override {
        // inputs: [query, key, value, (mask optional)]
        // query shape: [batch, seq_len, num_heads, head_dim]
        // key/value shape: [batch, kv_seq_len, num_heads, head_dim]
        auto query = inputs[0];
        auto key   = inputs[1];
        auto value = inputs[2];
        auto out   = outputs[0];

        int batch      = query->length(0);
        int seq_len    = query->length(1);
        int num_heads  = query->length(2);
        int head_dim   = query->length(3);
        int kv_seq_len = key->length(1);
        int kv_num_heads = key->length(2);

        if (!mPipeline) {
            if (!init(head_dim)) return NOT_SUPPORT;
        }

        auto param = reinterpret_cast<AttentionParams*>(mParam->map());
        param->batch        = batch;
        param->seq_len      = seq_len;
        param->kv_seq_len   = kv_seq_len;
        param->num_heads    = num_heads;
        param->kv_num_heads = kv_num_heads;
        param->head_dim     = head_dim;
        param->scale        = 1.0f / std::sqrt(static_cast<float>(head_dim));
        mParam->unmap();

        auto vkBn = static_cast<VulkanBackend*>(backend());
        mDescriptorSet->writeBuffer(vkBn->getBuffer(query), 0);
        mDescriptorSet->writeBuffer(vkBn->getBuffer(key),   1);
        mDescriptorSet->writeBuffer(vkBn->getBuffer(value), 2);
        mDescriptorSet->writeBuffer(vkBn->getBuffer(out),   3);
        mDescriptorSet->writeBuffer(mParam->buffer(), 4, mParam->size());

        mPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());

        int num_q_blocks = UP_DIV(seq_len, mBlockM);
        vkCmdDispatch(cmdBuffer->get(), num_q_blocks, num_heads, batch);

        return NO_ERROR;
    }

private:
    bool mKVCache;
    int  mImplType;
    int  mBlockM = BLOCK_M;
    const VulkanPipeline* mPipeline = nullptr;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    std::shared_ptr<VulkanBuffer> mParam;
};

class VulkanAttentionCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs,
                                           const std::vector<Tensor*>& outputs,
                                           const MNN::Op* op,
                                           Backend* backend) const override {
        auto param = op->main_as_AttentionParam();
        if (param->kv_cache()) {
            return nullptr; // fall back to CPU
        }
        return new VulkanAttention(backend, false, param->impl_type());
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Attention, new VulkanAttentionCreator);
    return true;
}();

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
