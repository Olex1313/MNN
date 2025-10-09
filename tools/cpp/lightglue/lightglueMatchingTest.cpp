#include <stdio.h>
#include "MNN/ImageProcess.hpp"
#include "MNN/Interpreter.hpp"
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include "MNN/AutoTime.hpp"
#include <iostream>
#include <cstring>
#include <chrono>
#include <thread>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "backend/cpu/ThreadPool.hpp"

struct Keypoints {
	std::vector<float> points;
	std::vector<float> descriptors;
	std::vector<float> angles;
	std::vector<float> sizes;
	int64_t K;
	int64_t D;
};

struct Matches {
	std::vector<int> m0, m1;
	std::vector<float> s0, s1;

	size_t size() const
	{
		size_t result = 0;

		size_t n = m0.size();
		for (int from = 0; from < n; ++from) {
			int to = m0[from];
			if (to < 0)
				continue;

			if (m1[to] != from)
				continue;

			result++;
		}

		return result;
	}
};

void resizeAllInputs(MNN::Interpreter* net, MNN::Session* sess,
					 const std::map<std::string, std::vector<int>>& inputShapes) {
	for (const auto& kv : inputShapes) {
		const std::string& name = kv.first;
		const std::vector<int>& shape = kv.second;
		MNN::Tensor* in = net->getSessionInput(sess, name.c_str());
		if (!in) {
			std::cerr << "Missing input tensor: " + name << std::endl;
			std::terminate();
		}
		net->resizeTensor(in, shape);
	}
	net->resizeSession(sess);
}

inline size_t elemCount(const std::vector<int>& dims) {
	size_t n = 1;
	for (int d : dims) n *= static_cast<size_t>(d);
	return n;
}

void copyToInput(MNN::Interpreter* net, MNN::Session* sess,
				 const std::string& name,
				 const float* data,
				 const std::vector<int>& shape) {
	MNN::Tensor* in = net->getSessionInput(sess, name.c_str());
	if (!in) {
		std::cerr << "Missing input tensor: " + name << std::endl;
		std::terminate();
	}
	// Host staging tensor with same shape & dim type as device tensor
	std::unique_ptr<MNN::Tensor> host(MNN::Tensor::create<float>(shape, nullptr, in->getDimensionType()));
	std::memcpy(host->host<float>(), data, elemCount(shape) * sizeof(float));
	in->copyFromHostTensor(host.get());
}

template <typename T>
std::pair<std::vector<T>, std::vector<int>> fetchOutput(MNN::Interpreter* net, MNN::Session* sess,
														const std::string& name) {
	MNN::Tensor* out = net->getSessionOutput(sess, name.c_str());
	if (!out) {
		std::cerr << "Missing output tensor: " + name << std::endl;
		std::terminate();
	}
	// Copy device → host
	std::unique_ptr<MNN::Tensor> host(new MNN::Tensor(out, out->getDimensionType()));
	out->copyToHostTensor(host.get());
	const size_t n = elemCount(host->shape());
	std::vector<T> buf(n);
	std::memcpy(buf.data(), host->host<T>(), n * sizeof(T));
	return {buf, host->shape()};
}

inline Keypoints ReadKeypoints(const std::string &path) {

	std::ifstream is(path, std::ios::binary);
	if (!is.good()) {
		std::cerr << "kpts file not found. specify working dir?" << std::endl;
		std::terminate();
	}

	Keypoints k;
	int64_t K, D;
	uint64_t n_points, n_desc, n_angles, n_sizes;

	is.read(reinterpret_cast<char*>(&K), sizeof(K));
	is.read(reinterpret_cast<char*>(&D), sizeof(D));
	is.read(reinterpret_cast<char*>(&n_points), sizeof(n_points));
	is.read(reinterpret_cast<char*>(&n_desc), sizeof(n_desc));
	is.read(reinterpret_cast<char*>(&n_angles), sizeof(n_angles));
	is.read(reinterpret_cast<char*>(&n_sizes), sizeof(n_sizes));

	k.K = K;
	k.D = D;
	k.points.resize(static_cast<size_t>(n_points));
	k.descriptors.resize(static_cast<size_t>(n_desc));
	k.angles.resize(static_cast<size_t>(n_angles));
	k.sizes.resize(static_cast<size_t>(n_sizes));

	is.read(reinterpret_cast<char*>(k.points.data()),      n_points * sizeof(float));
	is.read(reinterpret_cast<char*>(k.descriptors.data()), n_desc   * sizeof(float));
	is.read(reinterpret_cast<char*>(k.angles.data()),      n_angles * sizeof(float));
	is.read(reinterpret_cast<char*>(k.sizes.data()),       n_sizes  * sizeof(float));

	return k;
}

inline void WriteKeypoints(const Keypoints& k, const std::string& path) {
	std::ofstream os(path, std::ios::binary);
	const int64_t K = k.K;
	const int64_t D = k.D;
	const uint64_t n_points = static_cast<uint64_t>(k.points.size());
	const uint64_t n_desc   = static_cast<uint64_t>(k.descriptors.size());
	const uint64_t n_angles = static_cast<uint64_t>(k.angles.size());
	const uint64_t n_sizes  = static_cast<uint64_t>(k.sizes.size());

	os.write(reinterpret_cast<const char*>(&K), sizeof(K));
	os.write(reinterpret_cast<const char*>(&D), sizeof(D));
	os.write(reinterpret_cast<const char*>(&n_points), sizeof(n_points));
	os.write(reinterpret_cast<const char*>(&n_desc), sizeof(n_desc));
	os.write(reinterpret_cast<const char*>(&n_angles), sizeof(n_angles));
	os.write(reinterpret_cast<const char*>(&n_sizes), sizeof(n_sizes));

	os.write(reinterpret_cast<const char*>(k.points.data()),      n_points * sizeof(float));
	os.write(reinterpret_cast<const char*>(k.descriptors.data()), n_desc   * sizeof(float));
	os.write(reinterpret_cast<const char*>(k.angles.data()),      n_angles * sizeof(float));
	os.write(reinterpret_cast<const char*>(k.sizes.data()),       n_sizes  * sizeof(float));
}

Matches matchPairMNN(Keypoints &kpts0, Keypoints &kpts1, const std::string &model_path)
{
	if (kpts0.D != kpts1.D) {
		std::cerr << "kpts0.D != kpts1.D" << std::endl; std::terminate();
	}

	if (kpts0.K != kpts1.K) {
		std::cerr << "kpts0.K != kpts1.K" << std::endl; std::terminate();
	}

	int K = kpts0.K;
	int D = kpts0.D;

	// shapes: (N, K, 2) and (N, K, D)
	std::vector<int> kpt_shape = {1, K, 2};
	std::vector<int> desc_shape = {1, K, D};

	std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(model_path.c_str()));
	if (!net) {
		std::cerr << "Failed to create MNN interpreter" << std::endl; std::terminate();
	}

	MNN::ScheduleConfig config;
	// for opencl this is not num threads this is combination of flags from MNNGpuMode enum
	// performance: 0.6s/iter for no optimizations, 0.5s/iter for MNN_GPU_TUNING_WIDE and 0.17s/iter for MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_BUFFER
	config.type = MNN_FORWARD_OPENCL;
	config.numThread = MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_BUFFER;

//    config.type = MNN_FORWARD_VULKAN;
//    config.numThread = MNN_GPU_TUNING_WIDE | MNN_GPU_RECORD_BATCH;

//    config.type = MNN_FORWARD_CUDA;
//    config.numThread = MNN_GPU_TUNING_NONE;

	MNN::BackendConfig bconf;
	bconf.precision = MNN::BackendConfig::Precision_Normal;
	bconf.power     = MNN::BackendConfig::Power_Normal;
	bconf.memory    = MNN::BackendConfig::Memory_Normal;
	config.backendConfig = &bconf;

	MNN::Session* sess = net->createSession(config);
	if (!sess) {
		std::cerr << "Failed to create MNN session" << std::endl; std::terminate();
	}

	resizeAllInputs(net.get(), sess, {
			{"keypoints0",   kpt_shape},
			{"descriptors0", desc_shape},
			{"keypoints1",   kpt_shape},
			{"descriptors1", desc_shape},
	});

	copyToInput(net.get(), sess, "keypoints0",   kpts0.points.data(),  kpt_shape);
	copyToInput(net.get(), sess, "descriptors0", kpts0.descriptors.data(),  desc_shape);
	copyToInput(net.get(), sess, "keypoints1",   kpts1.points.data(),  kpt_shape);
	copyToInput(net.get(), sess, "descriptors1", kpts1.descriptors.data(),  desc_shape);

	int code = net->runSession(sess);
	if (code != 0) {
		std::cerr << "MNN runSession error code: " + std::to_string(code) << std::endl; std::terminate();
	}

	Matches result;
	std::vector<int> sh0, sh1, sh2, sh3;
	std::tie(result.m0, sh0) = fetchOutput<int>(net.get(), sess, "m0");
	std::tie(result.m1, sh1) = fetchOutput<int>(net.get(), sess, "m1");
	std::tie(result.s0, sh2) = fetchOutput<float>(net.get(), sess, "s0");
	std::tie(result.s1, sh3) = fetchOutput<float>(net.get(), sess, "s1");

	for (int iter = 0; iter < 100; ++iter) {
		auto t0 = std::chrono::steady_clock::now();
		net->runSession(sess);
		auto t1 = std::chrono::steady_clock::now();
		double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
		std::cout << "  mnn pass " << iter << ": " << ms * 0.000001 << " s\n";
	}

	return result;
}

int main(int argc, const char* argv[]) {

	// Keypoints generated by ALIKED detector/descriptor
	Keypoints kpts0 = ReadKeypoints("kpts0.bin");
	Keypoints kpts1 = ReadKeypoints("kpts1.bin");

	// MNN model generated from .onnx by MNNConverter with following flags:
	// -f ONNX
	// --modelFile /home/simiyutin/sources/image_matching/lightglue/LightGlue-ONNX/weights/lightglue_aliked_3840.onnx
	// --MNNModel weights/lightglue_matcher_large.mnn
	// --saveStaticModel --optimizeLevel 1 --transformerFuse
	std::string model_path = "lightglue_matcher_large.mnn";

	Matches matches = matchPairMNN(kpts0, kpts1, model_path);
	std::cout << "detected " << matches.size() << " MNN matches" << std::endl;
	return 0;
}
