/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/common_runtime/single_threaded_cpu_device.h"

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {

namespace {

static constexpr int kNumThreads = 1;

thread::ThreadPool* GraphRunnerThreadPool() {
  static thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "graph_runner", kNumThreads);
  return thread_pool;
}

// A simple single-threaded CPU device. This can be used to run inexpensive
// computations. In particular, using this avoids initializing the global thread
// pools in LocalDevice.
class SingleThreadedCpuDevice : public Device {
 public:
  explicit SingleThreadedCpuDevice(Env* env)
      : Device(env, Device::BuildDeviceAttributes("/device:CPU:0", DEVICE_CPU,
                                                  Bytes(256 << 20),
                                                  DeviceLocality())) {
    eigen_worker_threads_.num_threads = kNumThreads;
    eigen_worker_threads_.workers = GraphRunnerThreadPool();
    eigen_device_.reset(new Eigen::ThreadPoolDevice(
        eigen_worker_threads_.workers->AsEigenThreadPool(),
        eigen_worker_threads_.num_threads));
    set_tensorflow_cpu_worker_threads(&eigen_worker_threads_);
    set_eigen_cpu_device(eigen_device_.get());
  }

  ~SingleThreadedCpuDevice() override { eigen_device_.reset(); }

  Status Sync() override { return Status::OK(); }

  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override {
    Tensor parsed(tensor_proto.dtype());
    if (!parsed.FromProto(cpu_allocator(), tensor_proto)) {
      return errors::InvalidArgument("Cannot parse tensor from tensor_proto.");
    }
    *tensor = parsed;
    return Status::OK();
  }

  void CopyTensorInSameDevice(const Tensor* input_tensor, Tensor* output_tensor,
                              const DeviceContext*,
                              StatusCallback done) override {
    if (input_tensor->NumElements() != output_tensor->NumElements()) {
      done(errors::Internal(
          "SingleThreadedCPU->SingleThreadedCPU copy shape mismatch: input=",
          input_tensor->shape(), ", output=", output_tensor->shape()));
      return;
    }
    tensor::DeepCopy(*input_tensor, output_tensor);
    done(Status::OK());
  }

  Allocator* GetAllocator(AllocatorAttributes attr) override {
    return cpu_allocator();
  }

 private:
  DeviceBase::CpuWorkerThreads eigen_worker_threads_;
  std::unique_ptr<Eigen::ThreadPoolDevice> eigen_device_;
};

}  // namespace

Device* NewSingleThreadedCpuDevice(Env* env) {
  return new SingleThreadedCpuDevice(env);
}

}  // namespace tensorflow
