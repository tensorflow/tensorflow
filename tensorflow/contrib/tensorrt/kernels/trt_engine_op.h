/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_TENSORRT_KERNELS_TRT_ENGINE_OP_H_
#define TENSORFLOW_CONTRIB_TENSORRT_KERNELS_TRT_ENGINE_OP_H_

#include <memory>
#include <string>
#include <vector>

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "cuda/include/cuda_runtime_api.h"
#include "tensorflow/contrib/tensorrt/resources/trt_allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
class Logger;

class TRTEngineOp : public OpKernel {
 public:
  explicit TRTEngineOp(OpKernelConstruction* context);

  void Compute(OpKernelContext* context) override;
  ~TRTEngineOp();

 private:
  template <typename T>
  struct Destroyer {
    void operator()(T* d) { d->destroy(); }
  };

  template <typename T>
  using destroyed_ptr = std::unique_ptr<T, Destroyer<T>>;
  destroyed_ptr<nvinfer1::ICudaEngine> trt_engine_ptr_;
  // TODO(samikama): context should go to a resource manager!
  destroyed_ptr<nvinfer1::IExecutionContext> trt_execution_context_ptr_;

  std::vector<string> input_nodes_;
  std::vector<string> output_nodes_;
  std::shared_ptr<nvinfer1::IGpuAllocator> allocator_;
  string serialized_engine_;
};

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CONTRIB_TENSORRT_KERNELS_TRT_ENGINE_OP_H_
