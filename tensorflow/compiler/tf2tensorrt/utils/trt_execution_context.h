/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TRT_EXECUTION_CONTEXT_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TRT_EXECUTION_CONTEXT_H_

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

// A wrapper for the TensorRT execution context which will destroy the TensorRT
// execution context when the object goes out of scope.
class ExecutionContext : public TrtUniquePtrType<nvinfer1::IExecutionContext> {
 public:
  ExecutionContext(nvinfer1::IExecutionContext* context, bool has_memory)
      : TrtUniquePtrType<nvinfer1::IExecutionContext>(context),
        has_device_memory_(has_memory) {}
  static ExecutionContext Create(nvinfer1::ICudaEngine* cuda_engine);

  bool HasDeviceMemory() { return has_device_memory_; }

 private:
  bool has_device_memory_;
};

};  // namespace tensorrt
};  // namespace tensorflow
#endif
#endif
