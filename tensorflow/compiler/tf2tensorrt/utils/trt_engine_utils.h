/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TRT_ENGINE_UTILS_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TRT_ENGINE_UTILS_H_

#include <string>
#include <vector>

#include "tensorflow/compiler/tf2tensorrt/common/datavec.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_shape_optimization_profiles.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
using ::tsl::StatusOr;

// Creates a TensorRT execution context.
ExecutionContext CreateExecutionContext(nvinfer1::ICudaEngine* cuda_engine);

// Sets input buffers for TRT from a list of input tensors. The input tensors
// are either defined by ctx or by input_vec.
Status SetTrtEngineInputs(nvinfer1::ICudaEngine* cuda_engine,
                          nvinfer1::IExecutionContext* execution_context,
                          const int trt_profile_idx,
                          std::vector<void*>& buffers, bool use_implicit_batch,
                          int num_batch,
                          const TrtShapeOptimizationProfile& profiles,
                          OpKernelContext* ctx = nullptr,
                          const DataVec* input_vec = nullptr);

// Returns the shape of a binding from TensorRT.
//
// The binding is identified by its binding_index. The batch_size argument is
// ignored if use_implicit_batch==false. The shape is returned in the last
// argument.
Status GetTrtBindingShape(const nvinfer1::ICudaEngine* cuda_engine,
                          const nvinfer1::IExecutionContext* execution_context,
                          int binding_index, bool use_implicit_batch,
                          int batch_size, TensorShape& shape);

// Defines output buffers for TRT. The buffers are allocated by ctx, if ctx is
// not null. Otherwise it is expected that the outputs DataVec is not null, and
// the Tensors in outputs are already allocated.
Status SetTrtEngineOutputs(nvinfer1::ICudaEngine* cuda_engine,
                           nvinfer1::IExecutionContext* execution_context,
                           int trt_profile_idx, std::vector<void*>& buffers,
                           bool use_implicit_batch, int batch_size = 0,
                           OpKernelContext* ctx = nullptr,
                           DataVec* outputs = nullptr);

// Enqueues TensorRT inference job. The batch_size argument is only relevant in
// implicit batch mode.
Status TrtEnqueue(nvinfer1::IExecutionContext* execution_context,
                  std::vector<void*>& buffers, cudaStream_t stream,
                  bool use_implicit_batch, int batch_size = 1);

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TRT_ENGINE_UTILS_H_
