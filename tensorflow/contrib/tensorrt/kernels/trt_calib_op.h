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

#ifndef TENSORFLOW_CONTRIB_TENSORRT_KERNELS_TRT_CALIB_OP_H
#define TENSORFLOW_CONTRIB_TENSORRT_KERNELS_TRT_CALIB_OP_H

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
namespace tensorflow {
namespace tensorrt {
class TRTCalibrationResource;
class TRTCalibOp : public AsyncOpKernel {
 public:
  explicit TRTCalibOp(OpKernelConstruction* context);

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override;
  tensorflow::Status AllocateCalibrationResources(
      OpKernelContext*, tensorflow::tensorrt::TRTCalibrationResource** cr);

 private:
  string resource_name_;
  tensorflow::GraphDef segment_graph_;
  tensorflow::int64 workspace_size_;
  std::vector<tensorflow::TensorShape> shapes_;
  std::unordered_map<string, std::pair<void*, size_t>> device_buffers_;
  std::vector<tensorflow::PersistentTensor> dev_tensors_;
  tensorflow::FunctionLibraryRuntime::Options fopts_;
  tensorflow::FunctionLibraryRuntime::Handle native_func_;
};
}  // namespace tensorrt
}  // namespace tensorflow
#endif
#endif
#endif  // TENSORFLOW_CONTRIB_TENSORRT_KERNELS_TRT_CALIB_OP_H
