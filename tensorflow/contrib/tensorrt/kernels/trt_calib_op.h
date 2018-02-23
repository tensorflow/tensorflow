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

#ifndef TENSORFLOW_CONTRIB_TENSORRT_TRT_CALIB_OP_H
#define TENSORFLOW_CONTRIB_TENSORRT_TRT_CALIB_OP_H

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
namespace tensorflow {
namespace trt {
// TODO(sami): Convert this to async kernel!
class TRTCalibOp : public OpKernel {
 public:
  explicit TRTCalibOp(OpKernelConstruction* context);

  void Compute(OpKernelContext* context) override;

 private:
  std::string repo_name;
  std::vector<std::string> segment_nodes_;
  std::vector<std::string> input_names_;
  std::vector<tensorflow::TensorShape> shapes_;
  std::unordered_map<std::string, std::pair<void*, size_t>> device_buffers_;
  std::vector<tensorflow::PersistentTensor> dev_tensors_;
};
}  // namespace trt
}  // namespace tensorflow
#endif
#endif
#endif  // TENSORFLOW_CONTRIB_TENSORRT_TRT_CALIB_OP_H
