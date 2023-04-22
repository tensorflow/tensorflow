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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_COMPRESSION_OPS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_COMPRESSION_OPS_H_

#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {
namespace experimental {

class CompressElementOp : public OpKernel {
 public:
  explicit CompressElementOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;
};

class UncompressElementOp : public OpKernel {
 public:
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";

  explicit UncompressElementOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 private:
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

}  // namespace experimental
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_COMPRESSION_OPS_H_
