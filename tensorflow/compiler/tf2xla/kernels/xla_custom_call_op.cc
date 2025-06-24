/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "xla/tsl/platform/status.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

class XlaCustomCallOp : public XlaOpKernel {
 public:
  explicit XlaCustomCallOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("target_name", &target_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("backend_config", &backend_config_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &output_type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &output_shape_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    std::vector<xla::XlaOp> inputs(ctx->num_inputs());
    for (int i = 0, end = ctx->num_inputs(); i < end; ++i) {
      inputs[i] = ctx->Input(i);
    }

    xla::Shape output_shape;
    TF_CHECK_OK(
        TensorShapeToXLAShape(output_type_, output_shape_, &output_shape));
    xla::XlaOp output = xla::CustomCall(ctx->builder(), target_name_, inputs,
                                        output_shape, backend_config_);
    ctx->SetOutput(0, output);
  }

 private:
  string target_name_;
  string backend_config_;
  DataType output_type_;
  TensorShape output_shape_;
};

REGISTER_XLA_OP(Name("XlaCustomCall"), XlaCustomCallOp);
}  // namespace
}  // namespace tensorflow
