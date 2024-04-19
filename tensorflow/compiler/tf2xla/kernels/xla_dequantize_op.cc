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

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/lib/quantize.h"
#include "xla/client/xla_builder.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

class XlaDequantizeOp : public XlaOpKernel {
 public:
  explicit XlaDequantizeOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("min_range", &min_range_));
    OP_REQUIRES_OK(context, context->GetAttr("max_range", &max_range_));
    OP_REQUIRES_OK(context, context->GetAttr("mode", &mode_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("transpose_output", &transpose_output_));
  }

  void Compile(XlaOpKernelContext* context) override {
    const xla::XlaOp& input = context->Input(0);

    xla::QuantizedRange range(min_range_, max_range_);

    xla::XlaOp output =
        xla::Dequantize<uint8>(input, range, mode_, transpose_output_);
    context->SetOutput(0, output);
  }

 private:
  float min_range_;
  float max_range_;
  bool transpose_output_;
  string mode_;
  XlaDequantizeOp(const XlaDequantizeOp&) = delete;
  void operator=(const XlaDequantizeOp&) = delete;
};

REGISTER_XLA_OP(Name("XlaDequantize"), XlaDequantizeOp);

}  // namespace
}  // namespace tensorflow
