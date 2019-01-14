/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/trace.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/stream_executor_util.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal_util.h"

#include "absl/container/flat_hash_set.h"

using namespace xla::poplarplugin;

namespace tensorflow {
namespace {
class PopopsUnaryOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit PopopsUnaryOp(const PoplibsOp& op_type, OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), op_type_(op_type) {
    AddRequiredAttributesToMap();
  }

 public:
  ~PopopsUnaryOp() override{};

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp input = ctx->Input(0);

    // Get the input shape
    xla::PrimitiveType input_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(0), &input_type));
    xla::Shape input_shape =
        TensorShapeToXLAShape(input_type, ctx->InputShape(0));

    xla::XlaBuilder& b = *ctx->builder();

    std::vector<xla::XlaOp> args = {input};
    xla::XlaOp output = xla::CustomCall(
        &b, GetPoplibsCustomOpTargetString(PoplibsLib::Popops, op_type_), args,
        input_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 protected:
  const absl::flat_hash_set<int64> AllocatingIndexes() override { return {}; }

  const absl::flat_hash_map<int64, int64> LayoutDependencies() override {
    return {};
  };

  const uint64 NumberOfInplaceOperands() override { return 1; }

 private:
  PoplibsOp op_type_;
};
}  // namespace

class PopopsSqrtOp : public PopopsUnaryOp {
 public:
  explicit PopopsSqrtOp(OpKernelConstruction* ctx)
      : PopopsUnaryOp(PoplibsOp::Sqrt, ctx) {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsSqrtOp);
};
REGISTER_XLA_OP(Name("Sqrt").Device(DEVICE_IPU_XLA_JIT), PopopsSqrtOp);

class PopopsRsqrtOp : public PopopsUnaryOp {
 public:
  explicit PopopsRsqrtOp(OpKernelConstruction* ctx)
      : PopopsUnaryOp(PoplibsOp::Rsqrt, ctx) {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsRsqrtOp);
};
REGISTER_XLA_OP(Name("Rsqrt").Device(DEVICE_IPU_XLA_JIT), PopopsRsqrtOp);

}  // namespace tensorflow
