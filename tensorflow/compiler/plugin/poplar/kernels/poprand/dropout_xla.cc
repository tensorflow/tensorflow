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

#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
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

class PoprandDropoutOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit PoprandDropoutOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rate", &rate));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("scale", &scale));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("is_using_user_seed", &is_using_user_seed));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed_modifier", &seed_modifier));

    attribute_map_.AddAttribute("rate", rate);
    attribute_map_.AddAttribute("scale", scale);
    attribute_map_.AddAttribute("is_using_user_seed", is_using_user_seed);
    attribute_map_.AddAttribute("seed_modifier", seed_modifier);
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = input_type(0);
    xla::XlaBuilder* b = ctx->builder();

    auto input = ctx->Input(0);
    auto shape = ctx->InputShape(0);

    auto seed_input = ctx->Input(1);
    auto seed_shape = ctx->InputShape(1);
    const DataType seed_type = input_type(1);

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    xla::Shape xla_seed_shape;
    OP_REQUIRES_OK(
        ctx, TensorShapeToXLAShape(seed_type, seed_shape, &xla_seed_shape));

    // We are outputting both the output of the operation and the seed used so
    // we can reuse the seed in the backprop pass.
    xla::Shape output_tuple_shape =
        xla::ShapeUtil::MakeTupleShape({xla_shape, xla_seed_shape});

    xla::XlaOp call_output = xla::CustomCall(
        b,
        GetPoplibsCustomOpTargetString(PoplibsOp::Poprand, PoplibsOp::Dropout),
        {input, seed_input}, output_tuple_shape, attribute_map_.Serialise());

    // The actual dropout output.
    xla::XlaOp output = xla::GetTupleElement(call_output, 0);

    // Save the seed used so we can reference it in the backwards pass.
    xla::XlaOp seed_used = xla::GetTupleElement(call_output, 1);

    ctx->SetOutput(0, output);
    ctx->SetOutput(1, seed_used);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PoprandDropoutOp);

  // Param to scale the non-dropped outputs by.
  float scale;

  // To the user this is the probability that a node will be dropped but it has
  // been reversed by this point (for poplar) to represent the probability that
  // a node will be kept.
  float rate;

  // Modifier to apply to the random number generator seed value.
  int32_t seed_modifier;

  // Track if the user provided the seed value or whether we should use the
  // global seed we create.
  bool is_using_user_seed;
};

REGISTER_IPU_OP("IpuDropout", PoprandDropoutOp);

}  // namespace tensorflow
