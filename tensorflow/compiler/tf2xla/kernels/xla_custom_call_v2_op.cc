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

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {
namespace {

class XlaCustomCallV2Op : public XlaOpKernel {
 public:
  explicit XlaCustomCallV2Op(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("call_target_name", &call_target_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("has_side_effect", &has_side_effect_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("backend_config", &backend_config_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("result_dtypes", &result_dtypes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("result_shapes", &result_shapes_));
    OP_REQUIRES(ctx, result_shapes_.size() == result_dtypes_.size(),
                errors::InvalidArgument("Unexpected number of result shapes: ",
                                        result_shapes_.size(),
                                        " != ", result_dtypes_.size()));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx, CompileImpl(*ctx));
  }

 private:
  Status CompileImpl(XlaOpKernelContext& ctx) const {
    std::vector<xla::XlaOp> operands(ctx.num_inputs());
    std::vector<xla::Shape> operand_shapes(ctx.num_inputs());
    for (int i = 0; i < ctx.num_inputs(); ++i) {
      operands[i] = ctx.Input(i);
      TF_ASSIGN_OR_RETURN(operand_shapes[i], ctx.InputXlaShape(i));
      xla::LayoutUtil::SetToDefaultLayout(&operand_shapes[i]);
    }

    std::vector<xla::Shape> result_shapes(ctx.num_outputs());
    for (int i = 0; i < ctx.num_outputs(); ++i) {
      const DataType dt = result_dtypes_[i];
      const TensorShape& shape = result_shapes_[i];
      TF_RETURN_IF_ERROR(TensorShapeToXLAShape(dt, shape, &result_shapes[i]));
    }

    xla::XlaOp results = xla::CustomCallWithLayout(                      //
        ctx.builder(),                                                   //
        call_target_name_,                                               //
        operands,                                                        //
        xla::ShapeUtil::MakeMaybeTupleShape(result_shapes),              //
        operand_shapes,                                                  //
        backend_config_,                                                 //
        has_side_effect_,                                                //
        /*output_operand_aliasing=*/{},                                  //
        /*literal=*/nullptr,                                             //
        /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,             //
        xla::CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED  //
    );

    if (ctx.num_outputs() == 1) {
      ctx.SetOutput(0, results);
    } else {
      for (int i = 0; i < ctx.num_outputs(); ++i) {
        ctx.SetOutput(i, xla::GetTupleElement(results, i));
      }
    }

    return OkStatus();
  }

  std::string call_target_name_;
  std::string backend_config_;
  bool has_side_effect_;
  std::vector<DataType> result_dtypes_;
  std::vector<TensorShape> result_shapes_;
};

REGISTER_XLA_OP(Name("XlaCustomCallV2"), XlaCustomCallV2Op);

}  // namespace
}  // namespace tensorflow
