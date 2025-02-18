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

#include <cstdint>

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

class InTopKOp : public XlaOpKernel {
 public:
  explicit InTopKOp(OpKernelConstruction* context) : XlaOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("T", &targets_dtype_));
    OP_REQUIRES_OK(context,
                   DataTypeToPrimitiveType(targets_dtype_, &targets_type_));
  }

  void Compile(XlaOpKernelContext* context) override {
    int64_t k;
    OP_REQUIRES_OK(context, context->ConstantInputAsIntScalar(2, &k));
    OP_REQUIRES(context, k >= 0,
                errors::InvalidArgument("Need k >= 0, got ", k));
    const TensorShape predictions_shape = context->InputShape(0);
    OP_REQUIRES(
        context, predictions_shape.dims() == 2,
        errors::InvalidArgument("predictions must be == 2-D, got shape ",
                                predictions_shape.DebugString()));
    const TensorShape targets_shape = context->InputShape(1);
    OP_REQUIRES(context, targets_shape.dims() == 1,
                errors::InvalidArgument("targets must be == 1-D, got shape ",
                                        targets_shape.DebugString()));

    int64_t batch_size = predictions_shape.dim_size(0);
    OP_REQUIRES(context, batch_size == targets_shape.dim_size(0),
                errors::InvalidArgument(
                    "targets must have same elements as predictions rows. Had ",
                    targets_shape.dim_size(0), ", needed ", batch_size));

    // Given `predictions` with shape batch_size*num_classes and `target` with
    // shape num_classes, we generate `targets_values_r1` with shape num_classes
    // which the elements are the corresponding values of `targets` in
    // `predictions` for each example. This step can be done using xla::Gather
    // as well.
    xla::XlaOp predictions_r2 = context->Input(0);
    xla::XlaOp targets_r1 = context->Input(1);

    xla::XlaBuilder* xla_builder = context->builder();
    xla::XlaOp iota_r1 =
        xla::Iota(xla_builder, targets_type_, predictions_shape.dim_size(1));
    xla::XlaOp iota_r2 = xla::Broadcast(iota_r1, {batch_size});

    xla::XlaOp eq_r2 = xla::Eq(targets_r1, iota_r2, {0});
    xla::XlaOp zero_r0_f32 = xla::Zero(xla_builder, xla::F32);
    xla::XlaOp zero_r2_f32 = xla::ZerosLike(predictions_r2);
    xla::XlaOp select_r2 = xla::Select(eq_r2, predictions_r2, zero_r2_f32);
    xla::XlaOp targets_values_r1 = xla::Reduce(
        select_r2, zero_r0_f32,
        xla::CreateScalarAddComputation(xla::F32, xla_builder), {1});

    // Calculate in each row of `predictions`, how many values are larger than
    // the value of target class. Then return the result whether the count < k,
    // which indicates the target is in topk.
    xla::XlaOp gt_r2 = xla::Gt(predictions_r2, targets_values_r1, {0});
    xla::XlaOp zero_r0 = xla::Zero(xla_builder, xla::S32);
    xla::XlaOp zero_r2 = xla::Broadcast(zero_r0, predictions_shape.dim_sizes());
    xla::XlaOp one_r0 = xla::One(xla_builder, xla::S32);
    xla::XlaOp one_r2 = xla::Broadcast(one_r0, predictions_shape.dim_sizes());
    xla::XlaOp one_hot_r2 = xla::Select(gt_r2, one_r2, zero_r2);
    xla::XlaOp num_gt_r1 = xla::Reduce(
        one_hot_r2, zero_r0,
        xla::CreateScalarAddComputation(xla::S32, xla_builder), {1});

    xla::XlaOp result =
        xla::And(xla::Lt(num_gt_r1, xla::ConstantR0<int32>(xla_builder, k)),
                 xla::IsFinite(targets_values_r1));

    context->SetOutput(0, result);
  }

 protected:
  DataType targets_dtype_;
  xla::PrimitiveType targets_type_;

  InTopKOp(const InTopKOp&) = delete;
  void operator=(const InTopKOp&) = delete;
};

REGISTER_XLA_OP(Name("InTopKV2")
                    .CompileTimeConstantInput("k")
                    .TypeConstraint("T", {DT_INT32, DT_INT64}),
                InTopKOp);

}  // namespace
}  // namespace tensorflow
