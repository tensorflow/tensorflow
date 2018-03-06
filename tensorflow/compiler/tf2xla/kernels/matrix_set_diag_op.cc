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

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

namespace tensorflow {

class MatrixSetDiagOp : public XlaOpKernel {
 public:
  explicit MatrixSetDiagOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    const TensorShape input_shape = context->InputShape(0);
    const TensorShape diag_shape = context->InputShape(1);

    const int rank = input_shape.dims();

    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input_shape),
                errors::InvalidArgument(
                    "input must be at least 2-dim, received shape: ",
                    input_shape.DebugString()));

    // Check to make sure the last dimension of diag is equal to the smaller of
    // the last two dimensions of input.
    const int64 m = input_shape.dim_size(rank - 2);
    const int64 n = input_shape.dim_size(rank - 1);
    const int64 min_dim = std::min(m, n);

    TensorShape batch_shape = input_shape;
    batch_shape.RemoveLastDims(2);

    TensorShape expected_diag_shape = batch_shape;
    expected_diag_shape.AddDim(min_dim);
    OP_REQUIRES(context, expected_diag_shape == diag_shape,
                errors::InvalidArgument(
                    "must have diagonal.shape == input.shape[:-2] + "
                    "min(input.shape[-2:]), but received input shape: ",
                    input_shape.DebugString(),
                    " and diagonal shape: ", diag_shape.DebugString()));

    xla::ComputationBuilder* builder = context->builder();
    xla::ComputationDataHandle input = context->Input(0);
    xla::ComputationDataHandle diag = context->Input(1);

    auto zero = XlaHelpers::Zero(builder, context->input_type(0));

    // Create an indicator tensor that is true only on the diagonal.
    xla::ComputationDataHandle iota_m;
    OP_REQUIRES_OK(context, XlaHelpers::Iota(builder, DT_INT32, m, &iota_m));
    xla::ComputationDataHandle iota_n;
    OP_REQUIRES_OK(context, XlaHelpers::Iota(builder, DT_INT32, n, &iota_n));
    auto indicator = builder->Eq(iota_m,
                                 builder->Broadcast(iota_n, {m}),
                                 /*broadcast_dimensions=*/{0});
    indicator = builder->Broadcast(indicator, batch_shape.dim_sizes());

    // Broadcast diag up to the input shape. Use an implicit broadcast (Add)
    // because we need to broadcast on the right.
    std::vector<int64> diag_broadcast_dims(rank - 1);
    std::iota(diag_broadcast_dims.begin(), diag_broadcast_dims.end(), 0);
    if (min_dim != m) {
      diag_broadcast_dims.back() = rank - 1;
    }
    diag = builder->Add(diag, builder->Broadcast(zero, input_shape.dim_sizes()),
                        /*broadcast_dimensions=*/diag_broadcast_dims);

    auto output = builder->Select(indicator, diag, input);
    context->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MatrixSetDiagOp);
};

REGISTER_XLA_OP(Name("MatrixSetDiag"), MatrixSetDiagOp);

}  // namespace tensorflow
