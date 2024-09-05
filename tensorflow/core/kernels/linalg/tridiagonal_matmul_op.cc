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

// See docs in ../ops/linalg_ops.cc.

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// TODO(b/131583008): add broadcast support (for batch dimensions).
template <class Scalar>
class TridiagonalMatMulOp : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);

  explicit TridiagonalMatMulOp(OpKernelConstruction* context) : Base(context) {}

  void ValidateInputMatrixShapes(
      OpKernelContext* context,
      const TensorShapes& input_matrix_shapes) const final {
    auto num_inputs = input_matrix_shapes.size();
    OP_REQUIRES(
        context, num_inputs == 4,
        errors::InvalidArgument("Expected 4 inputs, got ", num_inputs, "."));

    auto n = input_matrix_shapes[3].dim_size(0);

    OP_REQUIRES(context,
                input_matrix_shapes[0].dim_size(0) == 1 &&
                    input_matrix_shapes[0].dim_size(1) == n,
                errors::InvalidArgument("Invalid superdiagonal shape."));

    OP_REQUIRES(context,
                input_matrix_shapes[1].dim_size(0) == 1 &&
                    input_matrix_shapes[1].dim_size(1) == n,
                errors::InvalidArgument("Invalid main diagonal shape."));

    OP_REQUIRES(context,
                input_matrix_shapes[2].dim_size(0) == 1 &&
                    input_matrix_shapes[2].dim_size(1) == n,
                errors::InvalidArgument("Invalid subdiagonal shape."));
  }

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
    return TensorShapes({input_matrix_shapes[3]});
  }

  int64_t GetCostPerUnit(const TensorShapes& input_matrix_shapes) const final {
    const int num_eqs = static_cast<int>(input_matrix_shapes[0].dim_size(1));
    const int num_rhss = static_cast<int>(input_matrix_shapes[3].dim_size(0));

    const double add_cost = Eigen::TensorOpCost::AddCost<Scalar>();
    const double mult_cost = Eigen::TensorOpCost::MulCost<Scalar>();

    const double cost = num_rhss * ((3 * num_eqs - 2) * mult_cost +
                                    (2 * num_eqs - 2) * add_cost);
    return cost >= static_cast<double>(kint64max) ? kint64max
                                                  : static_cast<int64_t>(cost);
  }

  // Needed to prevent writing result to the same location where input is.
  bool EnableInputForwarding() const final { return false; }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    // Superdiagonal elements. Must have length m.
    // Last element is ignored.
    const auto& superdiag = inputs[0].row(0);

    // Diagonal elements. Must have length m.
    const auto& maindiag = inputs[1].row(0);

    // Subdiagonal elements. Must have length m.
    // First element is ignored.
    const auto& subdiag = inputs[2].row(0);

    // Right-hand matrix. Size m x n.
    const auto& rhs = inputs[3];

    MatrixMap& result = outputs->at(0);

    const int m = rhs.rows();
    const int n = rhs.cols();

    ConstVectorMap subdiag_map(subdiag.data() + 1, m - 1);
    ConstVectorMap superdiag_map(superdiag.data(), m - 1);
    ConstMatrixMap rhs_except_first_row(rhs.data() + n, m - 1, n);
    ConstMatrixMap rhs_except_last_row(rhs.data(), m - 1, n);

    MatrixMap result_except_first_row(result.data() + n, m - 1, n);
    MatrixMap result_except_last_row(result.data(), m - 1, n);
    result.array() = rhs.array().colwise() * maindiag.transpose().array();
    result_except_first_row.noalias() +=
        (rhs_except_last_row.array().colwise() *
         subdiag_map.transpose().array())
            .matrix();
    result_except_last_row.noalias() +=
        (rhs_except_first_row.array().colwise() *
         superdiag_map.transpose().array())
            .matrix();
  }

 private:
  TridiagonalMatMulOp(const TridiagonalMatMulOp&) = delete;
  void operator=(const TridiagonalMatMulOp&) = delete;
};

REGISTER_LINALG_OP_CPU("TridiagonalMatMul", (TridiagonalMatMulOp<float>),
                       float);
REGISTER_LINALG_OP_CPU("TridiagonalMatMul", (TridiagonalMatMulOp<double>),
                       double);
REGISTER_LINALG_OP_CPU("TridiagonalMatMul", (TridiagonalMatMulOp<complex64>),
                       complex64);
REGISTER_LINALG_OP_CPU("TridiagonalMatMul", (TridiagonalMatMulOp<complex128>),
                       complex128);
}  // namespace tensorflow
