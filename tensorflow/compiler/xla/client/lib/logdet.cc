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

#include "tensorflow/compiler/xla/client/lib/logdet.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/loops.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/qr.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

// log(det(A)) = sum(log(vecdiag(QR(A).r))), since R is triangular and Q is
// orthonormal
XlaOp LogDet(XlaOp a) {
  return a.builder()->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape a_shape, a.builder()->GetShape(a));
    // Compute the number of Householder transformations required on 'a' by
    // determining the number of rows in 'a' that are already triangular. The
    // determinant of Q is -1 ^ (number of Householder transfomations)
    auto rows = Iota(a.builder(), ShapeUtil::ChangeElementType(a_shape, S32),
                     a_shape.rank() - 2);
    auto cols = Iota(a.builder(), ShapeUtil::ChangeElementType(a_shape, S32),
                     a_shape.rank() - 1);
    auto in_lower_triangle = Lt(cols, rows);
    auto is_zero = Eq(a, ScalarLike(a, 0));
    auto num_zeros_in_triangle_per_row = Einsum(
        ConvertElementType(And(in_lower_triangle, is_zero), S32), "...a->...");
    TF_ASSIGN_OR_RETURN(auto row_shape,
                        a.builder()->GetShape(num_zeros_in_triangle_per_row));
    rows = Iota(a.builder(), row_shape, row_shape.rank() - 1);
    auto num_triangle_rows =
        Einsum(ConvertElementType(Eq(rows, num_zeros_in_triangle_per_row), S32),
               "...a->...");
    auto num_rows =
        ScalarLike(num_triangle_rows, a_shape.dimensions(a_shape.rank() - 2));

    TF_ASSIGN_OR_RETURN(auto qr, QRDecomposition(a, true));
    // Get the and log of the determinant based on the values along the diagonal
    // of R.
    auto log_abs_det = Einsum(Log(Abs(qr.r)), "...aa->...");
    auto sign_diag = Reduce(
        Sign(Einsum(qr.r, "...aa->...a")),
        One(a.builder(), a_shape.element_type()),
        CreateScalarMultiplyComputation(a_shape.element_type(), a.builder()),
        {a_shape.rank() - 2});
    return sign_diag * log_abs_det *
           Select(ConvertElementType(Rem(num_rows - num_triangle_rows,
                                         ScalarLike(num_triangle_rows, 2)),
                                     PRED),
                  ScalarLike(sign_diag, -1.0), ScalarLike(sign_diag, 1.0));
  });
}

}  // namespace xla
