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
    auto qr = Qr(a);

    // Get the sign and logarithm of the determinant based on the values along
    // the diagonal of R and the number of zeros in taus.
    auto log_abs_det = Einsum(Log(Abs(qr.q_and_r)), "...aa->...");
    auto sign_diag = Reduce(
        Sign(Einsum(qr.q_and_r, "...aa->...a")),
        One(a.builder(), a_shape.element_type()),
        CreateScalarMultiplyComputation(a_shape.element_type(), a.builder()),
        {a_shape.rank() - 2});
    auto sign_taus = Reduce(
        Select(Eq(qr.taus, ZerosLike(qr.taus)), FullLike(qr.taus, -1),
               FullLike(qr.taus, 1)),
        One(a.builder(), a_shape.element_type()),
        CreateScalarMultiplyComputation(a_shape.element_type(), a.builder()),
        {a_shape.rank() - 2});
    return sign_diag * log_abs_det * sign_taus;
  });
}

}  // namespace xla
