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

SignAndLogDet SLogDet(XlaOp a) {
  StatusOr<SignAndLogDet> result = [&]() -> StatusOr<SignAndLogDet> {
    TF_ASSIGN_OR_RETURN(Shape a_shape, a.builder()->GetShape(a));
    auto qr = Qr(a);

    int64 m = ShapeUtil::GetDimension(a_shape, -2);
    int64 n = ShapeUtil::GetDimension(a_shape, -1);
    if (m != n) {
      return InvalidArgument(
          "Arguments to logdet must be (batched) square matrices, got: %s",
          a_shape.ToString());
    }
    // Get the sign and logarithm of the determinant based on the values along
    // the diagonal of R and the number of zeros in taus.
    auto log_abs_det = Einsum(Log(Abs(qr.q_and_r)), "...aa->...");
    auto sign_diag = Reduce(
        Sign(Einsum(qr.q_and_r, "...aa->...a")),
        One(a.builder(), a_shape.element_type()),
        CreateScalarMultiplyComputation(a_shape.element_type(), a.builder()),
        {a_shape.rank() - 2});
    auto sliced_taus = SliceInMinorDims(qr.taus, {0}, {n - 1});
    auto sign_taus = Reduce(
        Select(Ne(sliced_taus, ZerosLike(sliced_taus)),
               FullLike(sliced_taus, -1), FullLike(sliced_taus, 1)),
        One(a.builder(), a_shape.element_type()),
        CreateScalarMultiplyComputation(a_shape.element_type(), a.builder()),
        {a_shape.rank() - 2});
    return SignAndLogDet{sign_diag * sign_taus, log_abs_det};
  }();
  if (!result.ok()) {
    XlaOp error = a.builder()->ReportError(result.status());
    return SignAndLogDet{error, error};
  }
  return result.ValueOrDie();
}

XlaOp LogDet(XlaOp a) {
  SignAndLogDet slogdet = SLogDet(a);
  return Select(
      Ge(slogdet.sign, ZerosLike(slogdet.sign)), slogdet.logdet,
      FullLike(slogdet.logdet, std::numeric_limits<float>::quiet_NaN()));
}

}  // namespace xla
