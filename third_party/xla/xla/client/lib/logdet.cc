/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/client/lib/logdet.h"

#include <limits>
#include <memory>
#include <vector>

#include "xla/client/lib/arithmetic.h"
#include "xla/client/lib/constants.h"
#include "xla/client/lib/loops.h"
#include "xla/client/lib/math.h"
#include "xla/client/lib/matrix.h"
#include "xla/client/lib/qr.h"
#include "xla/client/lib/slicing.h"
#include "xla/client/xla_builder.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "tsl/platform/errors.h"

namespace xla {

SignAndLogDet SLogDet(XlaOp a) {
  absl::StatusOr<SignAndLogDet> result =
      [&]() -> absl::StatusOr<SignAndLogDet> {
    TF_ASSIGN_OR_RETURN(Shape a_shape, a.builder()->GetShape(a));
    auto qr = Qr(a);

    int64_t m = ShapeUtil::GetDimension(a_shape, -2);
    int64_t n = ShapeUtil::GetDimension(a_shape, -1);
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
  return result.value();
}

XlaOp LogDet(XlaOp a) {
  SignAndLogDet slogdet = SLogDet(a);
  return Select(
      Ge(slogdet.sign, ZerosLike(slogdet.sign)), slogdet.logdet,
      FullLike(slogdet.logdet, std::numeric_limits<float>::quiet_NaN()));
}

}  // namespace xla
