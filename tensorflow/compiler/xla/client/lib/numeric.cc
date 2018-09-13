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

#include <numeric>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/numeric.h"

namespace xla {

XlaOp IdentityMatrix(XlaBuilder* builder, PrimitiveType type, int64 m,
                     int64 n) {
  auto a = Iota(builder, type, m);
  auto b = Iota(builder, type, n);
  auto indicator = Eq(a, Broadcast(b, {m}), /*broadcast_dimensions=*/{0});
  return ConvertElementType(indicator, type);
}

XlaOp GetMatrixDiagonal(XlaOp x) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64 n_dims = ShapeUtil::Rank(shape);
    TF_RET_CHECK(n_dims >= 2);
    const int64 m = shape.dimensions(n_dims - 2);
    const int64 n = shape.dimensions(n_dims - 1);
    absl::Span<const int64> major_dims =
        AsInt64Slice(shape.dimensions()).subspan(/*pos=*/0, /*len=*/n_dims - 2);
    auto a = Iota(builder, U32, n);
    auto b = Iota(builder, U32, m);
    auto indicator = Eq(b, Broadcast(a, {m}), /*broadcast_dimensions=*/{0});
    auto mask = Broadcast(indicator, major_dims);

    // TPUs don't support S64 add reduction at the moment. But fortunately
    // OR-reductions work just as well for integers.
    XlaComputation reducer =
        primitive_util::IsIntegralType(shape.element_type())
            ? CreateScalarOrComputation(shape.element_type(), builder)
            : CreateScalarAddComputation(shape.element_type(), builder);

    return Reduce(Select(mask, x, Zeros(builder, shape)), ScalarLike(x, 0),
                  reducer, {m >= n ? n_dims - 2 : n_dims - 1});
  });
}

XlaOp Triangle(XlaOp x, bool lower) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64 n_dims = ShapeUtil::Rank(shape);
    TF_RET_CHECK(n_dims >= 2);
    const int64 m = shape.dimensions(n_dims - 2);
    const int64 n = shape.dimensions(n_dims - 1);
    absl::Span<const int64> major_dims =
        AsInt64Slice(shape.dimensions()).subspan(/*pos=*/0, /*len=*/n_dims - 2);
    auto a = Iota(builder, U32, n);
    auto b = Iota(builder, U32, m);
    xla::XlaOp indicator;
    if (lower) {
      indicator = Ge(b, Broadcast(a, {m}), /*broadcast_dimensions=*/{0});
    } else {
      indicator = Le(b, Broadcast(a, {m}), /*broadcast_dimensions=*/{0});
    }
    auto mask = Broadcast(indicator, major_dims);

    return Select(mask, x, Zeros(builder, shape));
  });
}

XlaOp UpperTriangle(XlaOp x) { return Triangle(x, false); }

XlaOp LowerTriangle(XlaOp x) { return Triangle(x, true); }

}  // namespace xla
