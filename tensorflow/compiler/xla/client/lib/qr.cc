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

#include "tensorflow/compiler/xla/client/lib/qr.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/loops.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

QrDecomposition Qr(XlaOp a) {
  auto result = [&]() -> StatusOr<QrDecomposition> {
    XlaBuilder* builder = a.builder();
    TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
    const int num_dims = a_shape.rank();
    if (num_dims < 2) {
      return InvalidArgument(
          "Arguments to QR must have rank >= 2: got shape %s",
          a_shape.ToString());
    }
    const int64_t m = ShapeUtil::GetDimension(a_shape, -2);
    const int64_t n = ShapeUtil::GetDimension(a_shape, -1);

    std::vector<int64_t> taus_dims(a_shape.dimensions().begin(),
                                   a_shape.dimensions().end());
    taus_dims.pop_back();
    taus_dims.back() = std::min(m, n);
    auto taus_shape = ShapeUtil::MakeShape(a_shape.element_type(), taus_dims);

    Shape qr_shape = ShapeUtil::MakeTupleShape({a_shape, taus_shape});
    auto qr = CustomCall(a.builder(), "Qr", {a}, qr_shape);
    a = GetTupleElement(qr, 0);
    auto taus = GetTupleElement(qr, 1);

    return QrDecomposition{a, taus};
  }();
  if (!result.ok()) {
    XlaOp error = a.builder()->ReportError(result.status());
    return QrDecomposition{error, error};
  }
  return result.value();
}

XlaOp ProductOfElementaryHouseholderReflectors(XlaOp a, XlaOp taus) {
  XlaBuilder* builder = a.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
    TF_ASSIGN_OR_RETURN(Shape taus_shape, builder->GetShape(taus));
    if (a_shape.rank() < 2) {
      return InvalidArgument(
          "Matrix `a` must have >= 2 dimensions: got shape %s",
          a_shape.ToString());
    }
    if (taus_shape.rank() + 1 != a_shape.rank()) {
      return InvalidArgument(
          "Matrix `taus` must have one fewer dimension than `a`: got shapes "
          "%s and %s",
          taus_shape.ToString(), a_shape.ToString());
    }
    const int64_t m = ShapeUtil::GetDimension(a_shape, -2);
    const int64_t n = ShapeUtil::GetDimension(a_shape, -1);
    if (m < n) {
      return InvalidArgument(
          "Argument to product of elementary Householder "
          "reflectors must have m >= n, got shape %s",
          a_shape.ToString());
    }
    absl::Span<const int64_t> a_batch_dims =
        absl::MakeConstSpan(a_shape.dimensions().begin(),
                            a_shape.dimensions().begin() + a_shape.rank() - 2);
    absl::Span<const int64_t> taus_batch_dims = absl::MakeConstSpan(
        taus_shape.dimensions().begin(),
        taus_shape.dimensions().begin() + taus_shape.rank() - 1);
    const int64_t k = ShapeUtil::GetDimension(taus_shape, -1);
    if (a_shape.element_type() != taus_shape.element_type() ||
        a_batch_dims != taus_batch_dims || k > n) {
      return InvalidArgument("Invalid shape for `taus`, got a=%s and taus=%s",
                             taus_shape.ToString(), a_shape.ToString());
    }
    return CustomCall(a.builder(), "ProductOfElementaryHouseholderReflectors",
                      {a, taus}, a_shape);
  });
}

void QrExplicit(XlaOp a, bool full_matrices, XlaOp& q, XlaOp& r) {
  StatusOr<Shape> a_shape_or = a.builder()->GetShape(a);
  if (!a_shape_or.ok()) {
    q = a.builder()->ReportError(a_shape_or.status());
    r = q;
    return;
  }
  Shape a_shape = a_shape_or.value();
  const int64_t m = ShapeUtil::GetDimension(a_shape, -2);
  const int64_t n = ShapeUtil::GetDimension(a_shape, -1);
  const int64_t p = std::min(m, n);

  auto qr = Qr(a);
  if (full_matrices) {
    XlaOp t;
    if (m < n) {
      t = SliceInMinorDims(qr.q_and_r, {0, 0}, {m, m});
    } else {
      t = PadInDim(qr.q_and_r, Zero(a.builder(), a_shape.element_type()),
                   a_shape.dimensions_size() - 1, /*pad_lo=*/0,
                   /*pad_hi=*/m - n);
    }
    q = ProductOfElementaryHouseholderReflectors(t, qr.taus);
    r = UpperTriangle(qr.q_and_r);
  } else {
    XlaOp t;
    if (m < n) {
      t = SliceInMinorDims(qr.q_and_r, {0, 0}, {m, m});
    } else {
      t = qr.q_and_r;
    }
    q = ProductOfElementaryHouseholderReflectors(t, qr.taus);
    q = SliceInMinorDims(q, {0, 0}, {m, p});
    r = UpperTriangle(SliceInMinorDims(qr.q_and_r, {0, 0}, {p, n}));
  }
}

}  // namespace xla
