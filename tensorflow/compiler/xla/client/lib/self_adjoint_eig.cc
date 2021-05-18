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

#include "tensorflow/compiler/xla/client/lib/self_adjoint_eig.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/loops.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

SelfAdjointEigResult SelfAdjointEig(XlaOp a, bool lower, int64 max_iter,
                                    float tol, bool sort_eigenvalues) {
  XlaBuilder* builder = a.builder();
  XlaOp result = builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
    const int64 num_dims = a_shape.rank();
    if (num_dims < 2) {
      return InvalidArgument(
          "Arguments to Eigen decomposition must have rank >= 2: got shape %s.",
          a_shape.ToString());
    }
    PrimitiveType type = a_shape.element_type();
    if (!primitive_util::IsFloatingPointType(type) &&
        !primitive_util::IsComplexType(type)) {
      return InvalidArgument(
          "Type of the input matrix must be floating point "
          "or complex: got %s.",
          a_shape.ToString());
    }

    const int64 m = ShapeUtil::GetDimension(a_shape, -2);
    const int64 n = ShapeUtil::GetDimension(a_shape, -1);

    if (m != n) {
      return InvalidArgument(
          "Arguments to symmetric eigendecomposition must be square matrices: "
          "got shape (%d, %d).",
          m, n);
    }

    const int num_batch_dims = a_shape.dimensions().size() - 2;
    const std::vector<int64> batch_dims(
        a_shape.dimensions().begin(),
        a_shape.dimensions().begin() + num_batch_dims);

    PrimitiveType eigvals_type =
        primitive_util::IsComplexType(type)
            ? primitive_util::ComplexComponentType(type)
            : type;
    std::vector<int64> eigvals_dims = batch_dims;
    eigvals_dims.push_back(m);
    Shape eigh_shape = ShapeUtil::MakeTupleShape(
        {a_shape, ShapeUtil::MakeShape(eigvals_type, eigvals_dims)});
    // TODO(phawkins): upgrade Eigh decomposition to a first-class HLO operator.
    std::string opaque =
        absl::StrFormat("%d,%d,%d,%f", lower, sort_eigenvalues, max_iter, tol);
    return CustomCall(a.builder(), "Eigh", {a}, eigh_shape, opaque);
  });
  return SelfAdjointEigResult{GetTupleElement(result, 0),
                              GetTupleElement(result, 1)};
}

}  // namespace xla
