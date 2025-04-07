/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/hlo/builder/lib/lu_decomposition.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {

LuDecompositionResult LuDecomposition(XlaOp a) {
  XlaBuilder* builder = a.builder();
  XlaOp result = builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
    const int ndims = a_shape.dimensions().size();
    TF_RET_CHECK(ndims >= 2);
    const int64_t m = ShapeUtil::GetDimension(a_shape, -2);
    const int64_t n = ShapeUtil::GetDimension(a_shape, -1);
    const int num_batch_dims = a_shape.dimensions().size() - 2;
    const std::vector<int64_t> batch_dims(
        a_shape.dimensions().begin(),
        a_shape.dimensions().begin() + num_batch_dims);

    std::vector<int64_t> pivot_dims = batch_dims;
    pivot_dims.push_back(std::min(m, n));
    std::vector<int64_t> perm_dims = batch_dims;
    perm_dims.push_back(m);
    Shape lu_shape = ShapeUtil::MakeTupleShape(
        {a_shape, ShapeUtil::MakeShape(S32, pivot_dims),
         ShapeUtil::MakeShape(S32, perm_dims)});
    // The TPU compiler has a rewrite pass that lowers an LuDecomposition
    // CustomCall.
    // TODO(phawkins): upgrade LU decomposition to a first-class HLO operator
    // and implement it on other backends.
    return CustomCall(a.builder(), "LuDecomposition", {a}, lu_shape);
  });
  return LuDecompositionResult{GetTupleElement(result, 0),
                               GetTupleElement(result, 1),
                               GetTupleElement(result, 2)};
}

}  // namespace xla
