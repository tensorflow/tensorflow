/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_CLIENT_LIB_COMPARATORS_H_
#define XLA_CLIENT_LIB_COMPARATORS_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Creates a scalar less-than computation and returns it. The created
// computation has 2 * 'operand_types.size()' many parameters, where parameters
// 2 * i and 2 * i + 1 are a scalar with primitive type 'operand_types[i]'. The
// computation compares the first two parameters. For floating point types, a
// total order is created where
// -NaN < -infinity < ... < -0 < 0 < ... < infinity < NaN
XlaComputation CreateScalarLtComputation(
    const std::vector<PrimitiveType>& operand_types, XlaBuilder* builder);

// Creates a scalar greater-than computation and returns it. The created
// computation has 2 * 'operand_types.size()' many parameters, where parameters
// 2 * i and 2 * i + 1 are a scalar with primitive type 'operand_types[i]'. The
// computation compares the first two parameters. For floating point types, a
// total order is created where
// NaN > infinity > ... > 0 > -0 > ... > -infinity > -NaN
XlaComputation CreateScalarGtComputation(
    const std::vector<PrimitiveType>& operand_types, XlaBuilder* builder);

// Creates a scalar comparison computation and returns it. This function takes
// a vector of comparator functions to compare the operands where the function
// isn't nullopt with the specified comparator at that location.
XlaComputation CreateScalarComparisonComputation(
    const std::string& name, const std::vector<PrimitiveType>& operand_types,
    const std::vector<
        std::optional<XlaOp (*)(XlaOp, XlaOp, absl::Span<const int64_t>)>>&
        generators,
    XlaBuilder* builder);

}  // namespace xla

#endif  // XLA_CLIENT_LIB_COMPARATORS_H_
