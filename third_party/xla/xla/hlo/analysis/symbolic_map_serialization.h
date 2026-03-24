/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_HLO_ANALYSIS_SYMBOLIC_MAP_SERIALIZATION_H_
#define XLA_HLO_ANALYSIS_SYMBOLIC_MAP_SERIALIZATION_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"

namespace xla {

// Returns string representation of binary functions like `max`, `min`,
// `floordiv`, etc.
std::string GetBinaryOpString(SymbolicExprType type);

// Prints symbolic expression to stream. If num_dims is provided, then the first
// num_dims variables are dimensions, and the rest are symbols. If var_names is
// provided, then variable names are taken from it.
void Print(SymbolicExpr expr, llvm::raw_ostream& os,
           std::optional<int64_t> num_dims = std::nullopt);
void Print(SymbolicExpr expr, llvm::raw_ostream& os,
           absl::Span<const std::string> var_names);

// Prints symbolic map to stream.
void Print(const SymbolicMap& map, llvm::raw_ostream& os);

// Parses symbolic map from string.
SymbolicMap ParseSymbolicMap(absl::string_view serialized_symbolic_map,
                             mlir::MLIRContext* mlir_context);

// Parses a symbolic map from `map_str`. Advances `map_str` past the
// parsed map. Returns the parsed map or null if parsing failed.
SymbolicMap ParseSymbolicMapAndAdvance(absl::string_view* map_str,
                                       mlir::MLIRContext* mlir_context);

// Parse a symbolic expression from `expr_str`. `num_dims` specifies the
// number of dimension variables. It is used to determine a variable index from
// a symbol id. For example, if `num_dims` is 2, 's0' parses to variable index
// 2, 's1' to 3, etc.
SymbolicExpr ParseSymbolicExpr(absl::string_view expr_str,
                               mlir::MLIRContext* mlir_context,
                               std::optional<int64_t> num_dims = std::nullopt);
// Parses a symbolic expression from `expr_str`. Advances `expr_str` past the
// parsed expression. Returns the parsed expression or null if parsing failed.
SymbolicExpr ParseSymbolicExprAndAdvance(
    absl::string_view* expr_str, mlir::MLIRContext* mlir_context,
    std::optional<int64_t> num_dims = std::nullopt);
// Uses `variable_map` to resolve variable names to symbolic expressions. If a
// variable name is found in the map, the corresponding SymbolicExpr is used.
SymbolicExpr ParseSymbolicExprAndAdvance(
    absl::string_view* expr_str, mlir::MLIRContext* mlir_context,
    const llvm::DenseMap<llvm::StringRef, SymbolicExpr>& variable_map);

}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_SYMBOLIC_MAP_SERIALIZATION_H_
