/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MODEL_AFFINE_MAP_PRINTER_H_
#define XLA_SERVICE_GPU_MODEL_AFFINE_MAP_PRINTER_H_

#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>

#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project

namespace xla {
namespace gpu {

// AffineMapPrinter allows to "pretty print" mlir::AffineMap by setting custom
// symbol and dimension names.
class AffineMapPrinter {
 public:
  AffineMapPrinter() = default;
  AffineMapPrinter(AffineMapPrinter&& other) = default;
  AffineMapPrinter& operator=(AffineMapPrinter&& other) = default;
  AffineMapPrinter(absl::Span<const std::string_view> dim_names,
                   absl::Span<const std::string_view> symbol_names);

  void SetSymbolName(int64_t symbol_id, llvm::StringRef name);
  void SetDimensionName(int64_t dim_id, llvm::StringRef name);

  std::string GetSymbolName(int64_t symbol_id) const;
  std::string GetDimensionName(int64_t dim_id) const;

  void Print(std::ostream& out, mlir::AffineMap affine_map) const;
  std::string ToString(mlir::AffineMap affine_map) const;

  void Print(std::ostream& out, mlir::AffineExpr affine_expr) const;
  std::string ToString(mlir::AffineExpr affine_expr) const;

 private:
  void PrintExprImpl(mlir::AffineExpr affine_expr, bool add_parentheses,
                     llvm::raw_ostream& os) const;

  llvm::DenseMap<unsigned, std::string> dim_id_to_name_;
  llvm::DenseMap<unsigned, std::string> symbol_id_to_name_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_AFFINE_MAP_PRINTER_H_
