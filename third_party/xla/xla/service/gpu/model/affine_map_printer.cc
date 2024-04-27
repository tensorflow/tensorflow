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

#include "xla/service/gpu/model/affine_map_printer.h"

#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace xla {
namespace gpu {
namespace {

using mlir::AffineBinaryOpExpr;
using mlir::AffineConstantExpr;
using mlir::AffineDimExpr;
using mlir::AffineExpr;
using mlir::AffineExprKind;
using mlir::AffineMap;
using mlir::AffineSymbolExpr;

}  // namespace

AffineMapPrinter::AffineMapPrinter(
    absl::Span<const std::string_view> dim_names,
    absl::Span<const std::string_view> symbol_names) {
  dim_id_to_name_.reserve(dim_names.size());
  for (const auto& [index, name] : llvm::enumerate(dim_names)) {
    dim_id_to_name_[index] = name;
  }
  symbol_id_to_name_.reserve(symbol_names.size());
  for (const auto& [index, name] : llvm::enumerate(symbol_names)) {
    symbol_id_to_name_[index] = name;
  }
}

void AffineMapPrinter::Print(std::ostream& out, AffineMap affine_map) const {
  out << ToString(affine_map);
}

std::string AffineMapPrinter::ToString(AffineMap affine_map) const {
  std::string s;
  llvm::raw_string_ostream ss(s);

  if (dim_id_to_name_.empty() && symbol_id_to_name_.empty()) {
    affine_map.print(ss);
    return s;
  }
  // Dimension identifiers.
  int dim_count = affine_map.getNumDims();
  ss << '(';
  for (int i = 0; i < dim_count - 1; ++i) {
    ss << GetDimensionName(i) << ", ";
  }
  if (dim_count >= 1) {
    ss << GetDimensionName(dim_count - 1);
  }
  ss << ')';
  // Symbolic identifiers.
  int symbol_count = affine_map.getNumSymbols();
  if (symbol_count != 0) {
    ss << '[';
    for (unsigned i = 0; i < symbol_count - 1; ++i) {
      ss << GetSymbolName(i) << ", ";
    }
    if (affine_map.getNumSymbols() >= 1) {
      ss << GetSymbolName(symbol_count - 1);
    }
    ss << ']';
  }
  // Result affine expressions.
  ss << " -> (";
  llvm::interleaveComma(affine_map.getResults(), ss, [&](AffineExpr expr) {
    PrintExprImpl(expr, /*add_parentheses=*/false, ss);
  });
  ss << ')';
  return s;
}

void AffineMapPrinter::Print(std::ostream& out,
                             mlir::AffineExpr affine_expr) const {
  out << ToString(affine_expr);
}

std::string AffineMapPrinter::ToString(mlir::AffineExpr affine_expr) const {
  std::string s;
  llvm::raw_string_ostream ss(s);
  PrintExprImpl(affine_expr, /*add_parentheses=*/false, ss);
  return s;
}

void AffineMapPrinter::PrintExprImpl(const mlir::AffineExpr affine_expr,
                                     bool add_parentheses,
                                     llvm::raw_ostream& os) const {
  const char* binopSpelling = nullptr;
  switch (affine_expr.getKind()) {
    case AffineExprKind::SymbolId: {
      unsigned symbol_id =
          mlir::cast<AffineSymbolExpr>(affine_expr).getPosition();
      os << GetSymbolName(symbol_id);
      return;
    }
    case AffineExprKind::DimId: {
      unsigned dim_id = mlir::cast<AffineDimExpr>(affine_expr).getPosition();
      os << GetDimensionName(dim_id);
      return;
    }
    case AffineExprKind::Constant:
      os << mlir::cast<AffineConstantExpr>(affine_expr).getValue();
      return;
    case AffineExprKind::Add:
      binopSpelling = " + ";
      break;
    case AffineExprKind::Mul:
      binopSpelling = " * ";
      break;
    case AffineExprKind::FloorDiv:
      binopSpelling = " floordiv ";
      break;
    case AffineExprKind::CeilDiv:
      binopSpelling = " ceildiv ";
      break;
    case AffineExprKind::Mod:
      binopSpelling = " mod ";
      break;
  }

  auto binOp = mlir::cast<AffineBinaryOpExpr>(affine_expr);
  AffineExpr lhsExpr = binOp.getLHS();
  AffineExpr rhsExpr = binOp.getRHS();

  // Handle tightly binding binary operators.
  if (binOp.getKind() != AffineExprKind::Add) {
    if (add_parentheses) {
      os << '(';
    }

    // Pretty print multiplication with -1.
    auto rhsConst = mlir::dyn_cast<AffineConstantExpr>(rhsExpr);
    if (rhsConst && binOp.getKind() == AffineExprKind::Mul &&
        rhsConst.getValue() == -1) {
      os << "-";
      PrintExprImpl(lhsExpr, /*add_parentheses=*/true, os);
      if (add_parentheses) {
        os << ')';
      }
      return;
    }

    PrintExprImpl(lhsExpr, /*add_parentheses=*/true, os);

    os << binopSpelling;
    PrintExprImpl(rhsExpr, /*add_parentheses=*/true, os);

    if (add_parentheses) {
      os << ')';
    }
    return;
  }

  // Print out special "pretty" forms for add.
  if (add_parentheses) {
    os << '(';
  }

  // Pretty print addition to a product that has a negative operand as a
  // subtraction.
  if (auto rhs = mlir::dyn_cast<AffineBinaryOpExpr>(rhsExpr)) {
    if (rhs.getKind() == AffineExprKind::Mul) {
      AffineExpr rrhsExpr = rhs.getRHS();
      if (auto rrhs = mlir::dyn_cast<AffineConstantExpr>(rrhsExpr)) {
        if (rrhs.getValue() == -1) {
          PrintExprImpl(lhsExpr, /*add_parentheses=*/false, os);
          os << " - ";
          if (rhs.getLHS().getKind() == AffineExprKind::Add) {
            PrintExprImpl(rhs.getLHS(), /*add_parentheses=*/true, os);
          } else {
            PrintExprImpl(rhs.getLHS(), /*add_parentheses=*/false, os);
          }

          if (add_parentheses) {
            os << ')';
          }
          return;
        }

        if (rrhs.getValue() < -1) {
          PrintExprImpl(lhsExpr, /*add_parentheses=*/false, os);
          os << " - ";
          PrintExprImpl(rhs.getLHS(), /*add_parentheses=*/true, os);
          os << " * " << -rrhs.getValue();
          if (add_parentheses) {
            os << ')';
          }
          return;
        }
      }
    }
  }

  // Pretty print addition to a negative number as a subtraction.
  if (auto rhsConst = mlir::dyn_cast<AffineConstantExpr>(rhsExpr)) {
    if (rhsConst.getValue() < 0) {
      PrintExprImpl(lhsExpr, /*add_parentheses=*/false, os);
      os << " - " << -rhsConst.getValue();
      if (add_parentheses) {
        os << ')';
      }
      return;
    }
  }

  PrintExprImpl(lhsExpr, /*add_parentheses=*/false, os);

  os << " + ";
  PrintExprImpl(rhsExpr, /*add_parentheses=*/false, os);

  if (add_parentheses) {
    os << ')';
  }
}

void AffineMapPrinter::SetSymbolName(int64_t symbol_id, llvm::StringRef name) {
  symbol_id_to_name_[symbol_id] = name;
}

void AffineMapPrinter::SetDimensionName(int64_t dim_id, llvm::StringRef name) {
  dim_id_to_name_[dim_id] = name;
}

std::string AffineMapPrinter::GetSymbolName(int64_t symbol_id) const {
  auto it = symbol_id_to_name_.find(symbol_id);
  if (it == symbol_id_to_name_.end()) {
    return absl::StrCat("s", symbol_id);
  }
  return it->second;
}

std::string AffineMapPrinter::GetDimensionName(int64_t dim_id) const {
  auto it = dim_id_to_name_.find(dim_id);
  if (it == dim_id_to_name_.end()) {
    return absl::StrCat("d", dim_id);
  }
  return it->second;
}

}  // namespace gpu
}  // namespace xla
