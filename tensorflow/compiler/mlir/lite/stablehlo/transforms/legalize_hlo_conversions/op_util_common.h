/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_OP_UTIL_COMMON_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_OP_UTIL_COMMON_H_

#include <cstdint>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir::odml {

// Class that encodes the "layout" of a tensor. Layouts, generically
// are some naming of the dimensions of a tensor. In all cases, 2 dimensions
// are "special" (e.g. batch / feature) and the rest are referred to as "spatial
// dims". When the special dims are batch and feature, batch is special dim 1
// and feature is special dim 2. When special dims are input and output features
// (conv filter), input features is special dim 1 and output features is special
// dim 2.
class Layout {
 public:
  llvm::ArrayRef<int64_t> Spatials() const { return spatials_; }

  int64_t NumSpatials() const { return spatials_.size(); }

  int64_t Rank() const { return NumSpatials() + 2; }

  Layout(int64_t special_dim1, int64_t special_dim2, ArrayRef<int64_t> spatials)
      : special_dim1_(special_dim1),
        special_dim2_(special_dim2),
        spatials_(spatials) {}

  // TODO: b/351437662 - Consider just using 2 arrays for the case where
  // there are more than 2 special dims.
  int64_t SpecialDim1() const { return special_dim1_; }

  // Conveniance accesor for getting the dimension size of the first
  // special dimension from a shape.
  int64_t SpecialDim1(llvm::ArrayRef<int64_t> shape) const {
    return shape[special_dim1_];
  }

  int64_t SpecialDim2() const { return special_dim2_; }

  // Convenience accesor for getting the dimension size of the second
  // special dimension from a shape.
  int64_t SpecialDim2(llvm::ArrayRef<int64_t> shape) const {
    return shape[special_dim2_];
  }

  // Conveniance method for equality checking special dims.
  bool HasSpecialDims(int64_t special_dim1, int64_t special_dim2) const;

  // Determines if the spatial dimensions are all adjacent and in
  // ascending order.
  bool AreSpatialsIota() const;

  // Gets a "permutation array" to be used for transposing a tensor
  // of "this" layout to the given layout. A permutation array is some
  // permutation of [0, 1, i...] for i < rank(layout). Assumes
  // "this" and given layout have the same rank.
  llvm::SmallVector<int64_t, 4> GetPermForReLayout(
      const Layout& to_layout) const;

  // Permutes given shape based on the permutaion implied to take this Layout to
  // the given one.
  llvm::SmallVector<int64_t, 4> PermuteShape(const Layout& to_layout,
                                             ArrayRef<int64_t> shape) const;

  bool operator==(const Layout& other) const {
    return SpecialDim1() == other.SpecialDim1() &&
           SpecialDim2() == other.SpecialDim2() &&
           Spatials() == other.Spatials();
  }

  bool operator!=(const Layout& other) const { return !(*this == other); }

 private:
  int64_t special_dim1_;
  int64_t special_dim2_;
  llvm::SmallVector<int64_t> spatials_;
};

// Wrapper for the padding attrs along a single dimension.
class DimPadding {
 public:
  int64_t Hi() const { return hi_; }

  int64_t Lo() const { return lo_; }

  bool Trivial() const { return Hi() == 0 && Lo() == 0; }

  DimPadding(int64_t lo, int64_t hi) : lo_(lo), hi_(hi) {}

 private:
  int64_t lo_;
  int64_t hi_;
};

inline llvm::SmallVector<int64_t> UnrollI64Splat(DenseElementsAttr data) {
  if (!data.isSplat()) {
    return llvm::SmallVector<int64_t>(data.getValues<int64_t>());
  }
  return llvm::SmallVector<int64_t>(data.getType().getNumElements(),
                                    data.getSplatValue<int64_t>());
}

// Resolves optional strides or dilations attributes. If not present,
// will return trivial 1's vector.
llvm::SmallVector<int64_t, 4> ResolveStridesOrDilations(
    int64_t rank, std::optional<mlir::DenseIntElementsAttr> opt_attr);

// Resolves optional paddings attributes. If not present, will return
// trivial [0, 0] paddings on each dim.
llvm::SmallVector<DimPadding, 4> ResolvePadding(
    int64_t rank, std::optional<mlir::DenseIntElementsAttr> opt_padding);

// Does the padding correspond to "SAME" on given dimension configuration.
// Assumes given dimension configuration is well formed.
bool IsSamePaddingOnDim(int64_t in, int64_t dilate, int64_t stride, int64_t k,
                        const DimPadding& pad);

template <typename T>
inline DenseElementsAttr BuildScalarDense(Type e_type, T val) {
  auto type = RankedTensorType::get({}, e_type);
  return DenseElementsAttr::get(type, val);
}

}  // namespace mlir::odml

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_OP_UTIL_COMMON_H_
