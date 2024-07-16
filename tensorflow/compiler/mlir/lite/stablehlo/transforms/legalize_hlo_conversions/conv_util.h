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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_CONV_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_CONV_UTIL_H_

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

// Helpers for working with mhlo.convolution attrs in the mlir api as
// native cc types.

namespace mlir::odml {

// Generic class that wraps the "layout" of a convolution parameter.
// Both kernel (e.g. [o, 0, 1, i]) and input/output (e.g. [b, 0, 1, f])
// share the same structure just with different terminology for the
// batch/feature/input_feature/output_feature dims.
class Layout {
 public:
  llvm::ArrayRef<int64_t> Spatials() const { return spatials_; }

  int64_t NumSpatials() const { return spatials_.size(); }

  int64_t Rank() const { return NumSpatials() + 2; }

  // This class should not be instantiated directly, so protect constructor.
  Layout(int64_t special_dim1, int64_t special_dim2, ArrayRef<int64_t> spatials)
      : special_dim1_(special_dim1),
        special_dim2_(special_dim2),
        spatials_(spatials) {}

  // Gets index of first special dim. The batch dim for input and outputs,
  // or the output feature dim for the kernel.
  int64_t SpecialDim1() const { return special_dim1_; }

  // Conveniance accesor for getting the dimension size of the first
  // special dimension from a shape.
  int64_t SpecialDim1(llvm::ArrayRef<int64_t> shape) const {
    return shape[special_dim1_];
  }

  // Gets index of second special dim. The feature dim for input and outputs,
  // or the input feature dim for the kernel.
  int64_t SpecialDim2() const { return special_dim2_; }

  // Convenience accesor for getting the dimension size of the second
  // special dimension from a shape.
  int64_t SpecialDim2(llvm::ArrayRef<int64_t> shape) const {
    return shape[special_dim2_];
  }

  // Conveniance method for equality checking special dims.
  bool HasSpecialDims(int64_t special_dim1, int64_t special_dim2) const;

  // Determines if the spatial dimensions are all adjacent and in
  // ascending order (HWD).
  bool AreSpatialsIota() const;

 private:
  int64_t special_dim1_;
  int64_t special_dim2_;
  llvm::SmallVector<int64_t> spatials_;
};

// [b, spatials..., f] / [o, spatials..., i].
inline bool IsTFLNativeLayout(const Layout& layout) {
  return layout.AreSpatialsIota() &&
         layout.HasSpecialDims(0, layout.Rank() - 1);
}

// Wrapper for the padding attrs along a single dimension.
class DimPadding {
 public:
  int64_t Hi() const { return hi_; }

  int64_t Lo() const { return lo_; }

  DimPadding(int64_t hi, int64_t lo) : hi_(hi), lo_(lo) {}

 private:
  int64_t hi_;
  int64_t lo_;
};

class ConvData {
 public:
  // int for each spatial dim. Default 1.
  llvm::ArrayRef<int64_t> Strides() const { return strides_; }

  // 2d array for each spatial dim. Default 0.
  llvm::ArrayRef<DimPadding> Padding() const { return padding_; }

  int64_t BatchGroupCount() const { return batch_group_count_; }

  int64_t FeatureGroupCount() const { return feature_group_count_; }

  // int for each spatial dim. Default 1.
  llvm::ArrayRef<int64_t> InputDilations() const { return input_dilations_; }

  // int for each spatial dim. Default 1.
  llvm::ArrayRef<int64_t> KernelDilations() const { return kernel_dilations_; }

  // bool for each spatial dim. Default false.
  llvm::ArrayRef<bool> WindowReversal() const { return window_reversal_; }

  llvm::ArrayRef<int64_t> InputShape() const { return input_shape_; }

  const Layout& InputLayout() const { return input_layout_; }

  llvm::ArrayRef<int64_t> KernelShape() const { return kernel_shape_; }

  const Layout& KernelLayout() const { return kernel_layout_; }

  llvm::ArrayRef<int64_t> OutputShape() const { return output_shape_; }

  const Layout& OutputLayout() const { return output_layout_; }

  mlir::Type ElementType() const { return element_type_; }

  explicit ConvData(mhlo::ConvolutionOp op);

 private:
  llvm::SmallVector<int64_t, 2> strides_;

  llvm::SmallVector<DimPadding, 2> padding_;

  llvm::SmallVector<int64_t, 2> input_dilations_;
  llvm::SmallVector<int64_t, 2> kernel_dilations_;

  llvm::SmallVector<bool, 2> window_reversal_;

  Layout input_layout_;
  Layout kernel_layout_;
  Layout output_layout_;

  llvm::SmallVector<int64_t, 4> input_shape_;
  llvm::SmallVector<int64_t, 4> kernel_shape_;
  llvm::SmallVector<int64_t, 4> output_shape_;

  int64_t batch_group_count_;
  int64_t feature_group_count_;

  mlir::Type element_type_;
};

}  // namespace mlir::odml

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_CONV_UTIL_H_
