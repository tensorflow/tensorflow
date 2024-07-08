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
// batch/feature/input_feature/output_feature dims. This class heiarchy
// reflects this.
class Layout {
 public:
  llvm::ArrayRef<int64_t> Spatials() const { return spatials_; }

  int64_t NumSpatials() const { return spatials_.size(); }

  int64_t Rank() const { return NumSpatials() + 2; }

 protected:
  // This class should not be instantiated directly, so protect constructor.
  Layout(int64_t special_dim1, int64_t special_dim2, ArrayRef<int64_t> spatials)
      : special_dim1_(special_dim1),
        special_dim2_(special_dim2),
        spatials_(spatials) {}

  // Gets index of first special dim.
  int64_t SpecialDim1() const { return special_dim1_; }

  // Gets dim value of first special dim from shape.
  int64_t SpecialDim1(llvm::ArrayRef<int64_t> shape) const {
    return shape[special_dim1_];
  }

  // Gets index of second special dim.
  int64_t SpecialDim2() const { return special_dim2_; }

  // Gets dim value of second special dim from shape.
  int64_t SpecialDim2(llvm::ArrayRef<int64_t> shape) const {
    return shape[special_dim2_];
  }

  bool IsTFLNativeLayout() const;

 private:
  int64_t special_dim1_;
  int64_t special_dim2_;
  llvm::ArrayRef<int64_t> spatials_;
};

// Layout wrapper for input/output mhlo.convolution parameters.
class IOLayout : public Layout {
 public:
  IOLayout(int64_t special_dim1, int64_t special_dim2,
           ArrayRef<int64_t> spatials)
      : Layout(special_dim1, special_dim2, spatials) {}

  int64_t Batch() const { return SpecialDim1(); }

  int64_t Batch(llvm::ArrayRef<int64_t> shape) const {
    return SpecialDim1(shape);
  }

  int64_t Feature() const { return SpecialDim2(); }

  int64_t Feature(llvm::ArrayRef<int64_t> shape) const {
    return SpecialDim2(shape);
  }

  bool IsNHWC() const { return IsTFLNativeLayout(); }
};

// Layout wrapper for kernel mhlo.convolution parameters.
class KernelLayout : public Layout {
 public:
  KernelLayout(int64_t special_dim1, int64_t special_dim2,
               ArrayRef<int64_t> spatials)
      : Layout(special_dim1, special_dim2, spatials) {}

  int64_t OutFeature() const { return SpecialDim1(); }

  int64_t OutFeature(llvm::ArrayRef<int64_t> shape) const {
    return SpecialDim1(shape);
  }

  int64_t InFeature() const { return SpecialDim2(); }

  int64_t InFeature(llvm::ArrayRef<int64_t> shape) const {
    return SpecialDim2(shape);
  }

  bool IsOHWI() const { return IsTFLNativeLayout(); }
};

class ConvLayouts {
 public:
  const IOLayout& Input() const { return input_; }

  const KernelLayout& Kernel() const { return kernel_; }

  const IOLayout& Output() const { return output_; }

  explicit ConvLayouts(mhlo::ConvDimensionNumbersAttr dnums)
      : input_(IOLayout{dnums.getInputBatchDimension(),
                        dnums.getInputFeatureDimension(),
                        dnums.getInputSpatialDimensions()}),
        kernel_(KernelLayout{dnums.getKernelOutputFeatureDimension(),
                             dnums.getKernelInputFeatureDimension(),
                             dnums.getKernelSpatialDimensions()}),
        output_(IOLayout{dnums.getOutputBatchDimension(),
                         dnums.getOutputFeatureDimension(),
                         dnums.getOutputSpatialDimensions()}) {}

 private:
  IOLayout input_;
  KernelLayout kernel_;
  IOLayout output_;
};

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

  // Layout for input, kernel, output.
  const ConvLayouts& Layouts() const { return layouts_; }

  int64_t BatchGroupCount() const { return batch_group_count_; }

  int64_t FeatureGroupCount() const { return feature_group_count_; }

  // int for each spatial dim. Default 1.
  llvm::ArrayRef<int64_t> InputDilations() const { return input_dilations_; }

  // int for each spatial dim. Default 1.
  llvm::ArrayRef<int64_t> KernelDilations() const { return kernel_dilations_; }

  // bool for each spatial dim. Default false.
  llvm::ArrayRef<bool> WindowReversal() const { return window_reversal_; }

  llvm::ArrayRef<int64_t> InputShape() const { return input_shape_; }

  llvm::ArrayRef<int64_t> KernelShape() const { return kernel_shape_; }

  llvm::ArrayRef<int64_t> OutputShape() const { return output_shape_; }

  mlir::Type ElementType() const { return element_type_; }

  explicit ConvData(mhlo::ConvolutionOp op);

 private:
  llvm::SmallVector<int64_t, 2> strides_;

  llvm::SmallVector<DimPadding, 2> padding_;

  ConvLayouts layouts_;

  int64_t batch_group_count_;
  int64_t feature_group_count_;

  llvm::SmallVector<int64_t, 2> input_dilations_;
  llvm::SmallVector<int64_t, 2> kernel_dilations_;

  llvm::SmallVector<bool, 2> window_reversal_;

  llvm::ArrayRef<int64_t> input_shape_;
  llvm::ArrayRef<int64_t> kernel_shape_;
  llvm::ArrayRef<int64_t> output_shape_;

  mlir::Type element_type_;
};

}  // namespace mlir::odml

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_CONV_UTIL_H_
