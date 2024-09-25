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

#include <cstddef>
#include <cstdint>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/op_util_common.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

// Helpers for working with mhlo.convolution attrs in the mlir api as
// native cc types.

namespace mlir::odml {

class ConvView {
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

  explicit ConvView(mhlo::ConvolutionOp op);

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

inline bool HasSupportedRank(const ConvView& data) {
  return data.InputLayout().Rank() == 4 || data.InputLayout().Rank() == 5;
}

inline bool HasSupportedOutFeatureDims(const ConvView& data) {
  const int64_t kernel_out_features =
      data.KernelLayout().SpecialDim2(data.KernelShape());
  const int64_t out_features =
      data.OutputLayout().SpecialDim2(data.OutputShape());
  return kernel_out_features == out_features;
}

inline bool IsTrivialConv(const ConvView& data) {
  return llvm::all_of(data.InputDilations(), [](auto d) { return d == 1; });
}

//
// Supported non-trivial conv predicates
//=-----

bool MatchWithResizeBilinearOp(const ConvView& data, bool& align_corners);

inline bool MatchWithResizeBilinearOp(const ConvView& data) {
  bool align_corners = false;
  return MatchWithResizeBilinearOp(data, align_corners);
}

bool IsTransposeConvPaddingValid(mhlo::ConvolutionOp conv_op,
                                 size_t num_spatial_dims,
                                 const ArrayRef<int64_t>& strides,
                                 const ArrayRef<int64_t>& padding);

bool IsTransposeConvPaddingSame(mhlo::ConvolutionOp conv_op,
                                size_t num_spatial_dims,
                                const ArrayRef<int64_t>& strides,
                                const ArrayRef<int64_t>& padding);

inline bool IsSupportedNonTrivialConv(const ConvView& data) {
  // Only non-trivial 2d convolutions are supported.
  const bool valid_rank = data.InputLayout().Rank() == 4;

  // Negative padding is unsupported.
  bool has_nagative_padding = llvm::all_of(
      data.Padding(),
      [](const DimPadding& p) { return p.Hi() < 0 || p.Lo() < 0; });

  return (valid_rank && !IsTrivialConv(data) && !has_nagative_padding);
}

inline bool IsSupportedNonTrivialConv(mhlo::ConvolutionOp op) {
  const ConvView data(op);
  return IsSupportedNonTrivialConv(data);
}

//
// Standard conv predicates
//=-----

inline bool HasStandardConvInFeatureDims(const ConvView& data) {
  // kernel_in_features * feature_groups = input_features by definition.
  const int64_t input_features =
      data.InputLayout().SpecialDim2(data.InputShape());

  const bool trivial_kernel_in_features =
      data.FeatureGroupCount() == input_features;
  const bool is_grouped_conv = data.FeatureGroupCount() != 1;

  const int64_t rank = data.InputLayout().Rank();
  return !trivial_kernel_in_features && (!is_grouped_conv || rank == 4);
}

inline bool IsStandardConv(const ConvView& data) {
  return HasSupportedRank(data) && IsTrivialConv(data) &&
         HasStandardConvInFeatureDims(data) && HasSupportedOutFeatureDims(data);
}

// Does this convolution map to a standard conv_2d or conv_3d
// (not depthwise or tranpose conv)?
inline bool IsStandardConv(mhlo::ConvolutionOp op) {
  const ConvView data(op);
  return IsStandardConv(data);
}

//
// Depthwise conv predicates
//=-----

inline bool IsDepthwiseConv(const ConvView& data) {
  const bool valid_rank = data.InputLayout().Rank() == 4;
  if (!valid_rank || !HasSupportedOutFeatureDims(data) ||
      !IsTrivialConv(data)) {
    return false;
  }
  const int64_t in_channel_dim =
      data.InputLayout().SpecialDim2(data.InputShape());
  return data.FeatureGroupCount() == in_channel_dim;
}

// Does this convolution map to depthwise conv?
inline bool IsDepthwiseConv(mhlo::ConvolutionOp op) {
  const ConvView data(op);
  return IsDepthwiseConv(data);
}

//
// Tfl native layouts
//=-----

inline int64_t DnumRank(mhlo::ConvDimensionNumbersAttr dnums) {
  return dnums.getInputSpatialDimensions().size() + 2;
}

inline Layout GetTFLNativeInputOrOutputLayout(int64_t rank) {
  auto spatials = llvm::to_vector(llvm::seq<int64_t>(1, rank - 1));
  return Layout(0, rank - 1, spatials);
}

inline Layout GetTFLNativeInputOrOutputLayout(
    mhlo::ConvDimensionNumbersAttr dnums) {
  return GetTFLNativeInputOrOutputLayout((DnumRank(dnums)));
}

inline Layout GetTFLNativeStandardConvKernelLayout(int64_t rank) {
  if (rank != 5) {
    auto spatials = llvm::to_vector(llvm::seq<int64_t>(1, rank - 1));
    return Layout(rank - 1, 0, spatials);
  }
  auto spatials = llvm::to_vector(llvm::seq(rank - 2));
  return Layout(rank - 2, rank - 1, spatials);
}

inline Layout GetTFLNativeDepthwiseConvKernelLayout() {
  return Layout(0, 3, {1, 2});
}

inline Layout GetTFLNativeStandardConvKernelLayout(
    mhlo::ConvDimensionNumbersAttr dnums) {
  return GetTFLNativeStandardConvKernelLayout(DnumRank(dnums));
}

inline bool IsTFLNativeLayout(const ConvView& data) {
  const int64_t rank = data.KernelLayout().Rank();
  const auto native_io_layout = GetTFLNativeInputOrOutputLayout(rank);

  std::optional<Layout> native_kernel_layout = std::nullopt;
  if (IsDepthwiseConv(data)) {
    native_kernel_layout = GetTFLNativeDepthwiseConvKernelLayout();
  } else if (IsStandardConv(data) || IsSupportedNonTrivialConv(data)) {
    native_kernel_layout = GetTFLNativeStandardConvKernelLayout(rank);
  }
  if (!native_kernel_layout.has_value()) {
    return false;
  }

  return data.InputLayout() == native_io_layout &&
         data.KernelLayout() == *native_kernel_layout &&
         data.OutputLayout() == native_io_layout;
}

//
// ConvDimensionNumbers utils
//=-----

inline mhlo::ConvDimensionNumbersAttr CloneDnumsWithInputLayout(
    OpBuilder& b, mhlo::ConvDimensionNumbersAttr dnums, const Layout& layout) {
  return mhlo::ConvDimensionNumbersAttr::get(
      b.getContext(), layout.SpecialDim1(), layout.SpecialDim2(),
      layout.Spatials(), dnums.getKernelInputFeatureDimension(),
      dnums.getKernelOutputFeatureDimension(),
      dnums.getKernelSpatialDimensions(), dnums.getOutputBatchDimension(),
      dnums.getOutputFeatureDimension(), dnums.getOutputSpatialDimensions());
}

inline mhlo::ConvDimensionNumbersAttr CloneDnumsWithKernelLayout(
    OpBuilder& b, mhlo::ConvDimensionNumbersAttr dnums, const Layout& layout) {
  return mhlo::ConvDimensionNumbersAttr::get(
      b.getContext(), dnums.getInputBatchDimension(),
      dnums.getInputFeatureDimension(), dnums.getInputSpatialDimensions(),
      layout.SpecialDim1(), layout.SpecialDim2(), layout.Spatials(),
      dnums.getOutputBatchDimension(), dnums.getOutputFeatureDimension(),
      dnums.getOutputSpatialDimensions());
}

inline mhlo::ConvDimensionNumbersAttr CloneDnumsWithOutputLayout(
    OpBuilder& b, mhlo::ConvDimensionNumbersAttr dnums, const Layout& layout) {
  return mhlo::ConvDimensionNumbersAttr::get(
      b.getContext(), dnums.getInputBatchDimension(),
      dnums.getInputFeatureDimension(), dnums.getInputSpatialDimensions(),
      dnums.getKernelInputFeatureDimension(),
      dnums.getKernelOutputFeatureDimension(),
      dnums.getKernelSpatialDimensions(), layout.SpecialDim1(),
      layout.SpecialDim2(), layout.Spatials());
}

// Wraps the lhs of given conv op in an explicit pad op matching the same
// behavior implicit in the paddings attribute. Gets result of new pad op.
Value CreatePadOpFromConvPadding(OpBuilder& b, mhlo::ConvolutionOp op);

}  // namespace mlir::odml

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_CONV_UTIL_H_
