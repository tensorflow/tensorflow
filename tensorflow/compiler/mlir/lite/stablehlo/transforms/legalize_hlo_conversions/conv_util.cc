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
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/conv_util.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/op_util_common.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {

llvm::SmallVector<bool, 2> ResolveWindowReversal(
    const int64_t num_spatials,
    std::optional<mlir::DenseElementsAttr> opt_reversals) {
  if (!opt_reversals.has_value()) {
    return llvm::SmallVector<bool, 2>(num_spatials, false);
  }
  auto reversals = opt_reversals.value();
  if (reversals.isSplat()) {
    return llvm::SmallVector<bool, 2>(num_spatials,
                                      reversals.getSplatValue<bool>());
  }
  return llvm::SmallVector<bool, 2>(reversals.getValues<bool>());
}

ConvView::ConvView(mhlo::ConvolutionOp op)
    : input_layout_(
          Layout{op.getDimensionNumbers().getInputBatchDimension(),
                 op.getDimensionNumbers().getInputFeatureDimension(),
                 op.getDimensionNumbers().getInputSpatialDimensions()}),
      kernel_layout_(
          Layout{op.getDimensionNumbers().getKernelInputFeatureDimension(),
                 op.getDimensionNumbers().getKernelOutputFeatureDimension(),
                 op.getDimensionNumbers().getKernelSpatialDimensions()}),
      output_layout_(
          Layout{op.getDimensionNumbers().getOutputBatchDimension(),
                 op.getDimensionNumbers().getOutputFeatureDimension(),
                 op.getDimensionNumbers().getOutputSpatialDimensions()}),
      input_shape_(
          llvm::SmallVector<int64_t, 4>(op.getLhs().getType().getShape())),
      kernel_shape_(
          llvm::SmallVector<int64_t, 4>(op.getRhs().getType().getShape())),
      output_shape_(
          llvm::SmallVector<int64_t, 4>(op.getResult().getType().getShape())),
      batch_group_count_(op.getBatchGroupCount()),
      feature_group_count_(op.getFeatureGroupCount()),
      element_type_(op.getLhs().getType().getElementType()) {
  const int64_t num_spatials = InputLayout().NumSpatials();

  strides_ = ResolveStridesOrDilations(num_spatials, op.getWindowStrides());

  input_dilations_ =
      ResolveStridesOrDilations(num_spatials, op.getLhsDilation());
  kernel_dilations_ =
      ResolveStridesOrDilations(num_spatials, op.getRhsDilation());

  padding_ = ResolvePadding(num_spatials, op.getPadding());

  window_reversal_ =
      ResolveWindowReversal(num_spatials, op.getWindowReversal());
}

Value CreatePadOpFromConvPadding(OpBuilder& b, mhlo::ConvolutionOp op) {
  const ConvView data(op);
  const auto rank = data.InputLayout().Rank();
  auto input_spatials = data.InputLayout().Spatials();

  llvm::SmallVector<int64_t, 4> hi_padding(rank, 0);
  llvm::SmallVector<int64_t, 4> lo_padding(rank, 0);

  for (const auto& [ind, dim_padding] : llvm::enumerate(data.Padding())) {
    const size_t cur_input_spatial = input_spatials[ind];
    hi_padding[cur_input_spatial] = dim_padding.Hi();
    lo_padding[cur_input_spatial] = dim_padding.Lo();
  }

  const llvm::SmallVector<int64_t, 4> interior_padding(rank, 0);

  auto padding_attr_type = RankedTensorType::get({rank}, b.getI64Type());
  auto hi_padding_attr =
      DenseIntElementsAttr::get(padding_attr_type, hi_padding);
  auto lo_padding_attr =
      DenseIntElementsAttr::get(padding_attr_type, lo_padding);
  auto interior_padding_attr =
      DenseIntElementsAttr::get(padding_attr_type, interior_padding);

  auto padding_value_type = RankedTensorType::get({}, data.ElementType());
  auto padding_value_attr = b.getZeroAttr(padding_value_type);
  auto padding_value_op =
      b.create<arith::ConstantOp>(op->getLoc(), padding_value_attr);

  auto pad_op = b.create<mhlo::PadOp>(padding_value_op->getLoc(), op.getLhs(),
                                      padding_value_op, lo_padding_attr,
                                      hi_padding_attr, interior_padding_attr);

  return pad_op;
}

bool MatchWithResizeBilinearOp(const ConvView& data, bool& align_corners) {
  if (data.InputLayout().Rank() != 4 || data.KernelLayout().Rank() != 4 ||
      data.OutputLayout().Rank() != 4 ||
      data.InputLayout().Spatials() != data.OutputLayout().Spatials()) {
    return false;
  }

  if (data.InputDilations().size() != 2 ||
      !(llvm::all_of(data.KernelDilations(), [](auto d) { return d == 1; })) ||
      data.Strides().size() != 2 || data.Padding().size() != 2) {
    return false;
  }

  // This is based on method in compiler/tf2xla/kernels/image_resize_ops.cc
  auto can_convert_to_bilinear =
      [](bool align_corners, int64_t dilation, int64_t padding, int64_t stride,
         int64_t input_spatial, int64_t output_spatial) {
        int64_t input_spatial_size =
            align_corners ? input_spatial - 1 : input_spatial;
        int64_t output_spatial_size =
            align_corners ? output_spatial - 1 : output_spatial;

        int64_t gcd = std::gcd(static_cast<uint64_t>(input_spatial_size),
                               static_cast<uint64_t>(output_spatial_size));

        if ((gcd == 0) || (input_spatial_size % gcd != 0) ||
            (input_spatial_size / gcd != stride) || (dilation - 1 != padding)) {
          return false;
        }
        return true;
      };

  if (data.InputDilations()[0] != 1 && data.InputDilations()[1] == 1) {
    if (can_convert_to_bilinear(
            /*align_corners=*/true, data.InputDilations()[0],
            data.Padding()[0].Lo(), data.Strides()[0],
            data.InputShape()[data.InputLayout().Spatials()[0]],
            data.OutputShape()[data.OutputLayout().Spatials()[0]])) {
      align_corners = true;
      return true;
    } else if (can_convert_to_bilinear(
                   /*align_corners=*/false, data.InputDilations()[0],
                   data.Padding()[0].Lo(), data.Strides()[0],
                   data.InputShape()[data.InputLayout().Spatials()[0]],
                   data.OutputShape()[data.OutputLayout().Spatials()[0]])) {
      align_corners = false;
      return true;
    };
  } else if (data.InputDilations()[0] == 1 && data.InputDilations()[1] != 1) {
    if (can_convert_to_bilinear(
            /*align_corners=*/true, data.InputDilations()[1],
            data.Padding()[1].Lo(), data.Strides()[1],
            data.InputShape()[data.InputLayout().Spatials()[1]],
            data.OutputShape()[data.OutputLayout().Spatials()[1]])) {
      align_corners = true;
      return true;
    } else if (can_convert_to_bilinear(
                   /*align_corners=*/false, data.InputDilations()[1],
                   data.Padding()[1].Lo(), data.Strides()[1],
                   data.InputShape()[data.InputLayout().Spatials()[1]],
                   data.OutputShape()[data.OutputLayout().Spatials()[1]])) {
      align_corners = false;
      return true;
    };
  }

  return false;
}

bool IsTransposeConvPaddingValid(mhlo::ConvolutionOp conv_op,
                                 size_t num_spatial_dims,
                                 const ArrayRef<int64_t>& strides,
                                 const ArrayRef<int64_t>& padding) {
  auto dnums = conv_op.getDimensionNumbers();
  // The newly added spatial dimension requires zero left and right padding.
  ArrayRef<int64_t> input_spatial_dims = dnums.getInputSpatialDimensions();
  ArrayRef<int64_t> kernel_spatial_dims = dnums.getKernelSpatialDimensions();
  ArrayRef<int64_t> output_spatial_dims = dnums.getOutputSpatialDimensions();

  for (size_t i = 0; i < num_spatial_dims; ++i) {
    int64_t stride = strides[i];
    int64_t input_size = mlir::cast<ShapedType>(conv_op.getLhs().getType())
                             .getDimSize(input_spatial_dims[i]);
    int64_t kernel_size = mlir::cast<ShapedType>(conv_op.getRhs().getType())
                              .getDimSize(kernel_spatial_dims[i]);
    int64_t output_size = mlir::cast<ShapedType>(conv_op.getType())
                              .getDimSize(output_spatial_dims[i]);

    // stablehlo.convolution op needs explicit padding to be set to model any
    // Transposed-Convolution in JAX/PT. Checking to see if-
    // 1. Pre set padding matches to the desired padding
    // 2. Output size respects the `VALID` padding scenario
    if ((padding[2 * i] == padding[2 * i + 1]) &&
        (((kernel_size - 1) != padding[2 * i]) ||
         (output_size != (stride * (input_size - 1)) + kernel_size))) {
      // padding[2 * i] == padding[2 * i + 1] means equal padding is applied
      // on both sides of a spatial dimension.
      // This happens when kernel_dim >= stride
      return false;
    } else if ((padding[2 * i] != padding[2 * i + 1]) &&
               (((kernel_size - 1) != padding[2 * i]) ||
                ((stride - 1) != padding[2 * i + 1]) ||
                (output_size != (stride * input_size)))) {
      return false;
    }
  }

  return true;
}

bool IsTransposeConvPaddingSame(mhlo::ConvolutionOp conv_op,
                                size_t num_spatial_dims,
                                const ArrayRef<int64_t>& strides,
                                const ArrayRef<int64_t>& padding) {
  auto dnums = conv_op.getDimensionNumbers();

  // The newly added spatial dimension requires zero left and right padding.
  ArrayRef<int64_t> input_spatial_dims = dnums.getInputSpatialDimensions();
  ArrayRef<int64_t> output_spatial_dims = dnums.getOutputSpatialDimensions();
  for (size_t i = 0; i < num_spatial_dims; ++i) {
    // In some cases the total padding is odd, so we have 1 leftover, which is
    // why below we check pad_delta > 1.
    int64_t pad_delta = std::abs(padding[2 * i] - padding[2 * i + 1]);
    if (pad_delta > 1) {
      return false;
    }
    int64_t stride = strides[i];
    int64_t input_size = mlir::cast<ShapedType>(conv_op.getLhs().getType())
                             .getDimSize(input_spatial_dims[i]);
    int64_t output_size = mlir::cast<ShapedType>(conv_op.getType())
                              .getDimSize(output_spatial_dims[i]);
    // The reason for the below check is as follows:
    // When computing the output, we have the following relation between
    // o - output dim size, i - input dim size, s - stride, P - total pads
    // o = (i-k+1) + (s-1)(i-1) + P
    // Where the first term is the kernel applications on the input,
    // the second term is the additional applications from the stride
    // and P is a term that captures the total padding. After expanding we get
    // o = si + k - s + 2 + P
    // Here JAX sets P to cancel k-s+2, leading to the expression below
    if (output_size != input_size * stride) {
      return false;
    }
  }
  return true;
}

}  // namespace mlir::odml
