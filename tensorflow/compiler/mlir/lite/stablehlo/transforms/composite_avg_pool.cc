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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/composite_avg_pool.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/composite_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"  // IWYU pragma: keep
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // IWYU pragma: keep

namespace mlir {
namespace odml {

using mhlo::CompositeOp;

// Struct for holding composite attrs for torch average pool as CC types.
struct TorchAvgPoolData {
  int n;  // Batch.
  int c;  // Channels.

  int h_in;  // Input height.
  int w_in;  // Input width.

  int h_out;  // Output height.
  int w_out;  // Output width.

  int kh;  // Kernel height.
  int kw;  // Kernel width.

  int ph;  // Padding on "height" dimension (both sides).
  int pw;  // Padding on "width" dimension (both sides).

  int sh;  // Stride on "height" dimension.
  int sw;  // Stride on "width" dimension.

  bool ceil_mode;  // Rounding strategy (ceil or floor).
};

// Rounds the dimension based on the ceil mode.
int RoundDim(float dim, bool ceil_mode) {
  if (ceil_mode) {
    return std::ceil(dim);
  }
  return std::floor(dim);
}

// For H or W, calculate the output dimension for average pool.
int CalculateSpatialOutDim(int in, int k, int p, int s, bool ceil_mode) {
  const float effective_size = in - k + (2 * p);
  int out = RoundDim(effective_size / (float)s, ceil_mode) + 1;
  // Only possible if rounder is ceil.
  if ((out - 1) * s >= in + p) {
    out -= 1;
  }
  return out;
}

// Builds a `TorchAvgPoolData` from composite op.
TorchAvgPoolData GetTorchAvgPoolData(CompositeOp op) {
  auto composite_attrs = op.getCompositeAttributes();
  TorchAvgPoolData data;

  auto op_type = mlir::cast<RankedTensorType>(op.getOperand(0).getType());

  data.n = op_type.getShape()[0];
  data.c = op_type.getShape()[1];
  data.h_in = op_type.getShape()[2];
  data.w_in = op_type.getShape()[3];

  std::vector<int32_t> kernel_size;
  GetI32VectorFromDenseI64CompositeAttr(composite_attrs, "kernel_size",
                                        &kernel_size);
  data.kh = kernel_size[0];
  data.kw = kernel_size[1];

  std::vector<int32_t> padding;
  GetI32VectorFromDenseI64CompositeAttr(composite_attrs, "padding", &padding);
  data.ph = padding[0];
  data.pw = padding[1];

  std::vector<int32_t> strides;
  GetI32VectorFromDenseI64CompositeAttr(composite_attrs, "stride", &strides);
  data.sh = strides[0];
  data.sw = strides[1];

  data.ceil_mode =
      GetBoolFromCompositeAttr(composite_attrs, "ceil_mode").value();

  data.h_out = CalculateSpatialOutDim(data.h_in, data.kh, data.ph, data.sh,
                                      data.ceil_mode);
  data.w_out = CalculateSpatialOutDim(data.w_in, data.kw, data.pw, data.sw,
                                      data.ceil_mode);

  return data;
}

// Determines the true number of present elements in the given window
// in input tensor with pytorch rounding behavior.
int ActualNumElementsInKernel(int k_row_start, int k_col_start,
                              const TorchAvgPoolData& pool) {
  int res = 0;
  for (int k_col = 0; k_col < pool.kw; ++k_col) {
    for (int k_row = 0; k_row < pool.kh; ++k_row) {
      const int target_col = k_col + k_col_start;
      const int target_row = k_row + k_row_start;

      const bool row_in_bound = target_row < pool.h_in + (2 * pool.ph);
      const bool col_in_bound = target_col < pool.w_in + (2 * pool.pw);
      const bool is_counted_in_original_input = row_in_bound && col_in_bound;

      res += is_counted_in_original_input;
    }
  }
  return res;
}

// Gets a matrix which corrects the overcounting of divisors when casting a
// average pool with ceil mode true as one with ceil mode false on a padded
// tensor.
DenseFPElementsAttr GetCorrectionMatrix(Builder& builder, CompositeOp op) {
  const TorchAvgPoolData pool = GetTorchAvgPoolData(op);

  llvm::SmallVector<int64_t, 4> nhwc_out_shape(4);
  nhwc_out_shape[0] = 1;  // Broadcast batch.
  nhwc_out_shape[1] = pool.h_out;
  nhwc_out_shape[2] = pool.w_out;
  nhwc_out_shape[3] = 1;  // Broadcast channels.

  auto out_shaped_type =
      RankedTensorType::get(nhwc_out_shape, builder.getF32Type());
  auto get_flat_ind = [&](int row, int col) -> size_t {
    return row * pool.w_out + col;
  };

  std::vector<float> correction_data(out_shaped_type.getNumElements(), 1.0);

  const float kern_size = pool.kh * pool.kw;

  // LEMMA 1: Changing the rounding mode from floor to ceil will increase an
  // output dimension by at most 1 (see `ComputeSpatialOutDim`). This is because
  // for any `x`, `ceil(x) - floor(x) <= 1`.

  // Consider that we pad the input of a average pool with floor rounding to the
  // appropriate size and switch the rounding mode to ceil. When computing the
  // average of a given window, the elements which exist in the newly padded
  // zones will be counted as present elements. Therefore in some windows we
  // will overcount the divisors.
  //    Following from (LEMMA 1) the only windows which contain overcounted
  // divisors are the ones on the outside right and bottom edge. We can iterate
  // over these windows and multiply the corresponding out element by
  // `kernel_size / X` where `X` is the number of elements in the padded input
  // tensor not in the newly padded zone. This corrects the overcounting of
  // divisors resulting in an equivalant computation.
  {
    const int right_col = pool.w_out - 1;
    const int k_col_start = right_col * pool.sw;
    for (int row = 0; row < pool.h_out; ++row) {
      const int k_row_start = row * pool.sh;
      const int correct_divisor =
          ActualNumElementsInKernel(k_row_start, k_col_start, pool);
      const size_t flat_ind = get_flat_ind(row, right_col);
      correction_data[flat_ind] = kern_size / correct_divisor;
    }
  }

  {
    const int bottom_row = pool.h_out - 1;
    const int k_row_start = bottom_row * pool.sh;
    for (int col = 0; col < pool.w_out; ++col) {
      const int k_col_start = col * pool.sw;
      const int correct_divisor =
          ActualNumElementsInKernel(k_row_start, k_col_start, pool);
      const size_t flat_ind = get_flat_ind(bottom_row, col);
      correction_data[flat_ind] = kern_size / correct_divisor;
    }
  }

  return DenseFPElementsAttr::get(out_shaped_type, correction_data);
}

std::array<int, 4> GetPadOpPaddingValues(const TorchAvgPoolData& pool) {
  int pad_bottom = pool.ph;
  int pad_right = pool.pw;

  if (pool.ceil_mode) {
    const int remaining_bottom = pool.h_in - ((pool.h_out - 1) * pool.sh);
    const int ceil_pad_bottom = pool.kh - remaining_bottom;
    pad_bottom = ceil_pad_bottom - pool.ph;

    const int remaining_right = pool.w_in - ((pool.w_out - 1) * pool.sw);
    const int ceil_pad_right = pool.kw - remaining_right;
    pad_right = ceil_pad_right - pool.pw;
  }

  return {pool.ph, pad_bottom, pool.pw, pad_right};
}

DenseIntElementsAttr GetPadOpAttr(Builder& builder, CompositeOp op) {
  const TorchAvgPoolData pool = GetTorchAvgPoolData(op);

  const auto values = GetPadOpPaddingValues(pool);

  llvm::SmallVector<int32_t> padding_vec(8, 0);  // NHWC
  for (auto [ind, val] : llvm::enumerate(values)) {
    padding_vec[ind + 2] = val;
  }

  auto padding_shape = RankedTensorType::get({4, 2}, builder.getI32Type());
  return DenseIntElementsAttr::get(padding_shape, padding_vec);
}

ShapedType GetPadOpType(CompositeOp op) {
  const TorchAvgPoolData pool = GetTorchAvgPoolData(op);

  const auto padding_values = GetPadOpPaddingValues(pool);
  const int64_t h = pool.h_in + padding_values[0] + padding_values[1];
  const int64_t w = pool.w_in + padding_values[2] + padding_values[3];
  llvm::SmallVector<int64_t> shape = {pool.n, h, w, pool.c};

  auto op_type = mlir::cast<RankedTensorType>(op->getResult(0).getType());
  return RankedTensorType::get(shape, op_type.getElementType());
}

StringAttr GetAvgPoolOpPadAttr(Builder& builder, CompositeOp op) {
  const TorchAvgPoolData pool = GetTorchAvgPoolData(op);

  if (pool.ph == 0 && pool.pw == 0) {
    return builder.getStringAttr("VALID");
  }
  if (pool.h_out == pool.h_in && pool.w_out == pool.w_in) {
    return builder.getStringAttr("SAME");
  }
  return builder.getStringAttr("CUSTOM");
}

}  // namespace odml
}  // namespace mlir
