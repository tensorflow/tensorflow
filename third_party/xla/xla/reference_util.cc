/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/reference_util.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/types/span.h"
#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/array4d.h"
#include "xla/hlo/builder/padding.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/tsl/lib/math/math_util.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"

namespace xla {

/* static */ std::unique_ptr<Array2D<double>> ReferenceUtil::Array2DF32ToF64(
    const Array2D<float>& input) {
  auto result =
      std::make_unique<Array2D<double>>(input.height(), input.width());
  for (int64_t rowno = 0; rowno < input.height(); ++rowno) {
    for (int64_t colno = 0; colno < input.width(); ++colno) {
      (*result)(rowno, colno) = input(rowno, colno);
    }
  }
  return result;
}

/*  static */ std::unique_ptr<Array3D<float>> ReferenceUtil::ConvArray3D(
    const Array3D<float>& lhs, const Array3D<float>& rhs, int64_t kernel_stride,
    Padding padding) {
  return ConvArray3DGeneralDimensionsDilated(
      lhs, rhs, kernel_stride, padding, 1, 1,
      XlaBuilder::CreateDefaultConvDimensionNumbers(1));
}

/*static*/ std::unique_ptr<Array3D<float>>
ReferenceUtil::ConvArray3DGeneralDimensionsDilated(
    const Array3D<float>& lhs, const Array3D<float>& rhs, int64_t kernel_stride,
    Padding padding, int64_t lhs_dilation, int64_t rhs_dilation,
    const ConvolutionDimensionNumbers& dnums) {
  CHECK_EQ(dnums.input_spatial_dimensions_size(), 1);
  CHECK_EQ(dnums.kernel_spatial_dimensions_size(), 1);
  CHECK_EQ(dnums.output_spatial_dimensions_size(), 1);
  // Reuse the code for Array4D-convolution by extending the 3D input into a 4D
  // array by adding a fourth dummy dimension of size 1 without stride, padding
  // and dilation.
  Array4D<float> a4dlhs(lhs.n1(), lhs.n2(), lhs.n3(), 1);
  a4dlhs.Each([&](absl::Span<const int64_t> indices, float* value_ptr) {
    CHECK_EQ(indices[3], 0);
    *value_ptr = lhs.operator()(indices[0], indices[1], indices[2]);
  });
  Array4D<float> a4drhs(rhs.n1(), rhs.n2(), rhs.n3(), 1);
  a4drhs.Each([&](absl::Span<const int64_t> indices, float* value_ptr) {
    CHECK_EQ(indices[3], 0);
    *value_ptr = rhs.operator()(indices[0], indices[1], indices[2]);
  });
  // Add a second dummy spatial dimensions.
  ConvolutionDimensionNumbers dnums2d = dnums;
  dnums2d.add_input_spatial_dimensions(3);
  dnums2d.add_kernel_spatial_dimensions(3);
  dnums2d.add_output_spatial_dimensions(3);
  std::unique_ptr<Array4D<float>> convr4 = ConvArray4DGeneralDimensionsDilated(
      a4dlhs, a4drhs, {kernel_stride, 1}, padding, {lhs_dilation, 1},
      {rhs_dilation, 1}, dnums2d);

  auto convr3 = std::make_unique<Array3D<float>>(
      convr4->planes(), convr4->depth(), convr4->height());
  convr4->Each([&](absl::Span<const int64_t> indices, float* value_ptr) {
    CHECK_EQ(indices[3], 0);
    convr3->operator()(indices[0], indices[1], indices[2]) = *value_ptr;
  });
  return convr3;
}

/* static */ std::unique_ptr<Array4D<float>> ReferenceUtil::ConvArray4D(
    const Array4D<float>& lhs, const Array4D<float>& rhs,
    std::pair<int64_t, int64_t> kernel_stride, Padding padding) {
  return ConvArray4DGeneralDimensions(
      lhs, rhs, kernel_stride, padding,
      XlaBuilder::CreateDefaultConvDimensionNumbers());
}

/* static */ std::unique_ptr<Array4D<float>>
ReferenceUtil::SeparableConvArray4D(const Array4D<float>& input,
                                    const Array4D<float>& depthwise_weights,
                                    const Array4D<float>& pointwise_weights,
                                    std::pair<int64_t, int64_t> kernel_stride,
                                    Padding padding) {
  const int64_t depth_multiplier = depthwise_weights.planes();
  CHECK_EQ(pointwise_weights.depth(), input.depth() * depth_multiplier);

  // Combine the two weights by reducing the depth_multiplier, so that we can
  // apply a single convolution on the combined weights.
  Array4D<float> weights(pointwise_weights.planes(), input.depth(),
                         depthwise_weights.height(), depthwise_weights.width());
  for (int64_t kx = 0; kx < depthwise_weights.width(); ++kx) {
    for (int64_t ky = 0; ky < depthwise_weights.height(); ++ky) {
      for (int64_t kz = 0; kz < input.depth(); ++kz) {
        for (int64_t out = 0; out < pointwise_weights.planes(); ++out) {
          float weight = 0.0;
          for (int64_t depth = 0; depth < depth_multiplier; ++depth) {
            weight +=
                depthwise_weights(depth, kz, ky, kx) *
                pointwise_weights(out, depth + kz * depth_multiplier, 0, 0);
          }
          weights(out, kz, ky, kx) = weight;
        }
      }
    }
  }

  return ConvArray4D(input, weights, kernel_stride, padding);
}

/* static */ int64_t ReferenceUtil::WindowCount(int64_t unpadded_width,
                                                int64_t window_len,
                                                int64_t stride,
                                                Padding padding) {
  if (padding == Padding::kValid) {
    return window_util::StridedBound(unpadded_width, window_len, stride);
  }
  return tsl::MathUtil::CeilOfRatio(unpadded_width, stride);
}

/* static  */ std::unique_ptr<std::vector<float>>
ReferenceUtil::ReduceWindow1DGeneric(
    absl::Span<const float> operand, float init,
    absl::FunctionRef<float(float, float)> reduce_func,
    absl::Span<const int64_t> window, absl::Span<const int64_t> stride,
    absl::Span<const std::pair<int64_t, int64_t>> padding) {
  CHECK_EQ(window.size(), 1);
  CHECK_EQ(stride.size(), 1);
  CHECK_EQ(padding.size(), 1);

  int64_t padded_width = padding[0].first + operand.size() + padding[0].second;
  int64_t stride_amount = stride[0];
  int64_t window_size = window[0];
  int64_t result_size =
      window_util::StridedBound(padded_width, window_size, stride_amount);
  int64_t pad_low = padding[0].first;
  auto result = std::make_unique<std::vector<float>>(result_size);

  // Do a full 1D reduce window.
  for (int64_t i0 = 0; i0 < result_size; ++i0) {
    int64_t i0_base = i0 * stride_amount - pad_low;
    float val = init;
    for (int64_t i0_win = 0; i0_win < window_size; ++i0_win) {
      if (i0_base + i0_win >= 0 && i0_base + i0_win < operand.size()) {
        val = reduce_func(val, operand[i0_base + i0_win]);
      }
    }
    (*result)[i0] = val;
  }
  return result;
}

/* static */ std::unique_ptr<Array4D<float>>
ReferenceUtil::ReduceWindow4DGeneric(
    const Array4D<float>& operand, float init,
    absl::FunctionRef<float(float, float)> reduce_func,
    absl::Span<const int64_t> window, absl::Span<const int64_t> stride,
    Padding padding) {
  std::vector<int64_t> dim_lengths{operand.n1(), operand.n2(), operand.n3(),
                                   operand.n4()};
  return ReduceWindow4DGeneric(
      operand, init, reduce_func, window, stride,
      xla::MakePadding(dim_lengths, window, stride, padding));
}

/* static */ std::unique_ptr<Array4D<float>>
ReferenceUtil::ReduceWindow4DGeneric(
    const Array4D<float>& operand, float init,
    absl::FunctionRef<float(float, float)> reduce_func,
    absl::Span<const int64_t> window, absl::Span<const int64_t> stride,
    absl::Span<const std::pair<int64_t, int64_t>> padding) {
  std::vector<int64_t> dim_lengths{operand.n1(), operand.n2(), operand.n3(),
                                   operand.n4()};

  std::vector<int64_t> window_counts(window.size(), 0);
  std::vector<int64_t> pad_low(window.size(), 0);
  for (int64_t i = 0; i < window.size(); ++i) {
    int64_t padded_width =
        padding[i].first + dim_lengths[i] + padding[i].second;
    window_counts[i] =
        window_util::StridedBound(padded_width, window[i], stride[i]);
    pad_low[i] = padding[i].first;
  }
  auto result = std::make_unique<Array4D<float>>(
      window_counts[0], window_counts[1], window_counts[2], window_counts[3]);
  // Do a full 4D reduce window.
  for (int64_t i0 = 0; i0 < window_counts[0]; ++i0) {
    for (int64_t i1 = 0; i1 < window_counts[1]; ++i1) {
      for (int64_t i2 = 0; i2 < window_counts[2]; ++i2) {
        for (int64_t i3 = 0; i3 < window_counts[3]; ++i3) {
          int64_t i0_base = i0 * stride[0] - pad_low[0];
          int64_t i1_base = i1 * stride[1] - pad_low[1];
          int64_t i2_base = i2 * stride[2] - pad_low[2];
          int64_t i3_base = i3 * stride[3] - pad_low[3];

          float val = init;
          for (int64_t i0_win = 0; i0_win < window[0]; ++i0_win) {
            for (int64_t i1_win = 0; i1_win < window[1]; ++i1_win) {
              for (int64_t i2_win = 0; i2_win < window[2]; ++i2_win) {
                for (int64_t i3_win = 0; i3_win < window[3]; ++i3_win) {
                  if (i0_base + i0_win >= 0 && i1_base + i1_win >= 0 &&
                      i2_base + i2_win >= 0 && i3_base + i3_win >= 0 &&
                      i0_base + i0_win < operand.n1() &&
                      i1_base + i1_win < operand.n2() &&
                      i2_base + i2_win < operand.n3() &&
                      i3_base + i3_win < operand.n4()) {
                    val = reduce_func(
                        val, operand(i0_base + i0_win, i1_base + i1_win,
                                     i2_base + i2_win, i3_base + i3_win));
                  }
                }
              }
            }
          }
          (*result)(i0, i1, i2, i3) = val;
        }
      }
    }
  }
  return result;
}

/* static  */ std::unique_ptr<Array4D<float>> ReferenceUtil::ReduceWindow4DAdd(
    const Array4D<float>& operand, float init, absl::Span<const int64_t> window,
    absl::Span<const int64_t> stride, Padding padding) {
  const auto add_reduce = [](float arg1, float arg2) { return arg1 + arg2; };
  return ReduceWindow4DGeneric(operand, init, add_reduce, window, stride,
                               padding);
}

/* static */ std::unique_ptr<Array4D<float>> ReferenceUtil::BatchNorm4D(
    const Array4D<float>& input, const Array4D<float>& mean,
    const Array4D<float>& var, const Array4D<float>& scale,
    const Array4D<float>& offset, float epsilon) {
  auto normalized =
      *MapArray4D(input, mean, [](float a, float b) { return a - b; });
  normalized = *MapArray4D(normalized, var, [&](float a, float b) {
    return a / std::sqrt(b + epsilon);
  });
  normalized =
      *MapArray4D(normalized, scale, [](float a, float b) { return a * b; });
  return MapArray4D(normalized, offset, [](float a, float b) { return a + b; });
}

/* static  */ std::unique_ptr<Array4D<float>>
ReferenceUtil::SelectAndScatter4DGePlus(const Array4D<float>& operand,
                                        const Array4D<float>& source,
                                        float init,
                                        absl::Span<const int64_t> window,
                                        absl::Span<const int64_t> stride,
                                        bool same_padding) {
  Padding padding = same_padding ? Padding::kSame : Padding::kValid;
  auto result = std::make_unique<Array4D<float>>(operand.n1(), operand.n2(),
                                                 operand.n3(), operand.n4());
  std::vector<int64_t> dim_lengths{operand.n1(), operand.n2(), operand.n3(),
                                   operand.n4()};
  auto padding_both = xla::MakePadding(dim_lengths, window, stride, padding);
  // Fill the output, with the initial value.
  result->Fill(init);

  std::vector<int64_t> window_counts(window.size(), 0);
  std::vector<int64_t> pad_low(window.size(), 0);
  for (int64_t i = 0; i < window.size(); ++i) {
    window_counts[i] =
        WindowCount(dim_lengths[i], window[i], stride[i], padding);
    pad_low[i] = padding_both[i].first;
  }
  CHECK_EQ(window_counts[0], source.n1());
  CHECK_EQ(window_counts[1], source.n2());
  CHECK_EQ(window_counts[2], source.n3());
  CHECK_EQ(window_counts[3], source.n4());

  // Do a full 4D select and Scatter.
  for (int64_t i0 = 0; i0 < window_counts[0]; ++i0) {
    for (int64_t i1 = 0; i1 < window_counts[1]; ++i1) {
      for (int64_t i2 = 0; i2 < window_counts[2]; ++i2) {
        for (int64_t i3 = 0; i3 < window_counts[3]; ++i3) {
          // Now we are inside a window and need to find the max and the argmax.
          int64_t i0_base = i0 * stride[0] - pad_low[0];
          int64_t i1_base = i1 * stride[1] - pad_low[1];
          int64_t i2_base = i2 * stride[2] - pad_low[2];
          int64_t i3_base = i3 * stride[3] - pad_low[3];
          int64_t scatter_0 = (i0_base >= 0) ? i0_base : 0;
          int64_t scatter_1 = (i1_base >= 0) ? i1_base : 0;
          int64_t scatter_2 = (i2_base >= 0) ? i2_base : 0;
          int64_t scatter_3 = (i3_base >= 0) ? i3_base : 0;
          float val = operand(scatter_0, scatter_1, scatter_2, scatter_3);
          for (int64_t i0_win = 0; i0_win < window[0]; ++i0_win) {
            for (int64_t i1_win = 0; i1_win < window[1]; ++i1_win) {
              for (int64_t i2_win = 0; i2_win < window[2]; ++i2_win) {
                for (int64_t i3_win = 0; i3_win < window[3]; ++i3_win) {
                  if (i0_base + i0_win >= 0 && i1_base + i1_win >= 0 &&
                      i2_base + i2_win >= 0 && i3_base + i3_win >= 0 &&
                      i0_base + i0_win < operand.n1() &&
                      i1_base + i1_win < operand.n2() &&
                      i2_base + i2_win < operand.n3() &&
                      i3_base + i3_win < operand.n4()) {
                    float tmp = operand(i0_base + i0_win, i1_base + i1_win,
                                        i2_base + i2_win, i3_base + i3_win);
                    if (tmp > val) {
                      val = tmp;
                      scatter_0 = i0_base + i0_win;
                      scatter_1 = i1_base + i1_win;
                      scatter_2 = i2_base + i2_win;
                      scatter_3 = i3_base + i3_win;
                    }
                  }
                }
              }
            }
          }
          (*result)(scatter_0, scatter_1, scatter_2, scatter_3) +=
              source(i0, i1, i2, i3);
        }
      }
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array4D<float>>
ReferenceUtil::ConvArray4DGeneralDimensions(
    const Array4D<float>& lhs, const Array4D<float>& rhs,
    std::pair<int64_t, int64_t> kernel_stride, Padding padding,
    ConvolutionDimensionNumbers dimension_numbers) {
  return ConvArray4DGeneralDimensionsDilated(lhs, rhs, kernel_stride, padding,
                                             {1, 1}, {1, 1},
                                             std::move(dimension_numbers));
}

/* static */ std::unique_ptr<Array4D<float>>
ReferenceUtil::ConvArray4DGeneralDimensionsDilated(
    const Array4D<float>& lhs, const Array4D<float>& rhs,
    std::pair<int64_t, int64_t> kernel_stride, Padding padding,
    std::pair<int64_t, int64_t> lhs_dilation,
    std::pair<int64_t, int64_t> rhs_dilation,
    ConvolutionDimensionNumbers dnums) {
  HloComputation::Builder b("ConvArray4DGeneralDimensionDilated");
  auto lhs_literal = LiteralUtil::CreateR4FromArray4D<float>(lhs);
  auto rhs_literal = LiteralUtil::CreateR4FromArray4D<float>(rhs);

  std::array<int64_t, 2> ordered_kernel_strides;
  std::array<int64_t, 2> ordered_input_dimensions;
  std::array<int64_t, 2> ordered_kernel_dimensions;
  if (dnums.kernel_spatial_dimensions(0) > dnums.kernel_spatial_dimensions(1)) {
    ordered_kernel_strides[0] = kernel_stride.second;
    ordered_kernel_strides[1] = kernel_stride.first;
  } else {
    ordered_kernel_strides[0] = kernel_stride.first;
    ordered_kernel_strides[1] = kernel_stride.second;
  }

  ordered_input_dimensions[0] =
      lhs_literal.shape().dimensions(dnums.input_spatial_dimensions(0));
  ordered_input_dimensions[1] =
      lhs_literal.shape().dimensions(dnums.input_spatial_dimensions(1));
  ordered_kernel_dimensions[0] =
      rhs_literal.shape().dimensions(dnums.kernel_spatial_dimensions(0));
  ordered_kernel_dimensions[1] =
      rhs_literal.shape().dimensions(dnums.kernel_spatial_dimensions(1));

  std::vector<std::pair<int64_t, int64_t>> paddings =
      MakePadding(ordered_input_dimensions, ordered_kernel_dimensions,
                  ordered_kernel_strides, padding);
  CHECK_EQ(paddings.size(), 2);

  Window window;

  WindowDimension dim;
  dim.set_size(
      rhs_literal.shape().dimensions(dnums.kernel_spatial_dimensions(0)));
  dim.set_stride(kernel_stride.first);
  dim.set_padding_low(paddings[0].first);
  dim.set_padding_high(paddings[0].second);
  dim.set_window_dilation(rhs_dilation.first);
  dim.set_base_dilation(lhs_dilation.first);
  *window.add_dimensions() = dim;

  WindowDimension dim2;
  dim2.set_size(
      rhs_literal.shape().dimensions(dnums.kernel_spatial_dimensions(1)));
  dim2.set_stride(kernel_stride.second);
  dim2.set_padding_low(paddings[1].first);
  dim2.set_padding_high(paddings[1].second);
  dim2.set_window_dilation(rhs_dilation.second);
  dim2.set_base_dilation(lhs_dilation.second);
  *window.add_dimensions() = dim2;

  const Shape shape =
      ShapeInference::InferConvolveShape(
          lhs_literal.shape(), rhs_literal.shape(),
          /*feature_group_count=*/1, /*batch_group_count=*/1, window, dnums,
          /*preferred_element_type=*/std::nullopt)
          .value();

  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  PrecisionConfig precision_config;
  precision_config.mutable_operand_precision()->Resize(
      /*new_size=*/2, PrecisionConfig::DEFAULT);
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums, precision_config));
  HloModuleConfig config;
  HloModule module("ReferenceUtil", config);
  auto computation = module.AddEntryComputation(b.Build());

  HloEvaluator evaluator;
  Literal result_literal = evaluator.Evaluate(*computation, {}).value();

  CHECK_EQ(result_literal.shape().rank(), 4);
  auto result =
      std::make_unique<Array4D<float>>(result_literal.shape().dimensions(0),
                                       result_literal.shape().dimensions(1),
                                       result_literal.shape().dimensions(2),
                                       result_literal.shape().dimensions(3));

  result->Each([&](absl::Span<const int64_t> indices, float* value) {
    *value = result_literal.Get<float>(indices);
  });

  return result;
}

/* static */ std::unique_ptr<std::vector<float>>
ReferenceUtil::ReduceToColArray2D(
    const Array2D<float>& matrix, float init,
    absl::FunctionRef<float(float, float)> reduce_function) {
  int64_t rows = matrix.height();
  int64_t cols = matrix.width();
  auto result = std::make_unique<std::vector<float>>();
  for (int64_t i = 0; i < rows; ++i) {
    float acc = init;
    for (int64_t j = 0; j < cols; ++j) {
      acc = reduce_function(acc, matrix(i, j));
    }
    result->push_back(acc);
  }
  return result;
}

/* static */ std::unique_ptr<std::vector<float>>
ReferenceUtil::ReduceToRowArray2D(
    const Array2D<float>& matrix, float init,
    absl::FunctionRef<float(float, float)> reduce_function) {
  int64_t rows = matrix.height();
  int64_t cols = matrix.width();
  auto result = std::make_unique<std::vector<float>>();
  for (int64_t i = 0; i < cols; ++i) {
    float acc = init;
    for (int64_t j = 0; j < rows; ++j) {
      acc = reduce_function(acc, matrix(j, i));
    }
    result->push_back(acc);
  }
  return result;
}

/*static*/ std::vector<float> ReferenceUtil::Reduce4DTo1D(
    const Array4D<float>& array, float init, absl::Span<const int64_t> dims,
    absl::FunctionRef<float(float, float)> reduce_function) {
  std::vector<float> result;
  CHECK_EQ(dims.size(), 3);
  const absl::flat_hash_set<int64_t> dim_set(dims.begin(), dims.end());
  CHECK_EQ(dim_set.size(), 3);
  for (int64_t a0 = 0; a0 == 0 || (!dim_set.contains(0) && a0 < array.n1());
       ++a0) {
    for (int64_t a1 = 0; a1 == 0 || (!dim_set.contains(1) && a1 < array.n2());
         ++a1) {
      for (int64_t a2 = 0; a2 == 0 || (!dim_set.contains(2) && a2 < array.n3());
           ++a2) {
        for (int64_t a3 = 0;
             a3 == 0 || (!dim_set.contains(3) && a3 < array.n4()); ++a3) {
          float accumulator = init;
          for (int64_t i0 = 0;
               i0 == 0 || (dim_set.contains(0) && i0 < array.n1()); ++i0) {
            for (int64_t i1 = 0;
                 i1 == 0 || (dim_set.contains(1) && i1 < array.n2()); ++i1) {
              for (int64_t i2 = 0;
                   i2 == 0 || (dim_set.contains(2) && i2 < array.n3()); ++i2) {
                for (int64_t i3 = 0;
                     i3 == 0 || (dim_set.contains(3) && i3 < array.n4());
                     ++i3) {
                  // Handle zero-sized arrays.
                  if (array.n1() > 0 && array.n2() > 0 && array.n3() > 0 &&
                      array.n4() > 0) {
                    accumulator = reduce_function(
                        accumulator, array(a0 + i0, a1 + i1, a2 + i2, a3 + i3));
                  }
                }
              }
            }
          }
          result.push_back(accumulator);
        }
      }
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array4D<float>> ReferenceUtil::Broadcast1DTo4D(
    const std::vector<float>& array, const std::vector<int64_t>& bounds,
    int64_t broadcast_from_dim) {
  auto result = std::make_unique<Array4D<float>>(bounds[0], bounds[1],
                                                 bounds[2], bounds[3]);
  for (int64_t i = 0; i < result->n1(); ++i) {
    for (int64_t j = 0; j < result->n2(); ++j) {
      for (int64_t k = 0; k < result->n3(); ++k) {
        for (int64_t l = 0; l < result->n4(); ++l) {
          switch (broadcast_from_dim) {
            case 0:
              (*result)(i, j, k, l) = array[i];
              break;
            case 1:
              (*result)(i, j, k, l) = array[j];
              break;
            case 2:
              (*result)(i, j, k, l) = array[k];
              break;
            case 3:
              (*result)(i, j, k, l) = array[l];
              break;
            default:
              break;
          }
        }
      }
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array2D<float>> ReferenceUtil::Reduce3DTo2D(
    const Array3D<float>& array, float init, absl::Span<const int64_t> dims,
    absl::FunctionRef<float(float, float)> reduce_function) {
  CHECK_EQ(dims.size(), 1);
  int64_t rows = dims[0] == 0 ? array.n2() : array.n1();
  int64_t cols = dims[0] == 2 ? array.n2() : array.n3();
  auto result = std::make_unique<Array2D<float>>(rows, cols);
  result->Fill(init);
  for (int i0 = 0; i0 < array.n1(); ++i0) {
    for (int i1 = 0; i1 < array.n2(); ++i1) {
      for (int i2 = 0; i2 < array.n3(); ++i2) {
        int64_t row = dims[0] == 0 ? i1 : i0;
        int64_t col = dims[0] == 2 ? i1 : i2;
        (*result)(row, col) =
            reduce_function((*result)(row, col), array(i0, i1, i2));
      }
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array2D<float>> ReferenceUtil::MapArray2D(
    const Array2D<float>& matrix,
    absl::FunctionRef<float(float)> map_function) {
  int64_t rows = matrix.height();
  int64_t cols = matrix.width();
  auto result = std::make_unique<Array2D<float>>(rows, cols);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      (*result)(i, j) = map_function(matrix(i, j));
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array2D<float>> ReferenceUtil::MapArray2D(
    const Array2D<float>& lhs, const Array2D<float>& rhs,
    absl::FunctionRef<float(float, float)> map_function) {
  CHECK_EQ(lhs.height(), rhs.height());
  CHECK_EQ(lhs.width(), rhs.width());
  int64_t rows = lhs.height();
  int64_t cols = rhs.width();
  auto result = std::make_unique<Array2D<float>>(rows, cols);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      (*result)(i, j) = map_function(lhs(i, j), rhs(i, j));
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array3D<float>> ReferenceUtil::MapArray3D(
    const Array3D<float>& array, absl::FunctionRef<float(float)> map_function) {
  int64_t n1 = array.n1();
  int64_t n2 = array.n2();
  int64_t n3 = array.n3();
  auto result = std::make_unique<Array3D<float>>(n1, n2, n3);
  for (int64_t i = 0; i < n1; ++i) {
    for (int64_t j = 0; j < n2; ++j) {
      for (int64_t k = 0; k < n3; ++k) {
        (*result)(i, j, k) = map_function(array(i, j, k));
      }
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array3D<float>> ReferenceUtil::MapArray3D(
    const Array3D<float>& lhs, const Array3D<float>& rhs,
    absl::FunctionRef<float(float, float)> map_function) {
  CHECK_EQ(lhs.n1(), rhs.n1());
  CHECK_EQ(lhs.n2(), rhs.n2());
  CHECK_EQ(lhs.n3(), rhs.n3());
  int64_t n1 = lhs.n1();
  int64_t n2 = rhs.n2();
  int64_t n3 = rhs.n3();
  auto result = std::make_unique<Array3D<float>>(n1, n2, n3);
  for (int64_t i = 0; i < n1; ++i) {
    for (int64_t j = 0; j < n2; ++j) {
      for (int64_t k = 0; k < n3; ++k) {
        (*result)(i, j, k) = map_function(lhs(i, j, k), rhs(i, j, k));
      }
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array2D<float>> ReferenceUtil::MapWithIndexArray2D(
    const Array2D<float>& matrix,
    absl::FunctionRef<float(float, int64_t, int64_t)> map_function) {
  int64_t rows = matrix.height();
  int64_t cols = matrix.width();
  auto result = std::make_unique<Array2D<float>>(rows, cols);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      (*result)(i, j) = map_function(matrix(i, j), i, j);
    }
  }
  return result;
}

}  // namespace xla
