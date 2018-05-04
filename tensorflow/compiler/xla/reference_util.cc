/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/reference_util.h"

#include <array>
#include <utility>

#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_single_threaded_matmul.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace {

template <typename T>
std::unique_ptr<Array2D<T>> MatmulArray2DImpl(
    const Array2D<T>& lhs, const Array2D<T>& rhs,
    const std::function<void(
        const void* run_options_ptr, T* out, T* lhs, T* rhs, int64 m, int64 n,
        int64 k, int32 transpose_lhs, int32 transpose_rhs)>& impl_fn) {
  CHECK_EQ(lhs.width(), rhs.height());
  int m = lhs.height();
  int n = rhs.width();
  int k = lhs.width();
  auto result = MakeUnique<Array2D<T>>(m, n);
  // Because Eigen is a header-oriented library, make sure that the Eigen code
  // is the same as the code used by the CPU backend (otherwise the linker will
  // randomly pick *some* definition).
  impl_fn(
      /*run_options_ptr=*/nullptr, result->data(), rhs.data(), lhs.data(), n, m,
      k,
      /*transpose_lhs=*/0,
      /*transpose_rhs=*/0);
  return result;
}

}  // namespace

/* static */ std::unique_ptr<Array2D<Eigen::half>> ReferenceUtil::MatmulArray2D(
    const Array2D<Eigen::half>& lhs, const Array2D<Eigen::half>& rhs) {
  return MatmulArray2DImpl<Eigen::half>(
      lhs, rhs, __xla_cpu_runtime_EigenSingleThreadedMatMulF16);
}

/* static */ std::unique_ptr<Array2D<float>> ReferenceUtil::MatmulArray2D(
    const Array2D<float>& lhs, const Array2D<float>& rhs) {
  return MatmulArray2DImpl<float>(
      lhs, rhs, __xla_cpu_runtime_EigenSingleThreadedMatMulF32);
}

/* static */ std::unique_ptr<Array2D<double>> ReferenceUtil::MatmulArray2D(
    const Array2D<double>& lhs, const Array2D<double>& rhs) {
  return MatmulArray2DImpl<double>(
      lhs, rhs, __xla_cpu_runtime_EigenSingleThreadedMatMulF64);
}

/* static */ std::unique_ptr<Array2D<double>> ReferenceUtil::Array2DF32ToF64(
    const Array2D<float>& input) {
  auto result = MakeUnique<Array2D<double>>(input.height(), input.width());
  for (int64 rowno = 0; rowno < input.height(); ++rowno) {
    for (int64 colno = 0; colno < input.height(); ++colno) {
      (*result)(rowno, colno) = input(rowno, colno);
    }
  }
  return result;
}

/*  static */ std::unique_ptr<Array3D<float>> ReferenceUtil::ConvArray3D(
    const Array3D<float>& lhs, const Array3D<float>& rhs, int64 kernel_stride,
    Padding padding) {
  return ConvArray3DGeneralDimensionsDilated(
      lhs, rhs, kernel_stride, padding, 1, 1,
      XlaBuilder::CreateDefaultConvDimensionNumbers(1));
}

/*static*/ std::unique_ptr<Array3D<float>>
ReferenceUtil::ConvArray3DGeneralDimensionsDilated(
    const Array3D<float>& lhs, const Array3D<float>& rhs, int64 kernel_stride,
    Padding padding, int64 lhs_dilation, int64 rhs_dilation,
    const ConvolutionDimensionNumbers& dnums) {
  CHECK_EQ(dnums.input_spatial_dimensions_size(), 1);
  CHECK_EQ(dnums.kernel_spatial_dimensions_size(), 1);
  CHECK_EQ(dnums.output_spatial_dimensions_size(), 1);
  // Reuse the code for Array4D-convolution by extending the 3D input into a 4D
  // array by adding a fourth dummy dimension of size 1 without stride, padding
  // and dilation.
  Array4D<float> a4dlhs(lhs.n1(), lhs.n2(), lhs.n3(), 1);
  a4dlhs.Each(
      [&](tensorflow::gtl::ArraySlice<int64> indices, float* value_ptr) {
        CHECK_EQ(indices[3], 0);
        *value_ptr = lhs.operator()(indices[0], indices[1], indices[2]);
      });
  Array4D<float> a4drhs(rhs.n1(), rhs.n2(), rhs.n3(), 1);
  a4drhs.Each(
      [&](tensorflow::gtl::ArraySlice<int64> indices, float* value_ptr) {
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

  auto convr3 = MakeUnique<Array3D<float>>(convr4->planes(), convr4->depth(),
                                           convr4->height());
  convr4->Each(
      [&](tensorflow::gtl::ArraySlice<int64> indices, float* value_ptr) {
        CHECK_EQ(indices[3], 0);
        convr3->operator()(indices[0], indices[1], indices[2]) = *value_ptr;
      });
  return convr3;
}

/* static */ std::unique_ptr<Array4D<float>> ReferenceUtil::ConvArray4D(
    const Array4D<float>& lhs, const Array4D<float>& rhs,
    std::pair<int64, int64> kernel_stride, Padding padding) {
  return ConvArray4DGeneralDimensions(
      lhs, rhs, kernel_stride, padding,
      XlaBuilder::CreateDefaultConvDimensionNumbers());
}

/* static */ std::unique_ptr<Array4D<float>>
ReferenceUtil::SeparableConvArray4D(const Array4D<float>& input,
                                    const Array4D<float>& depthwise_weights,
                                    const Array4D<float>& pointwise_weights,
                                    std::pair<int64, int64> kernel_stride,
                                    Padding padding) {
  const int64 depth_multiplier = depthwise_weights.planes();
  CHECK_EQ(pointwise_weights.depth(), input.depth() * depth_multiplier);

  // Combine the two weights by reducing the depth_multiplier, so that we can
  // apply a single convolution on the combined weights.
  Array4D<float> weights(pointwise_weights.planes(), input.depth(),
                         depthwise_weights.height(), depthwise_weights.width());
  for (int64 kx = 0; kx < depthwise_weights.width(); ++kx) {
    for (int64 ky = 0; ky < depthwise_weights.height(); ++ky) {
      for (int64 kz = 0; kz < input.depth(); ++kz) {
        for (int64 out = 0; out < pointwise_weights.planes(); ++out) {
          float weight = 0.0;
          for (int64 depth = 0; depth < depth_multiplier; ++depth) {
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

/* static */ int64 ReferenceUtil::WindowCount(int64 unpadded_width,
                                              int64 window_len, int64 stride,
                                              Padding padding) {
  if (padding == Padding::kValid) {
    return window_util::StridedBound(unpadded_width, window_len, stride);
  }
  return tensorflow::MathUtil::CeilOfRatio(unpadded_width, stride);
}

/* static  */ std::unique_ptr<std::vector<float>>
ReferenceUtil::ReduceWindow1DGeneric(
    const tensorflow::gtl::ArraySlice<float>& operand, float init,
    const std::function<float(float, float)>& reduce_func,
    const tensorflow::gtl::ArraySlice<int64>& window,
    const tensorflow::gtl::ArraySlice<int64>& stride,
    const tensorflow::gtl::ArraySlice<std::pair<int64, int64>>& padding) {
  std::vector<int64> dim_lengths{static_cast<int64>(operand.size())};
  std::vector<int64> window_counts(window.size(), 0);
  std::vector<int64> pad_low(window.size(), 0);
  for (int64 i = 0; i < window.size(); ++i) {
    int64 padded_width = padding[i].first + dim_lengths[i] + padding[i].second;
    window_counts[i] =
        window_util::StridedBound(padded_width, window[i], stride[i]);
    pad_low[i] = padding[i].first;
  }
  auto result = MakeUnique<std::vector<float>>(window_counts[0]);

  // Do a full 1D reduce window.
  for (int64 i0 = 0; i0 < window_counts[0]; ++i0) {
    int64 i0_base = i0 * stride[0] - pad_low[0];

    float val = init;
    for (int64 i0_win = 0; i0_win < window[0]; ++i0_win) {
      if (i0_base + i0_win >= 0 && i0_base + i0_win < dim_lengths[0]) {
        val = reduce_func(val, operand[i0_base + i0_win]);
      }
    }
    (*result)[i0] = val;
  }
  return result;
}

/* static  */ std::unique_ptr<std::vector<float>>
ReferenceUtil::ReduceWindow1DAdd(
    const tensorflow::gtl::ArraySlice<float>& operand, float init,
    const tensorflow::gtl::ArraySlice<int64>& window,
    const tensorflow::gtl::ArraySlice<int64>& stride, Padding padding) {
  const auto add_reduce = [](float arg1, float arg2) { return arg1 + arg2; };
  std::vector<int64> dim_lengths{static_cast<int64>(operand.size())};
  return ReduceWindow1DGeneric(
      operand, init, add_reduce, window, stride,
      xla::MakePadding(dim_lengths, window, stride, padding));
}

/* static */ std::unique_ptr<Array2D<float>>
ReferenceUtil::ReduceWindow2DGeneric(
    const Array2D<float>& operand, float init,
    const std::function<float(float, float)>& reduce_func,
    const tensorflow::gtl::ArraySlice<int64>& window,
    const tensorflow::gtl::ArraySlice<int64>& stride,
    const tensorflow::gtl::ArraySlice<std::pair<int64, int64>>& padding) {
  std::vector<int64> dim_lengths{operand.height(), operand.width()};

  std::vector<int64> window_counts(window.size(), 0);
  std::vector<int64> pad_low(window.size(), 0);
  for (int64 i = 0; i < window.size(); ++i) {
    int64 padded_width = padding[i].first + dim_lengths[i] + padding[i].second;
    window_counts[i] =
        window_util::StridedBound(padded_width, window[i], stride[i]);
    pad_low[i] = padding[i].first;
  }
  auto result = MakeUnique<Array2D<float>>(window_counts[0], window_counts[1]);

  // Do a full 2D reduce window.
  for (int64 i0 = 0; i0 < window_counts[0]; ++i0) {
    for (int64 i1 = 0; i1 < window_counts[1]; ++i1) {
      int64 i0_base = i0 * stride[0] - pad_low[0];
      int64 i1_base = i1 * stride[1] - pad_low[1];

      float val = init;
      for (int64 i0_win = 0; i0_win < window[0]; ++i0_win) {
        for (int64 i1_win = 0; i1_win < window[1]; ++i1_win) {
          if (i0_base + i0_win >= 0 && i1_base + i1_win >= 0 &&
              i0_base + i0_win < operand.n1() &&
              i1_base + i1_win < operand.n2()) {
            val = reduce_func(val, operand(i0_base + i0_win, i1_base + i1_win));
          }
        }
      }
      (*result)(i0, i1) = val;
    }
  }
  return result;
}

/* static  */ std::unique_ptr<Array2D<float>> ReferenceUtil::ReduceWindow2DAdd(
    const Array2D<float>& operand, float init,
    const tensorflow::gtl::ArraySlice<int64>& window,
    const tensorflow::gtl::ArraySlice<int64>& stride, Padding padding) {
  const auto add_reduce = [](float arg1, float arg2) { return arg1 + arg2; };
  std::vector<int64> dim_lengths{operand.height(), operand.width()};
  return ReduceWindow2DGeneric(
      operand, init, add_reduce, window, stride,
      xla::MakePadding(dim_lengths, window, stride, padding));
}

/* static  */ std::unique_ptr<Array3D<float>> ReferenceUtil::ReduceWindow3DAdd(
    const Array3D<float>& operand, float init,
    const tensorflow::gtl::ArraySlice<int64>& window,
    const tensorflow::gtl::ArraySlice<int64>& stride, Padding padding) {
  std::vector<int64> dim_lengths{operand.n1(), operand.n2(), operand.n3()};
  auto padding_both = xla::MakePadding(dim_lengths, window, stride, padding);

  std::vector<int64> window_counts(window.size(), 0);
  std::vector<int64> pad_low(window.size(), 0);
  for (int64 i = 0; i < window.size(); ++i) {
    window_counts[i] =
        WindowCount(dim_lengths[i], window[i], stride[i], padding);
    pad_low[i] = padding_both[i].first;
  }
  auto result = MakeUnique<Array3D<float>>(window_counts[0], window_counts[1],
                                           window_counts[2]);

  for (int64 i0 = 0; i0 < window_counts[0]; ++i0) {
    for (int64 i1 = 0; i1 < window_counts[1]; ++i1) {
      for (int64 i2 = 0; i2 < window_counts[2]; ++i2) {
        int64 i0_base = i0 * stride[0] - pad_low[0];
        int64 i1_base = i1 * stride[1] - pad_low[1];
        int64 i2_base = i2 * stride[2] - pad_low[2];

        float val = init;
        for (int64 i0_win = 0; i0_win < window[0]; ++i0_win) {
          for (int64 i1_win = 0; i1_win < window[1]; ++i1_win) {
            for (int64 i2_win = 0; i2_win < window[2]; ++i2_win) {
              if (i0_base + i0_win >= 0 && i1_base + i1_win >= 0 &&
                  i2_base + i2_win >= 0 && i0_base + i0_win < operand.n1() &&
                  i1_base + i1_win < operand.n2() &&
                  i2_base + i2_win < operand.n3()) {
                val += operand(i0_base + i0_win, i1_base + i1_win,
                               i2_base + i2_win);
              }
            }
          }
        }
        (*result)(i0, i1, i2) = val;
      }
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array4D<float>>
ReferenceUtil::ReduceWindow4DGeneric(
    const Array4D<float>& operand, float init,
    const std::function<float(float, float)>& reduce_func,
    const tensorflow::gtl::ArraySlice<int64>& window,
    const tensorflow::gtl::ArraySlice<int64>& stride, Padding padding) {
  std::vector<int64> dim_lengths{operand.n1(), operand.n2(), operand.n3(),
                                 operand.n4()};
  return ReduceWindow4DGeneric(
      operand, init, reduce_func, window, stride,
      xla::MakePadding(dim_lengths, window, stride, padding));
}

/* static */ std::unique_ptr<Array4D<float>>
ReferenceUtil::ReduceWindow4DGeneric(
    const Array4D<float>& operand, float init,
    const std::function<float(float, float)>& reduce_func,
    const tensorflow::gtl::ArraySlice<int64>& window,
    const tensorflow::gtl::ArraySlice<int64>& stride,
    const tensorflow::gtl::ArraySlice<std::pair<int64, int64>>& padding) {
  std::vector<int64> dim_lengths{operand.n1(), operand.n2(), operand.n3(),
                                 operand.n4()};

  std::vector<int64> window_counts(window.size(), 0);
  std::vector<int64> pad_low(window.size(), 0);
  for (int64 i = 0; i < window.size(); ++i) {
    int64 padded_width = padding[i].first + dim_lengths[i] + padding[i].second;
    window_counts[i] =
        window_util::StridedBound(padded_width, window[i], stride[i]);
    pad_low[i] = padding[i].first;
  }
  auto result = MakeUnique<Array4D<float>>(window_counts[0], window_counts[1],
                                           window_counts[2], window_counts[3]);
  // Do a full 4D reduce window.
  for (int64 i0 = 0; i0 < window_counts[0]; ++i0) {
    for (int64 i1 = 0; i1 < window_counts[1]; ++i1) {
      for (int64 i2 = 0; i2 < window_counts[2]; ++i2) {
        for (int64 i3 = 0; i3 < window_counts[3]; ++i3) {
          int64 i0_base = i0 * stride[0] - pad_low[0];
          int64 i1_base = i1 * stride[1] - pad_low[1];
          int64 i2_base = i2 * stride[2] - pad_low[2];
          int64 i3_base = i3 * stride[3] - pad_low[3];

          float val = init;
          for (int64 i0_win = 0; i0_win < window[0]; ++i0_win) {
            for (int64 i1_win = 0; i1_win < window[1]; ++i1_win) {
              for (int64 i2_win = 0; i2_win < window[2]; ++i2_win) {
                for (int64 i3_win = 0; i3_win < window[3]; ++i3_win) {
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
    const Array4D<float>& operand, float init,
    const tensorflow::gtl::ArraySlice<int64>& window,
    const tensorflow::gtl::ArraySlice<int64>& stride, Padding padding) {
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
ReferenceUtil::SelectAndScatter4DGePlus(
    const Array4D<float>& operand, const Array4D<float>& source, float init,
    const tensorflow::gtl::ArraySlice<int64>& window,
    const tensorflow::gtl::ArraySlice<int64>& stride, bool same_padding) {
  Padding padding = same_padding ? Padding::kSame : Padding::kValid;
  auto result = MakeUnique<Array4D<float>>(operand.n1(), operand.n2(),
                                           operand.n3(), operand.n4());
  std::vector<int64> dim_lengths{operand.n1(), operand.n2(), operand.n3(),
                                 operand.n4()};
  auto padding_both = xla::MakePadding(dim_lengths, window, stride, padding);
  // Fill the output, with the initial value.
  result->Fill(init);

  std::vector<int64> window_counts(window.size(), 0);
  std::vector<int64> pad_low(window.size(), 0);
  for (int64 i = 0; i < window.size(); ++i) {
    window_counts[i] =
        WindowCount(dim_lengths[i], window[i], stride[i], padding);
    pad_low[i] = padding_both[i].first;
  }
  CHECK_EQ(window_counts[0], source.n1());
  CHECK_EQ(window_counts[1], source.n2());
  CHECK_EQ(window_counts[2], source.n3());
  CHECK_EQ(window_counts[3], source.n4());

  // Do a full 4D select and Scatter.
  for (int64 i0 = 0; i0 < window_counts[0]; ++i0) {
    for (int64 i1 = 0; i1 < window_counts[1]; ++i1) {
      for (int64 i2 = 0; i2 < window_counts[2]; ++i2) {
        for (int64 i3 = 0; i3 < window_counts[3]; ++i3) {
          // Now we are inside a window and need to find the max and the argmax.
          int64 i0_base = i0 * stride[0] - pad_low[0];
          int64 i1_base = i1 * stride[1] - pad_low[1];
          int64 i2_base = i2 * stride[2] - pad_low[2];
          int64 i3_base = i3 * stride[3] - pad_low[3];
          int64 scatter_0 = (i0_base >= 0) ? i0_base : 0;
          int64 scatter_1 = (i1_base >= 0) ? i1_base : 0;
          int64 scatter_2 = (i2_base >= 0) ? i2_base : 0;
          int64 scatter_3 = (i3_base >= 0) ? i3_base : 0;
          float val = operand(scatter_0, scatter_1, scatter_2, scatter_3);
          for (int64 i0_win = 0; i0_win < window[0]; ++i0_win) {
            for (int64 i1_win = 0; i1_win < window[1]; ++i1_win) {
              for (int64 i2_win = 0; i2_win < window[2]; ++i2_win) {
                for (int64 i3_win = 0; i3_win < window[3]; ++i3_win) {
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
    std::pair<int64, int64> kernel_stride, Padding padding,
    ConvolutionDimensionNumbers dimension_numbers) {
  return ConvArray4DGeneralDimensionsDilated(lhs, rhs, kernel_stride, padding,
                                             {1, 1}, {1, 1},
                                             std::move(dimension_numbers));
}

/* static */ std::unique_ptr<Array4D<float>>
ReferenceUtil::ConvArray4DGeneralDimensionsDilated(
    const Array4D<float>& lhs, const Array4D<float>& rhs,
    std::pair<int64, int64> kernel_stride, Padding padding,
    std::pair<int64, int64> lhs_dilation, std::pair<int64, int64> rhs_dilation,
    ConvolutionDimensionNumbers dnums) {
  HloComputation::Builder b("ConvArray4DGeneralDimensionDilated");
  auto lhs_literal = Literal::CreateR4FromArray4D<float>(lhs);
  auto rhs_literal = Literal::CreateR4FromArray4D<float>(rhs);

  std::array<int64, 2> ordered_kernel_strides;
  std::array<int64, 2> ordered_input_dimensions;
  std::array<int64, 2> ordered_kernel_dimensions;
  if (dnums.kernel_spatial_dimensions(0) > dnums.kernel_spatial_dimensions(1)) {
    ordered_kernel_strides[0] = kernel_stride.second;
    ordered_kernel_strides[1] = kernel_stride.first;
  } else {
    ordered_kernel_strides[0] = kernel_stride.first;
    ordered_kernel_strides[1] = kernel_stride.second;
  }

  ordered_input_dimensions[0] =
      lhs_literal->shape().dimensions(dnums.input_spatial_dimensions(0));
  ordered_input_dimensions[1] =
      lhs_literal->shape().dimensions(dnums.input_spatial_dimensions(1));
  ordered_kernel_dimensions[0] =
      rhs_literal->shape().dimensions(dnums.kernel_spatial_dimensions(0));
  ordered_kernel_dimensions[1] =
      rhs_literal->shape().dimensions(dnums.kernel_spatial_dimensions(1));

  std::vector<std::pair<int64, int64>> paddings =
      MakePadding(ordered_input_dimensions, ordered_kernel_dimensions,
                  ordered_kernel_strides, padding);
  CHECK_EQ(paddings.size(), 2);

  Window window;

  WindowDimension dim;
  dim.set_size(
      rhs_literal->shape().dimensions(dnums.kernel_spatial_dimensions(0)));
  dim.set_stride(kernel_stride.first);
  dim.set_padding_low(paddings[0].first);
  dim.set_padding_high(paddings[0].second);
  dim.set_window_dilation(rhs_dilation.first);
  dim.set_base_dilation(lhs_dilation.first);
  *window.add_dimensions() = dim;

  WindowDimension dim2;
  dim2.set_size(
      rhs_literal->shape().dimensions(dnums.kernel_spatial_dimensions(1)));
  dim2.set_stride(kernel_stride.second);
  dim2.set_padding_low(paddings[1].first);
  dim2.set_padding_high(paddings[1].second);
  dim2.set_window_dilation(rhs_dilation.second);
  dim2.set_base_dilation(lhs_dilation.second);
  *window.add_dimensions() = dim2;

  const Shape& shape =
      ShapeInference::InferConvolveShape(lhs_literal->shape(),
                                         rhs_literal->shape(), window, dnums)
          .ConsumeValueOrDie();

  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, window, dnums));
  HloModuleConfig config;
  HloModule module("ReferenceUtil", config);
  auto computation = module.AddEntryComputation(b.Build());

  HloEvaluator evaluator;
  std::unique_ptr<Literal> result_literal =
      evaluator.Evaluate<const Literal*>(*computation, {}).ConsumeValueOrDie();

  CHECK_EQ(ShapeUtil::Rank(result_literal->shape()), 4);
  auto result =
      MakeUnique<Array4D<float>>(result_literal->shape().dimensions(0),
                                 result_literal->shape().dimensions(1),
                                 result_literal->shape().dimensions(2),
                                 result_literal->shape().dimensions(3));

  result->Each([&](tensorflow::gtl::ArraySlice<int64> indices, float* value) {
    *value = result_literal->Get<float>(indices);
  });

  return result;
}

/* static */ std::unique_ptr<std::vector<float>>
ReferenceUtil::ReduceToColArray2D(
    const Array2D<float>& matrix, float init,
    const std::function<float(float, float)>& reduce_function) {
  int64 rows = matrix.height();
  int64 cols = matrix.width();
  auto result = MakeUnique<std::vector<float>>();
  for (int64 i = 0; i < rows; ++i) {
    float acc = init;
    for (int64 j = 0; j < cols; ++j) {
      acc = reduce_function(acc, matrix(i, j));
    }
    result->push_back(acc);
  }
  return result;
}

/* static */ std::unique_ptr<std::vector<float>>
ReferenceUtil::ReduceToRowArray2D(
    const Array2D<float>& matrix, float init,
    const std::function<float(float, float)>& reduce_function) {
  int64 rows = matrix.height();
  int64 cols = matrix.width();
  auto result = MakeUnique<std::vector<float>>();
  for (int64 i = 0; i < cols; ++i) {
    float acc = init;
    for (int64 j = 0; j < rows; ++j) {
      acc = reduce_function(acc, matrix(j, i));
    }
    result->push_back(acc);
  }
  return result;
}

/*static*/ std::vector<float> ReferenceUtil::Reduce4DTo1D(
    const Array4D<float>& array, float init,
    tensorflow::gtl::ArraySlice<int64> dims,
    const std::function<float(float, float)>& reduce_function) {
  std::vector<float> result;
  CHECK_EQ(dims.size(), 3);
  const std::set<int64> dim_set(dims.begin(), dims.end());
  CHECK_EQ(dim_set.size(), 3);
  for (int64 a0 = 0; a0 == 0 || (!dim_set.count(0) && a0 < array.n1()); ++a0) {
    for (int64 a1 = 0; a1 == 0 || (!dim_set.count(1) && a1 < array.n2());
         ++a1) {
      for (int64 a2 = 0; a2 == 0 || (!dim_set.count(2) && a2 < array.n3());
           ++a2) {
        for (int64 a3 = 0; a3 == 0 || (!dim_set.count(3) && a3 < array.n4());
             ++a3) {
          float accumulator = init;
          for (int64 i0 = 0; i0 == 0 || (dim_set.count(0) && i0 < array.n1());
               ++i0) {
            for (int64 i1 = 0; i1 == 0 || (dim_set.count(1) && i1 < array.n2());
                 ++i1) {
              for (int64 i2 = 0;
                   i2 == 0 || (dim_set.count(2) && i2 < array.n3()); ++i2) {
                for (int64 i3 = 0;
                     i3 == 0 || (dim_set.count(3) && i3 < array.n4()); ++i3) {
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
    const std::vector<float>& array, const std::vector<int64>& bounds,
    int64 broadcast_from_dim) {
  auto result =
      MakeUnique<Array4D<float>>(bounds[0], bounds[1], bounds[2], bounds[3]);
  for (int64 i = 0; i < result->n1(); ++i) {
    for (int64 j = 0; j < result->n2(); ++j) {
      for (int64 k = 0; k < result->n3(); ++k) {
        for (int64 l = 0; l < result->n4(); ++l) {
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
    const Array3D<float>& array, float init,
    tensorflow::gtl::ArraySlice<int64> dims,
    const std::function<float(float, float)>& reduce_function) {
  CHECK_EQ(dims.size(), 1);
  int64 rows = dims[0] == 0 ? array.n2() : array.n1();
  int64 cols = dims[0] == 2 ? array.n2() : array.n3();
  auto result = MakeUnique<Array2D<float>>(rows, cols);
  result->Fill(init);
  for (int i0 = 0; i0 < array.n1(); ++i0) {
    for (int i1 = 0; i1 < array.n2(); ++i1) {
      for (int i2 = 0; i2 < array.n3(); ++i2) {
        int64 row = dims[0] == 0 ? i1 : i0;
        int64 col = dims[0] == 2 ? i1 : i2;
        (*result)(row, col) =
            reduce_function((*result)(row, col), array(i0, i1, i2));
      }
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array2D<float>> ReferenceUtil::MapArray2D(
    const Array2D<float>& matrix,
    const std::function<float(float)>& map_function) {
  int64 rows = matrix.height();
  int64 cols = matrix.width();
  auto result = MakeUnique<Array2D<float>>(rows, cols);
  for (int64 i = 0; i < rows; ++i) {
    for (int64 j = 0; j < cols; ++j) {
      (*result)(i, j) = map_function(matrix(i, j));
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array2D<float>> ReferenceUtil::MapArray2D(
    const Array2D<float>& lhs, const Array2D<float>& rhs,
    const std::function<float(float, float)>& map_function) {
  CHECK_EQ(lhs.height(), rhs.height());
  CHECK_EQ(lhs.width(), rhs.width());
  int64 rows = lhs.height();
  int64 cols = rhs.width();
  auto result = MakeUnique<Array2D<float>>(rows, cols);
  for (int64 i = 0; i < rows; ++i) {
    for (int64 j = 0; j < cols; ++j) {
      (*result)(i, j) = map_function(lhs(i, j), rhs(i, j));
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array2D<float>> ReferenceUtil::MapWithIndexArray2D(
    const Array2D<float>& matrix,
    const std::function<float(float, int64, int64)>& map_function) {
  int64 rows = matrix.height();
  int64 cols = matrix.width();
  auto result = MakeUnique<Array2D<float>>(rows, cols);
  for (int64 i = 0; i < rows; ++i) {
    for (int64 j = 0; j < cols; ++j) {
      (*result)(i, j) = map_function(matrix(i, j), i, j);
    }
  }
  return result;
}

}  // namespace xla
