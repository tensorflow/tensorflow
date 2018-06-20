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

#ifndef TENSORFLOW_COMPILER_XLA_REFERENCE_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_REFERENCE_UTIL_H_

#include <array>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Utility class for reference implementations of linear algebra routines.
class ReferenceUtil {
 public:
  // Returns the result of a transpose operation on the input matrix.
  template <typename T>
  static std::unique_ptr<Array2D<T>> TransposeArray2D(
      const Array2D<T>& operand) {
    auto result = MakeUnique<Array2D<T>>(operand.width(), operand.height());
    for (int64 w = 0; w < operand.width(); ++w) {
      for (int64 h = 0; h < operand.height(); ++h) {
        (*result)(w, h) = operand(h, w);
      }
    }

    return result;
  }

  // Returns the result of a matrix multiply `lhs x rhs`.
  static std::unique_ptr<Array2D<Eigen::half>> MatmulArray2D(
      const Array2D<Eigen::half>& lhs, const Array2D<Eigen::half>& rhs);
  static std::unique_ptr<Array2D<float>> MatmulArray2D(
      const Array2D<float>& lhs, const Array2D<float>& rhs);
  static std::unique_ptr<Array2D<double>> MatmulArray2D(
      const Array2D<double>& lhs, const Array2D<double>& rhs);

  // Converts the input operand to use f64 values instead of f32 values.
  static std::unique_ptr<Array2D<double>> Array2DF32ToF64(
      const Array2D<float>& input);

  // Returns the result of a convolution `lhs <conv> rhs`, with the default
  // convolution dimension numbers returned from
  // ComputationBuilder::CreateDefaultConvDimensionNumbers().
  static std::unique_ptr<Array4D<float>> ConvArray4D(
      const Array4D<float>& lhs, const Array4D<float>& rhs,
      std::pair<int64, int64> kernel_stride, Padding padding);

  // Returns the result of a convolution `lhs <conv> rhs`, with the given
  // convolution dimension numbers.
  static std::unique_ptr<Array4D<float>> ConvArray4DGeneralDimensions(
      const Array4D<float>& lhs, const Array4D<float>& rhs,
      std::pair<int64, int64> kernel_stride, Padding padding,
      ConvolutionDimensionNumbers dimension_numbers);

  // Returns the result of a convolution `lhs <conv> rhs`, with the given
  // dilation factors.
  static std::unique_ptr<Array4D<float>> ConvArray4DGeneralDimensionsDilated(
      const Array4D<float>& lhs, const Array4D<float>& rhs,
      std::pair<int64, int64> kernel_stride, Padding padding,
      std::pair<int64, int64> lhs_dilation,
      std::pair<int64, int64> rhs_dilation, ConvolutionDimensionNumbers dnums);

  // Returns the result of a convolution `lhs <conv> rhs`, with the default
  // convolution dimension numbers returned from
  // ComputationBuilder::CreateDefaultConvDimensionNumbers().
  static std::unique_ptr<Array3D<float>> ConvArray3D(const Array3D<float>& lhs,
                                                     const Array3D<float>& rhs,
                                                     int64 kernel_stride,
                                                     Padding padding);

  // Returns the result of a convolution `lhs <conv> rhs`.
  static std::unique_ptr<Array3D<float>> ConvArray3DGeneralDimensionsDilated(
      const Array3D<float>& lhs, const Array3D<float>& rhs, int64 kernel_stride,
      Padding padding, int64 lhs_dilation, int64 rhs_dilation,
      const ConvolutionDimensionNumbers& dnums);

  // Returns the result of a separable  convolution with the given parameters.
  // kernel_stride and padding applies to the depthwise convolution during
  // the separable convolution. pointwise_weights.depth() must be equal to
  // input.depth() * depthwise_weights.planes().
  static std::unique_ptr<Array4D<float>> SeparableConvArray4D(
      const Array4D<float>& input, const Array4D<float>& depthwise_weights,
      const Array4D<float>& pointwise_weights,
      std::pair<int64, int64> kernel_stride, Padding padding);

  // Returns the result of reducing a matrix to a column vector. init is the
  // initial value for the reduce operation, and reduce_function is the function
  // to apply for each reduction step.
  static std::unique_ptr<std::vector<float>> ReduceToColArray2D(
      const Array2D<float>& matrix, float init,
      const std::function<float(float, float)>& reduce_function);

  // Returns the result of reducing a matrix to a row vector. init is the
  // initial value for the reduce operation, and reduce_function is the function
  // to apply for each reduction step.
  static std::unique_ptr<std::vector<float>> ReduceToRowArray2D(
      const Array2D<float>& matrix, float init,
      const std::function<float(float, float)>& reduce_function);

  // Performs a R2=>R1 reduction by reducing away the dimension specified in
  // 'dimension_to_reduce'.
  template <typename T>
  static std::vector<T> ReduceR2ToR1(const Array2D<T>& input,
                                     int dimension_to_reduce, T init,
                                     const std::function<T(T, T)>& freduce) {
    std::vector<T> result(dimension_to_reduce == 0 ? input.n2() : input.n1(),
                          init);
    for (int i0 = 0; i0 < input.n1(); ++i0) {
      for (int i1 = 0; i1 < input.n2(); ++i1) {
        int output = dimension_to_reduce == 0 ? i1 : i0;
        result[output] = freduce(result[output], input(i0, i1));
      }
    }
    return result;
  }

  // Returns the result of reducing the 4D array to a vector, reducing away
  // the dimensions specified in dims.
  static std::vector<float> Reduce4DTo1D(
      const Array4D<float>& array, float init,
      tensorflow::gtl::ArraySlice<int64> dims,
      const std::function<float(float, float)>& reduce_function);

  // Broadcast 1D dimension to 4D, from the dimension `broadcast_from_dim`.
  static std::unique_ptr<Array4D<float>> Broadcast1DTo4D(
      const std::vector<float>& array, const std::vector<int64>& bounds,
      int64 broadcast_from_dim);

  // Returns the result of reducing the 3D array to a 2D array, reducing away
  // the dimensions specified in dims.
  static std::unique_ptr<Array2D<float>> Reduce3DTo2D(
      const Array3D<float>& array, float init,
      tensorflow::gtl::ArraySlice<int64> dims,
      const std::function<float(float, float)>& reduce_function);

  // Applies map_function to each element in the input (2D array) and returns
  // the result.
  static std::unique_ptr<Array2D<float>> MapArray2D(
      const Array2D<float>& matrix,
      const std::function<float(float)>& map_function);

  // Applies map_function to each pair of corresponding elements in the two
  // inputs arrays and returns the result.
  static std::unique_ptr<Array2D<float>> MapArray2D(
      const Array2D<float>& lhs, const Array2D<float>& rhs,
      const std::function<float(float, float)>& map_function);

  // Number of windows in a given dimension. Calculation taken from
  // xla::MakePadding().
  static int64 WindowCount(int64 unpadded_width, int64 window_len, int64 stride,
                           Padding padding);

  // Windowed reductions with Add as the function to apply.
  static std::unique_ptr<std::vector<float>> ReduceWindow1DAdd(
      const tensorflow::gtl::ArraySlice<float>& operand, float init,
      const tensorflow::gtl::ArraySlice<int64>& window,
      const tensorflow::gtl::ArraySlice<int64>& stride, Padding padding);
  static std::unique_ptr<Array2D<float>> ReduceWindow2DAdd(
      const Array2D<float>& operand, float init,
      const tensorflow::gtl::ArraySlice<int64>& window,
      const tensorflow::gtl::ArraySlice<int64>& stride, Padding padding);
  static std::unique_ptr<Array3D<float>> ReduceWindow3DAdd(
      const Array3D<float>& operand, float init,
      const tensorflow::gtl::ArraySlice<int64>& window,
      const tensorflow::gtl::ArraySlice<int64>& stride, Padding padding);
  static std::unique_ptr<Array4D<float>> ReduceWindow4DAdd(
      const Array4D<float>& operand, float init,
      const tensorflow::gtl::ArraySlice<int64>& window,
      const tensorflow::gtl::ArraySlice<int64>& stride, Padding padding);

  // Windowed reductions with a generic reduce function.
  static std::unique_ptr<std::vector<float>> ReduceWindow1DGeneric(
      const tensorflow::gtl::ArraySlice<float>& operand, float init,
      const std::function<float(float, float)>& reduce_func,
      const tensorflow::gtl::ArraySlice<int64>& window,
      const tensorflow::gtl::ArraySlice<int64>& stride,
      const tensorflow::gtl::ArraySlice<std::pair<int64, int64>>& padding);
  static std::unique_ptr<Array2D<float>> ReduceWindow2DGeneric(
      const Array2D<float>& operand, float init,
      const std::function<float(float, float)>& reduce_func,
      const tensorflow::gtl::ArraySlice<int64>& window,
      const tensorflow::gtl::ArraySlice<int64>& stride,
      const tensorflow::gtl::ArraySlice<std::pair<int64, int64>>& padding);
  static std::unique_ptr<Array4D<float>> ReduceWindow4DGeneric(
      const Array4D<float>& operand, float init,
      const std::function<float(float, float)>& reduce_func,
      const tensorflow::gtl::ArraySlice<int64>& window,
      const tensorflow::gtl::ArraySlice<int64>& stride, Padding padding);
  // With arbitrary padding.
  static std::unique_ptr<Array4D<float>> ReduceWindow4DGeneric(
      const Array4D<float>& operand, float init,
      const std::function<float(float, float)>& reduce_func,
      const tensorflow::gtl::ArraySlice<int64>& window,
      const tensorflow::gtl::ArraySlice<int64>& stride,
      const tensorflow::gtl::ArraySlice<std::pair<int64, int64>>& padding);

  // Batch normalize data.
  static std::unique_ptr<Array4D<float>> BatchNorm4D(
      const Array4D<float>& input, const Array4D<float>& mean,
      const Array4D<float>& var, const Array4D<float>& scale,
      const Array4D<float>& offset, float epsilon);

  // Performs select and scatter with Greater Than or equal as the select, plus
  // as the scatter, and Same Padding.
  // TODO(b/74533103) Switch tests to evaluator and remove this implementation.
  static std::unique_ptr<Array4D<float>> SelectAndScatter4DGePlus(
      const Array4D<float>& operand, const Array4D<float>& source, float init,
      const tensorflow::gtl::ArraySlice<int64>& window,
      const tensorflow::gtl::ArraySlice<int64>& stride, bool same_padding);

  // Concatenates the lhs and rhs arrays along the concatenate_dimension.
  // E.g. if concatenate_dimension is 0, the "n1"/height dimension is
  // concatenated, so the arrays are stacked on top of each other.
  template <typename T>
  static std::unique_ptr<Array2D<T>> Concat2D(const Array2D<T>& lhs,
                                              const Array2D<T>& rhs,
                                              int concatenate_dimension) {
    CHECK(0 <= concatenate_dimension && concatenate_dimension < 2);
    auto result = MakeUnique<Array2D<T>>(
        concatenate_dimension == 0 ? lhs.n1() + rhs.n1() : lhs.n1(),
        concatenate_dimension == 1 ? lhs.n2() + rhs.n2() : lhs.n2());
    for (int64 i0 = 0; i0 < result->n1(); ++i0) {
      for (int64 i1 = 0; i1 < result->n2(); ++i1) {
        // If we exceed the bounds of the LHS, draw from the RHS, where the
        // result index is adjusted by the number of values present in the LHS.
        (*result)(i0, i1) = i0 < lhs.n1() && i1 < lhs.n2()
                                ? lhs(i0, i1)
                                : rhs(i0 >= lhs.n1() ? i0 - lhs.n1() : i0,
                                      i1 >= lhs.n2() ? i1 - lhs.n2() : i1);
      }
    }
    return result;
  }

  // Concatenates the lhs and rhs 3D arrays along the concatenate_dimension. lhs
  // and rhs must have the same dimensions except for the concatenate dimension.
  template <typename T>
  static std::unique_ptr<Array3D<T>> Concat3D(const Array3D<T>& lhs,
                                              const Array3D<T>& rhs,
                                              int concatenate_dimension) {
    CHECK(0 <= concatenate_dimension && concatenate_dimension < 3);
    const int64 lhs_dims[] = {lhs.n1(), lhs.n2(), lhs.n3()};
    const int64 rhs_dims[] = {rhs.n1(), rhs.n2(), rhs.n3()};
    int64 out_dims[] = {rhs.n1(), rhs.n2(), rhs.n3()};
    for (int i = 0; i < 3; ++i) {
      if (i != concatenate_dimension) {
        out_dims[i] = lhs_dims[i];
        CHECK_EQ(lhs_dims[i], rhs_dims[i]);
      } else {
        out_dims[i] = lhs_dims[i] + rhs_dims[i];
      }
    }
    auto result = MakeUnique<Array3D<T>>(out_dims[0], out_dims[1], out_dims[2]);
    for (int64 i0 = 0; i0 < result->n1(); ++i0) {
      for (int64 i1 = 0; i1 < result->n2(); ++i1) {
        for (int64 i2 = 0; i2 < result->n3(); ++i2) {
          (*result)(i0, i1, i2) =
              i0 < lhs.n1() && i1 < lhs.n2() && i2 < lhs.n3()
                  ? lhs(i0, i1, i2)
                  : rhs(i0 >= lhs.n1() ? i0 - lhs.n1() : i0,
                        i1 >= lhs.n2() ? i1 - lhs.n2() : i1,
                        i2 >= lhs.n3() ? i2 - lhs.n3() : i2);
        }
      }
    }
    return result;
  }

  // Concatenates the lhs and rhs 4D arrays along the concatenate_dimension. lhs
  // and rhs must have the same dimensions except for the concatenate dimension.
  template <typename T>
  static std::unique_ptr<Array4D<T>> Concat4D(const Array4D<T>& lhs,
                                              const Array4D<T>& rhs,
                                              int concatenate_dimension) {
    CHECK(0 <= concatenate_dimension && concatenate_dimension < 4);
    const int64 lhs_dims[] = {lhs.n1(), lhs.n2(), lhs.n3(), lhs.n4()};
    const int64 rhs_dims[] = {rhs.n1(), rhs.n2(), rhs.n3(), rhs.n4()};
    int64 out_dims[] = {rhs.n1(), rhs.n2(), rhs.n3(), rhs.n4()};
    for (int i = 0; i < 4; ++i) {
      if (i != concatenate_dimension) {
        out_dims[i] = lhs_dims[i];
        CHECK_EQ(lhs_dims[i], rhs_dims[i]);
      } else {
        out_dims[i] = lhs_dims[i] + rhs_dims[i];
      }
    }
    auto result = MakeUnique<Array4D<T>>(out_dims[0], out_dims[1], out_dims[2],
                                         out_dims[3]);
    for (int64 i0 = 0; i0 < result->n1(); ++i0) {
      for (int64 i1 = 0; i1 < result->n2(); ++i1) {
        for (int64 i2 = 0; i2 < result->n3(); ++i2) {
          for (int64 i3 = 0; i3 < result->n4(); ++i3) {
            (*result)(i0, i1, i2, i3) =
                i0 < lhs.n1() && i1 < lhs.n2() && i2 < lhs.n3() && i3 < lhs.n4()
                    ? lhs(i0, i1, i2, i3)
                    : rhs(i0 >= lhs.n1() ? i0 - lhs.n1() : i0,
                          i1 >= lhs.n2() ? i1 - lhs.n2() : i1,
                          i2 >= lhs.n3() ? i2 - lhs.n3() : i2,
                          i3 >= lhs.n4() ? i3 - lhs.n4() : i3);
          }
        }
      }
    }
    return result;
  }

  // Slices with index clamping
  template <typename T>
  static std::vector<T> ClampSlice1D(
      const tensorflow::gtl::ArraySlice<T>& input, int64 start, int64 size) {
    start = std::min<int64>(std::max<int64>(0, start), input.size() - size);
    std::vector<T> result;
    for (int64 i = 0; i < size; ++i) {
      result.push_back(input[(start + i)]);
    }
    return result;
  }

  // Slices the input array given starting indices, limit indices, and strides
  // in each dimension.
  template <typename T>
  static std::unique_ptr<Array2D<T>> Slice2D(const Array2D<T>& input,
                                             std::array<int64, 2> starts,
                                             std::array<int64, 2> limits,
                                             std::array<int64, 2> strides) {
    CHECK_LE(starts[0], input.n1());
    CHECK_LE(starts[1], input.n2());
    CHECK_LE(limits[0], input.n1());
    CHECK_LE(limits[1], input.n2());
    CHECK_GE(strides[0], 1);
    CHECK_GE(strides[1], 1);
    auto result =
        MakeUnique<Array2D<T>>(CeilOfRatio(limits[0] - starts[0], strides[0]),
                               CeilOfRatio(limits[1] - starts[1], strides[1]));
    for (int64 i0 = 0; i0 < result->n1(); ++i0) {
      for (int64 i1 = 0; i1 < result->n2(); ++i1) {
        (*result)(i0, i1) =
            input(starts[0] + i0 * strides[0], starts[1] + i1 * strides[1]);
      }
    }
    return result;
  }

  template <typename T>
  static std::unique_ptr<Array3D<T>> Slice3D(const Array3D<T>& input,
                                             std::array<int64, 3> starts,
                                             std::array<int64, 3> limits,
                                             std::array<int64, 3> strides) {
    CHECK_LE(starts[0], input.n1());
    CHECK_LE(starts[1], input.n2());
    CHECK_LE(starts[2], input.n3());
    CHECK_LE(limits[0], input.n1());
    CHECK_LE(limits[1], input.n2());
    CHECK_LE(limits[2], input.n3());
    CHECK_GE(strides[0], 1);
    CHECK_GE(strides[1], 1);
    CHECK_GE(strides[2], 1);
    auto result =
        MakeUnique<Array3D<T>>(CeilOfRatio(limits[0] - starts[0], strides[0]),
                               CeilOfRatio(limits[1] - starts[1], strides[1]),
                               CeilOfRatio(limits[2] - starts[2], strides[2]));

    for (int64 i0 = 0; i0 < result->n1(); ++i0) {
      for (int64 i1 = 0; i1 < result->n2(); ++i1) {
        for (int64 i2 = 0; i2 < result->n3(); ++i2) {
          (*result)(i0, i1, i2) =
              input(starts[0] + i0 * strides[0], starts[1] + i1 * strides[1],
                    starts[2] + i2 * strides[2]);
        }
      }
    }
    return result;
  }

  template <typename T>
  static std::unique_ptr<Array4D<T>> Slice4D(const Array4D<T>& input,
                                             std::array<int64, 4> starts,
                                             std::array<int64, 4> limits,
                                             std::array<int64, 4> strides) {
    CHECK_LE(starts[0], input.n1());
    CHECK_LE(starts[1], input.n2());
    CHECK_LE(starts[2], input.n3());
    CHECK_LE(starts[3], input.n4());
    CHECK_LE(limits[0], input.n1());
    CHECK_LE(limits[1], input.n2());
    CHECK_LE(limits[2], input.n3());
    CHECK_LE(limits[3], input.n4());
    CHECK_GE(strides[0], 1);
    CHECK_GE(strides[1], 1);
    CHECK_GE(strides[2], 1);
    CHECK_GE(strides[3], 1);
    auto result =
        MakeUnique<Array4D<T>>(CeilOfRatio(limits[0] - starts[0], strides[0]),
                               CeilOfRatio(limits[1] - starts[1], strides[1]),
                               CeilOfRatio(limits[2] - starts[2], strides[2]),
                               CeilOfRatio(limits[3] - starts[3], strides[3]));
    for (int64 i0 = 0; i0 < result->n1(); ++i0) {
      for (int64 i1 = 0; i1 < result->n2(); ++i1) {
        for (int64 i2 = 0; i2 < result->n3(); ++i2) {
          for (int64 i3 = 0; i3 < result->n4(); ++i3) {
            (*result)(i0, i1, i2, i3) =
                input(starts[0] + i0 * strides[0], starts[1] + i1 * strides[1],
                      starts[2] + i2 * strides[2], starts[3] + i3 * strides[3]);
          }
        }
      }
    }
    return result;
  }

  // Applies map_function to each element in the input (2D array) and returns
  // the result.
  // (row, column) index of each element is also provided as arguments to
  // map_function.
  static std::unique_ptr<Array2D<float>> MapWithIndexArray2D(
      const Array2D<float>& matrix,
      const std::function<float(float, int64, int64)>& map_function);

  // Applies map_function to each element in the input (4D array) and returns
  // the result.
  template <typename F>
  static std::unique_ptr<Array4D<float>> MapArray4D(const Array4D<float>& input,
                                                    F&& map_function) {
    return MapWithIndexArray4D(input,
                               [&](float value, int64, int64, int64, int64) {
                                 return map_function(value);
                               });
  }

  // Applies map_function to each element in the input (4D array) and returns
  // the result.
  // (plane, depth, height, width) index of each element is also provided as
  // arguments to map_function.
  template <typename F>
  static std::unique_ptr<Array4D<float>> MapWithIndexArray4D(
      const Array4D<float>& input, F&& map_function) {
    auto result = MakeUnique<Array4D<float>>(input.planes(), input.depth(),
                                             input.height(), input.width());
    for (int64 plane = 0; plane < input.planes(); ++plane) {
      for (int64 depth = 0; depth < input.depth(); ++depth) {
        for (int64 height = 0; height < input.height(); ++height) {
          for (int64 width = 0; width < input.width(); ++width) {
            (*result)(plane, depth, height, width) =
                map_function(input(plane, depth, height, width), plane, depth,
                             height, width);
          }
        }
      }
    }
    return result;
  }

  // Applies map_function to each pair of elements in the input lhs and rhs
  // (4D array) and returns the result.
  template <typename F>
  static std::unique_ptr<Array4D<float>> MapArray4D(const Array4D<float>& lhs,
                                                    const Array4D<float>& rhs,
                                                    F&& map_function) {
    return MapWithIndexArray4D(
        lhs, rhs, [&](float lhs, float rhs, int64, int64, int64, int64) {
          return map_function(lhs, rhs);
        });
  }

  // Applies map_function to each pair of element in lhs and rhs (4D array) and
  // returns the result.
  // (plane, depth, height, width) index of each element is also provided as
  // arguments to map_function.
  template <typename F>
  static std::unique_ptr<Array4D<float>> MapWithIndexArray4D(
      const Array4D<float>& lhs, const Array4D<float>& rhs, F&& map_function) {
    auto result = MakeUnique<Array4D<float>>(lhs.planes(), lhs.depth(),
                                             lhs.height(), lhs.width());
    for (int64 plane = 0; plane < lhs.planes(); ++plane) {
      for (int64 depth = 0; depth < lhs.depth(); ++depth) {
        for (int64 height = 0; height < lhs.height(); ++height) {
          for (int64 width = 0; width < lhs.width(); ++width) {
            (*result)(plane, depth, height, width) = map_function(
                lhs(plane, depth, height, width),
                rhs(plane, depth, height, width), plane, depth, height, width);
          }
        }
      }
    }
    return result;
  }

  // Returns the result of a 2D pad on an input matrix.
  template <typename NativeT>
  static std::unique_ptr<Array2D<NativeT>> PadArray2D(
      const Array2D<NativeT>& operand, const PaddingConfig& padding,
      const NativeT pad) {
    int64 in0 = operand.n1();
    int64 high_padding0 = padding.dimensions(0).edge_padding_high();
    int64 low_padding0 = padding.dimensions(0).edge_padding_low();
    int64 interior_padding0 = padding.dimensions(0).interior_padding();
    int64 out0 =
        in0 + low_padding0 + high_padding0 + (in0 - 1) * interior_padding0;

    int64 in1 = operand.n2();
    int64 high_padding1 = padding.dimensions(1).edge_padding_high();
    int64 low_padding1 = padding.dimensions(1).edge_padding_low();
    int64 interior_padding1 = padding.dimensions(1).interior_padding();
    int64 out1 =
        in1 + low_padding1 + high_padding1 + (in1 - 1) * interior_padding1;

    auto result = MakeUnique<Array2D<NativeT>>(out0, out1);
    result->Fill(pad);
    int64 o0 = low_padding0;
    for (int64 i0 = 0; i0 < in0; ++i0) {
      int64 o1 = low_padding1;
      for (int64 i1 = 0; i1 < in1; ++i1) {
        if (o0 >= 0 && o1 >= 0 && o0 < out0 && o1 < out1) {
          (*result)(o0, o1) = operand(i0, i1);
        }
        o1 += interior_padding1 + 1;
      }
      o0 += interior_padding0 + 1;
    }
    return result;
  }

  // Returns the result of a 3D pad on an input matrix.
  template <typename NativeT>
  static Array3D<NativeT> PadArray3D(const Array3D<NativeT>& operand,
                                     const PaddingConfig& padding,
                                     const NativeT pad) {
    CHECK_EQ(padding.dimensions_size(), 3);

    const int64 input_bounds[] = {operand.n1(), operand.n2(), operand.n3()};
    int64 pad_low[3];
    int64 pad_high[3];
    int64 pad_interior[3];
    int64 output_bounds[3];
    for (int64 i = 0; i < 3; ++i) {
      pad_low[i] = padding.dimensions(i).edge_padding_low();
      pad_high[i] = padding.dimensions(i).edge_padding_high();
      CHECK_LE(0, pad_low[i]);
      CHECK_LE(0, pad_high[i]);
      CHECK_LE(0, padding.dimensions(i).interior_padding())
          << "not implemented";
      pad_interior[i] = padding.dimensions(i).interior_padding();

      output_bounds[i] = pad_low[i] + input_bounds[i] + pad_high[i] +
                         (input_bounds[i] - 1) * pad_interior[i];
    }

    Array3D<NativeT> result(output_bounds[0], output_bounds[1],
                            output_bounds[2]);
    int indices[] = {0, 0, 0};
    for (indices[0] = 0; indices[0] < output_bounds[0]; ++indices[0]) {
      for (indices[1] = 0; indices[1] < output_bounds[1]; ++indices[1]) {
        for (indices[2] = 0; indices[2] < output_bounds[2]; ++indices[2]) {
          NativeT* value = &result(indices[0], indices[1], indices[2]);
          bool value_padded = false;
          for (int i = 0; i < 3; ++i) {
            bool in_low_padding = indices[i] < pad_low[i];
            bool in_high_padding = indices[i] >= output_bounds[i] - pad_high[i];
            if (in_low_padding || in_high_padding) {
              *value = pad;
              value_padded = true;
            }
            if (pad_interior[i] &&
                (indices[i] - pad_low[i]) % (pad_interior[i] + 1)) {
              *value = pad;
              value_padded = true;
            }
          }
          if (value_padded) {
            continue;
          }
          *value = operand((indices[0] - pad_low[0]) / (pad_interior[0] + 1),
                           (indices[1] - pad_low[1]) / (pad_interior[1] + 1),
                           (indices[2] - pad_low[2]) / (pad_interior[2] + 1));
        }
      }
    }
    return result;
  }

  // Returns the result of a 4D pad on an input array.
  template <typename NativeT>
  static Array4D<NativeT> PadArray4D(const Array4D<NativeT>& operand,
                                     const PaddingConfig& padding,
                                     const NativeT pad) {
    CHECK_EQ(padding.dimensions_size(), 4);

    const int64 input_bounds[] = {operand.n1(), operand.n2(), operand.n3(),
                                  operand.n4()};
    int64 pad_low[4];
    int64 pad_high[4];
    int64 pad_interior[4];
    int64 output_bounds[4];
    for (int64 i = 0; i < 4; ++i) {
      pad_low[i] = padding.dimensions(i).edge_padding_low();
      pad_high[i] = padding.dimensions(i).edge_padding_high();
      CHECK_LE(0, padding.dimensions(i).interior_padding())
          << "not implemented";
      pad_interior[i] = padding.dimensions(i).interior_padding();

      output_bounds[i] = pad_low[i] + input_bounds[i] + pad_high[i] +
                         (input_bounds[i] - 1) * pad_interior[i];
    }

    Array4D<NativeT> result(output_bounds[0], output_bounds[1],
                            output_bounds[2], output_bounds[3]);
    result.Each(
        [&](tensorflow::gtl::ArraySlice<int64> indices, NativeT* value) {
          for (int i = 0; i < 4; ++i) {
            bool in_low_padding = indices[i] < pad_low[i];
            bool in_high_padding = indices[i] >= output_bounds[i] - pad_high[i];
            if (in_low_padding || in_high_padding) {
              *value = pad;
              return;
            }
            if (pad_interior[i] &&
                (indices[i] - pad_low[i]) % (pad_interior[i] + 1)) {
              *value = pad;
              return;
            }
          }
          *value = operand((indices[0] - pad_low[0]) / (pad_interior[0] + 1),
                           (indices[1] - pad_low[1]) / (pad_interior[1] + 1),
                           (indices[2] - pad_low[2]) / (pad_interior[2] + 1),
                           (indices[3] - pad_low[3]) / (pad_interior[3] + 1));
        });
    return result;
  }

  // ApplyElementwise2D(f, x, y, ...) returns the Array2D formed by running
  // f(x[i], y[i], ...) for each array element in the Array2Ds x, y, ....
  //
  // The given arrays must have the same size and element type, and the return
  // type of f must be implicitly convertible to the arrays' element type.
  //
  // Example usage:
  //
  //   Array2D<float> x, y, z = ...;
  //   std::unique_ptr<Array2D> result = ReferenceUtil::ApplyElementwise2D(
  //     [](float a, float b, float c) { return a * b + c; }, x, y, z);
  //
  template <typename F, typename T1, typename... Ts>
  static std::unique_ptr<Array2D<T1>> ApplyElementwise2D(
      F&& f, const Array2D<T1>& array1, const Array2D<Ts>&... arrays) {
    AssertSameSize2D(array1, arrays...);
    auto result = MakeUnique<Array2D<T1>>(array1.n1(), array1.n2());
    for (int64 i = 0; i < array1.n1(); ++i) {
      for (int64 j = 0; j < array1.n2(); ++j) {
        (*result)(i, j) = f(array1(i, j), arrays(i, j)...);
      }
    }
    return result;
  }

 private:
  template <typename T1, typename T2, typename... Ts>
  static void AssertSameSize2D(const Array2D<T1>& array1,
                               const Array2D<T2>& array2,
                               const Array2D<Ts>&... arrays) {
    static_assert(std::is_same<T1, T2>::value, "Args must be same type.");
    CHECK_EQ(array1.n1(), array2.n1());
    CHECK_EQ(array1.n2(), array2.n2());
    AssertSameSize2D(array2, arrays...);
  }

  // Recursive base case for AssertSameSize2D.
  template <typename Array1>
  static void AssertSameSize2D(const Array1& array1) {}

  TF_DISALLOW_COPY_AND_ASSIGN(ReferenceUtil);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_REFERENCE_UTIL_H_
