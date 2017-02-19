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
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Utility class for reference implementations of linear algebra routines.
class ReferenceUtil {
 public:
  // Returns the result of a transpose operation on the input matrix.
  static std::unique_ptr<Array2D<float>> TransposeArray2D(
      const Array2D<float>& operand);

  // Returns the result of a matrix multiply `lhs x rhs`.
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
      std::pair<int64, int64> stride, Padding padding,
      std::pair<int64, int64> lhs_dilation,
      std::pair<int64, int64> rhs_dilation, ConvolutionDimensionNumbers dnums);

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
      std::function<float(float, float)> reduce_function);

  // Returns the result of reducing a matrix to a row vector. init is the
  // initial value for the reduce operation, and reduce_function is the function
  // to apply for each reduction step.
  static std::unique_ptr<std::vector<float>> ReduceToRowArray2D(
      const Array2D<float>& matrix, float init,
      std::function<float(float, float)> reduce_function);

  // Performs a R2=>R1 reduction by reducing away the dimension specified in
  // 'dimension_to_reduce'.
  template <typename T>
  static std::vector<T> ReduceR2ToR1(const Array2D<T>& input,
                                     int dimension_to_reduce, T init,
                                     std::function<T(T, T)> freduce) {
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
      std::function<float(float, float)> reduce_function);

  // Returns the result of reducing the 3D array to a 2D array, reducing away
  // the dimensions specified in dims.
  static std::unique_ptr<Array2D<float>> Reduce3DTo2D(
      const Array3D<float>& array, float init,
      tensorflow::gtl::ArraySlice<int64> dims,
      std::function<float(float, float)> reduce_function);

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

  // Performs a 2D window reduction with Add as the function to apply.
  static std::unique_ptr<Array2D<float>> ReduceWindow2DAdd(
      const Array2D<float>& operand, float init,
      const tensorflow::gtl::ArraySlice<int64>& window,
      const tensorflow::gtl::ArraySlice<int64>& stride, Padding padding);

  // Performs a 4D window reduction with Add as the function to apply.
  static std::unique_ptr<Array4D<float>> ReduceWindow4DAdd(
      const Array4D<float>& operand, float init,
      const tensorflow::gtl::ArraySlice<int64>& window,
      const tensorflow::gtl::ArraySlice<int64>& stride, Padding padding);

  // Performs a 4D window reduction with a generic reduce function.
  static std::unique_ptr<Array4D<float>> ReduceWindow4DGeneric(
      const Array4D<float>& operand, float init,
      const std::function<float(float, float)>& reduce_func,
      const tensorflow::gtl::ArraySlice<int64>& window,
      const tensorflow::gtl::ArraySlice<int64>& stride, Padding padding);

  // Performs select and scatter with Greater Than or equal as the select, plus
  // as the scatter, and Same Padding.
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
    std::vector<int64> lhs_dims = {lhs.n1(), lhs.n2(), lhs.n3()};
    std::vector<int64> rhs_dims = {rhs.n1(), rhs.n2(), rhs.n3()};
    std::vector<int64> out_dims = {rhs.n1(), rhs.n2(), rhs.n3()};
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
    std::vector<int64> lhs_dims = {lhs.n1(), lhs.n2(), lhs.n3(), lhs.n4()};
    std::vector<int64> rhs_dims = {rhs.n1(), rhs.n2(), rhs.n3(), rhs.n4()};
    std::vector<int64> out_dims = {rhs.n1(), rhs.n2(), rhs.n3(), rhs.n4()};
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

  // Slices with modulo-wrapping.
  template <typename T>
  static std::vector<T> ModSlice1D(const tensorflow::gtl::ArraySlice<T>& input,
                                   int64 start, int64 size) {
    std::vector<T> result;
    for (int64 i = 0; i < size; ++i) {
      result.push_back(input[(start + i) % input.size()]);
    }
    return result;
  }

  // Slices the input array given starting indices in each dimension and limit
  // indices in each dimension.
  template <typename T>
  static std::unique_ptr<Array2D<T>> Slice2D(const Array2D<T>& input,
                                             std::array<int64, 2> starts,
                                             std::array<int64, 2> limits) {
    CHECK_LE(starts[0], input.n1());
    CHECK_LE(starts[1], input.n2());
    CHECK_LE(limits[0], input.n1());
    CHECK_LE(limits[1], input.n2());
    auto result =
        MakeUnique<Array2D<T>>(limits[0] - starts[0], limits[1] - starts[1]);
    for (int64 i0 = 0; i0 < result->n1(); ++i0) {
      for (int64 i1 = 0; i1 < result->n2(); ++i1) {
        (*result)(i0, i1) = input(starts[0] + i0, starts[1] + i1);
      }
    }
    return result;
  }

  template <typename T>
  static std::unique_ptr<Array4D<T>> Slice4D(const Array4D<T>& input,
                                             std::array<int64, 4> starts,
                                             std::array<int64, 4> limits) {
    CHECK_LE(starts[0], input.n1());
    CHECK_LE(starts[1], input.n2());
    CHECK_LE(starts[2], input.n3());
    CHECK_LE(starts[3], input.n4());
    CHECK_LE(limits[0], input.n1());
    CHECK_LE(limits[1], input.n2());
    CHECK_LE(limits[2], input.n3());
    CHECK_LE(limits[3], input.n4());
    auto result =
        MakeUnique<Array4D<T>>(limits[0] - starts[0], limits[1] - starts[1],
                               limits[2] - starts[2], limits[3] - starts[3]);
    for (int64 i0 = 0; i0 < result->n1(); ++i0) {
      for (int64 i1 = 0; i1 < result->n2(); ++i1) {
        for (int64 i2 = 0; i2 < result->n3(); ++i2) {
          for (int64 i3 = 0; i3 < result->n4(); ++i3) {
            (*result)(i0, i1, i2, i3) = input(starts[0] + i0, starts[1] + i1,
                                              starts[2] + i2, starts[3] + i3);
          }
        }
      }
    }
    return result;
  }

  template <typename T>
  static std::unique_ptr<Array3D<T>> Slice3D(const Array3D<T>& input,
                                             std::array<int64, 3> starts,
                                             std::array<int64, 3> limits) {
    CHECK_LE(starts[0], input.n1());
    CHECK_LE(starts[1], input.n2());
    CHECK_LE(starts[2], input.n3());
    CHECK_LE(limits[0], input.n1());
    CHECK_LE(limits[1], input.n2());
    CHECK_LE(limits[2], input.n3());
    auto result = MakeUnique<Array3D<T>>(
        limits[0] - starts[0], limits[1] - starts[1], limits[2] - starts[2]);
    for (int64 i0 = 0; i0 < result->n1(); ++i0) {
      for (int64 i1 = 0; i1 < result->n2(); ++i1) {
        for (int64 i2 = 0; i2 < result->n3(); ++i2) {
          (*result)(i0, i1, i2) =
              input(starts[0] + i0, starts[1] + i1, starts[2] + i2);
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

  // Returns the result of a 2D pad on an input matrix.
  static std::unique_ptr<Array2D<float>> PadArray2D(
      const Array2D<float>& operand, const PaddingConfig& padding,
      const float pad);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ReferenceUtil);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_REFERENCE_UTIL_H_
