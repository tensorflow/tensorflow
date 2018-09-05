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

// Utilities for dealing with Literal protobufs.

#ifndef TENSORFLOW_COMPILER_XLA_LITERAL_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_LITERAL_UTIL_H_

#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/index_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/sparse_index_array.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/bitmap.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

class LiteralUtil {
 public:
  LiteralUtil() = delete;

  // Returns a literal scalar representing the first element.
  static Literal GetFirstScalarLiteral(const LiteralSlice& literal);

  // Creates a new literal of a given rank. To minimize ambiguity (for users
  // and the compiler) these CreateR[0-2] methods should explicitly specify the
  // native type. For example:
  //
  //  CreateR1<float>({1.0, 42.0});
  //  CreateR2<uint32>({{1, 2}, {3, 4}});
  //
  // The variants not ending with WithLayout use the default XLA layout for the
  // literal's linear representation in memory.
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR0(NativeT value);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR1(absl::Span<const NativeT> values);
  static std::unique_ptr<Literal> CreateR1(
      const tensorflow::core::Bitmap& values);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR2(
      std::initializer_list<std::initializer_list<NativeT>> values);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR2WithLayout(
      std::initializer_list<std::initializer_list<NativeT>> values,
      const Layout& layout);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR3(
      std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>
          values);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR3WithLayout(
      std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>
          values,
      const Layout& layout);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR4(
      std::initializer_list<std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>>
          values);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR4WithLayout(
      std::initializer_list<std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>>
          values,
      const Layout& layout);

  // Creates a literal with a sparse layout and the given indices and values.
  // The shape is initialized from the given dimensions.  The minor dimension of
  // the indices array must equal the rank of the shape (i.e. size of the
  // dimensions array). The major dimension of the indices array must equal the
  // number of elements in the values array. The maximum number of elements in
  // the array is taken from the max_indices() value of the index array.
  //
  // XLA assumes that sparse literals are in sorted order for all operations. If
  // the `sort` argument is true, then the indices and values will be sorted
  // while copying them into the literal. If you have ensured that the indices
  // and values are already sorted, then you may set the `sort` argument to
  // false to skip the sorting step.
  //
  // For example:
  //
  //   CreateSparse(
  //     {12, 12, 12},
  //     SparseIndexArray(10, 3,
  //                      Array2D{
  //                        {0, 1, 2},
  //                        {3, 4, 5},
  //                        {6, 7, 8},
  //                        {9, 10, 11},
  //                      }),
  //     {1.0, 2.0 3.0, 4.0})
  //
  // This creates an array with shape F64[12,12,12]sparse{10}, that has the
  // following non-zero values:
  //
  //     [0,  1,  2]: 1.0
  //     [3,  4,  5]: 2.0
  //     [6,  7,  8]: 3.0
  //     [9, 10, 11]: 4.0
  //
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateSparse(
      absl::Span<const int64> dimensions, SparseIndexArray indices,
      absl::Span<const NativeT> values, bool sort = true);

  // Creates a scalar literal value zero of the given primitive type.
  static Literal Zero(PrimitiveType primitive_type);
  // Creates a scalar literal value one of the given primitive type.
  static Literal One(PrimitiveType primitive_type);
  // Creates a scalar literal value containing the minimum value of the given
  // primitive type. For floating-point types, returns -inf.
  static Literal MinValue(PrimitiveType primitive_type);
  // Creates a scalar literal value containing the maximum value of the given
  // primitive type. For floating-point types, returns inf.
  static Literal MaxValue(PrimitiveType primitive_type);
  // Creates a literal of the given shape where each element is `value`.
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateFullWithDescendingLayout(
      absl::Span<const int64> dimensions, NativeT value);

  // Creates a new literal from an Array type. The variants not ending with
  // WithLayout use the default XLA layout for the literal's linear
  // representation in memory.
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateFromArray(const Array<NativeT>& values);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateFromArrayWithLayout(
      const Array<NativeT>& values, const Layout& layout);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR2FromArray2D(
      const Array2D<NativeT>& values);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR2FromArray2DWithLayout(
      const Array2D<NativeT>& values, const Layout& layout);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR3FromArray3D(
      const Array3D<NativeT>& values);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR3FromArray3DWithLayout(
      const Array3D<NativeT>& values, const Layout& layout);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR4FromArray4D(
      const Array4D<NativeT>& values);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR4FromArray4DWithLayout(
      const Array4D<NativeT>& values, const Layout& layout);

  // Creates a new vector of U8s literal value from a string.
  static std::unique_ptr<Literal> CreateR1U8(absl::string_view value);

  // Creates a linspace-populated literal with the given number of rows and
  // columns.
  static std::unique_ptr<Literal> CreateR2F32Linspace(float from, float to,
                                                      int64 rows, int64 cols);

  // Creates a literal that projects the (x, y) dimensions given in values into
  // the z dimension given by "projection".
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR3Projected(
      std::initializer_list<std::initializer_list<NativeT>> values,
      int64 projection);

  // Creates a literal that projects the (x, y) dimensions given in values into
  // the z and p dimensions given.
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR4Projected(
      std::initializer_list<std::initializer_list<NativeT>> values,
      int64 projection_p, int64 projection_z);

  // Returns an identity matrix (rank 2) with the given row and column count.
  template <typename NativeT>
  static std::unique_ptr<Literal> MakeIdentityR2(int64 size);

  // Returns a tuple literal composed of given literals. Data is copied from the
  // given elements into the returned literal.
  static std::unique_ptr<Literal> MakeTuple(
      absl::Span<const Literal* const> elements);

  static std::unique_ptr<Literal> MakeTupleFromSlices(
      absl::Span<const LiteralSlice> elements);

  // As above, but intended to be invoked with move semantics; i.e.
  //
  //  std::vector<std::unique_ptr<Literal>> elements = ...;
  //  auto result = LiteralUtil::MakeTupleOwned(std::move(elements));
  //
  // This would have been declared as an overload, but there is ambiguity
  // in invocation between the above signature and this one.
  static std::unique_ptr<Literal> MakeTupleOwned(
      std::vector<std::unique_ptr<Literal>> elements);

  // This overload lets you pass a braced list of unique_ptr<Literal>s to
  // MakeTupleOwned:
  //
  //   LiteralUtil::MakeTupleOwned(LiteralUtil::CreateR1(...), ...).
  //
  // Simply relying on the MakeTupleOwned(std::vector<unique_ptr<Literal>>)
  // overload doesn't work because std::initializer_list's elements are always
  // const.
  //
  // The arguments to this function must all be unique_ptr<Literal>.
  template <typename... Ts>
  static std::unique_ptr<Literal> MakeTupleOwned(
      std::unique_ptr<Ts>... elements) {
    std::array<std::unique_ptr<Literal>, sizeof...(Ts)> arr{
        std::move(elements)...};
    std::vector<std::unique_ptr<Literal>> v;
    v.insert(v.begin(), std::make_move_iterator(arr.begin()),
             std::make_move_iterator(arr.end()));
    return MakeTupleOwned(std::move(v));
  }

  // Create a constant token literal. Token types have no value.
  static std::unique_ptr<Literal> CreateToken();

  // Creates a new Literal object with its values havings the primitive_type
  // type, and with dimensions defined by the dimensions parameter.
  // The content of the literal values is the default value of the primitive
  // type of literal itself (0 for numeric types, and false for predicates).
  static std::unique_ptr<Literal> CreateFromDimensions(
      PrimitiveType primitive_type, absl::Span<const int64> dimensions);

  // If the given literal's data type is bfloat16, converts it to a float
  // literal; otherwise, returns a copy of it. If the literal is a tuple,
  // recursively converts its elements.
  static std::unique_ptr<Literal> ConvertBF16ToF32(
      const LiteralSlice& bf16_literal);

  // If the given literal's data type is float, converts it to a bfloat16
  // literal; otherwise, returns a copy of it. If the literal is a tuple,
  // recursively converts its elements.
  static std::unique_ptr<Literal> ConvertF32ToBF16(
      const LiteralSlice& f32_literal);

  // Creates a literal with a new shape with the given new dimensions using the
  // data in the given input literal. For reshaping purposes the (flat) data
  // buffer of the input literal is assumed to have the given minor_to_major
  // layout order.
  static std::unique_ptr<Literal> ReshapeSlice(
      absl::Span<const int64> new_dimensions,
      absl::Span<const int64> minor_to_major, const LiteralSlice& literal);

  // Creates a literal with the supplied shape, and uses the provided value
  // generator to populate the literal's values.
  // Returns the new literal object, or an error Status if failed.
  template <
      PrimitiveType type,
      typename T = typename primitive_util::PrimitiveTypeToNative<type>::type>
  static StatusOr<std::unique_ptr<Literal>> CreateRandomLiteral(
      const Shape& shape,
      const std::function<T(absl::Span<const int64>)>& generator);

  // Creates a literal with the supplied shape, and initializes the literal
  // values using a normal distribution with given mean and stddev standard
  // deviation, and using the engine as entropy generator.
  // Returns the new literal object, or an error Status if failed.
  template <
      PrimitiveType type, typename E,
      typename T = typename primitive_util::PrimitiveTypeToNative<type>::type>
  static StatusOr<std::unique_ptr<Literal>> CreateRandomLiteral(
      const Shape& shape, E* engine, T mean, T stddev);

  // Creates a literal with the supplied shape, and initializes the literal
  // values using a normal distribution with given mean and stddev standard
  // deviation.
  // Returns the new literal object, or an error Status if failed.
  template <
      PrimitiveType type,
      typename T = typename primitive_util::PrimitiveTypeToNative<type>::type>
  static StatusOr<std::unique_ptr<Literal>> CreateRandomLiteral(
      const Shape& shape, T mean, T stddev);

  //
  // End of factory methods.

  // Returns a multi-dimensional index as a string. For example: '{7, 8}' will
  // be returned for a 2-dimensional index with dimension 0 index equal to 7,
  // dimension 1 equal to 8.
  static string MultiIndexAsString(absl::Span<const int64> multi_index);
};

std::ostream& operator<<(std::ostream& out, const Literal& literal);

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR0(NativeT value) {
  auto literal = absl::make_unique<Literal>(ShapeUtil::MakeShape(
      primitive_util::NativeToPrimitiveType<NativeT>(), {}));
  literal->Set({}, value);
  return literal;
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR1(
    absl::Span<const NativeT> values) {
  auto literal = absl::make_unique<Literal>(
      ShapeUtil::MakeShape(primitive_util::NativeToPrimitiveType<NativeT>(),
                           {static_cast<int64>(values.size())}));
  literal->PopulateR1(values);
  return literal;
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR2WithLayout(
    std::initializer_list<std::initializer_list<NativeT>> values,
    const Layout& layout) {
  auto literal = absl::make_unique<Literal>(ShapeUtil::MakeShapeWithLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(),
      {static_cast<int64>(values.size()),
       static_cast<int64>(values.begin()->size())},
      AsInt64Slice(layout.minor_to_major())));
  literal->PopulateR2(values);
  return literal;
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR2(
    std::initializer_list<std::initializer_list<NativeT>> values) {
  return CreateR2WithLayout(values, LayoutUtil::GetDefaultLayoutForR2());
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR3WithLayout(
    std::initializer_list<std::initializer_list<std::initializer_list<NativeT>>>
        values,
    const Layout& layout) {
  const int64 d0 = values.size();
  const int64 d1 = values.begin()->size();
  const int64 d2 = values.begin()->begin()->size();
  Array3D<NativeT> tmp(d0, d1, d2);
  int64 i0 = 0;
  for (auto d1_values : values) {
    int64 i1 = 0;
    for (auto d2_values : d1_values) {
      int64 i2 = 0;
      for (auto value : d2_values) {
        tmp(i0, i1, i2) = value;
        ++i2;
      }
      ++i1;
    }
    ++i0;
  }
  return CreateR3FromArray3DWithLayout(tmp, layout);
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR3(
    std::initializer_list<std::initializer_list<std::initializer_list<NativeT>>>
        values) {
  return CreateR3WithLayout(values, LayoutUtil::GetDefaultLayoutForR3());
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR4WithLayout(
    std::initializer_list<std::initializer_list<
        std::initializer_list<std::initializer_list<NativeT>>>>
        values,
    const Layout& layout) {
  const int64 d0 = values.size();
  const int64 d1 = values.begin()->size();
  const int64 d2 = values.begin()->begin()->size();
  const int64 d3 = values.begin()->begin()->begin()->size();
  Array4D<NativeT> tmp(d0, d1, d2, d3);
  int64 i0 = 0;
  for (auto d1_values : values) {
    int64 i1 = 0;
    for (auto d2_values : d1_values) {
      int64 i2 = 0;
      for (auto d3_values : d2_values) {
        int64 i3 = 0;
        for (auto value : d3_values) {
          tmp(i0, i1, i2, i3) = value;
          ++i3;
        }
        ++i2;
      }
      ++i1;
    }
    ++i0;
  }
  return CreateR4FromArray4DWithLayout(tmp, layout);
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateSparse(
    absl::Span<const int64> dimensions, SparseIndexArray indices,
    absl::Span<const NativeT> values, bool sort) {
  int64 num_elements = values.size();
  int64 rank = dimensions.size();
  CHECK_EQ(num_elements, indices.index_count());
  CHECK_EQ(rank, indices.rank());
  auto literal =
      absl::make_unique<Literal>(ShapeUtil::MakeShapeWithSparseLayout(
          primitive_util::NativeToPrimitiveType<NativeT>(), dimensions,
          indices.max_indices()));
  literal->PopulateSparse(indices, values, sort);
  return literal;
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR4(
    std::initializer_list<std::initializer_list<
        std::initializer_list<std::initializer_list<NativeT>>>>
        values) {
  return CreateR4WithLayout(values, LayoutUtil::GetDefaultLayoutForR4());
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateFromArrayWithLayout(
    const Array<NativeT>& values, const Layout& layout) {
  auto literal = absl::make_unique<Literal>(ShapeUtil::MakeShapeWithLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(), values.dimensions(),
      AsInt64Slice(layout.minor_to_major())));
  literal->PopulateFromArray(values);
  return literal;
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateFromArray(
    const Array<NativeT>& values) {
  return CreateFromArrayWithLayout(
      values, LayoutUtil::GetDefaultLayoutForRank(values.num_dimensions()));
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal>
LiteralUtil::CreateR2FromArray2DWithLayout(const Array2D<NativeT>& values,
                                           const Layout& layout) {
  return CreateFromArrayWithLayout(values, layout);
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR2FromArray2D(
    const Array2D<NativeT>& values) {
  return CreateFromArray(values);
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal>
LiteralUtil::CreateR3FromArray3DWithLayout(const Array3D<NativeT>& values,
                                           const Layout& layout) {
  return CreateFromArrayWithLayout(values, layout);
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR3FromArray3D(
    const Array3D<NativeT>& values) {
  return CreateFromArray(values);
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR3Projected(
    std::initializer_list<std::initializer_list<NativeT>> values,
    int64 projection) {
  int64 dim0_size = projection;
  int64 dim1_size = values.size();
  int64 dim2_size = values.begin()->size();

  Array3D<NativeT> array(dim0_size, dim1_size, dim2_size);
  for (int64 dim0 = 0; dim0 < dim0_size; ++dim0) {
    int64 dim1 = 0;
    for (auto inner_list : values) {
      int64 dim2 = 0;
      for (auto value : inner_list) {
        array(dim0, dim1, dim2) = value;
        ++dim2;
      }
      CHECK_EQ(dim2_size, dim2);
      ++dim1;
    }
    CHECK_EQ(dim1_size, dim1);
  }
  return CreateR3FromArray3D(array);
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR4Projected(
    std::initializer_list<std::initializer_list<NativeT>> values,
    int64 projection_p, int64 projection_z) {
  int64 dim0_size = projection_p;
  int64 dim1_size = projection_z;
  int64 dim2_size = values.size();
  int64 dim3_size = values.begin()->size();

  Array4D<NativeT> array(dim0_size, dim1_size, dim2_size, dim3_size);
  for (int64 dim0 = 0; dim0 < dim0_size; ++dim0) {
    for (int64 dim1 = 0; dim1 < dim1_size; ++dim1) {
      int64 dim2 = 0;
      for (auto inner_list : values) {
        int64 dim3 = 0;
        for (auto value : inner_list) {
          array(dim0, dim1, dim2, dim3) = value;
          ++dim3;
        }
        CHECK_EQ(dim3_size, dim3);
        ++dim2;
      }
      CHECK_EQ(dim2_size, dim2);
    }
  }
  return CreateR4FromArray4D(array);
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR4FromArray4D(
    const Array4D<NativeT>& values) {
  return CreateFromArray(values);
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal>
LiteralUtil::CreateR4FromArray4DWithLayout(const Array4D<NativeT>& values,
                                           const Layout& layout) {
  return CreateFromArrayWithLayout(values, layout);
}

// Returns an identity matrix (rank 2) with the given row and column count.
template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::MakeIdentityR2(int64 size) {
  Array2D<NativeT> array(size, size, 0);
  for (int64 i = 0; i < size; ++i) {
    array(i, i) = 1;
  }
  return CreateR2FromArray2D(array);
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal>
LiteralUtil::CreateFullWithDescendingLayout(absl::Span<const int64> dimensions,
                                            NativeT value) {
  auto literal =
      absl::make_unique<Literal>(ShapeUtil::MakeShapeWithDescendingLayout(
          primitive_util::NativeToPrimitiveType<NativeT>(), dimensions));
  literal->PopulateWithValue(value);
  return literal;
}

template <PrimitiveType type, typename T>
/* static */ StatusOr<std::unique_ptr<Literal>>
LiteralUtil::CreateRandomLiteral(
    const Shape& shape,
    const std::function<T(absl::Span<const int64>)>& generator) {
  using NativeT = typename primitive_util::PrimitiveTypeToNative<type>::type;
  TF_RET_CHECK(shape.element_type() == type);
  auto literal = absl::make_unique<Literal>(shape);
  TF_RETURN_IF_ERROR(literal.get()->Populate<NativeT>(
      [&](absl::Span<const int64> indexes) { return generator(indexes); }));
  return std::move(literal);
}

template <PrimitiveType type, typename E, typename T>
/* static */ StatusOr<std::unique_ptr<Literal>>
LiteralUtil::CreateRandomLiteral(const Shape& shape, E* engine, T mean,
                                 T stddev) {
  using NativeT = typename primitive_util::PrimitiveTypeToNative<type>::type;
  std::normal_distribution<NativeT> generator(mean, stddev);
  return CreateRandomLiteral<type, NativeT>(
      shape,
      [&](absl::Span<const int64> /*indexes*/) { return generator(*engine); });
}

template <PrimitiveType type, typename T>
/* static */ StatusOr<std::unique_ptr<Literal>>
LiteralUtil::CreateRandomLiteral(const Shape& shape, T mean, T stddev) {
  std::minstd_rand0 engine;
  return CreateRandomLiteral<type>(shape, &engine, mean, stddev);
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_LITERAL_UTIL_H_
