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
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/index_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/bitmap.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Utility class for dealing with XLA literal values.  Most methods are
// templated by native (host) type which corresponds to a unique XLA
// PrimitiveType. See ComputationBuilder for details.  Not all primitive types
// defined in xla_data.proto have a corresponding native type or even have a
// storage location in the Literal proto yet (for example, primitive type F16).
class LiteralUtil {
 public:
  // Create new literal of a given rank. To minimize ambiguity (for users and
  // the compiler) these CreateR[0-2] methods should explicitly specify the
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
  static std::unique_ptr<Literal> CreateR1(
      tensorflow::gtl::ArraySlice<NativeT> values);
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

  // Creates a new value that has the equivalent value as literal, but conforms
  // to new_layout; e.g. a literal matrix that was in {0, 1} minor-to-major
  // dimension layout can be re-layed-out as {1, 0} minor-to-major dimension
  // layout and the value in the cell at any given logical index (i0, i1) will
  // be the same.
  //
  // Note: this is useful when the client wants to ensure that a value placed in
  // the XLA allocation tracker has a particular layout; for efficiency
  // purposes or avoiding unimplemented operation/layout combinations.
  static std::unique_ptr<Literal> Relayout(const Literal& literal,
                                           const Layout& new_layout);

  // Reshapes literal 'input' to have 'shape'. Both the original shape and
  // 'shape' must contain the same number of elements. The implementation
  // currently only supports monotonic dim0-major layouts.
  static StatusOr<std::unique_ptr<Literal>> Reshape(
      const xla::Literal& input, tensorflow::gtl::ArraySlice<int64> shape);

  // Creates a new literal by reordering the dimensions of the original literal.
  // The given `permutation` must be a permutation of the dimension numbers
  // in the original literal, and it specifies the order of the new dimensions
  // in the result literal (i.e., new_order[i] = old_order[permutation[i]]).
  // For example, a transpose call on a literal of shape [3 x 8 x 4] and
  // `permutation` = {2, 0, 1} returns a new literal of shape [4 x 3 x 8].
  static std::unique_ptr<Literal> Transpose(
      const Literal& literal, tensorflow::gtl::ArraySlice<int64> permutation);

  // Creates a sub-array from the the given literal by extracting the indices
  // [start_index, limit_index) of each dimension. The result literal has the
  // same rank and layout as for the given literal. The number of indices in
  // start_indices and limit_indices must be the rank of the literal, and the
  // indices follow the order of the dimensions.
  static std::unique_ptr<Literal> Slice(
      const Literal& literal, tensorflow::gtl::ArraySlice<int64> start_indices,
      tensorflow::gtl::ArraySlice<int64> limit_indices);

  // Creates a literal with a prepended dimension with bound "times"; e.g. a
  // f32[3x2] with times=4 will produce a f32[4x3x2] with the 3x2 from the input
  // literal replicated four times.
  template <typename NativeT>
  static std::unique_ptr<Literal> Replicate(const Literal& input, int64 times);

  // Create a literal by converting each element in an original literal to a new
  // type.
  template <typename NativeSrcT, typename NativeDestT>
  static std::unique_ptr<Literal> Convert(const Literal& literal);

  // Create a literal value zero of the given primitive type.
  static Literal Zero(PrimitiveType primitive_type);

  // Create a literal value one of the given primitive type.
  static Literal One(PrimitiveType primitive_type);

  // Creates a literal value containing the minimum value of the given
  // primitive type. For floating-point types, returns -inf.
  static Literal MinValue(PrimitiveType primitive_type);

  // Create a literal value containing the maximum value of the given
  // primitive type. For floating-point types, returns inf.
  static Literal MaxValue(PrimitiveType primitive_type);

  // Create a literal of the given shape where each element is `value`.
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateFullWithMonotonicDim0MajorLayout(
      tensorflow::gtl::ArraySlice<int64> dimensions, NativeT value);

  // Create a new literal from an array. The variants not ending with WithLayout
  // use the default XLA layout for the literal's linear representation in
  // memory.
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
  static std::unique_ptr<Literal> CreateR1U8(tensorflow::StringPiece value);

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

  // Clones literal into an owned unique_ptr version.
  static std::unique_ptr<Literal> CloneToUnique(const Literal& literal);

  // Gets or sets an element in the literal at the given index. The index is
  // CHECKed against the dimension sizes.
  template <typename NativeT>
  static NativeT Get(const Literal& literal,
                     tensorflow::gtl::ArraySlice<int64> multi_index);
  template <typename NativeT>
  static void Set(Literal* literal,
                  tensorflow::gtl::ArraySlice<int64> multi_index,
                  NativeT value);

  // Returns the element value at index (0, ..., 0), however many zeroes are
  // required for that index.
  template <typename NativeT>
  static NativeT GetFirstElement(const Literal& literal);

  // As Get(), but determines the correct type and converts the value
  // into text.
  static string GetAsString(const Literal& literal,
                            tensorflow::gtl::ArraySlice<int64> multi_index);

  // Returns an identity matrix (rank 2) with the given row and column count.
  template <typename NativeT>
  static std::unique_ptr<Literal> MakeIdentityR2(int64 size);

  // Returns a tuple literal composed of given literals.
  static std::unique_ptr<Literal> MakeTuple(
      tensorflow::gtl::ArraySlice<const Literal*> elements);

  // Validates that the data payload of the literal matches the literal shape;
  // if it does not, an appropriate status is returned.
  static tensorflow::Status ValidateLiteral(const Literal& literal);

  // Returns a string representation of the literal value.
  static string ToString(const Literal& literal);

  // Invokes the "per cell" callback for each element in the provided
  // literal with the element's indices and a string representation of
  // the element's value.
  //
  // This function is useful if you want a polymorphic representation
  // of the tensor's elements (turning it to a string for something
  // like representation in a protobuf).
  static void EachCellAsString(
      const Literal& literal,
      std::function<void(tensorflow::gtl::ArraySlice<int64> indices,
                         const string& value)>
          per_cell);
  template <typename NativeT>
  static void EachCell(
      const Literal& literal,
      std::function<void(tensorflow::gtl::ArraySlice<int64> indices,
                         NativeT value)>
          per_cell);

  // Templated methods which populate the given repeated field in the Literal
  // proto with the given value(s). The Shape field of the Literal proto is set
  // to match the array dimensions and type. Examples:
  //
  //   // Populate with floats.
  //   Array2D<float> float_values = ...
  //   PopulateR2FromArray2D(values, literal);
  //
  //   // Populate with int32s.
  //   PopulateR2({{1, 2}, {3, 4}}, literal);
  //
  template <typename NativeT>
  static void PopulateR0(NativeT values, Literal* literal);
  template <typename NativeT>
  static void PopulateR1(tensorflow::gtl::ArraySlice<NativeT> values,
                         Literal* literal);
  static void PopulateR1(const tensorflow::core::Bitmap& values,
                         Literal* literal);
  template <typename NativeT>
  static void PopulateR2(
      std::initializer_list<std::initializer_list<NativeT>> values,
      Literal* literal);
  template <typename NativeT>
  static void PopulateR2WithLayout(
      std::initializer_list<std::initializer_list<NativeT>> values,
      const Layout& layout, Literal* literal);
  template <typename NativeT>
  static void PopulateR2FromArray2D(const Array2D<NativeT>& values,
                                    Literal* literal);
  template <typename NativeT>
  static void PopulateR2FromArray2DWithLayout(const Array2D<NativeT>& values,
                                              const Layout& layout,
                                              Literal* literal);
  template <typename NativeT>
  static void PopulateR3FromArray3D(const Array3D<NativeT>& values,
                                    Literal* literal);
  template <typename NativeT>
  static void PopulateR3FromArray3DWithLayout(const Array3D<NativeT>& values,
                                              const Layout& layout,
                                              Literal* literal);
  template <typename NativeT>
  static void PopulateR4FromArray4D(const Array4D<NativeT>& values,
                                    Literal* literal);
  template <typename NativeT>
  static void PopulateR4FromArray4DWithLayout(const Array4D<NativeT>& values,
                                              const Layout& layout,
                                              Literal* literal);

  // Creates a Literal of the given dimensions with all elements set to the
  // given value.
  template <typename NativeT>
  static void PopulateWithValue(NativeT value,
                                tensorflow::gtl::ArraySlice<int64> dimensions,
                                Literal* literal);

  // Returns a pointer to the underlying buffer in the protobuf containing the
  // array data. Use with care.
  static const void* InternalData(const Literal& literal);
  static void* MutableInternalData(Literal* literal);

  // Allocates space in the repeated_field of the literal sufficient to hold
  // num_elements of the literal's primitive type. Values in the buffer are set
  // to zero. num_elements must equal the number of elements in the literals
  // shape.
  static void Reserve(int64 num_elements, Literal* literal);

  // Allocates space in the repeated_field of the literal sufficient to hold
  // num_elements of the literal's primitive type and sets each element in the
  // literal to the given value. num_elements must equal the number of elements
  // in the literals shape.
  template <typename NativeT>
  static void Resize(int64 num_elements, NativeT value, Literal* literal);

  // Returns true if the two given literals have the same shape and
  // values. Layout is not considered in the comparison.
  static bool Equal(const Literal& literal1, const Literal& literal2);

  // Returns whether every element in the given literal is equal to value.
  //
  // value is an int8 because we expect this to be called with small
  // compile-time constants (0, -1, etc.) and so that whatever value you pass
  // can be represented exactly by floating-point types as small as 16 bits.
  //
  // If value doesn't fit in literal's type, returns false.  Values of 1/0 are
  // considered equal to true/false; other values are not considered equal to
  // true.
  static bool IsAll(const Literal& literal, int8 value);

  // Like IsAll(const Literal&, int8), except we check whether the literal is
  // equal to a particular floating-point number.
  //
  // If the literal is not a floating-point value, this always returns false.
  //
  // This casts value to the type of literal, then compares using ==.  The usual
  // admonishments about floating-point equality checks apply.  We expect you to
  // use this to check for values that can be expressed precisely as a float,
  // e.g. -0.5.
  static bool IsAllFloat(const Literal& literal, float value);

  // Returns whether the literal is zero at the specified index. The literal
  // must be an array.
  static bool IsZero(const Literal& literal,
                     tensorflow::gtl::ArraySlice<int64> indices);

 private:
  // Returns an ArraySlice view of the array for the given literal for the
  // given NativeT (e.g., float). These
  // functions map native type to XLA PrimitiveType via template
  // specialization. The unspecialized forms below aborts to handle the error
  // case where the given native type does not map to an XLA primitive type.
  template <typename NativeT>
  static tensorflow::gtl::ArraySlice<NativeT> GetArraySlice(
      const Literal& literal) {
    static_assert(!std::is_same<NativeT, NativeT>::value,
                  "Cannot map native type to primitive type.");
  }
  template <typename NativeT>
  static tensorflow::protobuf::RepeatedField<NativeT>* GetMutableRepeatedField(
      Literal* literal) {
    // Make the expression depend on the template parameter NativeT so
    // that this compile-time error only apperas if this function is
    // instantiated with some concrete type that is not specialized
    // below.
    static_assert(!std::is_same<NativeT, NativeT>::value,
                  "Cannot map native type to primitive type.");
  }

  // Returns the linear index of the given index within the literal's
  // element_type repeated field.
  static int64 LinearIndex(const Literal& literal,
                           tensorflow::gtl::ArraySlice<int64> multi_index);

  TF_DISALLOW_COPY_AND_ASSIGN(LiteralUtil);
};

// Declarations of template specializations for GetArraySlice and
// GetMutableRepeatedField. The specializations map native type to XLA primitive
// type.
template <>
/* static */ tensorflow::gtl::ArraySlice<bool> LiteralUtil::GetArraySlice<bool>(
    const Literal& literal);

template <>
/* static */ tensorflow::protobuf::RepeatedField<bool>*
LiteralUtil::GetMutableRepeatedField<bool>(Literal* literal);

template <>
/* static */ tensorflow::gtl::ArraySlice<uint32>
LiteralUtil::GetArraySlice<uint32>(const Literal& literal);

template <>
/* static */ tensorflow::protobuf::RepeatedField<uint32>*
LiteralUtil::GetMutableRepeatedField<uint32>(Literal* literal);

template <>
/* static */ tensorflow::gtl::ArraySlice<uint64>
LiteralUtil::GetArraySlice<uint64>(const Literal& literal);

template <>
/* static */ tensorflow::protobuf::RepeatedField<tensorflow::protobuf_uint64>*
LiteralUtil::GetMutableRepeatedField<tensorflow::protobuf_uint64>(
    Literal* literal);

template <>
/* static */ tensorflow::gtl::ArraySlice<int32>
LiteralUtil::GetArraySlice<int32>(const Literal& literal);

template <>
/* static */ tensorflow::protobuf::RepeatedField<int32>*
LiteralUtil::GetMutableRepeatedField<int32>(Literal* literal);

template <>
/* static */ tensorflow::gtl::ArraySlice<int64>
LiteralUtil::GetArraySlice<int64>(const Literal& literal);

template <>
/* static */ tensorflow::protobuf::RepeatedField<tensorflow::protobuf_int64>*
LiteralUtil::GetMutableRepeatedField<tensorflow::protobuf_int64>(
    Literal* literal);

template <>
/* static */ tensorflow::gtl::ArraySlice<float>
LiteralUtil::GetArraySlice<float>(const Literal& literal);

template <>
/* static */ tensorflow::protobuf::RepeatedField<float>*
LiteralUtil::GetMutableRepeatedField<float>(Literal* literal);

template <>
/* static */ tensorflow::gtl::ArraySlice<double>
LiteralUtil::GetArraySlice<double>(const Literal& literal);

template <>
/* static */ tensorflow::protobuf::RepeatedField<double>*
LiteralUtil::GetMutableRepeatedField<double>(Literal* literal);

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR0(NativeT value) {
  auto literal = MakeUnique<Literal>();
  PopulateR0(value, literal.get());
  return literal;
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR1(
    tensorflow::gtl::ArraySlice<NativeT> values) {
  auto literal = MakeUnique<Literal>();
  PopulateR1(values, literal.get());
  return literal;
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR2WithLayout(
    std::initializer_list<std::initializer_list<NativeT>> values,
    const Layout& layout) {
  auto literal = MakeUnique<Literal>();
  PopulateR2WithLayout(values, layout, literal.get());
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
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR4(
    std::initializer_list<std::initializer_list<
        std::initializer_list<std::initializer_list<NativeT>>>>
        values) {
  return CreateR4WithLayout(values, LayoutUtil::GetDefaultLayoutForR4());
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal>
LiteralUtil::CreateR2FromArray2DWithLayout(const Array2D<NativeT>& values,
                                           const Layout& layout) {
  auto literal = MakeUnique<Literal>();
  PopulateR2FromArray2DWithLayout(values, layout, literal.get());
  return literal;
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR2FromArray2D(
    const Array2D<NativeT>& values) {
  return CreateR2FromArray2DWithLayout(values,
                                       LayoutUtil::GetDefaultLayoutForR2());
}
template <typename NativeT>
/* static */ std::unique_ptr<Literal>
LiteralUtil::CreateR3FromArray3DWithLayout(const Array3D<NativeT>& values,
                                           const Layout& layout) {
  auto literal = MakeUnique<Literal>();
  PopulateR3FromArray3DWithLayout(values, layout, literal.get());
  return literal;
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR3FromArray3D(
    const Array3D<NativeT>& values) {
  return CreateR3FromArray3DWithLayout(values,
                                       LayoutUtil::GetDefaultLayoutForR3());
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
  return CreateR4FromArray4DWithLayout(values,
                                       LayoutUtil::GetDefaultLayoutForR4());
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal>
LiteralUtil::CreateR4FromArray4DWithLayout(const Array4D<NativeT>& values,
                                           const Layout& layout) {
  auto literal = MakeUnique<Literal>();
  PopulateR4FromArray4DWithLayout(values, layout, literal.get());
  return literal;
}

template <typename NativeT>
/* static */ NativeT LiteralUtil::Get(
    const Literal& literal, tensorflow::gtl::ArraySlice<int64> multi_index) {
  int64 linear_index = LinearIndex(literal, multi_index);
  return GetArraySlice<NativeT>(literal).at(linear_index);
}

template <typename NativeT>
/* static */ NativeT LiteralUtil::GetFirstElement(const Literal& literal) {
  return GetArraySlice<NativeT>(literal).at(0);
}

template <>
/* static */ inline uint8 LiteralUtil::Get<uint8>(
    const Literal& literal, tensorflow::gtl::ArraySlice<int64> multi_index) {
  CHECK(literal.shape().element_type() == U8);
  int64 linear_index = LinearIndex(literal, multi_index);
  return literal.u8s()[linear_index];
}

template <>
/* static */ inline int8 LiteralUtil::Get<int8>(
    const Literal& literal, tensorflow::gtl::ArraySlice<int64> multi_index) {
  CHECK(literal.shape().element_type() == S8);
  int64 linear_index = LinearIndex(literal, multi_index);
  return literal.u8s()[linear_index];
}

template <typename NativeT>
/* static */ void LiteralUtil::Set(
    Literal* literal, tensorflow::gtl::ArraySlice<int64> multi_index,
    NativeT value) {
  int64 linear_index = LinearIndex(*literal, multi_index);
  GetMutableRepeatedField<NativeT>(literal)->Set(linear_index, value);
}

template <>
/* static */ inline void LiteralUtil::Set(
    Literal* literal, tensorflow::gtl::ArraySlice<int64> multi_index,
    uint8 value) {
  int64 linear_index = LinearIndex(*literal, multi_index);
  (*literal->mutable_u8s())[linear_index] = value;
}

template <>
/* static */ inline void LiteralUtil::Set(
    Literal* literal, tensorflow::gtl::ArraySlice<int64> multi_index,
    int8 value) {
  return Set<uint8>(literal, multi_index, value);
}

template <>
/* static */ inline void LiteralUtil::Set(
    Literal* literal, tensorflow::gtl::ArraySlice<int64> multi_index,
    int64 value) {
  int64 linear_index = LinearIndex(*literal, multi_index);
  (*literal->mutable_s64s())[linear_index] = value;
}

template <>
/* static */ inline void LiteralUtil::Set(
    Literal* literal, tensorflow::gtl::ArraySlice<int64> multi_index,
    uint64 value) {
  int64 linear_index = LinearIndex(*literal, multi_index);
  (*literal->mutable_u64s())[linear_index] = value;
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
/* static */ void LiteralUtil::EachCell(
    const Literal& literal,
    std::function<void(tensorflow::gtl::ArraySlice<int64> indices,
                       NativeT value)>
        per_cell) {
  if (ShapeUtil::HasZeroElements(literal.shape())) {
    return;
  }
  std::vector<int64> indices(ShapeUtil::Rank(literal.shape()), 0);
  do {
    per_cell(indices, Get<NativeT>(literal, indices));
  } while (IndexUtil::BumpIndices(literal.shape(), &indices));
}

template <typename NativeT>
/* static */ void LiteralUtil::PopulateR0(NativeT value, Literal* literal) {
  *literal->mutable_shape() = ShapeUtil::MakeShape(
      primitive_util::NativeToPrimitiveType<NativeT>(), {});
  tensorflow::protobuf::RepeatedField<NativeT>* repeated_field =
      GetMutableRepeatedField<NativeT>(literal);
  repeated_field->Add(value);
}

template <>
/* static */ inline void LiteralUtil::PopulateR0<uint8>(uint8 value,
                                                        Literal* literal) {
  *literal->mutable_shape() =
      ShapeUtil::MakeShape(primitive_util::NativeToPrimitiveType<uint8>(), {});
  literal->mutable_u8s()->push_back(value);
}

template <>
/* static */ inline void LiteralUtil::PopulateR0<int8>(int8 value,
                                                       Literal* literal) {
  *literal->mutable_shape() =
      ShapeUtil::MakeShape(primitive_util::NativeToPrimitiveType<int8>(), {});
  literal->mutable_u8s()->push_back(value);
}

template <>
/* static */ inline void LiteralUtil::PopulateR0<uint64>(uint64 value,
                                                         Literal* literal) {
  *literal->mutable_shape() =
      ShapeUtil::MakeShape(primitive_util::NativeToPrimitiveType<uint64>(), {});
  literal->mutable_u64s()->Add(value);
}

template <>
/* static */ inline void LiteralUtil::PopulateR0<int64>(int64 value,
                                                        Literal* literal) {
  *literal->mutable_shape() =
      ShapeUtil::MakeShape(primitive_util::NativeToPrimitiveType<int64>(), {});
  literal->mutable_s64s()->Add(value);
}

template <typename NativeT>
/* static */ void LiteralUtil::PopulateR1(
    tensorflow::gtl::ArraySlice<NativeT> values, Literal* literal) {
  *literal->mutable_shape() =
      ShapeUtil::MakeShape(primitive_util::NativeToPrimitiveType<NativeT>(),
                           {static_cast<int64>(values.size())});
  Reserve(values.size(), literal);
  for (int64 i = 0; i < values.size(); ++i) {
    Set(literal, {i}, values[i]);
  }
}

/* static */ inline void LiteralUtil::PopulateR1(
    const tensorflow::core::Bitmap& values, Literal* literal) {
  *literal->mutable_shape() =
      ShapeUtil::MakeShape(PRED, {static_cast<int64>(values.bits())});
  Reserve(values.bits(), literal);
  for (int64 i = 0; i < values.bits(); ++i) {
    Set(literal, {i}, values.get(i));
  }
}

template <typename NativeT>
/* static */ void LiteralUtil::PopulateR2WithLayout(
    std::initializer_list<std::initializer_list<NativeT>> values,
    const Layout& layout, Literal* literal) {
  *literal->mutable_shape() = ShapeUtil::MakeShapeWithLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(),
      {static_cast<int64>(values.size()),
       static_cast<int64>(values.begin()->size())},
      AsInt64Slice(layout.minor_to_major()));

  const int64 dim0_size = values.size();
  const int64 dim1_size = values.begin()->size();
  CHECK_EQ(dim0_size, literal->shape().dimensions(0));
  CHECK_EQ(dim1_size, literal->shape().dimensions(1));

  const int64 num_elements = dim1_size * dim0_size;
  Reserve(num_elements, literal);

  int64 dim0 = 0;
  for (auto inner_list : values) {
    int64 dim1 = 0;
    for (auto value : inner_list) {
      Set(literal, {dim0, dim1}, value);
      ++dim1;
    }
    CHECK_EQ(dim1_size, dim1);
    ++dim0;
  }
}

template <typename NativeT>
/* static */ void LiteralUtil::PopulateR2(
    std::initializer_list<std::initializer_list<NativeT>> values,
    Literal* literal) {
  PopulateR2WithLayout(values, LayoutUtil::GetDefaultLayoutForR2(), literal);
}

template <typename NativeT>
/* static */ void LiteralUtil::PopulateR2FromArray2DWithLayout(
    const Array2D<NativeT>& values, const Layout& layout, Literal* literal) {
  *literal->mutable_shape() = ShapeUtil::MakeShapeWithLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(),
      {values.height(), values.width()}, AsInt64Slice(layout.minor_to_major()));

  const int64 dim1_size = values.width();
  const int64 dim0_size = values.height();
  CHECK_EQ(dim0_size, literal->shape().dimensions(0));
  CHECK_EQ(dim1_size, literal->shape().dimensions(1));
  Reserve(dim1_size * dim0_size, literal);
  for (int64 dim0 = 0; dim0 < dim0_size; ++dim0) {
    for (int64 dim1 = 0; dim1 < dim1_size; ++dim1) {
      Set(literal, {dim0, dim1}, values(dim0, dim1));
    }
  }
}

template <typename NativeT>
/* static */ void LiteralUtil::PopulateR2FromArray2D(
    const Array2D<NativeT>& values, Literal* literal) {
  PopulateR2FromArray2DWithLayout(values, LayoutUtil::GetDefaultLayoutForR2(),
                                  literal);
}
template <typename NativeT>
/* static */ void LiteralUtil::PopulateR3FromArray3DWithLayout(
    const Array3D<NativeT>& values, const Layout& layout, Literal* literal) {
  *literal->mutable_shape() = ShapeUtil::MakeShapeWithLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(),
      {values.n1(), values.n2(), values.n3()},
      AsInt64Slice(layout.minor_to_major()));

  CHECK_EQ(values.n1(), literal->shape().dimensions(0));
  CHECK_EQ(values.n2(), literal->shape().dimensions(1));
  CHECK_EQ(values.n3(), literal->shape().dimensions(2));
  Reserve(values.n1() * values.n2() * values.n3(), literal);
  for (int64 dim0 = 0; dim0 < values.n1(); ++dim0) {
    for (int64 dim1 = 0; dim1 < values.n2(); ++dim1) {
      for (int64 dim2 = 0; dim2 < values.n3(); ++dim2) {
        Set(literal, {dim0, dim1, dim2}, values(dim0, dim1, dim2));
      }
    }
  }
}

template <typename NativeT>
/* static */ void LiteralUtil::PopulateR3FromArray3D(
    const Array3D<NativeT>& values, Literal* literal) {
  PopulateR3FromArray3DWithLayout(values, LayoutUtil::GetDefaultLayoutForR3(),
                                  literal);
}

template <typename NativeT>
/* static */ void LiteralUtil::PopulateR4FromArray4DWithLayout(
    const Array4D<NativeT>& values, const Layout& layout, Literal* literal) {
  *literal->mutable_shape() = ShapeUtil::MakeShapeWithLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(),
      {values.planes(), values.depth(), values.height(), values.width()},
      AsInt64Slice(layout.minor_to_major()));

  CHECK_EQ(values.n1(), literal->shape().dimensions(0));
  CHECK_EQ(values.n2(), literal->shape().dimensions(1));
  CHECK_EQ(values.n3(), literal->shape().dimensions(2));
  CHECK_EQ(values.n4(), literal->shape().dimensions(3));
  Reserve(values.n1() * values.n2() * values.n3() * values.n4(), literal);
  for (int64 dim0 = 0; dim0 < values.n1(); ++dim0) {
    for (int64 dim1 = 0; dim1 < values.n2(); ++dim1) {
      for (int64 dim2 = 0; dim2 < values.n3(); ++dim2) {
        for (int64 dim3 = 0; dim3 < values.n4(); ++dim3) {
          Set(literal, {dim0, dim1, dim2, dim3},
              values(dim0, dim1, dim2, dim3));
        }
      }
    }
  }
}

template <typename NativeT>
/* static */ void LiteralUtil::PopulateR4FromArray4D(
    const Array4D<NativeT>& values, Literal* literal) {
  PopulateR4FromArray4DWithLayout(values, LayoutUtil::GetDefaultLayoutForR4(),
                                  literal);
}

template <typename NativeT>
/* static */ void LiteralUtil::PopulateWithValue(
    NativeT value, tensorflow::gtl::ArraySlice<int64> dimensions,
    Literal* literal) {
  *literal->mutable_shape() = ShapeUtil::MakeShape(
      primitive_util::NativeToPrimitiveType<NativeT>(), dimensions);
  tensorflow::protobuf::RepeatedField<NativeT>* repeated_field =
      GetMutableRepeatedField<NativeT>(literal);
  for (int64 i = 0; i < ShapeUtil::ElementsIn(literal->shape()); ++i) {
    repeated_field->Add(value);
  }
}

template <>
/* static */ void LiteralUtil::PopulateWithValue(
    int64 value, tensorflow::gtl::ArraySlice<int64> dimensions,
    Literal* literal);

template <>
/* static */ void LiteralUtil::PopulateWithValue(
    uint64 value, tensorflow::gtl::ArraySlice<int64> dimensions,
    Literal* literal);

template <typename NativeSrcT, typename NativeDestT>
/* static */ std::unique_ptr<Literal> LiteralUtil::Convert(
    const Literal& literal) {
  auto result_literal = MakeUnique<Literal>();
  Shape result_shape = literal.shape();
  result_shape.set_element_type(
      primitive_util::NativeToPrimitiveType<NativeDestT>());
  *result_literal->mutable_shape() = result_shape;
  LiteralUtil::Reserve(ShapeUtil::ElementsIn(result_shape),
                       result_literal.get());
  LiteralUtil::EachCell<NativeSrcT>(
      literal,
      [&](tensorflow::gtl::ArraySlice<int64> indices, NativeSrcT value) {
        LiteralUtil::Set<NativeDestT>(result_literal.get(), indices,
                                      static_cast<NativeDestT>(value));
      });
  return result_literal;
}

template <typename NativeT>
/* static */ void LiteralUtil::Resize(int64 num_elements, NativeT value,
                                      Literal* literal) {
  CHECK_EQ(ShapeUtil::ElementsIn(literal->shape()), num_elements);
  tensorflow::protobuf::RepeatedField<NativeT>* repeated_field =
      GetMutableRepeatedField<NativeT>(literal);
  repeated_field->Resize(num_elements, value);
}

template <>
/* static */ void LiteralUtil::Resize(int64 num_elements, int64 value,
                                      Literal* literal);

template <>
/* static */ void LiteralUtil::Resize(int64 num_elements, uint64 value,
                                      Literal* literal);

template <typename NativeT>
/* static */ std::unique_ptr<Literal>
LiteralUtil::CreateFullWithMonotonicDim0MajorLayout(
    tensorflow::gtl::ArraySlice<int64> dimensions, NativeT value) {
  Shape shape = ShapeUtil::MakeShapeWithMonotonicDim0MajorLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(), dimensions);
  auto literal = MakeUnique<Literal>();
  *literal->mutable_shape() = shape;
  Reserve(ShapeUtil::ElementsIn(shape), literal.get());
  std::vector<int64> index(dimensions.size(), 0);
  do {
    Set(literal.get(), index, value);
  } while (IndexUtil::BumpIndices(shape, &index));
  return literal;
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> LiteralUtil::Replicate(
    const Literal& input, int64 times) {
  std::vector<int64> bounds = {times};
  bounds.insert(bounds.end(), input.shape().dimensions().begin(),
                input.shape().dimensions().end());
  auto literal = MakeUnique<Literal>();
  *literal->mutable_shape() =
      ShapeUtil::MakeShape(input.shape().element_type(), bounds);
  Reserve(ShapeUtil::ElementsIn(literal->shape()), literal.get());
  for (int64 index = 0; index < ShapeUtil::ElementsIn(input.shape()); ++index) {
    const std::vector<int64> element_indices =
        IndexUtil::LinearIndexToMultidimensionalIndex(input.shape(), index);
    const auto element = Get<NativeT>(input, element_indices);
    for (int64 sample = 0; sample < times; ++sample) {
      std::vector<int64> output_indices = {sample};
      output_indices.insert(output_indices.end(), element_indices.begin(),
                            element_indices.end());
      Set<NativeT>(literal.get(), output_indices, element);
    }
  }
  return literal;
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_LITERAL_UTIL_H_
