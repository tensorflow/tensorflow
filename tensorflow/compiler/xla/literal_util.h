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
#include "tensorflow/compiler/xla/status_macros.h"
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

// This class is a simple vector of boolean values. It's used to workaround some
// implementations of std::vector<bool> that use a bitset which does not have
// the semantics expected by Literal::preds().
class BoolVector {
 public:
  typedef bool* iterator;
  typedef const bool* const_iterator;

  BoolVector() : bits_(nullptr), size_(0), capacity_(0) {}

  BoolVector(const_iterator other_begin, const_iterator other_end)
      : bits_(nullptr), size_(0), capacity_(0) {
    if (other_begin && other_end) {
      resize(other_end - other_begin);
      memcpy(begin(), other_begin, size());
    }
  }

  BoolVector(const BoolVector& other) { CopyFrom(other); }

  BoolVector& operator=(const BoolVector& other) {
    CopyFrom(other);
    return *this;
  }

  void push_back(const bool& value) {
    resize(size_ + 1);
    bits_[size_ - 1] = value;
  }

  bool* data() const { return bits_.get(); }

  size_t size() const { return size_; }

  size_t capacity() const { return capacity_; }

  void resize(size_t new_size, bool val = false) {
    if (new_size == 0) {
      bits_.reset(nullptr);
      size_ = 0;
      capacity_ = 0;
    } else {
      size_t old_size = size();
      if (new_size > old_size) {
        grow(new_size);
      }
      if (old_size < new_size) {
        memset(&bits_[old_size], val, new_size - old_size);
      }
      size_ = new_size;
    }
  }

  void clear() {
    bits_.reset(nullptr);
    size_ = 0;
    capacity_ = 0;
  }

  iterator begin() { return &bits_[0]; }
  iterator end() { return &bits_[size()]; }
  const_iterator begin() const { return &bits_[0]; }
  const_iterator end() const { return &bits_[size()]; }

 private:
  void grow(size_t n) {
    if (capacity_ < n) {
      capacity_ = 2 * n;
      bool* new_bits = new bool[capacity_]();
      if (size_ > 0) {
        memcpy(new_bits, bits_.get(), size_);
      }
      bits_.reset(new_bits);
    }
  }

  void CopyFrom(const BoolVector& other) {
    bits_ = MakeUnique<bool[]>(other.capacity());
    memcpy(begin(), other.begin(), other.size());
    size_ = other.size();
    capacity_ = other.capacity();
  }

  std::unique_ptr<bool[]> bits_;
  size_t size_;
  size_t capacity_;
};

// Utility class for dealing with XLA literal values.  Most methods are
// templated by native (host) type which corresponds to a unique XLA
// PrimitiveType. See ComputationBuilder for details.  Not all primitive types
// defined in xla_data.proto have a corresponding native type or even have a
// storage location in the Literal proto yet (for example, primitive type F16).
class Literal {
 public:
  Literal() {}

  Literal(const Literal& other) = default;

  explicit Literal(const LiteralProto& other) { CopyFromProto(other); }

  Literal& operator=(const Literal& other) = default;

  LiteralProto ToProto() const;

  bool has_shape() const {
    return shape_.element_type() != PRIMITIVE_TYPE_INVALID;
  }

  // Basic accessor functions.  Names mirror the original protobuf
  // functions for convenience.
  string DebugString() const { return ToProto().DebugString(); }
  string ShortDebugString() const { return ToProto().ShortDebugString(); }

  void Clear() {
    shape_.Clear();
    preds_.clear();
    u8s_.clear();
    s32s_.clear();
    s64s_.clear();
    u32s_.clear();
    u64s_.clear();
    f16s_.clear();
    f32s_.clear();
    f64s_.clear();
    tuple_literals_.clear();
  }

  int preds_size() const { return preds().size(); }
  const BoolVector& preds() const { return preds_; }
  BoolVector* mutable_preds() { return &preds_; }

  int s32s_size() const { return s32s().size(); }
  int32 s32s(int i) const { return s32s_[i]; }
  const std::vector<int32>& s32s() const { return s32s_; }
  std::vector<int32>* mutable_s32s() { return &s32s_; }

  int s64s_size() const { return s64s().size(); }
  void add_s64s(int64 value) { s64s_.push_back(value); }
  const std::vector<int64>& s64s() const { return s64s_; }
  std::vector<int64>* mutable_s64s() { return &s64s_; }

  int u32s_size() const { return u32s().size(); }
  uint32 u32s(int i) const { return u32s_[i]; }
  const std::vector<uint32>& u32s() const { return u32s_; }
  std::vector<uint32>* mutable_u32s() { return &u32s_; }

  int u64s_size() const { return u64s().size(); }
  const std::vector<uint64>& u64s() const { return u64s_; }
  std::vector<uint64>* mutable_u64s() { return &u64s_; }

  int f16s_size() const { return f16s().size(); }
  half f16s(int i) const { return f16s_[i]; }
  const std::vector<half>& f16s() const { return f16s_; }
  std::vector<half>* mutable_f16s() { return &f16s_; }

  int f32s_size() const { return f32s().size(); }
  float f32s(int i) const { return f32s_[i]; }
  void add_f32s(float value) { f32s_.push_back(value); }
  const std::vector<float>& f32s() const { return f32s_; }
  std::vector<float>& f32s() { return f32s_; }
  std::vector<float>* mutable_f32s() { return &f32s_; }

  int f64s_size() const { return f64s().size(); }
  const std::vector<double>& f64s() const { return f64s_; }
  std::vector<double>* mutable_f64s() { return &f64s_; }

  int tuple_literals_size() const { return tuple_literals().size(); }
  const Literal& tuple_literals(int i) const { return tuple_literals_[i]; }
  Literal* add_tuple_literals() {
    tuple_literals_.push_back(Literal());
    return &tuple_literals_.back();
  }
  std::vector<Literal>* mutable_tuple_literals() { return &tuple_literals_; }
  const std::vector<Literal>& tuple_literals() const { return tuple_literals_; }

  int u8s_size() const { return u8s().size(); }
  const std::vector<uint8>& u8s() const { return u8s_; }
  void set_u8s(const std::vector<uint8>& value) { u8s_ = value; }
  void set_u8s(tensorflow::StringPiece value) {
    u8s_ = std::vector<uint8>(value.size());
    u8s_.clear();
    append_u8s(value);
  }

  void append_u8s(tensorflow::StringPiece value) {
    u8s_.insert(u8s_.end(), value.begin(), value.end());
  }

  string u8s_string() const { return string(u8s().begin(), u8s().end()); }

  std::vector<uint8>* mutable_u8s() { return &u8s_; }

  const Shape& shape() const { return shape_; }
  Shape* mutable_shape() { return &shape_; }

  void Swap(Literal* other) {
    Literal temp = *this;
    *this = *other;
    *other = temp;
  }

  // CreatesCreate new literal of a given rank. To minimize ambiguity (for users
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

  // Creates a new Literal object with the shape specified as parameter.
  // The content of the literal values is the default value of the primitive
  // type of literal itself (0 for numeric types, and false for predicates).
  static std::unique_ptr<Literal> CreateFromShape(const Shape& shape);

  // Creates a new Literal object with its values havings the primitive_type
  // type, and with dimensions defined by the dimensions parameter.
  // The content of the literal values is the default value of the primitive
  // type of literal itself (0 for numeric types, and false for predicates).
  static std::unique_ptr<Literal> CreateFromDimensions(
      PrimitiveType primitive_type,
      tensorflow::gtl::ArraySlice<int64> dimensions);

  // Copies the values from src_literal, starting at src_base shape indexes,
  // to this literal, starting at dest_base, where the copy size in each
  // dimension is specified by copy_size.
  // The src_literal and this literal must have the same primitive type,
  // src_base+copy_size must fit the source literal dimensions, as well as
  // dest_base+copy_size must fit the destination literal dimensions.
  Status Copy(const Literal& src_literal,
              tensorflow::gtl::ArraySlice<int64> src_base,
              tensorflow::gtl::ArraySlice<int64> dest_base,
              tensorflow::gtl::ArraySlice<int64> copy_size);

  // Creates a new value that has the equivalent value as this literal, but
  // conforms to new_layout; e.g. a literal matrix that was in {0, 1}
  // minor-to-major dimension layout can be re-layed-out as {1, 0}
  // minor-to-major dimension layout and the value in the cell at any given
  // logical index (i0, i1) will be the same.
  //
  // Note: this is useful when the client wants to ensure that a value placed in
  // the XLA allocation tracker has a particular layout; for efficiency
  // purposes or avoiding unimplemented operation/layout combinations.
  std::unique_ptr<Literal> Relayout(const Layout& new_layout) const;

  // Creates a new literal by reshaping this literal to have 'shape'. Both the
  // original shape and 'shape' must contain the same number of elements. The
  // implementation currently only supports monotonic dim0-major layouts.
  StatusOr<std::unique_ptr<Literal>> Reshape(
      tensorflow::gtl::ArraySlice<int64> shape) const;

  // Creates a new literal by reordering the dimensions of this literal.
  // The given `permutation` must be a permutation of the dimension numbers
  // in the original literal, and it specifies the order of the new dimensions
  // in the result literal (i.e., new_order[i] = old_order[permutation[i]]).
  // For example, a transpose call on a literal of shape [3 x 8 x 4] and
  // `permutation` = {2, 0, 1} returns a new literal of shape [4 x 3 x 8].
  std::unique_ptr<Literal> Transpose(
      tensorflow::gtl::ArraySlice<int64> permutation) const;

  // Creates a sub-array from this literal by extracting the indices
  // [start_index, limit_index) of each dimension. The result literal has the
  // same rank and layout as for the given literal. The number of indices in
  // start_indices and limit_indices must be the rank of the literal, and the
  // indices follow the order of the dimensions.
  std::unique_ptr<Literal> Slice(
      tensorflow::gtl::ArraySlice<int64> start_indices,
      tensorflow::gtl::ArraySlice<int64> limit_indices) const;

  // Creates a literal with a prepended dimension with bound "times"; e.g. a
  // f32[3x2] with times=4 will produce a f32[4x3x2] with the 3x2 from this
  // literal replicated four times.
  template <typename NativeT>
  std::unique_ptr<Literal> Replicate(int64 times) const;

  // Creates a literal by converting each element in this literal to a new
  // type.
  template <typename NativeSrcT, typename NativeDestT>
  std::unique_ptr<Literal> Convert() const;

  // Creates a literal value zero of the given primitive type.
  static Literal Zero(PrimitiveType primitive_type);

  // Creates a literal value one of the given primitive type.
  static Literal One(PrimitiveType primitive_type);

  // Creates a literal value containing the minimum value of the given
  // primitive type. For floating-point types, returns -inf.
  static Literal MinValue(PrimitiveType primitive_type);

  // Creates a literal value containing the maximum value of the given
  // primitive type. For floating-point types, returns inf.
  static Literal MaxValue(PrimitiveType primitive_type);

  // Creates a literal of the given shape where each element is `value`.
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateFullWithMonotonicDim0MajorLayout(
      tensorflow::gtl::ArraySlice<int64> dimensions, NativeT value);

  // Creates a new literal from an array. The variants not ending with
  // WithLayout use the default XLA layout for the literal's linear
  // representation in memory.
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

  // Clones this literal into an owned unique_ptr version.
  std::unique_ptr<Literal> CloneToUnique() const;

  // Returns the linear index of the given index within this literal's
  // element_type repeated field.
  int64 LinearIndex(tensorflow::gtl::ArraySlice<int64> multi_index) const;

  // Gets or sets an element in the literal at the given index. The index is
  // CHECKed against the dimension sizes.
  template <typename NativeT>
  NativeT Get(tensorflow::gtl::ArraySlice<int64> multi_index) const;
  template <typename NativeT>
  void Set(tensorflow::gtl::ArraySlice<int64> multi_index, NativeT value);

  // Retrieves the mutable array slice interface which can be used to manipulate
  // pre-allocated literal values.
  template <typename NativeT>
  tensorflow::gtl::MutableArraySlice<NativeT> GetMutableArraySlice();

  // Returns the element value at index (0, ..., 0), however many zeroes are
  // required for that index.
  template <typename NativeT>
  NativeT GetFirstElement() const;

  // As Get(), but determines the correct type and converts the value
  // into text.
  string GetAsString(tensorflow::gtl::ArraySlice<int64> multi_index) const;

  // Returns an identity matrix (rank 2) with the given row and column count.
  template <typename NativeT>
  static std::unique_ptr<Literal> MakeIdentityR2(int64 size);

  // Returns a tuple literal composed of given literals.
  static std::unique_ptr<Literal> MakeTuple(
      tensorflow::gtl::ArraySlice<const Literal*> elements);

  // Validates that the data payload of the literal matches the literal shape;
  // if it does not, an appropriate status is returned.
  tensorflow::Status ValidateLiteral() const;

  // Returns a string representation of the literal value.
  string ToString() const;

  // Invokes the "per cell" callback for each element in the provided
  // literal with the element's indices and a string representation of
  // the element's value.
  //
  // This function is useful if you want a polymorphic representation
  // of the tensor's elements (turning it to a string for something
  // like representation in a protobuf).
  void EachCellAsString(
      const std::function<void(tensorflow::gtl::ArraySlice<int64> indices,
                               const string& value)>& per_cell) const;
  template <typename NativeT>
  void EachCell(std::function<void(tensorflow::gtl::ArraySlice<int64> indices,
                                   NativeT value)>
                    per_cell) const;

  // Templated methods which populate the given repeated field in this literal
  // with the given value(s). The Shape field of this literal is set
  // to match the array dimensions and type. Examples:
  //
  //   // Populate with floats.
  //   Array2D<float> float_values = ...
  //   literal.PopulateR2FromArray2D(values);
  //
  //   // Populate with int32s.
  //   literal.PopulateR2({{1, 2}, {3, 4}});
  //
  template <typename NativeT>
  void PopulateR0(NativeT values);
  template <typename NativeT>
  void PopulateR1(tensorflow::gtl::ArraySlice<NativeT> values);
  void PopulateR1(const tensorflow::core::Bitmap& values);
  template <typename NativeT>
  void PopulateR2(std::initializer_list<std::initializer_list<NativeT>> values);
  template <typename NativeT>
  void PopulateR2WithLayout(
      std::initializer_list<std::initializer_list<NativeT>> values,
      const Layout& layout);
  template <typename NativeT>
  void PopulateR2FromArray2D(const Array2D<NativeT>& values);
  template <typename NativeT>
  void PopulateR2FromArray2DWithLayout(const Array2D<NativeT>& values,
                                       const Layout& layout);
  template <typename NativeT>
  void PopulateR3FromArray3D(const Array3D<NativeT>& values);
  template <typename NativeT>
  void PopulateR3FromArray3DWithLayout(const Array3D<NativeT>& values,
                                       const Layout& layout);
  template <typename NativeT>
  void PopulateR4FromArray4D(const Array4D<NativeT>& values);
  template <typename NativeT>
  void PopulateR4FromArray4DWithLayout(const Array4D<NativeT>& values,
                                       const Layout& layout);

  // Populates literal values by calling the generator function for every cell
  // in this literal object.
  template <typename NativeT>
  Status Populate(
      const std::function<NativeT(tensorflow::gtl::ArraySlice<int64> indexes)>&
          generator);

  // Creates a Literal of the given dimensions with all elements set to the
  // given value.
  template <typename NativeT>
  void PopulateWithValue(NativeT value,
                         tensorflow::gtl::ArraySlice<int64> dimensions);

  // Returns a pointer to the underlying vector corresponding to the Literal's
  // shape.
  const void* InternalData() const;
  void* MutableInternalData();

  // Allocates space in the underlying vector of this literal sufficient to hold
  // num_elements of this literal's primitive type. Values in the vector are set
  // to zero. num_elements must equal the number of elements in the literal's
  // shape.
  void Reserve(int64 num_elements);

  // Allocates space in the underlying vector of this literal sufficient to hold
  // num_elements of this literal's primitive type and sets each element in this
  // literal to the given value. num_elements must equal the number of elements
  // in this literal's shape.
  template <typename NativeT>
  void Resize(int64 num_elements, NativeT value);

  // Returns true if this literal has the same shape and value as the given
  // literal. Layout is not considered in the comparison.
  bool Equal(const Literal& literal2) const;

  // Returns whether every element in this literal is equal to value.
  //
  // value is an int8 because we expect this to be called with small
  // compile-time constants (0, -1, etc.) and so that whatever value you pass
  // can be represented exactly by floating-point types as small as 16 bits.
  //
  // If value doesn't fit in this literal's type, returns false.  Values of 1/0
  // are considered equal to true/false; other values are not considered equal
  // to true.
  bool IsAll(int8 value) const;

  // Like IsAll(const Literal&, int8), except we check whether the literal is
  // equal to a particular floating-point number.
  //
  // If the literal is not a floating-point value, this always returns false.
  //
  // This casts value to the type of literal, then compares using ==.  The usual
  // admonishments about floating-point equality checks apply.  We expect you to
  // use this to check for values that can be expressed precisely as a float,
  // e.g. -0.5.
  bool IsAllFloat(float value) const;

  // Returns whether this literal is zero at the specified index. This literal
  // must be an array.
  bool IsZero(tensorflow::gtl::ArraySlice<int64> indices) const;

 private:
  // Returns an ArraySlice view of the array for this literal for the given
  // NativeT (e.g., float). These functions map native type to XLA PrimitiveType
  // via template specialization. The unspecialized forms below aborts to handle
  // the error case where the given native type does not map to an XLA primitive
  // type.
  template <typename NativeT>
  tensorflow::gtl::ArraySlice<NativeT> GetArraySlice() const {
    static_assert(!std::is_same<NativeT, NativeT>::value,
                  "Cannot map native type to primitive type.");
  }

  // Copy from a LiteralProto instance.
  void CopyFromProto(const LiteralProto& literal_proto);

  // Internal template helper for the Copy() API, matching its arguments one by
  // one.
  template <typename T>
  Status CopyRange(const Literal& src_literal,
                   tensorflow::gtl::ArraySlice<int64> src_base,
                   tensorflow::gtl::ArraySlice<int64> dest_base,
                   tensorflow::gtl::ArraySlice<int64> copy_size);

  // Utility structure which is used to create the optimal configuration for
  // a ShapeUtil::ForEachIndex() scan across two literals.
  struct StrideConfig {
    StrideConfig(const Shape& source_shape, const Shape& dest_shape,
                 tensorflow::gtl::ArraySlice<int64> dimensions);

    // The dimensions of the stride operation. Essentially every dimension
    // will be iterated from base[i] to base[i]+dimensions[i], in step[i]
    // steps.
    tensorflow::gtl::ArraySlice<int64> dimensions;
    DimensionVector base;
    DimensionVector step;
    int64 minor_dimension = 0;
    // The size of the strides for source and destination. One of the two
    // (the one looping through its most minor dimension) will be 1, while
    // the other will be the stride size at the dimension matching the other
    // shape most minor dimension being scanned.
    int64 dest_stride = 1;
    int64 source_stride = 1;
    // The size of the inner loop on the most minor dimension.
    int64 minor_loop_size = 1;
  };

  Shape shape_;
  BoolVector preds_;
  std::vector<uint8> u8s_;
  std::vector<int32> s32s_;
  std::vector<int64> s64s_;
  std::vector<uint32> u32s_;
  std::vector<uint64> u64s_;
  std::vector<half> f16s_;
  std::vector<float> f32s_;
  std::vector<double> f64s_;
  std::vector<Literal> tuple_literals_;
};

// Utility class for dealing with XLA literal values.  Most methods are
// templated by native (host) type which corresponds to a unique XLA
// PrimitiveType. See ComputationBuilder for details.  Not all primitive types
// defined in xla_data.proto have a corresponding native type or even have a
// storage location in the Literal proto yet (for example, primitive type F16).
//
// TODO(dnovillo) - All functions in this class simply redirect to the
// corresponding function in class Literal. Remove this class after converting
// all user code to use Literal directly.
class LiteralUtil {
 public:
  // Creates new literal of a given rank. To minimize ambiguity (for users and
  // the compiler) these CreateR[0-2] methods should explicitly specify the
  // native type. For example:
  //
  //  CreateR1<float>({1.0, 42.0});
  //  CreateR2<uint32>({{1, 2}, {3, 4}});
  //
  // The variants not ending with WithLayout use the default XLA layout for the
  // literal's linear representation in memory.
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR0(NativeT value) {
    return Literal::CreateR0(value);
  }

  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR1(
      tensorflow::gtl::ArraySlice<NativeT> values) {
    return Literal::CreateR1(values);
  }

  static std::unique_ptr<Literal> CreateR1(
      const tensorflow::core::Bitmap& values) {
    return Literal::CreateR1(values);
  }

  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR2(
      std::initializer_list<std::initializer_list<NativeT>> values) {
    return Literal::CreateR2(values);
  }

  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR2WithLayout(
      std::initializer_list<std::initializer_list<NativeT>> values,
      const Layout& layout) {
    return Literal::CreateR2WithLayout(values, layout);
  }

  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR3(
      std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>
          values) {
    return Literal::CreateR3(values);
  }

  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR3WithLayout(
      std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>
          values,
      const Layout& layout) {
    return Literal::CreateR3WithLayout(values, layout);
  }

  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR4(
      std::initializer_list<std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>>
          values) {
    return Literal::CreateR4(values);
  }

  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR4WithLayout(
      std::initializer_list<std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>>
          values,
      const Layout& layout) {
    return Literal::CreateR4WithLayout(values, layout);
  }

  // Creates a new Literal object with the shape specified as parameter.
  // The content of the literal values is the default value of the primitive
  // type of literal itself (0 for numeric types, and false for predicates).
  static std::unique_ptr<Literal> CreateFromShape(const Shape& shape) {
    return Literal::CreateFromShape(shape);
  }

  // Creates a new Literal object with its values havings the primitive_type
  // type, and with dimensions defined by the dimensions parameter.
  // The content of the literal values is the default value of the primitive
  // type of literal itself (0 for numeric types, and false for predicates).
  static std::unique_ptr<Literal> CreateFromDimensions(
      PrimitiveType primitive_type,
      tensorflow::gtl::ArraySlice<int64> dimensions) {
    return Literal::CreateFromDimensions(primitive_type, dimensions);
  }

  // Copies the values from src_literal, starting at src_base shape indexes,
  // to dest_literal, starting at dest_base, where the copy size in each
  // dimension is specified by copy_size.
  //
  // The src_literal and dest_literal must have the same primitive type,
  // src_base+copy_size must fit the source literal dimensions, as well as
  // dest_base+copy_size must fit the destination literal dimensions.
  static Status Copy(const Literal& src_literal,
                     tensorflow::gtl::ArraySlice<int64> src_base,
                     Literal* dest_literal,
                     tensorflow::gtl::ArraySlice<int64> dest_base,
                     tensorflow::gtl::ArraySlice<int64> copy_size) {
    return dest_literal->Copy(src_literal, src_base, dest_base, copy_size);
  }

  // Creates a new value that has the equivalent value as literal, but conforms
  // to new_layout; e.g. a literal matrix that was in {0, 1} minor-to-major
  // dimension layout can be re-laid-out as {1, 0} minor-to-major dimension
  // layout and the value in the cell at any given logical index (i0, i1) will
  // be the same.
  //
  // Note: this is useful when the client wants to ensure that a value placed in
  // the XLA allocation tracker has a particular layout; for efficiency
  // purposes or avoiding unimplemented operation/layout combinations.
  static std::unique_ptr<Literal> Relayout(const Literal& literal,
                                           const Layout& new_layout) {
    return literal.Relayout(new_layout);
  }

  // Reshapes literal 'input' to have 'shape'. Both the original shape and
  // 'shape' must contain the same number of elements. The implementation
  // currently only supports monotonic dim0-major layouts.
  static StatusOr<std::unique_ptr<Literal>> Reshape(
      const xla::Literal& input, tensorflow::gtl::ArraySlice<int64> shape) {
    return input.Reshape(shape);
  }

  // Creates a new literal by reordering the dimensions of the original literal.
  // The given `permutation` must be a permutation of the dimension numbers
  // in the original literal, and it specifies the order of the new dimensions
  // in the result literal (i.e., new_order[i] = old_order[permutation[i]]).
  // For example, a transpose call on a literal of shape [3 x 8 x 4] and
  // `permutation` = {2, 0, 1} returns a new literal of shape [4 x 3 x 8].
  static std::unique_ptr<Literal> Transpose(
      const Literal& literal, tensorflow::gtl::ArraySlice<int64> permutation) {
    return literal.Transpose(permutation);
  }

  // Creates a sub-array from the given literal by extracting the indices
  // [start_index, limit_index) of each dimension. The result literal has the
  // same rank and layout as for the given literal. The number of indices in
  // start_indices and limit_indices must be the rank of the literal, and the
  // indices follow the order of the dimensions.
  static std::unique_ptr<Literal> Slice(
      const Literal& literal, tensorflow::gtl::ArraySlice<int64> start_indices,
      tensorflow::gtl::ArraySlice<int64> limit_indices) {
    return literal.Slice(start_indices, limit_indices);
  }

  // Creates a literal with a prepended dimension with bound "times"; e.g. a
  // f32[3x2] with times=4 will produce a f32[4x3x2] with the 3x2 from the input
  // literal replicated four times.
  template <typename NativeT>
  static std::unique_ptr<Literal> Replicate(const Literal& input, int64 times) {
    return input.Replicate<NativeT>(times);
  }

  // Creates a literal by converting each element in an original literal to a
  // new type.
  template <typename NativeSrcT, typename NativeDestT>
  static std::unique_ptr<Literal> Convert(const Literal& literal) {
    return literal.Convert<NativeSrcT, NativeDestT>();
  }

  // Convert a literal to another primitive type, but only if the literal
  // type is connvertable into the destination type
  static StatusOr<std::unique_ptr<Literal>> ConvertIfSrcTypeMatches(
      const Literal& src_literal, PrimitiveType primitive_dest_type);

  // Creates a literal value zero of the given primitive type.
  static Literal Zero(PrimitiveType primitive_type) {
    return Literal::Zero(primitive_type);
  }

  // Creates a literal value one of the given primitive type.
  static Literal One(PrimitiveType primitive_type) {
    return Literal::One(primitive_type);
  }

  // Creates a literal value containing the minimum value of the given
  // primitive type. For floating-point types, returns -inf.
  static Literal MinValue(PrimitiveType primitive_type) {
    return Literal::MinValue(primitive_type);
  }

  // Creates a literal value containing the maximum value of the given
  // primitive type. For floating-point types, returns inf.
  static Literal MaxValue(PrimitiveType primitive_type) {
    return Literal::MaxValue(primitive_type);
  }

  // Creates a literal of the given shape where each element is `value`.
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateFullWithMonotonicDim0MajorLayout(
      tensorflow::gtl::ArraySlice<int64> dimensions, NativeT value) {
    return Literal::CreateFullWithMonotonicDim0MajorLayout(dimensions, value);
  }

  // Creates a new literal from an array. The variants not ending with
  // WithLayout use the default XLA layout for the literal's linear
  // representation in memory.
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR2FromArray2D(
      const Array2D<NativeT>& values) {
    return Literal::CreateR2FromArray2D(values);
  }

  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR2FromArray2DWithLayout(
      const Array2D<NativeT>& values, const Layout& layout) {
    return Literal::CreateR2FromArray2DWithLayout(values, layout);
  }

  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR3FromArray3D(
      const Array3D<NativeT>& values) {
    return Literal::CreateR3FromArray3D(values);
  }

  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR3FromArray3DWithLayout(
      const Array3D<NativeT>& values, const Layout& layout) {
    return Literal::CreateR3FromArray3DWithLayout(values, layout);
  }

  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR4FromArray4D(
      const Array4D<NativeT>& values) {
    return Literal::CreateR4FromArray4D(values);
  }

  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR4FromArray4DWithLayout(
      const Array4D<NativeT>& values, const Layout& layout) {
    return Literal::CreateR4FromArray4DWithLayout(values, layout);
  }

  // Creates a new vector of U8s literal value from a string.
  static std::unique_ptr<Literal> CreateR1U8(tensorflow::StringPiece value) {
    return Literal::CreateR1U8(value);
  }

  // Creates a linspace-populated literal with the given number of rows and
  // columns.
  static std::unique_ptr<Literal> CreateR2F32Linspace(float from, float to,
                                                      int64 rows, int64 cols) {
    return Literal::CreateR2F32Linspace(from, to, rows, cols);
  }

  // Creates a literal that projects the (x, y) dimensions given in values into
  // the z dimension given by "projection".
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR3Projected(
      std::initializer_list<std::initializer_list<NativeT>> values,
      int64 projection) {
    return Literal::CreateR3Projected(values, projection);
  }

  // Creates a literal that projects the (x, y) dimensions given in values into
  // the z and p dimensions given.
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR4Projected(
      std::initializer_list<std::initializer_list<NativeT>> values,
      int64 projection_p, int64 projection_z) {
    return Literal::CreateR4Projected(values, projection_p, projection_z);
  }

  // Clones literal into an owned unique_ptr version.
  static std::unique_ptr<Literal> CloneToUnique(const Literal& literal) {
    return literal.CloneToUnique();
  }

  // Returns the linear index of the given index within the literal's
  // element_type repeated field.
  static int64 LinearIndex(const Literal& literal,
                           tensorflow::gtl::ArraySlice<int64> multi_index) {
    return literal.LinearIndex(multi_index);
  }

  // Gets or sets an element in the literal at the given index. The index is
  // CHECKed against the dimension sizes.
  template <typename NativeT>
  static NativeT Get(const Literal& literal,
                     tensorflow::gtl::ArraySlice<int64> multi_index) {
    return literal.Get<NativeT>(multi_index);
  }

  template <typename NativeT>
  static void Set(Literal* literal,
                  tensorflow::gtl::ArraySlice<int64> multi_index,
                  NativeT value) {
    literal->Set(multi_index, value);
  }

  // Retrieves the mutable array slice interface which can be used to manipulate
  // pre-allocated literal values.
  template <typename NativeT>
  static tensorflow::gtl::MutableArraySlice<NativeT> GetMutableArraySlice(
      Literal* literal) {
    return literal->GetMutableArraySlice<NativeT>();
  }

  // Returns the element value at index (0, ..., 0), however many zeroes are
  // required for that index.
  template <typename NativeT>
  static NativeT GetFirstElement(const Literal& literal) {
    return literal.GetFirstElement<NativeT>();
  }

  // As Get(), but determines the correct type and converts the value
  // into text.
  static string GetAsString(const Literal& literal,
                            tensorflow::gtl::ArraySlice<int64> multi_index) {
    return literal.GetAsString(multi_index);
  }

  // Returns an identity matrix (rank 2) with the given row and column count.
  template <typename NativeT>
  static std::unique_ptr<Literal> MakeIdentityR2(int64 size) {
    return Literal::MakeIdentityR2<NativeT>(size);
  }

  // Returns a tuple literal composed of given literals.
  static std::unique_ptr<Literal> MakeTuple(
      tensorflow::gtl::ArraySlice<const Literal*> elements) {
    return Literal::MakeTuple(elements);
  }

  // Validates that the data payload of the literal matches the literal shape;
  // if it does not, an appropriate status is returned.
  static tensorflow::Status ValidateLiteral(const Literal& literal) {
    return literal.ValidateLiteral();
  }

  // Returns a string representation of the literal value.
  static string ToString(const Literal& literal) { return literal.ToString(); }

  // Invokes the "per cell" callback for each element in the provided
  // literal with the element's indices and a string representation of
  // the element's value.
  //
  // This function is useful if you want a polymorphic representation
  // of the tensor's elements (turning it to a string for something
  // like representation in a protobuf).
  static void EachCellAsString(
      const Literal& literal,
      const std::function<void(tensorflow::gtl::ArraySlice<int64> indices,
                               const string& value)>& per_cell) {
    literal.EachCellAsString(per_cell);
  }

  template <typename NativeT>
  static void EachCell(
      const Literal& literal,
      std::function<void(tensorflow::gtl::ArraySlice<int64> indices,
                         NativeT value)>
          per_cell) {
    literal.EachCell<NativeT>(per_cell);
  }

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
  static void PopulateR0(NativeT values, Literal* literal) {
    literal->PopulateR0(values);
  }

  template <typename NativeT>
  static void PopulateR1(tensorflow::gtl::ArraySlice<NativeT> values,
                         Literal* literal) {
    literal->PopulateR1(values);
  }

  static void PopulateR1(const tensorflow::core::Bitmap& values,
                         Literal* literal) {
    literal->PopulateR1(values);
  }

  template <typename NativeT>
  static void PopulateR2(
      std::initializer_list<std::initializer_list<NativeT>> values,
      Literal* literal) {
    literal->PopulateR2(values);
  }

  template <typename NativeT>
  static void PopulateR2WithLayout(
      std::initializer_list<std::initializer_list<NativeT>> values,
      const Layout& layout, Literal* literal) {
    literal->PopulateR2WithLayout(values, layout);
  }

  template <typename NativeT>
  static void PopulateR2FromArray2D(const Array2D<NativeT>& values,
                                    Literal* literal) {
    literal->PopulateR2FromArray2D(values);
  }

  template <typename NativeT>
  static void PopulateR2FromArray2DWithLayout(const Array2D<NativeT>& values,
                                              const Layout& layout,
                                              Literal* literal) {
    literal->PopulateR2FromArray2DWithLayout(values, layout);
  }

  template <typename NativeT>
  static void PopulateR3FromArray3D(const Array3D<NativeT>& values,
                                    Literal* literal) {
    literal->PopulateR3FromArray3D(values);
  }

  template <typename NativeT>
  static void PopulateR3FromArray3DWithLayout(const Array3D<NativeT>& values,
                                              const Layout& layout,
                                              Literal* literal) {
    literal->PopulateR3FromArray3DWithLayout(values, layout);
  }

  template <typename NativeT>
  static void PopulateR4FromArray4D(const Array4D<NativeT>& values,
                                    Literal* literal) {
    literal->PopulateR4FromArray4D(values);
  }

  template <typename NativeT>
  static void PopulateR4FromArray4DWithLayout(const Array4D<NativeT>& values,
                                              const Layout& layout,
                                              Literal* literal) {
    literal->PopulateR4FromArray4DWithLayout(values, layout);
  }

  // Populates literal values by calling the generator function for every cell
  // in the literal object.
  template <typename NativeT>
  static Status Populate(
      Literal* literal,
      const std::function<NativeT(tensorflow::gtl::ArraySlice<int64> indexes)>&
          generator) {
    return literal->Populate(generator);
  }

  // Creates a Literal of the given dimensions with all elements set to the
  // given value.
  template <typename NativeT>
  static void PopulateWithValue(NativeT value,
                                tensorflow::gtl::ArraySlice<int64> dimensions,
                                Literal* literal) {
    return literal->PopulateWithValue(value, dimensions);
  }

  // Returns a pointer to the underlying vector containing the array data. Use
  // with care.
  static const void* InternalData(const Literal& literal) {
    return literal.InternalData();
  }

  static void* MutableInternalData(Literal* literal) {
    return literal->MutableInternalData();
  }

  // Allocates space in the underlying vector of the literal sufficient to hold
  // num_elements of the literal's primitive type. Values in the vector are set
  // to zero. num_elements must equal the number of elements in the literals
  // shape.
  static void Reserve(int64 num_elements, Literal* literal) {
    literal->Reserve(num_elements);
  }

  // Allocates space in the underlying vector of the literal sufficient to hold
  // num_elements of the literal's primitive type and sets each element in the
  // literal to the given value. num_elements must equal the number of elements
  // in the literals shape.
  template <typename NativeT>
  static void Resize(int64 num_elements, NativeT value, Literal* literal) {
    literal->Resize(num_elements, value);
  }

  // Returns true if the two given literals have the same shape and
  // values. Layout is not considered in the comparison.
  static bool Equal(const Literal& literal1, const Literal& literal2) {
    return literal1.Equal(literal2);
  }

  // Returns whether every element in the given literal is equal to value.
  //
  // value is an int8 because we expect this to be called with small
  // compile-time constants (0, -1, etc.) and so that whatever value you pass
  // can be represented exactly by floating-point types as small as 16 bits.
  //
  // If value doesn't fit in literal's type, returns false.  Values of 1/0 are
  // considered equal to true/false; other values are not considered equal to
  // true.
  static bool IsAll(const Literal& literal, int8 value) {
    return literal.IsAll(value);
  }

  // Like IsAll(const Literal&, int8), except we check whether the literal is
  // equal to a particular floating-point number.
  //
  // If the literal is not a floating-point value, this always returns false.
  //
  // This casts value to the type of literal, then compares using ==.  The usual
  // admonishments about floating-point equality checks apply.  We expect you to
  // use this to check for values that can be expressed precisely as a float,
  // e.g. -0.5.
  static bool IsAllFloat(const Literal& literal, float value) {
    return literal.IsAllFloat(value);
  }

  // Returns whether the literal is zero at the specified index. The literal
  // must be an array.
  static bool IsZero(const Literal& literal,
                     tensorflow::gtl::ArraySlice<int64> indices) {
    return literal.IsZero(indices);
  }

  TF_DISALLOW_COPY_AND_ASSIGN(LiteralUtil);
};

// Declarations of template specializations for GetArraySlice and
// GetMutableArraySlice. The specializations map native type to XLA primitive
// type.
template <>
tensorflow::gtl::ArraySlice<bool> Literal::GetArraySlice<bool>() const;

template <>
tensorflow::gtl::ArraySlice<uint8> Literal::GetArraySlice<uint8>() const;

template <>
tensorflow::gtl::ArraySlice<int8> Literal::GetArraySlice<int8>() const;

template <>
tensorflow::gtl::ArraySlice<uint32> Literal::GetArraySlice<uint32>() const;

template <>
tensorflow::gtl::ArraySlice<uint64> Literal::GetArraySlice<uint64>() const;

template <>
tensorflow::gtl::ArraySlice<int32> Literal::GetArraySlice<int32>() const;

template <>
tensorflow::gtl::ArraySlice<int64> Literal::GetArraySlice<int64>() const;

template <>
inline tensorflow::gtl::ArraySlice<float> Literal::GetArraySlice<float>()
    const {
  DCHECK(shape().element_type() == F32);
  return f32s();
}

template <>
tensorflow::gtl::ArraySlice<double> Literal::GetArraySlice<double>() const;

template <>
tensorflow::gtl::ArraySlice<half> Literal::GetArraySlice<half>() const;

template <>
tensorflow::gtl::MutableArraySlice<bool> Literal::GetMutableArraySlice();

template <>
tensorflow::gtl::MutableArraySlice<int8> Literal::GetMutableArraySlice();

template <>
tensorflow::gtl::MutableArraySlice<uint8> Literal::GetMutableArraySlice();

template <>
tensorflow::gtl::MutableArraySlice<int32> Literal::GetMutableArraySlice();

template <>
tensorflow::gtl::MutableArraySlice<uint32> Literal::GetMutableArraySlice();

template <>
tensorflow::gtl::MutableArraySlice<int64> Literal::GetMutableArraySlice();

template <>
tensorflow::gtl::MutableArraySlice<uint64> Literal::GetMutableArraySlice();

template <>
tensorflow::gtl::MutableArraySlice<float> Literal::GetMutableArraySlice();

template <>
tensorflow::gtl::MutableArraySlice<double> Literal::GetMutableArraySlice();

template <>
tensorflow::gtl::MutableArraySlice<half> Literal::GetMutableArraySlice();

template <>
void Literal::Resize<bool>(int64 num_elements, bool value);

template <>
void Literal::Resize<int8>(int64 num_elements, int8 value);

template <>
void Literal::Resize<uint8>(int64 num_elements, uint8 value);

template <>
void Literal::Resize<int32>(int64 num_elements, int32 value);

template <>
void Literal::Resize<uint32>(int64 num_elements, uint32 value);

template <>
void Literal::Resize<int64>(int64 num_elements, int64 value);

template <>
void Literal::Resize<uint64>(int64 num_elements, uint64 value);

template <>
void Literal::Resize<float>(int64 num_elements, float value);

template <>
void Literal::Resize<double>(int64 num_elements, double value);

template <>
void Literal::Resize<half>(int64 num_elements, half value);

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR0(NativeT value) {
  auto literal = MakeUnique<Literal>();
  literal->PopulateR0<NativeT>(value);
  return literal;
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR1(
    tensorflow::gtl::ArraySlice<NativeT> values) {
  auto literal = MakeUnique<Literal>();
  literal->PopulateR1(values);
  return literal;
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR2WithLayout(
    std::initializer_list<std::initializer_list<NativeT>> values,
    const Layout& layout) {
  auto literal = MakeUnique<Literal>();
  literal->PopulateR2WithLayout(values, layout);
  return literal;
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR2(
    std::initializer_list<std::initializer_list<NativeT>> values) {
  return CreateR2WithLayout(values, LayoutUtil::GetDefaultLayoutForR2());
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR3WithLayout(
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
/* static */ std::unique_ptr<Literal> Literal::CreateR3(
    std::initializer_list<std::initializer_list<std::initializer_list<NativeT>>>
        values) {
  return CreateR3WithLayout(values, LayoutUtil::GetDefaultLayoutForR3());
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR4WithLayout(
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
/* static */ std::unique_ptr<Literal> Literal::CreateR4(
    std::initializer_list<std::initializer_list<
        std::initializer_list<std::initializer_list<NativeT>>>>
        values) {
  return CreateR4WithLayout(values, LayoutUtil::GetDefaultLayoutForR4());
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR2FromArray2DWithLayout(
    const Array2D<NativeT>& values, const Layout& layout) {
  auto literal = MakeUnique<Literal>();
  literal->PopulateR2FromArray2DWithLayout(values, layout);
  return literal;
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR2FromArray2D(
    const Array2D<NativeT>& values) {
  return CreateR2FromArray2DWithLayout(values,
                                       LayoutUtil::GetDefaultLayoutForR2());
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR3FromArray3DWithLayout(
    const Array3D<NativeT>& values, const Layout& layout) {
  auto literal = MakeUnique<Literal>();
  literal->PopulateR3FromArray3DWithLayout(values, layout);
  return literal;
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR3FromArray3D(
    const Array3D<NativeT>& values) {
  return CreateR3FromArray3DWithLayout(values,
                                       LayoutUtil::GetDefaultLayoutForR3());
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR3Projected(
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
/* static */ std::unique_ptr<Literal> Literal::CreateR4Projected(
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
/* static */ std::unique_ptr<Literal> Literal::CreateR4FromArray4D(
    const Array4D<NativeT>& values) {
  return CreateR4FromArray4DWithLayout(values,
                                       LayoutUtil::GetDefaultLayoutForR4());
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR4FromArray4DWithLayout(
    const Array4D<NativeT>& values, const Layout& layout) {
  auto literal = MakeUnique<Literal>();
  literal->PopulateR4FromArray4DWithLayout(values, layout);
  return literal;
}

template <typename NativeT>
NativeT Literal::Get(tensorflow::gtl::ArraySlice<int64> multi_index) const {
  int64 linear_index = LinearIndex(multi_index);
  return GetArraySlice<NativeT>().at(linear_index);
}

template <typename NativeT>
NativeT Literal::GetFirstElement() const {
  return GetArraySlice<NativeT>().at(0);
}

template <>
inline uint8 Literal::Get<uint8>(
    tensorflow::gtl::ArraySlice<int64> multi_index) const {
  CHECK(shape().element_type() == U8);
  int64 linear_index = LinearIndex(multi_index);
  return u8s()[linear_index];
}

template <>
inline int8 Literal::Get<int8>(
    tensorflow::gtl::ArraySlice<int64> multi_index) const {
  CHECK(shape().element_type() == S8);
  int64 linear_index = LinearIndex(multi_index);
  return u8s()[linear_index];
}

template <>
inline half Literal::Get<half>(
    tensorflow::gtl::ArraySlice<int64> multi_index) const {
  CHECK(shape().element_type() == F16);
  int64 linear_index = LinearIndex(multi_index);
  return GetArraySlice<half>()[linear_index];
}

template <typename NativeT>
void Literal::Set(tensorflow::gtl::ArraySlice<int64> multi_index,
                  NativeT value) {
  int64 linear_index = LinearIndex(multi_index);
  GetMutableArraySlice<NativeT>().at(linear_index) = value;
}

template <>
inline void Literal::Set(tensorflow::gtl::ArraySlice<int64> multi_index,
                         uint8 value) {
  int64 linear_index = LinearIndex(multi_index);
  (*mutable_u8s())[linear_index] = value;
}

template <>
inline void Literal::Set(tensorflow::gtl::ArraySlice<int64> multi_index,
                         int8 value) {
  return Set<uint8>(multi_index, value);
}

template <>
inline void Literal::Set(tensorflow::gtl::ArraySlice<int64> multi_index,
                         int64 value) {
  int64 linear_index = LinearIndex(multi_index);
  (*mutable_s64s())[linear_index] = value;
}

template <>
/* static */ inline void Literal::Set(
    tensorflow::gtl::ArraySlice<int64> multi_index, uint64 value) {
  int64 linear_index = LinearIndex(multi_index);
  (*mutable_u64s())[linear_index] = value;
}

// Returns an identity matrix (rank 2) with the given row and column count.
template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::MakeIdentityR2(int64 size) {
  Array2D<NativeT> array(size, size, 0);
  for (int64 i = 0; i < size; ++i) {
    array(i, i) = 1;
  }
  return CreateR2FromArray2D(array);
}

template <typename NativeT>
void Literal::EachCell(
    std::function<void(tensorflow::gtl::ArraySlice<int64> indices,
                       NativeT value)>
        per_cell) const {
  if (ShapeUtil::HasZeroElements(shape())) {
    return;
  }
  std::vector<int64> indices(ShapeUtil::Rank(shape()), 0);
  do {
    per_cell(indices, Get<NativeT>(indices));
  } while (IndexUtil::BumpIndices(shape(), &indices));
}

template <typename NativeT>
inline void Literal::PopulateR0(NativeT value) {
  *mutable_shape() = ShapeUtil::MakeShape(
      primitive_util::NativeToPrimitiveType<NativeT>(), {});
  Resize<NativeT>(1, value);
}

template <typename NativeT>
inline void Literal::PopulateR1(tensorflow::gtl::ArraySlice<NativeT> values) {
  *mutable_shape() =
      ShapeUtil::MakeShape(primitive_util::NativeToPrimitiveType<NativeT>(),
                           {static_cast<int64>(values.size())});
  Reserve(values.size());
  for (int64 i = 0; i < values.size(); ++i) {
    Set({i}, values[i]);
  }
}

inline void Literal::PopulateR1(const tensorflow::core::Bitmap& values) {
  *mutable_shape() =
      ShapeUtil::MakeShape(PRED, {static_cast<int64>(values.bits())});
  Reserve(values.bits());
  for (int64 i = 0; i < static_cast<int64>(values.bits()); ++i) {
    Set({i}, values.get(i));
  }
}

template <typename NativeT>
void Literal::PopulateR2WithLayout(
    std::initializer_list<std::initializer_list<NativeT>> values,
    const Layout& layout) {
  *mutable_shape() = ShapeUtil::MakeShapeWithLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(),
      {static_cast<int64>(values.size()),
       static_cast<int64>(values.begin()->size())},
      AsInt64Slice(layout.minor_to_major()));

  const int64 dim0_size = values.size();
  const int64 dim1_size = values.begin()->size();
  CHECK_EQ(dim0_size, shape().dimensions(0));
  CHECK_EQ(dim1_size, shape().dimensions(1));

  const int64 num_elements = dim1_size * dim0_size;
  Reserve(num_elements);

  int64 dim0 = 0;
  for (auto inner_list : values) {
    int64 dim1 = 0;
    for (auto value : inner_list) {
      Set({dim0, dim1}, value);
      ++dim1;
    }
    CHECK_EQ(dim1_size, dim1);
    ++dim0;
  }
}

template <typename NativeT>
void Literal::PopulateR2(
    std::initializer_list<std::initializer_list<NativeT>> values) {
  PopulateR2WithLayout(values, LayoutUtil::GetDefaultLayoutForR2());
}

template <typename NativeT>
void Literal::PopulateR2FromArray2DWithLayout(const Array2D<NativeT>& values,
                                              const Layout& layout) {
  *mutable_shape() = ShapeUtil::MakeShapeWithLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(),
      {values.height(), values.width()}, AsInt64Slice(layout.minor_to_major()));

  const int64 dim1_size = values.width();
  const int64 dim0_size = values.height();
  CHECK_EQ(dim0_size, shape().dimensions(0));
  CHECK_EQ(dim1_size, shape().dimensions(1));
  Reserve(dim1_size * dim0_size);
  for (int64 dim0 = 0; dim0 < dim0_size; ++dim0) {
    for (int64 dim1 = 0; dim1 < dim1_size; ++dim1) {
      Set({dim0, dim1}, values(dim0, dim1));
    }
  }
}

template <typename NativeT>
void Literal::PopulateR2FromArray2D(const Array2D<NativeT>& values) {
  PopulateR2FromArray2DWithLayout(values, LayoutUtil::GetDefaultLayoutForR2());
}

template <typename NativeT>
void Literal::PopulateR3FromArray3DWithLayout(const Array3D<NativeT>& values,
                                              const Layout& layout) {
  *mutable_shape() = ShapeUtil::MakeShapeWithLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(),
      {values.n1(), values.n2(), values.n3()},
      AsInt64Slice(layout.minor_to_major()));

  CHECK_EQ(values.n1(), shape().dimensions(0));
  CHECK_EQ(values.n2(), shape().dimensions(1));
  CHECK_EQ(values.n3(), shape().dimensions(2));
  Reserve(values.n1() * values.n2() * values.n3());
  for (int64 dim0 = 0; dim0 < values.n1(); ++dim0) {
    for (int64 dim1 = 0; dim1 < values.n2(); ++dim1) {
      for (int64 dim2 = 0; dim2 < values.n3(); ++dim2) {
        Set({dim0, dim1, dim2}, values(dim0, dim1, dim2));
      }
    }
  }
}

template <typename NativeT>
void Literal::PopulateR3FromArray3D(const Array3D<NativeT>& values) {
  PopulateR3FromArray3DWithLayout(values, LayoutUtil::GetDefaultLayoutForR3());
}

template <typename NativeT>
void Literal::PopulateR4FromArray4DWithLayout(const Array4D<NativeT>& values,
                                              const Layout& layout) {
  *mutable_shape() = ShapeUtil::MakeShapeWithLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(),
      {values.planes(), values.depth(), values.height(), values.width()},
      AsInt64Slice(layout.minor_to_major()));

  CHECK_EQ(values.n1(), shape().dimensions(0));
  CHECK_EQ(values.n2(), shape().dimensions(1));
  CHECK_EQ(values.n3(), shape().dimensions(2));
  CHECK_EQ(values.n4(), shape().dimensions(3));
  Reserve(values.n1() * values.n2() * values.n3() * values.n4());
  for (int64 dim0 = 0; dim0 < values.n1(); ++dim0) {
    for (int64 dim1 = 0; dim1 < values.n2(); ++dim1) {
      for (int64 dim2 = 0; dim2 < values.n3(); ++dim2) {
        for (int64 dim3 = 0; dim3 < values.n4(); ++dim3) {
          Set({dim0, dim1, dim2, dim3}, values(dim0, dim1, dim2, dim3));
        }
      }
    }
  }
}

template <typename NativeT>
void Literal::PopulateR4FromArray4D(const Array4D<NativeT>& values) {
  PopulateR4FromArray4DWithLayout(values, LayoutUtil::GetDefaultLayoutForR4());
}

template <typename NativeT>
Status Literal::Populate(
    const std::function<NativeT(tensorflow::gtl::ArraySlice<int64> indexes)>&
        generator) {
  const Shape& this_shape = shape();
  int64 rank = ShapeUtil::Rank(this_shape);
  TF_RET_CHECK(this_shape.element_type() ==
               primitive_util::NativeToPrimitiveType<NativeT>());
  tensorflow::gtl::MutableArraySlice<NativeT> data =
      GetMutableArraySlice<NativeT>();
  if (rank > 0) {
    StrideConfig stride_config(this_shape, this_shape,
                               AsInt64Slice(this_shape.dimensions()));
    DimensionVector minor_scan_indexes(rank, 0);
    int64 minor_dimension_size =
        ShapeUtil::GetDimension(this_shape, stride_config.minor_dimension);

    auto init_function = [&](const std::vector<int64>& indexes) {
      int64 index = LinearIndex(indexes);
      std::copy(indexes.begin(), indexes.end(), minor_scan_indexes.begin());
      for (int64 i = 0; i < minor_dimension_size; ++i) {
        minor_scan_indexes[stride_config.minor_dimension] = i;
        data.at(index + i) = generator(minor_scan_indexes);
      }
      return true;
    };
    ShapeUtil::ForEachIndex(this_shape, stride_config.base,
                            stride_config.dimensions, stride_config.step,
                            init_function);
  } else {
    // For scalars.
    data.at(0) = generator({});
  }
  return Status::OK();
}

template <typename NativeT>
void Literal::PopulateWithValue(NativeT value,
                                tensorflow::gtl::ArraySlice<int64> dimensions) {
  *mutable_shape() = ShapeUtil::MakeShape(
      primitive_util::NativeToPrimitiveType<NativeT>(), dimensions);
  Resize<NativeT>(ShapeUtil::ElementsIn(shape()), value);
}

template <typename NativeSrcT, typename NativeDestT>
std::unique_ptr<Literal> Literal::Convert() const {
  const Shape& this_shape = shape();
  auto result_literal = MakeUnique<Literal>();
  Shape* result_shape = result_literal->mutable_shape();
  *result_shape = this_shape;
  result_shape->set_element_type(
      primitive_util::NativeToPrimitiveType<NativeDestT>());
  result_literal->Reserve(ShapeUtil::ElementsIn(*result_shape));
  tensorflow::gtl::ArraySlice<NativeSrcT> src_data =
      GetArraySlice<NativeSrcT>();
  tensorflow::gtl::MutableArraySlice<NativeDestT> dest_data =
      result_literal->GetMutableArraySlice<NativeDestT>();
  int64 num_elements = ShapeUtil::ElementsIn(this_shape);

  for (int64 i = 0; i < num_elements; ++i) {
    dest_data[i] = static_cast<NativeDestT>(src_data[i]);
  }
  return result_literal;
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal>
Literal::CreateFullWithMonotonicDim0MajorLayout(
    tensorflow::gtl::ArraySlice<int64> dimensions, NativeT value) {
  Shape this_shape = ShapeUtil::MakeShapeWithMonotonicDim0MajorLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(), dimensions);
  auto literal = MakeUnique<Literal>();
  *literal->mutable_shape() = this_shape;
  literal->Reserve(ShapeUtil::ElementsIn(this_shape));
  std::vector<int64> index(dimensions.size(), 0);
  do {
    literal->Set(index, value);
  } while (IndexUtil::BumpIndices(this_shape, &index));
  return literal;
}

template <typename NativeT>
std::unique_ptr<Literal> Literal::Replicate(int64 times) const {
  DimensionVector bounds = {times};
  bounds.reserve(shape().dimensions_size() + 1);
  for (int64 bound : shape().dimensions()) {
    bounds.push_back(bound);
  }
  auto literal = MakeUnique<Literal>();
  *literal->mutable_shape() =
      ShapeUtil::MakeShape(shape().element_type(), bounds);
  int64 elements = ShapeUtil::ElementsIn(literal->shape());
  if (elements == 0) {
    return literal;
  }
  literal->Reserve(elements);

  DimensionVector output_indices(bounds.size(), 0);
  tensorflow::gtl::ArraySlice<int64> input_indices = output_indices;
  input_indices.remove_prefix(1);

  bool done = false;
  while (!done) {
    const auto element = Get<NativeT>(input_indices);
    literal->Set<NativeT>(output_indices, element);

    done = true;
    for (int n = 0; n < output_indices.size(); ++n) {
      ++output_indices[n];
      if (output_indices[n] < bounds[n]) {
        done = false;
        break;
      }
      output_indices[n] = 0;
    }
  }
  return literal;
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_LITERAL_UTIL_H_
