/* Copyright 2016 The OpenXLA Authors.

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

#ifndef XLA_LITERAL_H_
#define XLA_LITERAL_H_

#include <algorithm>
#include <climits>
#include <complex>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/casts.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/array4d.h"
#include "xla/index_util.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/maybe_owning.h"
#include "xla/primitive_util.h"
#include "xla/printer.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/lib/core/bitmap.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Forward declare Literal and LiteralSlice class to be used by the creation
// methods in the base class.
class Literal;
class LiteralSlice;

// Abstract base class for literals.
class LiteralBase {
 public:
  using DynamicSizeType = ShapeUtil::DynamicSizeType;

  virtual ~LiteralBase() = 0;

  // Literals are equal if they have compatible shapes and the same data
  // values. Layout is not compared. For a layout sensitive comparison
  // call Equal() with layout_sensitive=true.
  bool operator==(const LiteralBase& other) const {
    return Equal(other, false);
  }
  bool operator!=(const LiteralBase& other) const { return !(*this == other); }

  // Compares two literals with optional layout sensitivity. If you use
  // literals in a hash map, together with AbslHashValue or Hash defined below,
  // you must use this method instead of operator== to ensure proper layout
  // handling.
  bool Equal(const LiteralBase& other, bool layout_sensitive) const;

  // Returns the shape of the literal.
  const Shape& shape() const;

  // Serialize to proto.
  LiteralProto ToProto() const;

  // Returns a Span of the array for this literal for the given NativeT
  // (e.g., float). CHECKs if the subshape of the literal at the given
  // ShapeIndex is not array. See primitive_util.h for the mapping from XLA type
  // to native type.
  template <typename NativeT>
  absl::Span<const NativeT> data(const ShapeIndex& shape_index = {}) const;

  // Returns a const pointer to (or size of) the underlying buffer holding the
  // array at the given shape index. CHECKs if the subshape of the literal at
  // the given ShapeIndex is not array.
  const void* untyped_data(const ShapeIndex& shape_index = {}) const;
  int64_t size_bytes(const ShapeIndex& shape_index = {}) const;

  // Computes the size in bytes of the output of the Serialize method.
  absl::StatusOr<int64_t> SerializedSize() const {
    return ShapeUtil::SerializedSize(shape());
  }

  // Serialize the Literal into the given output iterator, whose value_type must
  // be char.  It's up to the caller to ensure that output can store
  // SerializedSize() bytes of data.  This can be ensured by using
  // std::back_inserter, or by manually resizing the target container.
  // This serializer is useful for bypassing the 2GB protobuf serialization
  // limit with very large literals, and it should be faster than protobuf
  // serialization when performance is a concern.
  // The serialization format should not be relied on for forward/backward
  // compatibility.  If compatibility is required, you should use protobuf
  // serialization instead.
  template <typename OutputIterator>
  absl::Status Serialize(OutputIterator output) const {
    return SerializeWithShapeProto(shape().ToProto(), output);
  }

  // Serialize the Literal into the given string.  This method has the same
  // caveats as the Serialize() method above.
  absl::Status SerializeToString(std::string* output) const;

  // Serialize the Literal into a string and return it.  This method has the
  // same caveats as the Serialize() method above.
  absl::StatusOr<std::string> SerializeAsString() const;

  // Returns this literal's data as a string. This literal must be a rank-1 U8
  // array.
  std::string GetR1U8AsString() const;

  // Prints a string representation of the literal value. The Shape of the
  // literal is a prefix of the literal value in the string.
  //
  // Warning: this function can take minutes for multi-million element Literals.
  void Print(Printer* printer) const;

  // Similar to Print, but prints the result in a compact one-line form.
  void PrintOneline(Printer* printer) const;

  // Prints a string representation of the literal value which does *not*
  // include the shape string.
  void PrintWithoutShape(Printer* printer) const;

  // Similar to PrintWithoutShape, but prints the result in a compact one-line
  // form.
  void PrintWithoutShapeOneline(Printer* printer) const;

  // Prints a string representation of the literal value which includes the
  // shape string with its layout.does *not* include the shape string.
  void PrintWithLayout(Printer* printer) const;

  // Similar to PrintWithLayout, but prints the result in a compact one-line
  // form.
  void PrintWithLayoutOneline(Printer* printer) const;

  // Returns a string representation of the literal value. The Shape of the
  // literal is a prefix of the literal value in the string.
  //
  // Warning: this function can take minutes for multi-million element Literals.
  std::string ToString() const;

  // Similar to ToString, but return the result in a compact one-line form.
  std::string ToStringOneline() const;

  // Returns a string representation of the literal value which does *not*
  // include the shape string.
  std::string ToStringWithoutShape() const;

  // Similar to ToStringWithoutShape, but return the result in a compact
  // one-line form.
  std::string ToStringWithoutShapeOneline() const;

  // Returns a string representation of the literal value which includes the
  // shape string with its layout.does *not* include the shape string.
  std::string ToStringWithLayout() const;

  // Similar to ToStringWithLayout, but return the result in a compact one-line
  // form.
  std::string ToStringWithLayoutOneline() const;

  // Gets an element in the literal at the given index. The multi_index is
  // CHECKed against the dimension sizes.
  template <typename NativeT>
  NativeT Get(absl::Span<const int64_t> multi_index,
              const ShapeIndex& shape_index) const;

  // Overloads of Get for array literals. CHECKs if the literal is not
  // array-shaped and dense.
  template <typename NativeT>
  NativeT Get(absl::Span<const int64_t> multi_index) const;

  // Gets an element in the literal at the given linear index. Linear index is
  // CHECKed against the literal size.
  template <typename NativeT>
  NativeT GetLinear(int64_t linear_index, const ShapeIndex& shape_index) const;

  // Overloads of GetLinear for array literals. CHECKs if the literal is
  // not array-shaped and dense.
  template <typename NativeT>
  NativeT GetLinear(int64_t linear_index) const;

  // Get the dynamic size on dim_index in the literal at the given shape_index.
  DynamicSizeType GetDynamicSize(int64_t dim_index,
                                 const ShapeIndex& shape_index) const;
  DynamicSizeType GetDynamicSize(int64_t dim_index) const;

  // Returns the element value at index (0, ..., 0), however many zeroes are
  // required for that index.
  template <typename NativeT>
  NativeT GetFirstElement() const;

  // As above but returns any integer type casted to an int64_t.
  std::optional<int64_t> GetFirstInteger() const;

  // As Get(), but determines the correct type and converts the value
  // into text.
  std::string GetAsString(absl::Span<const int64_t> multi_index,
                          const ShapeIndex& shape_index = {}) const;

  // Return whether the value at the specified index is equal to the provided
  // generic `value` (T must be an arithmetic type).
  //
  // Precondition: must be an array.
  template <typename T>
  typename std::enable_if<std::numeric_limits<T>::is_specialized, bool>::type
  IsEqualAt(absl::Span<const int64_t> multi_index, T value) const {
    if (auto as_s64 = GetIntegralAsS64(multi_index)) {
      return *as_s64 == value;
    }
    complex128 as_complex128 = *GetAsComplex128(multi_index);
    return as_complex128.imag() == 0 && as_complex128.real() == value;
  }

  bool IsEqualAt(absl::Span<const int64_t> multi_index,
                 complex128 value) const {
    if (auto as_s64 = GetIntegralAsS64(multi_index)) {
      return *as_s64 == value.real() && value.imag() == 0;
    }
    auto as_complex128 = GetAsComplex128(multi_index);
    return *as_complex128 == value;
  }

  // As Get(), but determines the correct type and converts the value into
  // int64_t.  This literal must be an array.
  std::optional<int64_t> GetIntegralAsS64(
      absl::Span<const int64_t> multi_index) const;

  // As Get(), but determines the correct type, and converts the value into
  // double. This literal must be an array.
  std::optional<double> GetAsDouble(
      absl::Span<const int64_t> multi_index) const;

  // As Get(), but determines the correct type, and converts the value into
  // complex128. All floating point types can be converted into complex128.
  //
  // This literal must be an array.
  std::optional<complex128> GetAsComplex128(
      absl::Span<const int64_t> multi_index) const;

  // Convert each element whose *linear* index is listed in "linear_indices"
  // to a double and return the sum of all of these elements.
  std::optional<double> GetSumAsDouble(
      absl::Span<const int64_t> linear_indices) const;

  // Invokes the "per cell" callback for each element in the provided
  // literal with the element's indices and a string representation of
  // the element's value.
  //
  // This function is useful if you want a polymorphic representation
  // of the tensor's elements (turning it to a string for something
  // like representation in a protobuf).
  //
  // This literal must have a dense layout.
  void EachCellAsString(
      absl::FunctionRef<void(absl::Span<const int64_t> indices,
                             const std::string& value)>
          per_cell) const;
  template <typename NativeT>
  void EachCell(
      absl::FunctionRef<void(absl::Span<const int64_t> indices, NativeT value)>
          per_cell) const;

  // Checks whether all of this literal's values are equal to the given scalar
  // literal.
  //
  // If `this` is not an array (e.g. it's a tuple), returns false.  This is
  // simpler than trying to handle subshapes here, and it's almost always what
  // you want.
  //
  // Preconditions:
  //  - `scalar` is a scalar.
  //  - `scalar` has the same element-type as `this`.
  bool IsAll(const Literal& scalar) const;

  // Returns whether every element in this literal is equal to value.
  //
  // value is an int8_t because we expect this to be called with small
  // compile-time constants (0, -1, etc.) and so that whatever value you pass
  // can be represented exactly by floating-point types as small as 16 bits.
  //
  // If value doesn't fit in this literal's type, returns false.  Values of 1/0
  // are considered equal to true/false; other values are not considered equal
  // to true.
  //
  // Returns false if this literal is not array-shaped.
  bool IsAll(int8_t value) const;

  // Like IsAll(int8_t), except we check whether the literal is equal to a
  // particular floating-point or complex number.
  //
  // Returns false if this literal is not a floating-point / complex value, or
  // if it's not an array.
  //
  // This casts value to the type of literal, then compares using ==, with the
  // caveat that NaNs are considered equal. Unlike IsAll, this does not
  // necessarily return false if the value does not fit in this literal's type.
  bool IsAllFloat(float value) const;
  bool IsAllComplex(complex64 value) const;

  // Determines if this literal consists entirely of the first element of the
  // literal.
  //
  // Returns false if this literal is not an array.
  bool IsAllFirst() const;

  // Returns the number of elements that have value equal to the given value.
  // Returns 0 if value does not fit in this literal's type or if the literal
  // is not an array.
  template <typename T>
  int64_t CountEqual(T value) const;

  // Returns the number of elements that have value equal to the given complex
  // value. Returns 0 if value does not fit in this literal's type or if the
  // literal is not an array.
  template <typename T>
  int64_t CountEqual(std::complex<T> value) const;

  // Literal consists entirely of an iota.
  bool IsR1Iota() const;

  // Returns the stride if the literal is a strided iota.
  std::optional<int64_t> IsR1StridedIota() const;

  // Returns whether this literal is zero at the specified index. This literal
  // must be an array with a dense layout.
  bool IsZero(absl::Span<const int64_t> indices) const;

  // Returns the count of the elements in the array at the given shape index in
  // this literal.
  int64_t element_count(const ShapeIndex& index = {}) const {
    if (index.empty()) {
      // Common case, avoid GetSubshape().
      return ShapeUtil::ElementsIn(shape());
    }
    return ShapeUtil::ElementsIn(ShapeUtil::GetSubshape(shape(), index));
  }

  // This definition is here to ensure that nobody accidentally implements this
  // function which would lead to inconsistencies. Use Hash instead.
  //
  // Note: code below should really be static_assert(false, ...), but that is
  // unfortunately not possible, as some compilers consider it invalid code,
  // see https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2593r0.html.
  template <typename H>
  friend H AbslHashValue(H state, const LiteralBase& value) {
    static_assert(sizeof(H) == 0,
                  "Do not use Literal directly as a hash key, because it has "
                  "multiple definitions of equality - layout sensitive or "
                  "insensitive. Instead, use AbslHashable<...>() to create a "
                  "wrapper with layout sensitivity specified suitable for "
                  "passing to Absl::Hash");
  }

  // Always use this together with the Equal method and not operator== in order
  // to handle layout sensitivity properly.
  template <typename H, bool kIsLayoutSensitive = true,
            int64_t kByteLimit = std::numeric_limits<int64_t>::max()>
  static H Hash(H state, const LiteralBase& literal) {
    state =
        Shape::Hash<H, kIsLayoutSensitive>(std::move(state), literal.shape());

    ShapeUtil::ForEachSubshape(literal.shape(), [&](const Shape& subshape,
                                                    const ShapeIndex& index) {
      if (!subshape.IsArray()) {
        return;
      }

      CHECK(LayoutUtil::IsDenseArray(subshape));
      const int64_t size_bytes = literal.size_bytes(index);
      const int64_t bytes_to_hash = std::min(size_bytes, kByteLimit);
      // When layout insensitive, we need to hash the data bytes in logical
      // order rather than physical order.
      const bool use_physical_order =
          kIsLayoutSensitive || !subshape.has_layout();
      auto data = absl::MakeConstSpan(
          static_cast<const char*>(literal.untyped_data(index)), size_bytes);
      if (use_physical_order) {
        state = H::combine(std::move(state), data.first(bytes_to_hash));
        return;
      }
      const int64_t elem_size =
          ShapeUtil::ByteSizeOfPrimitiveType(subshape.element_type());
      absl::Span<const int64_t> minor_to_major =
          subshape.layout().minor_to_major();
      DimensionVector elem_index(subshape.rank());
      absl::Span<int64_t> elem_index_span(elem_index.data(), elem_index.size());
      int64_t bytes_hashed = 0;
      while (bytes_hashed < bytes_to_hash) {
        int64_t offset =
            elem_size * IndexUtil::MultidimensionalIndexToLinearIndex(
                            subshape, minor_to_major, elem_index);
        state = H::combine(std::move(state), data.subspan(offset, elem_size));
        if (!IndexUtil::BumpIndices(subshape, elem_index_span)) return;
        bytes_hashed += elem_size;
      }
    });

    return std::move(state);
  }

  // Templated wrapper struct to control layout sensitivity during Absl::Hash.
  template <bool layout_sensitive>
  struct AbslHashable {
    const LiteralBase& literal;
    explicit AbslHashable(const LiteralBase& l) : literal(l) {}
    template <typename H>
    friend H AbslHashValue(H h, const AbslHashable& w) {
      return LiteralBase::Hash<H, layout_sensitive>(std::move(h), w.literal);
    }
  };

  // Converts this literal to the given shape. Returns an error is the
  // conversion is not possible.
  absl::StatusOr<Literal> ConvertToShape(const Shape& dest_shape) const;

  // Converts this literal to another primitive type using a bitcast
  // conversion. Returns an error if the conversion is not possible. This
  // literal must be array-shaped.
  absl::StatusOr<Literal> BitcastConvert(const Shape& dest_shape) const;

  // Converts this literal to another primitive type. Returns an error if the
  // conversion is not possible. This literal must be array-shaped.
  absl::StatusOr<Literal> Convert(PrimitiveType primitive_dest_type) const;

  // Clones the underlying buffers into a new Literal.
  Literal Clone() const;
  std::unique_ptr<Literal> CloneToUnique() const;

  // TODO(b/67651157): The methods below which perform computation on Literals
  // (Reshape, Slice, etc) should be moved elsewhere, and perhaps combined with
  // evaluator code which operates on Literals.
  //
  // Creates a new value that has the equivalent value as this
  // literal, but conforms to new_layout; e.g. a literal matrix that was in {0,
  // 1} minor-to-major dimension layout can be re-layed-out as {1, 0}
  // minor-to-major dimension layout and the value in the cell at any given
  // logical index (i0, i1) will be the same.
  //
  // For tuple shaped literals, shape_index should be used to select the inner
  // array that the new layout applies to.
  //
  // Note: this is useful when the client wants to ensure that a value placed in
  // the XLA allocation tracker has a particular layout; for efficiency
  // purposes or avoiding unimplemented operation/layout combinations.
  Literal Relayout(const Layout& new_layout,
                   const ShapeIndex& shape_index = {}) const;

  // An overload of Relayout which changes the layout of the entire shape rather
  // than being limited to a single array within the shape.
  Literal Relayout(const Shape& shape_with_layout) const;

  // Generate a new literal whose static sizes are equal to the previous
  // literal's dynamic sizes.
  Literal ToStatic() const;

  // Expand a static literal into a new one with a bounded dynamic literal. The
  // static dimensions of the original literal becomes dynamic dimensions of the
  // new literal, where the argument `bounded_shape` becomes the bounded shape
  // of the new literal.
  //
  // Precondition: bounded_shape.is_dynamic()
  Literal ToBoundedDynamic(const Shape& bounded_shape) const;

  // Creates a new literal by reshaping this literal to have the given
  // dimensions. The total number of elements must not change; The
  // implementation currently only supports monotonic dim0-major layouts.
  // This literal must be an array.
  absl::StatusOr<Literal> Reshape(absl::Span<const int64_t> dimensions) const;

  // Creates a new literal by broadcasting this literal with `dimensions` to
  // yield a literal of shape `result_shape`.
  absl::StatusOr<Literal> Broadcast(const Shape& result_shape,
                                    absl::Span<const int64_t> dimensions) const;

  // Creates a new literal by reordering the dimensions of this literal.
  // The given `permutation` must be a permutation of the dimension numbers
  // in the original literal, and it specifies the order of the new dimensions
  // in the result literal (i.e., new_order[i] = old_order[permutation[i]]).
  // For example, a transpose call on a literal of shape [3 x 8 x 4] and
  // `permutation` = {2, 0, 1} returns a new literal of shape [4 x 3 x 8].
  // This literal must be an array.
  Literal Transpose(absl::Span<const int64_t> permutation) const;

  // Creates a sub-array from this literal by extracting the indices
  // [start_index, limit_index) of each dimension. The result literal has the
  // same rank and layout as for the given literal. The number of indices in
  // start_indices and limit_indices must be the rank of the literal, and the
  // indices follow the order of the dimensions.
  // This literal must be an array.
  Literal Slice(absl::Span<const int64_t> start_indices,
                absl::Span<const int64_t> limit_indices) const;

  // Creates a literal with a prepended dimension with bound "times"; e.g. a
  // f32[3x2] with times=4 will produce a f32[4x3x2] with the 3x2 from this
  // literal replicated four times.
  // This literal must be an array.
  template <typename NativeT>
  Literal Replicate(int64_t times) const;

  // Returns true if the leaf arrays of the literal within the given shape index
  // are all determined.
  // See comments on ArrayValueState for detailed explanation.
  bool IsDetermined(const ShapeIndex& shape_index = {}) const;

  // Returns true if the leaf arrays of the literal within the given shape index
  // are all known.
  // See comments on ArrayValueState for detailed explanation.
  bool IsKnown(const ShapeIndex& shape_index = {}) const;

  // Creates a new Literal object with the shape specified as parameter.
  // The content of the literal values is the default value of the primitive
  // type of literal itself (0 for numeric types, and false for predicates).
  //
  // Note: It's an antipattern to use this method then immediately call
  // MutableLiteralBase::Populate on the result (since that results in zero
  // initialization, then reinitialization. Consider if a call to
  // std::make_unique<Literal>(shape), followed by the call to
  // MutableLiteralBase::Populate can be used instead.
  static Literal CreateFromShape(const Shape& shape);

  // WARNING: These two functions are only supposed to be used by HloEvaluator.
  // The rest of XLA assumes all literals are known.
  // Similar to CreateFromShape() but marks all leaf arrays as unknown.
  static Literal CreateFromShapeWithUnknownLeafArrays(const Shape& shape);
  // Similar to CreateFromShape() but marks all leaf arrays as undetermined.
  static Literal CreateFromShapeWithUndeterminedLeafArrays(const Shape& shape);

 protected:
  class Piece;

  // Recursively builds the subtree for the given piece and sets the subshapes
  // of the given piece with the given shape.
  void BuildPieceSubtree(const Shape& shape, Piece* piece);

  template <typename OutputIterator>
  absl::Status SerializeWithShapeProto(const ShapeProto& proto,
                                       OutputIterator output) const;

  template <typename OutputIterator>
  class SerializeState {
   public:
    SerializeState(const ShapeProto& shape, OutputIterator output)
        : output_(output) {
      WriteShape(shape);
    }

    int64_t num_written() const { return num_written_; }

    template <typename NativeT>
    void WriteElement(NativeT element) {
      constexpr PrimitiveType primitive_type =
          primitive_util::NativeToPrimitiveType<NativeT>();
      static_assert(primitive_util::BitWidth(primitive_type) % 8 == 0);
      if constexpr (primitive_util::IsComplexType(primitive_type)) {
        WriteElement(element.real());
        WriteElement(element.imag());
      } else {
        constexpr PrimitiveType unsigned_type =
            primitive_util::UnsignedIntegralTypeForBitWidth(
                primitive_util::BitWidth(primitive_type));
        using UnsignedT = primitive_util::NativeTypeOf<unsigned_type>;
        UnsignedT unsigned_element = absl::bit_cast<UnsignedT>(element);
        if constexpr (sizeof(UnsignedT) == 1) {
          *output_++ = absl::bit_cast<char>(unsigned_element);
          ++num_written_;
        } else {
          for (int i = 0; i < sizeof unsigned_element; ++i) {
            *output_++ = static_cast<char>(unsigned_element);
            unsigned_element >>= CHAR_BIT;
            ++num_written_;
          }
        }
      }
    }

    template <typename NativeT>
    void WriteElements(absl::Span<const NativeT> elements) {
      constexpr PrimitiveType primitive_type =
          primitive_util::NativeToPrimitiveType<NativeT>();
      constexpr int bits_per_element = primitive_util::BitWidth(primitive_type);
      if constexpr (bits_per_element < 8) {
        static_assert(!primitive_util::IsComplexType(primitive_type));
        static_assert(8 % bits_per_element == 0);

        constexpr int elements_per_byte = 8 / bits_per_element;
        int64_t bytes = elements.size() / elements_per_byte;
        for (int64_t i = 0; i < bytes; ++i) {
          uint8_t byte = 0;
          for (int b = 0; b < elements_per_byte; ++b) {
            uint8_t src = Eigen::numext::bit_cast<uint8_t>(
                              elements[i * elements_per_byte + b]) &
                          LsbMask<uint8_t>(bits_per_element);
            byte |= src << (b * bits_per_element);
          }
          WriteElement(byte);
        }
        int64_t rest = elements.size() % elements_per_byte;
        if (rest != 0) {
          uint8_t byte = 0;
          for (int64_t b = 0; b < rest; ++b) {
            uint8_t src = Eigen::numext::bit_cast<uint8_t>(
                              elements[bytes * elements_per_byte + b]) &
                          LsbMask<uint8_t>(bits_per_element);
            byte |= src << (b * bits_per_element);
          }
          WriteElement(byte);
        }
      } else {
        for (NativeT element : elements) {
          WriteElement(element);
        }
      }
    }

    void WriteDynamicSizes(absl::Span<const DynamicSizeType> sizes) {
      WriteElements(sizes);
    }

   private:
    void WriteShape(const ShapeProto& proto) {
      std::string shape_bytes = proto.SerializeAsString();
      uint64_t shape_size = shape_bytes.size();
      WriteElement(shape_size);
      output_ = std::copy(shape_bytes.begin(), shape_bytes.end(), output_);
      num_written_ += shape_bytes.size();
    }

    OutputIterator output_;
    int64_t num_written_ = 0;
  };

  template <typename InputIterator>
  class DeserializeState {
   public:
    DeserializeState(InputIterator input, InputIterator end)
        : input_(input), end_(end) {}

    int64_t num_read() const { return num_read_; }

    template <typename NativeT>
    ABSL_MUST_USE_RESULT bool ReadElement(NativeT& element) {
      constexpr PrimitiveType primitive_type =
          primitive_util::NativeToPrimitiveType<NativeT>();
      static_assert(primitive_util::BitWidth(primitive_type) % 8 == 0);
      if constexpr (primitive_util::IsComplexType(primitive_type)) {
        using ComponentT =
            primitive_util::NativeTypeOf<primitive_util::ComplexComponentType(
                primitive_type)>;
        ComponentT real;
        if (!ReadElement(real)) {
          return false;
        }
        ComponentT imag;
        if (!ReadElement(imag)) {
          return false;
        }
        element = NativeT(real, imag);
      } else {
        constexpr PrimitiveType unsigned_type =
            primitive_util::UnsignedIntegralTypeForBitWidth(
                primitive_util::BitWidth(primitive_type));
        using UnsignedT = primitive_util::NativeTypeOf<unsigned_type>;
        if constexpr (sizeof(UnsignedT) == 1) {
          if (at_end()) {
            return false;
          }
          element = absl::bit_cast<NativeT>(*input_++);
          ++num_read_;
        } else {
          UnsignedT unsigned_element = 0;
          for (int i = 0, shift = 0; i < sizeof unsigned_element;
               ++i, shift += CHAR_BIT) {
            if (at_end()) {
              return false;
            }
            unsigned_element |=
                static_cast<UnsignedT>(static_cast<unsigned char>(*input_++))
                << shift;
            ++num_read_;
          }
          element = absl::bit_cast<NativeT>(unsigned_element);
        }
      }
      return true;
    }

    template <typename NativeT>
    ABSL_MUST_USE_RESULT bool ReadElements(absl::Span<NativeT> elements) {
      constexpr PrimitiveType primitive_type =
          primitive_util::NativeToPrimitiveType<NativeT>();
      constexpr int bits_per_element = primitive_util::BitWidth(primitive_type);
      if constexpr (bits_per_element < 8) {
        static_assert(!primitive_util::IsComplexType(primitive_type));
        static_assert(8 % bits_per_element == 0);

        constexpr auto cast = [](uint8_t x) -> NativeT {
          if constexpr (primitive_util::IsFloatingPointType(primitive_type)) {
            return Eigen::numext::bit_cast<NativeT>(x);
          }
          return static_cast<NativeT>(x);
        };

        constexpr int elements_per_byte = 8 / bits_per_element;
        int64_t bytes = elements.size() / elements_per_byte;
        for (int64_t i = 0; i < bytes; ++i) {
          uint8_t byte;
          if (!ReadElement(byte)) {
            return false;
          }
          for (int b = 0; b < elements_per_byte; ++b) {
            elements[i * elements_per_byte + b] =
                cast(byte & LsbMask<uint8_t>(bits_per_element));
            byte >>= bits_per_element;
          }
        }
        int64_t rest = elements.size() % elements_per_byte;
        if (rest != 0) {
          uint8_t byte;
          if (!ReadElement(byte)) {
            return false;
          }
          for (int64_t b = 0; b < rest; ++b) {
            elements[bytes * elements_per_byte + b] =
                cast(byte & LsbMask<uint8_t>(bits_per_element));
            byte >>= bits_per_element;
          }
        }
      } else {
        for (NativeT& element : elements) {
          if (!ReadElement(element)) {
            return false;
          }
        }
      }
      return true;
    }

    bool ReadDynamicSizes(absl::Span<DynamicSizeType> sizes) {
      return ReadElements(sizes);
    }

    absl::StatusOr<Shape> ReadShape(uint64_t size) {
      std::string shape_bytes;
      shape_bytes.reserve(size);
      while (shape_bytes.size() < size) {
        if (at_end()) {
          return InvalidArgument("Failed to read shape data");
        }
        shape_bytes.push_back(*input_++);
        ++num_read_;
      }
      ShapeProto proto;
      if (!proto.ParseFromString(shape_bytes)) {
        return InvalidArgument("Failed to parse shape protobuf");
      }
      Shape shape(proto);
      TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(shape));
      return std::move(shape);
    }

    bool at_end() const { return input_ == end_; }

   private:
    InputIterator input_;
    InputIterator end_;
    int64_t num_read_ = 0;
  };

  // Array literals could be in one of the following three states:
  //   1) Known: we have evaluated and known the value of the array literal.
  //   2) Unknown: we have tried to evaluate the array literal, but its value
  //               cannot be evaluated statically.
  //   3) Undetermined: we haven't tried to evaluate the array literal.
  //  Unknown and Undetermined states are only meant to be used within
  //  HloEvaluator. The rest of XLA assumes array literals are all known.
  //  Literals that are unknown or undetermined can be copied from, using
  //  CopyFrom and Clone, or moved from using move constructor. Accessing values
  //  of such literals causes undefined behavior.
  enum class ArrayValueState { kKnown = 0, kUnknown = 1, kUndetermined = 2 };

  // A data structure representing a subshape at a particular ShapeIndex within
  // the literal. For array-shaped ShapeIndexes, this data structure holds the
  // pointer to the memory allocated for the array data.
  class Piece {
   public:
    ArrayValueState get_array_value_state() const;
    void set_array_value_state(ArrayValueState state);
    // Returns the buffer holding the array data for this piece as an array
    // slice. This piece must be array-shaped.
    template <typename NativeT>
    absl::Span<const NativeT> data() const;
    template <typename NativeT>
    absl::Span<NativeT> data();

    // Returns the buffer holding the array data for this piece as a void*. This
    // piece must be array-shaped.
    void* untyped_data();
    const void* untyped_data() const;

    // Gets or sets an element in the array at the given index. The multi_index
    // is CHECKed against the dimension sizes of the array.  This piece must be
    // array-shaped.
    template <typename NativeT>
    NativeT Get(absl::Span<const int64_t> index) const;
    template <typename NativeT>
    void Set(absl::Span<const int64_t> index, NativeT value);

    // Gets or sets an element in the array at the given linear index. The
    // linear index is CHECKed against the total number of elements in the
    // array. This piece must be array-shaped.
    template <typename NativeT>
    NativeT GetLinear(int64_t linear_index) const;
    template <typename NativeT>
    void SetLinear(int64_t linear_index, NativeT value);

    DynamicSizeType GetDynamicSize(int64_t dim_index) const;
    void SetDynamicSize(int64_t dim_index, DynamicSizeType size);
    void AllocateBuffers();
    void DeallocateBuffers();
    // Gets/sets the buffer holding the array data.
    const char* buffer() const;
    char* buffer() {
      return const_cast<char*>(const_cast<const Piece*>(this)->buffer());
    }
    void set_buffer(char* buffer) {
      DCHECK(LayoutUtil::IsDenseArray(*subshape_));
      storage_.Emplace<DenseRep>(buffer);
    }
    void MoveDataFrom(Piece& from) {
      DCHECK(!storage_.Isa<DenseRep>());
      DCHECK(!storage_.Isa<TupleRep>());
      if (auto* dense_rep = from.storage_.GetDenseRep()) {
        storage_.Emplace<DenseRep>(dense_rep->data);
      } else if (auto* inlined_rep = from.storage_.GetDenseInlinedRep()) {
        storage_.Emplace<DenseInlinedRep>(inlined_rep->data,
                                          from.total_bytes_dense());
      }
      from.storage_.Emplace<Uninitialized>();
    }

    // Gets/sets the buffer holding dynamic sizes.
    const DynamicSizeType* dynamic_size_buffer() const {
      DCHECK(LayoutUtil::IsDenseArray(*subshape_));
      return reinterpret_cast<const DynamicSizeType*>(
          buffer() + dynamic_size_buffer_offset());
    }
    DynamicSizeType* dynamic_size_buffer() {
      return const_cast<DynamicSizeType*>(
          const_cast<const Piece*>(this)->dynamic_size_buffer());
    }

    int64_t dynamic_size_buffer_bytes() const {
      DCHECK(LayoutUtil::IsDenseArray(*subshape_));
      return subshape().rank() * sizeof(DynamicSizeType);
    }

    // Gets or sets the subshape of this piece. This reference points to a
    // subshape within the shape in the containing Literal (Literal::shape_).
    const Shape& subshape() const { return *subshape_; }
    void set_subshape(const Shape* subshape) {
      subshape_ = subshape;
      if (storage_.Isa<Uninitialized>()) {
        if (subshape_->IsTuple()) {
          storage_.Emplace<TupleRep>();
        }
      }
    }

    // Returns the size in bytes of the buffer holding the dense array data.
    int64_t size_bytes_dense() const {
      DCHECK(LayoutUtil::IsDenseArray(*subshape_));
      return ShapeUtil::ByteSizeOf(subshape());
    }

    // The dynamic metadata starts at the end of the data in the literal.
    // The literal can have any number of bytes. For example, it could be a PRED
    // with 7 elements. `dynamic_size_buffer_offset` returns the number of bytes
    // before the dynamic size information including whatever padding is needed
    // to align the start of the dynamic size information so that it is aligned
    // to a multiple of `sizeof(DynamicSizeType)`.
    int64_t dynamic_size_buffer_offset() const {
      // Make sure the dynamic buffer starts on a boundary aligned to
      // `sizeof(DynamicSizeType)`.
      return RoundUpTo<int64_t>(size_bytes_dense(), sizeof(DynamicSizeType));
    }

    // Total size in bytes, including the dynamic size addition.
    //
    // The shape can become dynamic after this literal is allocated, so we
    // over-allocate the margin for the dynamic shape description in case we
    // need it.
    int64_t total_bytes_dense() const {
      return dynamic_size_buffer_offset() + dynamic_size_buffer_bytes();
    }

    // Returns the number of elements in this piece's array.
    int64_t element_count() const { return ShapeUtil::ElementsIn(subshape()); }

    // Returns the child piece at 'index' of this piece.
    Piece& child(int64_t index) {
      return const_cast<Piece&>(const_cast<const Piece*>(this)->child(index));
    }
    const Piece& child(int64_t index) const {
      auto* tuple_rep = storage_.GetTupleRep();
      DCHECK(tuple_rep);
      return tuple_rep->children[index];
    }

    // Adds a child piece to this piece's children.
    void emplace_back(Piece child_piece) {
      auto* tuple_rep = storage_.GetTupleRep();
      DCHECK(tuple_rep);
      tuple_rep->children.emplace_back(std::move(child_piece));
    }

    // Returns the size of children pieces of this piece.
    int64_t children_size() const {
      if (auto* tuple_rep = storage_.GetTupleRep()) {
        return tuple_rep->children.size();
      }
      return 0;
    }

    // Visitor functions that recursively traverses the piece and calls the
    // given function at each child piece. The function has the type:
    //    void (const ShapeIndex& index, const Piece& piece)
    template <typename Fn>
    void ForEachSubpiece(const Fn& func) const {
      ShapeIndex index;
      return ForEachHelper(
                 [&func](const ShapeIndex& index, const Piece& piece) {
                   func(index, piece);
                   return absl::OkStatus();
                 },
                 *this, &index)
          .IgnoreError();
    }
    // Same as above, but the function has the type:
    //    absl::Status (const ShapeIndex& index, const Piece& piece)
    // The first non-OK return value is returned by the function.
    template <typename Fn>
    absl::Status ForEachSubpieceWithStatus(const Fn& func) const {
      ShapeIndex index;
      return ForEachHelper(func, *this, &index);
    }
    // Same as above, but the function has the type:
    //    Bool (const ShapeIndex& index, const Piece& piece)
    // The first non-true return value is returned by the function.
    template <typename Fn>
    bool ForEachSubpieceWithBool(const Fn& func) const {
      ShapeIndex index;
      return ForEachHelperBool(func, *this, &index);
    }
    // Same as above, but the function has the type:
    //    Void (const ShapeIndex& index, Piece& piece)
    template <typename Fn>
    void ForEachMutableSubpiece(const Fn& func) {
      ShapeIndex index;
      return ForEachMutableHelper(
                 [&func](const ShapeIndex& index, Piece* piece) {
                   func(index, piece);
                   return absl::OkStatus();
                 },
                 const_cast<xla::LiteralBase::Piece*>(this), &index)
          .IgnoreError();
    }
    // Same as above, but the function has the type:
    //    absl::Status (const ShapeIndex& index, Piece& piece)
    // The first non-OK return value is returned by the function.
    template <typename Fn>
    absl::Status ForEachMutableSubpieceWithStatus(const Fn& func) {
      ShapeIndex index;
      return ForEachMutableHelper(
          func, const_cast<xla::LiteralBase::Piece*>(this), &index);
    }

    // Checks whether all elements of this Piece are equal to the given literal.
    //
    // Returns false if this Piece is not an array.
    //
    // Preconditions:
    //  - `scalar` is a scalar.
    //  - `scalar`'s type matches that of `this`.
    bool IsAll(const Literal& scalar) const;

    // Returns the number of elements with equal value to the given literal.
    // Returns 0 if this Piece is not an array.
    int64_t CountAll(const Literal& scalar) const;

    // Returns true if this piece and 'other' contain the same data. This piece
    // and 'other' must be array-shaped and compatible. If a literal has dynamic
    // shape, comparison is done only for the valid elements.
    bool EqualElements(const Piece& other) const;

    // Returns true if this piece and other pieces have the same dynamic
    // dimension sizes.
    bool EqualDynamicSize(const Piece& other) const;

    // Writes the shape and data (if array-shaped) into the given proto.
    void WriteToProto(LiteralProto* proto) const;

    // Copy the data from 'src' into this piece's buffer. Shapes of this piece
    // and src must be compatible. If only_dynamic_bound is true, only elements
    // within dynamic bounds will be copied.
    absl::Status CopyFrom(const Piece& src, bool only_dynamic_bound);

    // Copies the data from the given proto into this piece. The shape of this
    // piece must be equal (not just compatible) to the shape of the proto.
    absl::Status CopyFromProto(const LiteralProto& proto);

    // See comments on ArrayValueState for detailed explanation.
    bool IsDetermined() const;

    bool IsKnown() const;

    // Serialize the data contained by this Piece into the given serialization
    // state.
    template <typename NativeT, typename OutputIterator>
    void SerializeData(SerializeState<OutputIterator>& state) const;

    // Deserialize the data for this Piece from the given serialization state.
    template <typename NativeT, typename InputIterator>
    bool DeserializeData(DeserializeState<InputIterator>& state);

   private:
    // Literals can be used as DMA targets, which can require alignment. We
    // force a tsl::Allocator::kAllocatorAlignment-byte minimum alignment.
    static constexpr size_t kMinimumAlignment = 64;

    // The maximum number of bytes that can be inlined in the DenseInlinedRep.
    static constexpr size_t kMaxInlinedBytes = 24;

    // Uninitialized state representation.
    struct Uninitialized {};

    // Children pieces for tuple shaped pieces.
    struct TupleRep {
      std::vector<Piece> children;
    };

    // Out of line dense array storage.
    struct DenseRep {
      DenseRep() = default;
      explicit DenseRep(char* data) : data(data) {}

      char* data = nullptr;
    };

    // Inlined dense array storage.
    struct DenseInlinedRep {
      DenseInlinedRep() = default;
      DenseInlinedRep(const char* init, size_t size) {
        DCHECK_LE(size, kMaxInlinedBytes);
        std::memcpy(data, init, size);
      }

      alignas(kMinimumAlignment) char data[kMaxInlinedBytes];
    };

    // A wrapper around the piece representations with cached data pointer.
    class Storage {
     public:
      Storage() = default;

      Storage(Storage&& other) { *this = std::move(other); }
      Storage& operator=(Storage&& other) {
        rep_ = std::move(other.rep_);
        data_ = other.data_;

        if (auto* inline_rep = GetDenseInlinedRep()) {
          data_ = inline_rep->data;
        }

        other.rep_.emplace<Uninitialized>();
        other.data_ = nullptr;

        return *this;
      }

      template <typename Rep>
      bool Isa() const {
        return std::holds_alternative<Rep>(rep_);
      }

      template <typename Rep, typename... Args>
      Rep& Emplace(Args... args) {
        Rep& emplaced = rep_.emplace<Rep>(std::forward<Args>(args)...);
        if constexpr (std::is_same_v<Rep, DenseRep> ||
                      std::is_same_v<Rep, DenseInlinedRep>) {
          data_ = emplaced.data;
        } else {
          data_ = nullptr;
        }
        return emplaced;
      }

      const DenseInlinedRep* GetDenseInlinedRep() const {
        return std::get_if<DenseInlinedRep>(&rep_);
      }

      DenseInlinedRep* GetDenseInlinedRep() {
        return std::get_if<DenseInlinedRep>(&rep_);
      }

      const DenseRep* GetDenseRep() const {
        return std::get_if<DenseRep>(&rep_);
      }

      DenseRep* GetDenseRep() { return std::get_if<DenseRep>(&rep_); }

      const TupleRep* GetTupleRep() const {
        return std::get_if<TupleRep>(&rep_);
      }

      TupleRep* GetTupleRep() { return std::get_if<TupleRep>(&rep_); }

      const char* data() const {
        DCHECK_EQ(dense_data(), data_) << "cached data pointer is stale";
        return data_;
      }

      char* data() {
        DCHECK_EQ(dense_data(), data_) << "cached data pointer is stale";
        return data_;
      }

     private:
      const char* dense_data() const {
        if (auto* rep = GetDenseRep()) return rep->data;
        if (auto* rep = GetDenseInlinedRep()) return rep->data;
        return nullptr;
      }

      std::variant<Uninitialized, TupleRep, DenseRep, DenseInlinedRep> rep_;
      char* data_ = nullptr;  // cached `rep_.data` value for dense reps
    };

    // Helpers for traversing the piece via ForEachSubpiece rooted at 'index'.
    // The first non-OK (or non-true) value is returned by the function.
    // The callable 'func' has the same signature as described above in
    // ForEachSubpiece*.
    template <typename Fn>
    absl::Status ForEachHelper(const Fn& func, const Piece& piece,
                               ShapeIndex* index) const {
      TF_RETURN_IF_ERROR(func(*index, piece));
      if (auto* tuple_rep = piece.storage_.GetTupleRep()) {
        for (int64_t i = 0; i < tuple_rep->children.size(); ++i) {
          index->push_back(i);
          TF_RETURN_IF_ERROR(
              ForEachHelper(func, tuple_rep->children[i], index));
          index->pop_back();
        }
      }
      return absl::OkStatus();
    }
    template <typename Fn>
    bool ForEachHelperBool(const Fn& func, const Piece& piece,
                           ShapeIndex* index) const {
      if (!func(*index, piece)) {
        return false;
      }
      if (auto* tuple_rep = piece.storage_.GetTupleRep()) {
        for (int64_t i = 0; i < tuple_rep->children.size(); ++i) {
          index->push_back(i);
          if (!ForEachHelperBool(func, tuple_rep->children[i], index)) {
            return false;
          }
          index->pop_back();
        }
      }
      return true;
    }
    template <typename Fn>
    absl::Status ForEachMutableHelper(const Fn& func, Piece* piece,
                                      ShapeIndex* index) {
      TF_RETURN_IF_ERROR(func(*index, piece));
      if (auto* tuple_rep = piece->storage_.GetTupleRep()) {
        for (int64_t i = 0; i < tuple_rep->children.size(); ++i) {
          index->push_back(i);
          TF_RETURN_IF_ERROR(
              ForEachMutableHelper(func, &tuple_rep->children[i], index));
          index->pop_back();
        }
      }
      return absl::OkStatus();
    }

    // Recursive helper for EqualElements.
    template <typename NativeT>
    bool EqualElementsInternal(const Piece& other,
                               std::vector<int64_t>* multi_index) const;

    // Internal helper to copy elements from another given piece
    template <typename NativeT>
    void CopyElementsWithDynamicBound(const LiteralBase::Piece& src);

    // Storage for this piece.
    Storage storage_;

    // The shape of piece. This points into the shape of the containing Literal
    // (Literal::shape_).
    const Shape* subshape_ = nullptr;

    ArrayValueState array_value_state_ = ArrayValueState::kKnown;
  };  // class Piece

  const Piece& piece(const ShapeIndex& shape_index) const;

  // Returns the piece at the root of the shape.
  virtual const Piece& root_piece() const = 0;

  // LiteralSlice and Literal must access Pieces of other Literals.
  friend class MutableLiteralBase;
  friend class LiteralSlice;
  friend class BorrowingLiteral;

 private:
  // Like IsAllFloat, but if round_value is false and the value is not
  // representable with the literal's type (e.g., due to rounding error or
  // overflow/underflow when casting the value to the literal's type), returns
  // false.
  bool IsAllFloatImpl(float value, bool round_value) const;
};

// Abstract base class representing a mutable literal in XLA.
class MutableLiteralBase : public LiteralBase {
 public:
  ~MutableLiteralBase() override = 0;

  // Returns a Span view of the array for this literal for the
  // given NativeT (e.g., float). CHECKs if the subshape of the literal at the
  // given ShapeIndex is not array. See primitive_util.h for the mapping from
  // XLA type to native type.
  template <typename NativeT>
  absl::Span<NativeT> data(const ShapeIndex& shape_index = {});
  // Unhide const method from parent class.
  using LiteralBase::data;

  // TODO(b/67651157): Remove this accessor. Literal users should not be able to
  // mutate the shape as this can produce malformed Literals.
  Shape* mutable_shape_do_not_use();

  // Set the dynamic size on dim_index in the literal at the given shape_index.
  void SetDynamicSize(int64_t dim_index, const ShapeIndex& shape_index,
                      DynamicSizeType size);
  void SetDynamicSize(int64_t dim_index, DynamicSizeType size);

  // Returns a pointer to the underlying buffer holding the array at the given
  // shape index. CHECKs if the subshape of the literal at the given ShapeIndex
  // is not array.
  void* untyped_data(const ShapeIndex& shape_index = {});
  // Unhide const method from parent class.
  using LiteralBase::untyped_data;

  template <typename NativeT>
  void MutableEachCell(absl::FunctionRef<NativeT(
                           absl::Span<const int64_t> indices, NativeT value)>
                           per_cell);

  // Copy values from 'src_literal' rooted at 'src_shape_index' into this
  // literal rooted at 'dest_shape_index'. The subshape of this literal rooted
  // at 'dest_shape_index' must be compatible with the subshape of 'src_literal'
  // rooted at 'src_shape_index', but need not be arrays. If only_dynamic_bound
  // is true, only elements within dynamic bounds will be copied.
  absl::Status CopyFrom(const LiteralSlice& src_literal,
                        const ShapeIndex& dest_shape_index = {},
                        const ShapeIndex& src_shape_index = {},
                        bool only_dynamic_bound = false);

  // Copies the values from src_literal, starting at src_base shape indexes,
  // to this literal, starting at dest_base, where the copy size in each
  // dimension is specified by copy_size.
  // The src_literal and this literal must have the same primitive type,
  // src_base+copy_size must fit the source literal dimensions, as well as
  // dest_base+copy_size must fit the destination literal dimensions.
  // Note: if either src_literal or this literal contains dimensions with zero
  // element, then copy_size must be 0 in these dimensions while the
  // corresponding base indices being 0.
  // This literal and 'src_literal' must be arrays.
  absl::Status CopySliceFrom(const LiteralSlice& src_literal,
                             absl::Span<const int64_t> src_base,
                             absl::Span<const int64_t> dest_base,
                             absl::Span<const int64_t> copy_size);

  // Copies one element from src_literal[src_index] to (*this)[dest_index].
  void CopyElementFrom(const LiteralSlice& src_literal,
                       absl::Span<const int64_t> src_index,
                       absl::Span<const int64_t> dest_index);

  // Sets an element in the literal at the given index. The multi_index is
  // CHECKed against the dimension sizes.
  template <typename NativeT>
  void Set(absl::Span<const int64_t> multi_index, const ShapeIndex& shape_index,
           NativeT value);
  // Overloads of Set for array literals. CHECKs if the literal is not
  // array-shaped and dense.
  template <typename NativeT>
  void Set(absl::Span<const int64_t> multi_index, NativeT value);

  // As Set(), but truncates `value` to the literal element type before storing.
  // This literal must be an array.
  absl::Status SetIntegralAsS64(absl::Span<const int64_t> multi_index,
                                int64_t value);

  // As Set(), but truncates `value` to the literal element type before storing.
  // This literal must be an array.
  absl::Status SetFromDouble(absl::Span<const int64_t> multi_index,
                             double value);

  // Populate this literal with the given values. Examples:
  //
  //   // Populate with floats.
  //   Array2D<float> float_values = ...
  //   literal.PopulateR2FromArray2D(values);
  //
  //   // Populate with int32s.
  //   literal.PopulateR2<int32_t>({{1, 2}, {3, 4}});
  //
  // The shape and element type of this literal must match given values. For
  // example, in the call above to literal.PopulateR2(), 'literal' must be a 2x2
  // array of S32.
  template <typename NativeT>
  void PopulateR1(absl::Span<const NativeT> values);
  void PopulateR1(const tsl::core::Bitmap& values);
  template <typename NativeT>
  void PopulateR2(std::initializer_list<std::initializer_list<NativeT>> values);
  template <typename NativeT>
  void PopulateFromArray(const Array<NativeT>& values);
  template <typename NativeT>
  void PopulateR2FromArray2D(const Array2D<NativeT>& values);
  template <typename NativeT>
  void PopulateR3FromArray3D(const Array3D<NativeT>& values);
  template <typename NativeT>
  void PopulateR4FromArray4D(const Array4D<NativeT>& values);

  // Collection of type aliases for static type checking Literal::Populate(.*)
  // functions defined below. We rely on templates to be able to inline
  // generator and populator functions into the call sites.

  template <typename NativeT, typename Generator>
  using IsGenerator = std::enable_if_t<std::is_convertible_v<
      NativeT, std::invoke_result_t<Generator, absl::Span<const int64_t>>>>;

  template <typename NativeT, typename Generator>
  using IsParallelGenerator = std::enable_if_t<std::is_convertible_v<
      NativeT,
      std::invoke_result_t<Generator, absl::Span<const int64_t>, int>>>;

  template <typename Populator>
  using IsPopulator = std::enable_if_t<
      std::is_invocable_v<Populator, void*, absl::Span<const int64_t>>>;

  template <typename Populator>
  using IsParallelPopulator = std::enable_if_t<
      std::is_invocable_v<Populator, void*, absl::Span<const int64_t>, int>>;

  template <typename NativeT, typename Generator>
  using IsLinearGenerator = std::enable_if_t<
      std::is_convertible_v<NativeT, std::invoke_result_t<Generator, int64_t>>>;

  template <typename NativeT, typename Generator>
  using IsLinearParallelGenerator = std::enable_if_t<std::is_convertible_v<
      NativeT, std::invoke_result_t<Generator, int64_t, int>>>;

  template <typename Populator>
  using IsLinearPopulator =
      std::enable_if_t<std::is_invocable_v<Populator, void*, int64_t>>;

  template <typename Populator>
  using IsLinearParallelPopulator =
      std::enable_if_t<std::is_invocable_v<Populator, void*, int64_t, int>>;

  // Populates literal values by calling the generator function for every cell
  // in this literal object.
  //
  // This literal must have a dense layout.
  template <typename NativeT, typename Generator,
            IsGenerator<NativeT, Generator>* = nullptr>
  absl::Status Populate(Generator&& generator);

  // A parallel version of Populate(). This can be used if the generator is
  // thread-safe and the values for the shape's different elements are
  // independent.
  template <typename NativeT, typename Generator,
            IsParallelGenerator<NativeT, Generator>* = nullptr>
  absl::Status PopulateParallel(Generator&& generator);

  // Similar to Populate() but takes a populator function that allows caller
  // specify how to write to the destination buffer rather than a generator that
  // returns the values. This is useful when the value population simply does
  // memcpy without compute therefore can be written in a type agnostic way, so
  // that we can avoid templatizing the method for better code size.
  //
  // This literal must have a dense layout.
  template <typename Populator, IsPopulator<Populator>* = nullptr>
  absl::Status PopulateInplace(Populator&& populator);

  // A parallel version of PopulateInplace(). This can be used if the generator
  // is thread-safe and the values for the shape's different elements are
  // independent.
  template <typename Populator, IsParallelPopulator<Populator>* = nullptr>
  absl::Status PopulateInplaceParallel(Populator&& populator);

  // Overload of Populate() that takes a linear index generator.
  template <typename NativeT, typename Generator,
            IsLinearGenerator<NativeT, Generator>* = nullptr>
  absl::Status PopulateLinear(Generator&& generator);

  // Overload of PopulateParallel() that takes a linear index generator.
  template <typename NativeT, typename Generator,
            IsLinearParallelGenerator<NativeT, Generator>* = nullptr>
  absl::Status PopulateLinearParallel(Generator&& generator);

  // Overload of PopulateInplace() that takes a linear index generator.
  template <typename Populator, IsLinearPopulator<Populator>* = nullptr>
  absl::Status PopulateLinearInplace(Populator&& populator);

  // Overload of PopulateInplaceParallel() that takes a linear index generator.
  template <typename Populator, IsLinearParallelPopulator<Populator>* = nullptr>
  absl::Status PopulateLinearInplaceParallel(Populator&& populator);

  // Fills this literal with the given value.
  template <typename NativeT>
  void PopulateWithValue(NativeT value);

  // This operation is the inverse of DecomposeTuple. The given elements are
  // moved into the tuple elements of a new tuple-shaped Literal which is
  // returned. Upon return, each of the Literals in 'elements' is set to a nil
  // shape (empty tuple).
  static Literal MoveIntoTuple(absl::Span<Literal> elements);

  // Serialize from a proto.
  static absl::StatusOr<Literal> CreateFromProto(
      const LiteralProto& proto, bool prohibit_empty_literal = true);

 protected:
  // Returns the piece at the given ShapeIndex.
  Piece& piece(const ShapeIndex& shape_index) {
    return const_cast<Piece&>(LiteralBase::piece(shape_index));
  }

  Piece& mutable_root_piece() { return const_cast<Piece&>(root_piece()); }

  // Internal template helper for the Literal::CopySliceFrom(), matching its
  // arguments one by one.
  template <typename NativeT>
  absl::Status CopySliceFromInternal(const LiteralBase& src_literal,
                                     absl::Span<const int64_t> src_base,
                                     absl::Span<const int64_t> dest_base,
                                     absl::Span<const int64_t> copy_size);

  // The literal may or may not own the storage of the shape. Creating/copying a
  // shape can incur significant overhead which in many case we'd like to avoid,
  // esp. for small literals.
  using MaybeOwningShapePtr = MaybeOwning<Shape>;

  // The parent class borrows this shape.
  MaybeOwningShapePtr shape_;

  // We do not add static type checking for internal generators as these
  // functions are not part of the public API and we construct generators from
  // already type checked generators passed by the user.

  // Implementation details shared between Populate() and PopulateParallel().
  template <typename NativeT, typename Generator>
  absl::Status PopulateInternal(Generator&& generator, bool parallel);
  void PopulateInplaceInternal(
      absl::FunctionRef<void(void*, absl::Span<const int64_t>, int)> populator,
      bool parallel);

  // Implementation details shared between PopulateLinear() and
  // PopulateLinearParallel().
  template <typename NativeT, typename Generator>
  absl::Status PopulateLinearInternal(Generator&& generator, bool parallel);
  void PopulateLinearInplaceInternal(
      absl::FunctionRef<void(void*, int64_t, int)> populator, bool parallel);

  friend class LiteralBase;
  friend class MutableBorrowingLiteral;
};
std::ostream& operator<<(std::ostream& out, const Literal& literal);

// The underlying buffer and shape is always owned by this class.
class Literal : public MutableLiteralBase {
 public:
  Literal();

  // Create a literal of the given shape. The literal is allocated sufficient
  // memory to hold the shape. Memory is uninitialized.
  explicit Literal(const Shape& shape);
  ~Literal() override;

  // Literals are moveable, but not copyable. To copy a literal use
  // Literal::Clone or Literal::CloneToUnique. This prevents inadvertent copies
  // of literals which can be expensive.
  Literal(const Literal& other) = delete;
  Literal& operator=(const Literal& other) = delete;
  Literal(Literal&& other);
  // 'allocate_arrays' indicates whether to allocate memory for the arrays in
  // the shape. If false, buffer pointers inside of the Literal::Pieces are set
  // to nullptr.
  Literal(const Shape& shape, bool allocate_arrays,
          ArrayValueState leaf_array_value_state = ArrayValueState::kKnown);
  Literal& operator=(Literal&& other);

  // Similar to CopyFrom, but with move semantics. The subshape of this literal
  // rooted at 'dest_shape_index' must be *equal* to the shape 'src_literal'
  // (layouts and shapes must match), but need not be arrays. The memory
  // allocated in this literal for the subshape at dest_shape_index is
  // deallocated, and the respective buffers are replaced with those in
  // src_literal. Upon return, src_literal is set to a nil shape (empty tuple).
  virtual absl::Status MoveFrom(Literal&& src_literal,
                                const ShapeIndex& dest_shape_index);
  absl::Status MoveFrom(Literal&& src_literal) {
    return MoveFrom(std::move(src_literal), /*dest_shape_index=*/{});
  }

  // Returns a vector containing the tuple elements of this Literal as separate
  // Literals. This Literal must be tuple-shaped and can be a nested tuple. The
  // elements are moved into the new Literals; no data is copied. Upon return
  // this Literal is set to a nil shape (empty tuple)
  //
  // TODO(jlebar): Because this function invalidates `this`, it should be
  // ref-qualified with &&.
  std::vector<Literal> DecomposeTuple();

  // Returns a subliteral specified by given shape_index. No data is copied, the
  // current literal becomes invalid after this function call.
  //
  // TODO(jlebar): Because this function invalidates `this`, it should be
  // ref-qualified with &&.
  Literal SubLiteral(ShapeIndexView shape_index);

  // Deserialize a Literal from the given iterator range, whose value type must
  // be char.  See the comments on the Serialize() method for caveats.
  template <typename InputIterator>
  static absl::StatusOr<Literal> Deserialize(InputIterator begin,
                                             InputIterator end);

  static absl::StatusOr<Literal> DeserializeFromString(absl::string_view data) {
    return Deserialize(data.data(), data.data() + data.size());
  }

 private:
  friend class LiteralBase;
  friend class MutableLiteralBase;
  const Piece& root_piece() const final { return root_piece_; };
  // Deallocate the buffers held by this literal.
  void DeallocateBuffers();

  // Sets the shape_ field from a Shape. shape_'s element_size_in_bits field
  // on the layout is always set to 0 since Literals do not support packed
  // subbyte elements.
  void SetShape(const Shape& shape);

  // Recursively sets the subshapes and buffers of all subpieces rooted at
  // 'piece'. If 'allocate_array' is true, memory is allocated for the arrays in
  // the shape.
  void SetPiece(
      const Shape& shape, Piece* piece, bool allocate_arrays,
      ArrayValueState leaf_array_value_state = ArrayValueState::kKnown);
  Piece root_piece_;
};

// The underlying buffer is not owned by this class and is always owned by
// others. The shape is not owned by this class and not mutable.
class MutableBorrowingLiteral : public MutableLiteralBase {
 public:
  ~MutableBorrowingLiteral() override;

  MutableBorrowingLiteral() : MutableLiteralBase() {}

  MutableBorrowingLiteral(const MutableBorrowingLiteral& literal);
  MutableBorrowingLiteral& operator=(const MutableBorrowingLiteral& literal);

  // Implicit conversion constructors.
  // NOLINTNEXTLINE(google-explicit-constructor)
  MutableBorrowingLiteral(MutableLiteralBase* literal);
  MutableBorrowingLiteral(MutableBorrowingLiteral literal,
                          const ShapeIndex& view_root);

  // 'src_buf_ptr' is not owned by this class and must outlive the
  // lifetime of this class. It points to an appropriately sized buffer with
  // data interpreted as indicated by 'shape'.
  // This constructor is only used for array shapes.
  MutableBorrowingLiteral(const char* src_buf_ptr, const Shape& shape);

  // Similar as above, except to be used for constructing non-nested tuples.
  MutableBorrowingLiteral(absl::Span<char*> src_buf_ptrs, const Shape& shape);

  // Similar as above, except to be used for constructing literals with
  // potentially nested tuples (same shape as `src_buf_ptrs`) with borrowed
  // buffers for each shape index.
  explicit MutableBorrowingLiteral(ShapeTree<char*> src_buf_ptrs);

 private:
  const Piece& root_piece() const final { return *root_piece_; };
  // Recursively copies the subtree from the `src_piece` at the given child
  // index to the `dest_piece`. For buffers only the pointers are copied, but
  // not the content.
  void CopyPieceSubtree(const Shape& shape, const Piece* src_piece,
                        Piece* dest_piece);
  Piece* root_piece_ = nullptr;
};

// A read-only view of a Literal. A LiteralSlice contains pointers to shape and
// literal buffers always owned by others.
class LiteralSlice : public LiteralBase {
 public:
  LiteralSlice() : LiteralBase() {}

  // Implicit conversion constructors.
  // NOLINTNEXTLINE(google-explicit-constructor)
  LiteralSlice(const LiteralBase& literal);
  LiteralSlice(const LiteralBase& literal, const ShapeIndex& view_root);

 private:
  const Piece& root_piece() const final { return *root_piece_; };

  const Piece* root_piece_;  // Not owned.
};

// A read-only Literal where the underlying buffers are never owned by this
// class.
class BorrowingLiteral : public LiteralBase {
 public:
  BorrowingLiteral() : LiteralBase() {}

  // 'src_buf_ptr' is not owned by this class and must outlive the
  // lifetime of this class. It points to an appropriately sized buffer with
  // data interpreted as indicated by 'shape'.
  // This constructor is only used for array shapes.
  BorrowingLiteral(const char* src_buf_ptr, const Shape& shape);

  // Similar as above, except to be used for constructing non-nested tuples.
  BorrowingLiteral(absl::Span<const char* const> src_buf_ptrs,
                   const Shape& shape);

  // Similar as above, except to be used for constructing literals with
  // potentially nested tuples (same shape as `src_buf_ptrs`) with borrowed
  // buffers for each shape index.
  explicit BorrowingLiteral(ShapeTree<const char*> src_buf_ptrs);

 private:
  // Accessor for the root piece of this literal.
  const Piece& root_piece() const final { return root_piece_; };
  Piece root_piece_;

  // Shape of this literal. Stored as unique_ptr such that the (default) move
  // construction of this class would be trivially correct: the pointer to Shape
  // root_piece_ stores will still point to the correct address.
  std::unique_ptr<Shape> shape_;
};

template <typename NativeT, typename OutputIterator>
void LiteralBase::Piece::SerializeData(
    SerializeState<OutputIterator>& state) const {
  CHECK_EQ(subshape().element_type(),
           primitive_util::NativeToPrimitiveType<NativeT>());
  if (subshape().is_dynamic()) {
    absl::Span<const DynamicSizeType> sizes(dynamic_size_buffer(),
                                            subshape().rank());
    state.WriteDynamicSizes(sizes);
  }
  state.WriteElements(data<NativeT>());
}

template <typename NativeT, typename InputIterator>
bool LiteralBase::Piece::DeserializeData(
    DeserializeState<InputIterator>& state) {
  CHECK_EQ(subshape().element_type(),
           primitive_util::NativeToPrimitiveType<NativeT>());
  if (subshape().is_dynamic()) {
    absl::Span<DynamicSizeType> sizes(dynamic_size_buffer(), subshape().rank());
    if (!state.ReadDynamicSizes(sizes)) {
      return false;
    }
  }
  return state.ReadElements(data<NativeT>());
}

// Description of the native serialization format:
//
// - All data are stored in little-endian order.
//
// - The serialized format begins with a header.
//
// - The first 8 bytes (int64_t) of the header are the size of the serialized
//   ShapeProto that provides the shape of the literal.
//
// - The remaining bytes of the header provide the serialized ShapeProto itself.
//
// - After the header, each piece of the literal is serialized, as produced
//   through a depth-first traversal of the tuple tree.
//
// - If a piece is dynamic, we first write the sizes of the dynamic dimensions.
//
// - The elements of the piece are then written.  Elements smaller than a single
//   byte (PRED, S4, U4) are packed into bytes.  Otherwise, they are written in
//   little-endian byte order.
template <typename OutputIterator>
absl::Status LiteralBase::SerializeWithShapeProto(const ShapeProto& shape_proto,
                                                  OutputIterator output) const {
  SerializeState<OutputIterator> state(shape_proto, output);
  TF_RETURN_IF_ERROR(root_piece().ForEachSubpieceWithStatus(
      [&](const ShapeIndex& shape_index, const Piece& piece) -> absl::Status {
        const Shape& subshape = piece.subshape();
        if (subshape.IsTuple()) {
          return absl::OkStatus();
        }
        if (!subshape.IsArray()) {
          return InvalidArgument("Shape cannot be serialized: %s",
                                 shape().ToString());
        }
        primitive_util::ArrayTypeSwitch<void>(
            [&](auto primitive_type) {
              using NativeT = primitive_util::NativeTypeOf<primitive_type>;
              piece.SerializeData<NativeT>(state);
            },
            subshape.element_type());
        return absl::OkStatus();
      }));
  DCHECK_EQ(state.num_written(), SerializedSize().value())
      << shape().ToString();
  return absl::OkStatus();
}

template <typename InputIterator>
absl::StatusOr<Literal> Literal::Deserialize(InputIterator begin,
                                             InputIterator end) {
  DeserializeState<InputIterator> state(begin, end);
  uint64_t shape_size;
  if (!state.ReadElement(shape_size)) {
    return InvalidArgument("Failed to read shape size");
  }
  TF_ASSIGN_OR_RETURN(Shape shape, state.ReadShape(shape_size));
  Literal literal(shape);
  TF_RETURN_IF_ERROR(
      literal.mutable_root_piece().ForEachMutableSubpieceWithStatus(
          [&](const ShapeIndex& shape_index, Piece* piece) -> absl::Status {
            const Shape& subshape = piece->subshape();
            if (subshape.IsTuple()) {
              return absl::OkStatus();
            }
            if (!subshape.IsArray()) {
              return InvalidArgument("Shape cannot be deserialized: %s",
                                     shape.ToString());
            }
            bool ok = primitive_util::ArrayTypeSwitch<bool>(
                [&](auto primitive_type) {
                  using NativeT = primitive_util::NativeTypeOf<primitive_type>;
                  return piece->DeserializeData<NativeT>(state);
                },
                subshape.element_type());
            if (!ok) {
              return InvalidArgument(
                  "Failed to deserialize all data for shape: %s",
                  shape.ToString());
            }
            return absl::OkStatus();
          }));
  DCHECK_EQ(state.num_read(), ShapeUtil::SerializedSize(shape).value())
      << shape.ToString();
  if (!state.at_end()) {
    return InvalidArgument("Did not consume all input data");
  }
  return std::move(literal);
}

template <typename NativeT>
absl::Span<const NativeT> LiteralBase::Piece::data() const {
  DCHECK(LayoutUtil::IsDenseArray(subshape()))
      << __func__ << " is only supported for dense arrays: " << subshape();
  DCHECK(!subshape().has_layout() ||
         subshape().layout().element_size_in_bits() == 0)
      << __func__
      << " is not supported for layouts with custom bit size: " << subshape();
  DCHECK_EQ(subshape().element_type(),
            primitive_util::NativeToPrimitiveType<NativeT>())
      << "Attempting to access "
      << PrimitiveType_Name(primitive_util::NativeToPrimitiveType<NativeT>())
      << " type, but literal element type is "
      << PrimitiveType_Name(subshape().element_type());
  return absl::Span<const NativeT>(reinterpret_cast<const NativeT*>(buffer()),
                                   element_count());
}

template <typename NativeT>
absl::Span<NativeT> LiteralBase::Piece::data() {
  DCHECK(LayoutUtil::IsDenseArray(subshape()))
      << __func__ << " is only supported for dense arrays: " << subshape();
  DCHECK(!subshape().has_layout() ||
         subshape().layout().element_size_in_bits() == 0)
      << __func__
      << " is not supported for layouts with custom bit size: " << subshape();
  DCHECK_EQ(subshape().element_type(),
            primitive_util::NativeToPrimitiveType<NativeT>())
      << "Attempting to access "
      << PrimitiveType_Name(primitive_util::NativeToPrimitiveType<NativeT>())
      << " type, but literal element type is "
      << PrimitiveType_Name(subshape().element_type());
  return absl::Span<NativeT>(reinterpret_cast<NativeT*>(buffer()),
                             element_count());
}

template <typename NativeT>
NativeT LiteralBase::Piece::Get(absl::Span<const int64_t> multi_index) const {
  DCHECK(LayoutUtil::IsDenseArray(subshape()))
      << __func__ << " is only supported for dense arrays: " << subshape();
  return GetLinear<NativeT>(
      IndexUtil::MultidimensionalIndexToLinearIndex(subshape(), multi_index));
}

template <typename NativeT>
void LiteralBase::Piece::Set(absl::Span<const int64_t> multi_index,
                             NativeT value) {
  DCHECK(LayoutUtil::IsDenseArray(subshape()))
      << __func__ << " is only supported for dense arrays: " << subshape();
  return SetLinear<NativeT>(
      IndexUtil::MultidimensionalIndexToLinearIndex(subshape(), multi_index),
      value);
}

template <typename NativeT>
NativeT LiteralBase::Piece::GetLinear(int64_t linear_index) const {
  DCHECK(LayoutUtil::IsDenseArray(subshape()))
      << __func__ << " is only supported for dense arrays: " << subshape();
  DCHECK_LT(linear_index, element_count()) << "linear_index out of bounds";
  return data<NativeT>().data()[linear_index];
}

template <typename NativeT>
void LiteralBase::Piece::SetLinear(int64_t linear_index, NativeT value) {
  DCHECK(LayoutUtil::IsDenseArray(subshape()))
      << __func__ << " is only supported for dense arrays: " << subshape();
  DCHECK_LT(linear_index, element_count()) << "linear_index out of bounds";
  data<NativeT>().data()[linear_index] = value;
}

template <typename NativeT>
absl::Span<const NativeT> LiteralBase::data(
    const ShapeIndex& shape_index) const {
  return piece(shape_index).data<NativeT>();
}

template <typename NativeT>
absl::Span<NativeT> MutableLiteralBase::data(const ShapeIndex& shape_index) {
  return piece(shape_index).data<NativeT>();
}

template <typename NativeT>
inline NativeT LiteralBase::Get(absl::Span<const int64_t> multi_index,
                                const ShapeIndex& shape_index) const {
  return piece(shape_index).Get<NativeT>(multi_index);
}

template <typename NativeT>
inline NativeT LiteralBase::Get(absl::Span<const int64_t> multi_index) const {
  return root_piece().Get<NativeT>(multi_index);
}

template <typename NativeT>
inline NativeT LiteralBase::GetLinear(int64_t linear_index,
                                      const ShapeIndex& shape_index) const {
  return piece(shape_index).GetLinear<NativeT>(linear_index);
}

template <typename NativeT>
inline NativeT LiteralBase::GetLinear(int64_t linear_index) const {
  return root_piece().GetLinear<NativeT>(linear_index);
}

template <typename NativeT>
inline void MutableLiteralBase::Set(absl::Span<const int64_t> multi_index,
                                    const ShapeIndex& shape_index,
                                    NativeT value) {
  return piece(shape_index).Set<NativeT>(multi_index, value);
}

template <typename NativeT>
inline void MutableLiteralBase::Set(absl::Span<const int64_t> multi_index,
                                    NativeT value) {
  return mutable_root_piece().Set<NativeT>(multi_index, value);
}

template <typename NativeT>
NativeT LiteralBase::GetFirstElement() const {
  CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  return data<NativeT>().at(0);
}

template <typename T>
int64_t LiteralBase::CountEqual(T value) const {
  PrimitiveType ty = shape().element_type();
  if (!primitive_util::IsArrayType(ty)) {
    return 0;
  }
  Literal scalar(ShapeUtil::MakeScalarShape(ty));
  return primitive_util::ArrayTypeSwitch<int64_t>(
      [&](auto primitive_type_constant) -> int64_t {
        using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
        scalar.Set<NativeT>({}, static_cast<NativeT>(value));
        return root_piece().CountAll(scalar);
      },
      ty);
}

template <typename T>
int64_t LiteralBase::CountEqual(std::complex<T> value) const {
  PrimitiveType ty = shape().element_type();
  if (!primitive_util::IsComplexType(ty)) {
    return 0;
  }
  Literal scalar(ShapeUtil::MakeScalarShape(ty));
  return primitive_util::ComplexTypeSwitch<int64_t>(
      [&](auto primitive_type_constant) -> int64_t {
        using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
        scalar.Set<NativeT>({}, static_cast<NativeT>(value));
        return root_piece().CountAll(scalar);
      },
      ty);
}

template <typename NativeT>
TF_ATTRIBUTE_NOINLINE void LiteralBase::EachCell(
    absl::FunctionRef<void(absl::Span<const int64_t> indices, NativeT value)>
        per_cell) const {
  CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  if (ShapeUtil::IsZeroElementArray(shape())) {
    return;
  }
  std::vector<int64_t> indices(shape().rank(), 0);

  Shape shape_dynamic = shape();
  for (int64_t i = 0; i < shape_dynamic.rank(); ++i) {
    shape_dynamic.set_dimensions(i, GetDynamicSize(i));
  }
  do {
    per_cell(indices, Get<NativeT>(indices));
  } while (IndexUtil::BumpIndices(shape_dynamic, absl::MakeSpan(indices)));
}

template <typename NativeT>
TF_ATTRIBUTE_NOINLINE void MutableLiteralBase::MutableEachCell(
    absl::FunctionRef<NativeT(absl::Span<const int64_t> indices, NativeT value)>
        per_cell) {
  CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  if (ShapeUtil::IsZeroElementArray(shape())) {
    return;
  }
  std::vector<int64_t> indices(shape().rank(), 0);
  Shape shape_dynamic = shape();
  for (int64_t i = 0; i < shape_dynamic.rank(); ++i) {
    shape_dynamic.set_dimensions(i, GetDynamicSize(i));
  }
  do {
    Set<NativeT>(indices, per_cell(indices, Get<NativeT>(indices)));
  } while (IndexUtil::BumpIndices(shape_dynamic, absl::MakeSpan(indices)));
}

template <typename NativeT>
TF_ATTRIBUTE_NOINLINE void MutableLiteralBase::PopulateR1(
    absl::Span<const NativeT> values) {
  CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  CHECK_EQ(shape().rank(), 1);
  if (shape().is_static()) {
    CHECK_EQ(ShapeUtil::ElementsIn(shape()), values.size());
  } else {
    CHECK_EQ(GetDynamicSize(0), values.size());
  }
  CHECK_EQ(shape().element_type(),
           primitive_util::NativeToPrimitiveType<NativeT>());
  auto data_span = data<NativeT>();
  std::copy(values.begin(), values.end(), data_span.begin());
}

template <typename NativeT>
TF_ATTRIBUTE_NOINLINE void MutableLiteralBase::PopulateR2(
    std::initializer_list<std::initializer_list<NativeT>> values) {
  CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  CHECK_EQ(shape().rank(), 2);
  CHECK_EQ(shape().element_type(),
           primitive_util::NativeToPrimitiveType<NativeT>());

  const int64_t values_dim0_size = values.size();
  const int64_t values_dim1_size = values.begin()->size();
  const int64_t literal_dim0_size = shape().is_dynamic_dimension(0)
                                        ? GetDynamicSize(0)
                                        : shape().dimensions(0);
  const int64_t literal_dim1_size = shape().is_dynamic_dimension(1)
                                        ? GetDynamicSize(1)
                                        : shape().dimensions(1);

  CHECK_EQ(values_dim0_size, literal_dim0_size);
  CHECK_EQ(values_dim1_size, literal_dim1_size);

  int64_t dim0 = 0;
  for (auto inner_list : values) {
    int64_t dim1 = 0;
    for (auto value : inner_list) {
      Set({dim0, dim1}, value);
      ++dim1;
    }
    CHECK_EQ(values_dim1_size, dim1);
    ++dim0;
  }
}

template <typename NativeT>
TF_ATTRIBUTE_NOINLINE void MutableLiteralBase::PopulateFromArray(
    const Array<NativeT>& values) {
  CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  CHECK(shape().IsArray());
  CHECK_EQ(shape().element_type(),
           primitive_util::NativeToPrimitiveType<NativeT>());
  CHECK_EQ(shape().rank(), values.num_dimensions());
  for (int dim = 0; dim < values.num_dimensions(); ++dim) {
    int64_t shape_size = shape().is_dynamic_dimension(dim)
                             ? GetDynamicSize(dim)
                             : shape().dimensions(dim);
    CHECK_EQ(values.dim(dim), shape_size);
  }
  values.Each([this](absl::Span<const int64_t> indices, NativeT value) {
    this->Set(indices, value);
  });
}

template <typename NativeT>
void MutableLiteralBase::PopulateR2FromArray2D(const Array2D<NativeT>& values) {
  PopulateFromArray(values);
}

template <typename NativeT>
void MutableLiteralBase::PopulateR3FromArray3D(const Array3D<NativeT>& values) {
  PopulateFromArray(values);
}

template <typename NativeT>
void MutableLiteralBase::PopulateR4FromArray4D(const Array4D<NativeT>& values) {
  PopulateFromArray(values);
}

template <typename NativeT, typename Generator>
absl::Status MutableLiteralBase::PopulateInternal(Generator&& generator,
                                                  bool parallel) {
  const Shape& this_shape = shape();
  DCHECK(LayoutUtil::IsDenseArray(this_shape));
  TF_RET_CHECK(this_shape.element_type() ==
               primitive_util::NativeToPrimitiveType<NativeT>())
      << "Failing to populate literal with element type "
      << primitive_util::LowercasePrimitiveTypeName(this_shape.element_type())
      << " using data of type "
      << primitive_util::LowercasePrimitiveTypeName(
             primitive_util::NativeToPrimitiveType<NativeT>());
  PopulateInplaceInternal(
      [&](void* dest, absl::Span<const int64_t> indices, int thread_id) {
        *static_cast<NativeT*>(dest) = generator(indices, thread_id);
      },
      parallel);
  return absl::OkStatus();
}

template <typename NativeT, typename Generator,
          MutableLiteralBase::IsGenerator<NativeT, Generator>*>
absl::Status MutableLiteralBase::Populate(Generator&& generator) {
  TF_RET_CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  return PopulateInternal<NativeT>(
      [&](absl::Span<const int64_t> indexes, int /*thread_id*/) {
        return generator(indexes);
      },
      /*parallel=*/false);
}
template <typename NativeT, typename Generator,
          MutableLiteralBase::IsParallelGenerator<NativeT, Generator>*>
absl::Status MutableLiteralBase::PopulateParallel(Generator&& generator) {
  TF_RET_CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  return PopulateInternal<NativeT>(generator,
                                   /*parallel=*/data<NativeT>().size() > 32);
}

template <typename NativeT, typename Generator>
absl::Status MutableLiteralBase::PopulateLinearInternal(Generator&& generator,
                                                        bool parallel) {
  const Shape& this_shape = shape();
  DCHECK(LayoutUtil::IsDenseArray(this_shape));
  TF_RET_CHECK(this_shape.element_type() ==
               primitive_util::NativeToPrimitiveType<NativeT>())
      << "Failing to populate literal with element type "
      << primitive_util::LowercasePrimitiveTypeName(this_shape.element_type())
      << " using data of type "
      << primitive_util::LowercasePrimitiveTypeName(
             primitive_util::NativeToPrimitiveType<NativeT>());
  PopulateLinearInplaceInternal(
      [&](void* dest, int64_t linear_index, int thread_id) {
        *static_cast<NativeT*>(dest) = generator(linear_index, thread_id);
      },
      parallel);
  return absl::OkStatus();
}

template <typename NativeT, typename Generator,
          MutableLiteralBase::IsLinearGenerator<NativeT, Generator>*>
absl::Status MutableLiteralBase::PopulateLinear(Generator&& generator) {
  TF_RET_CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  return PopulateLinearInternal<NativeT>(
      [&](int64_t linear_index, int /*thread_id*/) {
        return generator(linear_index);
      },
      /*parallel=*/false);
}
template <typename NativeT, typename Generator,
          MutableLiteralBase::IsLinearParallelGenerator<NativeT, Generator>*>
absl::Status MutableLiteralBase::PopulateLinearParallel(Generator&& generator) {
  TF_RET_CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  return PopulateLinearInternal<NativeT>(
      std::forward<Generator>(generator),
      /*parallel=*/data<NativeT>().size() > 32);
}

template <typename Populator, MutableLiteralBase::IsPopulator<Populator>*>
absl::Status MutableLiteralBase::PopulateInplace(Populator&& populator) {
  TF_RET_CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  PopulateInplaceInternal(
      [&](void* dest, absl::Span<const int64_t> indexes, int /*thread_id*/) {
        return populator(dest, indexes);
      },
      /*parallel=*/false);
  return absl::OkStatus();
}

template <typename Populator,
          MutableLiteralBase::IsParallelPopulator<Populator>*>
absl::Status MutableLiteralBase::PopulateInplaceParallel(
    Populator&& populator) {
  TF_RET_CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  PopulateInplaceInternal(std::forward<Populator>(populator),
                          /*parallel=*/element_count() > 32);
  return absl::OkStatus();
}

template <typename Populator, MutableLiteralBase::IsLinearPopulator<Populator>*>
absl::Status MutableLiteralBase::PopulateLinearInplace(Populator&& populator) {
  TF_RET_CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  PopulateLinearInplaceInternal(
      [&](void* dest, int64_t linear_index, int /*thread_id*/) {
        return populator(dest, linear_index);
      },
      /*parallel=*/false);
  return absl::OkStatus();
}

template <typename Populator,
          MutableLiteralBase::IsLinearParallelPopulator<Populator>*>
absl::Status MutableLiteralBase::PopulateLinearInplaceParallel(
    Populator&& populator) {
  TF_RET_CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  PopulateLinearInplaceInternal(std::forward<Populator>(populator),
                                /*parallel=*/element_count() > 32);
  return absl::OkStatus();
}

template <typename NativeT>
void MutableLiteralBase::PopulateWithValue(NativeT value) {
  CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  CHECK_EQ(shape().element_type(),
           primitive_util::NativeToPrimitiveType<NativeT>());
  for (NativeT& element : data<NativeT>()) {
    element = value;
  }
}

template <typename NativeT>
Literal LiteralBase::Replicate(int64_t times) const {
  CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  DimensionVector bounds = {times};
  bounds.reserve(shape().rank() + 1);
  for (int64_t bound : shape().dimensions()) {
    bounds.push_back(bound);
  }
  Literal literal(ShapeUtil::MakeShape(shape().element_type(), bounds));
  int64_t elements = ShapeUtil::ElementsIn(literal.shape());
  if (elements == 0) {
    return literal;
  }

  DimensionVector output_indices(bounds.size(), 0);
  absl::Span<const int64_t> input_indices = output_indices;
  input_indices.remove_prefix(1);

  bool done = false;
  while (!done) {
    const auto element = Get<NativeT>(input_indices);
    literal.Set<NativeT>(output_indices, element);

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

#endif  // XLA_LITERAL_H_
