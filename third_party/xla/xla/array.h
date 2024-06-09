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

#ifndef XLA_ARRAY_H_
#define XLA_ARRAY_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>

#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/types.h"

namespace xla {

namespace array_impl {

template <typename T, typename T2>
using overload_for_float = std::enable_if_t<
    is_specialized_floating_point_v<T> && std::is_same<T2, float>::value, bool>;

// A type trait that is valid when all elements in a parameter pack are of
// integral type. Not using an alias template to work around MSVC 14.00 bug.
template <typename... Ts>
struct pack_is_integral : std::conjunction<std::is_integral<Ts>...> {};

// Compares three same-sized vectors elementwise. For each item in `values`,
// returns false if any of values[i] is outside the half-open range [starts[i],
// ends[i]).
template <typename C1, typename C2, typename C3>
bool all_inside_range(const C1& values, const C2& range_starts,
                      const C3& range_ends) {
  for (size_t i = 0, e = values.size(); i < e; ++i) {
    if (values[i] < range_starts[i] || values[i] >= range_ends[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace array_impl

// General N dimensional array class with arbitrary value type.
template <typename T>
class Array {
 public:
  // Type inference can have a hard time parsing very deep initializer list
  // nests, especially if one or more dimensions is one as the compiler just
  // sees a single-element integer initializer. These typedefs allow casting
  // explicitly with less typing.
  template <typename D>
  using InitializerList1D = std::initializer_list<D>;
  template <typename D>
  using InitializerList2D = std::initializer_list<InitializerList1D<D>>;
  template <typename D>
  using InitializerList3D = std::initializer_list<InitializerList2D<D>>;
  template <typename D>
  using InitializerList4D = std::initializer_list<InitializerList3D<D>>;

  using value_type = T;

  // Creates a new array with the specified dimensions and initialized elements.
  explicit Array(absl::Span<const int64_t> sizes)
      : sizes_(sizes.size()),
        values_(calculate_elements(sizes), default_init_t{}) {
    std::memcpy(sizes_.data.get(), sizes.data(),
                sizeof(int64_t) * sizes.size());
  }

  // Creates a new array with the specified dimensions and specified value for
  // every cell.
  Array(absl::Span<const int64_t> sizes, T value)
      : Array(sizes, no_default_init_t{}) {
    Fill(value);
  }

  // Creates a 2D array from the given nested initializer list. The outer
  // initializer list is the first dimension, the inner is the second dimension.
  // For example, {{1, 2, 3}, {4, 5, 6}} results in an array with n1=2 and n2=3.
  Array(InitializerList2D<T> values)
      : Array(ToInt64Array(values), no_default_init_t{}) {
    int64_t idx = 0;
    for (const auto& it1 : values) {
      for (const auto& it2 : it1) {
        values_[idx] = it2;
        ++idx;
      }
    }
    CHECK(idx == num_elements());
  }

  // Creates a 1D array of a floating-point type (half, bfloat16, float,
  // or double) from an initializer list of float values.
  template <typename T2, array_impl::overload_for_float<T, T2> = true>
  Array(std::initializer_list<T2> values)
      : Array(ToInt64Array(values), no_default_init_t{}) {
    int64_t idx = 0;
    for (const auto& it1 : values) {
      values_[idx] = static_cast<T>(it1);
      ++idx;
    }
    CHECK(idx == num_elements());
  }

  // Creates a 2D array of a floating-point type (float8, half, bfloat16, float,
  // or double) from an initializer list of float values.
  template <typename T2, array_impl::overload_for_float<T, T2> = true>
  Array(std::initializer_list<std::initializer_list<T2>> values)
      : Array(ToInt64Array(values), no_default_init_t{}) {
    int64_t idx = 0;
    for (const auto& it1 : values) {
      for (const auto& it2 : it1) {
        values_[idx] = static_cast<T>(it2);
        ++idx;
      }
    }
    CHECK(idx == num_elements());
  }

  // Creates a 3D array from the given nested initializer list. The outer
  // initializer list is the first dimension, and so on.
  Array(InitializerList3D<T> values)
      : Array(ToInt64Array(values), no_default_init_t{}) {
    int64_t idx = 0;
    for (const auto& it1 : values) {
      for (const auto& it2 : it1) {
        for (const auto& it3 : it2) {
          values_[idx] = it3;
          ++idx;
        }
      }
    }
    CHECK(idx == num_elements());
  }

  // Creates a 3D array of a floating-point type (half, bfloat16, float,
  // or double) from an initializer list of float values.
  template <typename T2, array_impl::overload_for_float<T, T2> = true>
  Array(std::initializer_list<std::initializer_list<std::initializer_list<T2>>>
            values)
      : Array(ToInt64Array(values), no_default_init_t{}) {
    int64_t idx = 0;
    for (const auto& it1 : values) {
      for (const auto& it2 : it1) {
        for (const auto& it3 : it2) {
          values_[idx] = static_cast<T>(it3);
          ++idx;
        }
      }
    }
    CHECK(idx == num_elements());
  }

  // Creates a 4D array from the given nested initializer list. The outer
  // initializer list is the first dimension, and so on.
  Array(InitializerList4D<T> values)
      : Array(ToInt64Array(values), no_default_init_t{}) {
    int64_t idx = 0;
    for (const auto& it1 : values) {
      for (const auto& it2 : it1) {
        for (const auto& it3 : it2) {
          for (const auto& it4 : it3) {
            values_[idx] = it4;
            ++idx;
          }
        }
      }
    }
    CHECK(idx == num_elements());
  }

  // Creates a 4D array of a floating-point type (half, bfloat16, float,
  // or double) from an initializer list of float values.
  template <typename T2, array_impl::overload_for_float<T, T2> = true>
  Array(std::initializer_list<
        std::initializer_list<std::initializer_list<std::initializer_list<T2>>>>
            values)
      : Array(ToInt64Array(values), no_default_init_t{}) {
    int64_t idx = 0;
    for (const auto& it1 : values) {
      for (const auto& it2 : it1) {
        for (const auto& it3 : it2) {
          for (const auto& it4 : it3) {
            values_[idx] = static_cast<T>(it4);
            ++idx;
          }
        }
      }
    }
    CHECK(idx == num_elements());
  }

  Array(const Array<T>& other)
      : sizes_(other.sizes_.Clone()), values_(other.values_.Clone()) {}

  Array(Array<T>&& other) = default;

  Array<T>& operator=(const Array<T>& other) {
    sizes_ = other.sizes_.Clone();
    values_ = other.values_.Clone();
    return *this;
  }

  Array<T>& operator=(Array<T>&& other) = default;

  // Fills the array with the specified value.
  void Fill(const T& value) { std::fill(begin(), end(), value); }

  // Fills the array with sequentially increasing values.
  void FillIota(const T& value) { std::iota(begin(), end(), value); }

  // Fills the array with a repeating sequence:
  //   [value, value + 1, ..., value + length - 1, value, ... ]
  void FillRepeatedIota(const T& value, int64_t length) {
    for (int64_t i = 0; i < num_elements(); i += length) {
      std::iota(begin() + i, begin() + std::min(i + length, num_elements()),
                value);
    }
  }

  // Fills the array with the sequence i*multiplier for i=0,1,...
  void FillWithMultiples(const T& multiplier) {
    for (int64_t i = 0; i < num_elements(); ++i) {
      values_[i] = static_cast<T>(i) * multiplier;
    }
  }

  // Fills the array with random normal variables with the specified mean.
  void FillRandom(const T& stddev, double mean = 0.0, int seed = 12345) {
    FillRandomDouble(static_cast<double>(stddev), mean, seed);
  }

  void FillRandomDouble(double stddev, double mean = 0.0, int seed = 12345) {
    std::mt19937 g(seed);
    std::normal_distribution<double> distribution(mean, stddev);
    for (int64_t i = 0; i < num_elements(); ++i) {
      if (std::is_same<T, bool>()) {
        values_[i] = static_cast<T>(distribution(g) > 0.0);
      } else {
        values_[i] = static_cast<T>(distribution(g));
      }
    }
  }

  // Fills the array with random uniform variables in the [min_value, max_value]
  // range. Defined for integral types.
  template <typename = typename std::enable_if<std::is_integral<T>::value>>
  void FillRandomUniform(const T& min_value, const T& max_value,
                         int seed = 12345) {
    std::mt19937 g(seed);
    std::uniform_int_distribution<T> distribution(min_value, max_value);
    for (int64_t i = 0; i < num_elements(); ++i) {
      values_[i] = static_cast<T>(distribution(g));
    }
  }

  // Fills the array with random uniform variables that's either True or False.
  // Defined for boolean type.
  void FillRandomBool(int seed = 12345) {
    std::mt19937 g(seed);
    std::uniform_int_distribution<int32_t> distribution(0, 1);
    for (int64_t i = 0; i < num_elements(); ++i) {
      values_[i] = static_cast<bool>(distribution(g));
    }
  }

  // Sets all the values in the array to values specified in the container.
  template <typename Container = std::initializer_list<T>>
  void SetValues(const Container& container) {
    CHECK_EQ(std::distance(std::begin(container), std::end(container)),
             num_elements());
    std::copy(std::begin(container), std::end(container), begin());
  }

  // Invokes a callback with the (indices, value_ptr) for each cell in the
  // array.
  void Each(absl::FunctionRef<void(absl::Span<const int64_t>, T*)> f) {
    OwnedBuffer<int64_t> index(sizes_.size, default_init_t{});
    for (int64_t i = 0; i < num_elements(); ++i, next_index(&index)) {
      f(index.span(), &values_[i]);
    }
  }

  // Invokes a callback with the (indices, value) for each cell in the array.
  void Each(absl::FunctionRef<void(absl::Span<const int64_t>, T)> f) const {
    OwnedBuffer<int64_t> index(sizes_.size, default_init_t{});
    for (int64_t i = 0; i < num_elements(); ++i, next_index(&index)) {
      f(index.span(), values_[i]);
    }
  }

  // Invokes a callback with the (indices, value_ptr) for each cell in the
  // array. If a callback returns a non-OK status, returns that else returns
  // absl::OkStatus().
  absl::Status EachStatus(
      absl::FunctionRef<absl::Status(absl::Span<const int64_t>, T*)> f) {
    OwnedBuffer<int64_t> index(sizes_.size, default_init_t{});
    for (int64_t i = 0; i < num_elements(); ++i, next_index(&index)) {
      absl::Status s = f(index.span(), &values_[i]);
      if (!s.ok()) {
        return s;
      }
    }
    return absl::OkStatus();
  }

  // Invokes a callback with the (indices, value) for each cell in the array.
  // If a callback returns a non-OK status, returns that else returns
  // absl::OkStatus().
  absl::Status EachStatus(
      absl::FunctionRef<absl::Status(absl::Span<const int64_t>, T)> f) const {
    OwnedBuffer<int64_t> index(sizes_.size, default_init_t{});
    for (int64_t i = 0; i < num_elements(); ++i, next_index(&index)) {
      absl::Status s = f(index.span(), values_[i]);
      if (!s.ok()) {
        return s;
      }
    }
    return absl::OkStatus();
  }

  // Returns the value at the cell specified by the indexes. The number of
  // arguments have to match with the number of dimensions for the array.
  //
  // The type trait is required to avoid this overload participating too
  // eagerly; a parameter pack can take zero or more elements, so we must
  // restrict this to only parameter packs that are all of integral type.
  template <typename... Dims>
  typename std::enable_if<array_impl::pack_is_integral<Dims...>::value,
                          const T&>::type
  operator()(Dims... dims) const {
    CHECK_EQ(sizeof...(dims), num_dimensions());
    // We are using a std::array to avoid having to allocate memory in this
    // function for performance reasons.
    std::array<int64_t, sizeof...(dims)> indexes{
        {static_cast<int64_t>(dims)...}};
    return values_[calculate_index(indexes)];
  }

  // Returns the value at the cell specified by the indexes. The number of
  // arguments have to match with the number of dimensions for the array.
  template <typename... Dims>
  typename std::enable_if<array_impl::pack_is_integral<Dims...>::value,
                          T&>::type
  operator()(Dims... dims) {
    return const_cast<T&>(const_cast<const Array*>(this)->operator()(
        std::forward<Dims>(dims)...));
  }

  // Returns the value at the cell specified by the indexes. The number of
  // arguments have to match with the number of dimensions for the array.
  const T& operator()(absl::Span<const int64_t> indexes) const {
    CHECK_EQ(indexes.size(), num_dimensions());
    return values_[calculate_index(indexes)];
  }

  // Returns the value at the cell specified by the indexes. The number of
  // arguments have to match with the number of dimensions for the array.
  T& operator()(absl::Span<const int64_t> indexes) {
    return const_cast<T&>(const_cast<const Array*>(this)->operator()(indexes));
  }

  // Low-level accessor for stuff like memcmp, handle with care. Returns pointer
  // to the underlying storage of the array (similarly to std::vector::data()).
  T* data() const {
    // TODO(tberghammer): Get rid of the const_cast. Currently it is needed
    // because the Eigen backend needs a non-const pointers even for reading
    // from the array.
    return const_cast<Array*>(this)->values_.data.get();
  }

  // Returns the size of the dimension at the given index.
  int64_t dim(int64_t n) const {
    DCHECK_LT(n, sizes_.size);
    return sizes_[n];
  }

  // Returns a vector containing the dimensions of the array.
  absl::Span<const int64_t> dimensions() const { return sizes_.span(); }

  int64_t num_dimensions() const { return sizes_.size; }

  // Returns the total number of elements in the array.
  int64_t num_elements() const { return values_.size; }

  const T* begin() const { return values_.data.get(); }
  T* begin() { return values_.data.get(); }
  const T* end() const { return values_.data.get() + num_elements(); }
  T* end() { return values_.data.get() + num_elements(); }

  bool operator==(const Array<T>& other) const {
    if (sizes_.size != other.sizes_.size) {
      return false;
    }
    for (int64_t i = 0, end = sizes_.size; i < end; ++i) {
      if (sizes_[i] != other.sizes_[i]) {
        return false;
      }
    }
    for (int64_t i = 0; i < num_elements(); ++i) {
      if (values_[i] != other.values_[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const Array<T>& other) const { return !(*this == other); }

  // Performs the equivalent of a slice operation on this array.
  Array<T> Slice(absl::Span<const int64_t> starts,
                 absl::Span<const int64_t> limits,
                 bool out_of_bounds_ok = false) const {
    CHECK_EQ(starts.size(), num_dimensions());
    CHECK_EQ(limits.size(), num_dimensions());

    OwnedBuffer<int64_t> sizes(starts.size());
    for (int64_t i = 0; i < starts.size(); ++i) {
      CHECK_GE(starts[i], 0);
      if (!out_of_bounds_ok) {
        CHECK_LE(limits[i], dim(i));
      }
      sizes[i] = limits[i] - starts[i];
    }
    Array<T> result(sizes.span());
    if (result.num_elements() == 0) {
      return result;
    }
    // Initializes the slice to the first value if out of bounds access are ok.
    if (out_of_bounds_ok) {
      CHECK_GT(num_elements(), 0);
      for (int64_t i = 0; i < result.num_elements(); ++i) {
        result.values_[i] = values_[0];
      }
    }

    OwnedBuffer<int64_t> index(sizes_.size, default_init_t{});
    int64_t slice_i = 0;
    for (int64_t i = 0; i < num_elements(); ++i, next_index(&index)) {
      if (array_impl::all_inside_range(index.span(), starts, limits)) {
        // Even though the bounds of result are different to our bounds, we're
        // iterating in the same order. So we can simply write successive linear
        // indices instead of recalculating a multi-dimensional index.
        result.values_[slice_i++] = values_[i];
      }
    }
    return result;
  }

  // Performs the equivalent of a DynamicUpdateSlice in-place on this array.
  void UpdateSlice(const Array<T>& from,
                   absl::Span<const int64_t> start_indices) {
    CHECK_EQ(from.num_dimensions(), num_dimensions());
    OwnedBuffer<int64_t> limit_indices(start_indices.size());
    for (int64_t i = 0; i < start_indices.size(); ++i) {
      limit_indices[i] = from.sizes_[i] + start_indices[i];
    }
    OwnedBuffer<int64_t> index(sizes_.size, default_init_t{});
    int64_t from_i = 0;
    for (int64_t i = 0; i < num_elements(); ++i, next_index(&index)) {
      if (array_impl::all_inside_range(index.span(), start_indices,
                                       limit_indices)) {
        // Even though the bounds of from are different to our bounds, we're
        // iterating in the same order. So we can simply write successive linear
        // indices instead of recalculating a multi-dimensional index.
        values_[i] = from.values_[from_i++];
      }
    }
  }

  // Performs an in-place reshape, modifying the dimensions but not the
  // underlying data.
  void Reshape(absl::Span<const int64_t> new_dimensions) {
    const int64_t new_num_elements =
        std::accumulate(new_dimensions.begin(), new_dimensions.end(), 1LL,
                        std::multiplies<int64_t>());
    CHECK_EQ(new_num_elements, num_elements());
    if (sizes_.size != new_dimensions.size()) {
      sizes_ = OwnedBuffer<int64_t>(new_dimensions.size());
    }
    std::memcpy(sizes_.data.get(), new_dimensions.data(),
                new_dimensions.size() * sizeof(int64_t));
  }

  // Performs a permutation of dimensions.
  void TransposeDimensions(absl::Span<const int64_t> permutation) {
    return TransposeDimensionsImpl<int64_t>(permutation);
  }
  void TransposeDimensions(absl::Span<const int> permutation) {
    return TransposeDimensionsImpl<int>(permutation);
  }
  void TransposeDimensions(std::initializer_list<int> permutation) {
    return TransposeDimensionsImpl<int>(permutation);
  }
  template <typename IntT,
            std::enable_if_t<std::is_integral_v<IntT>>* = nullptr>
  void TransposeDimensionsImpl(absl::Span<const IntT> permutation) {
    CHECK_EQ(sizes_.size, permutation.size());
    OwnedBuffer<int64_t> permuted_dims(permutation.size());
    for (int64_t i = 0; i < permutation.size(); ++i) {
      permuted_dims[i] = this->dim(permutation[i]);
    }
    Array<T> permuted(permuted_dims.span());
    OwnedBuffer<int64_t> src_indices(sizes_.size, -1);
    permuted.Each([&](absl::Span<const int64_t> indices, T* value) {
      for (int64_t i = 0; i < sizes_.size; ++i) {
        src_indices[permutation[i]] = indices[i];
      }
      *value = (*this)(src_indices.span());
    });
    *this = std::move(permuted);
  }

  template <typename H>
  friend H AbslHashValue(H h, const Array& array) {
    return H::combine(std::move(h), array.values_.span(), array.dimensions());
  }

  // Returns a string representation of the array suitable for debugging.
  std::string ToString() const {
    if (sizes_.size == 0) {
      return "";
    }
    std::string result;
    OwnedBuffer<int64_t> index(sizes_.size, default_init_t{});
    do {
      // Emit leading spaces and opening square brackets
      if (index[index.size - 1] == 0) {
        for (int64_t i = sizes_.size - 1; i >= 0; --i) {
          if (i == 0 || index[i - 1] != 0) {
            for (int64_t j = 0; j < sizes_.size; ++j) {
              absl::StrAppend(&result, j < i ? " " : "[");
            }
            break;
          }
        }
      }
      int value_index = calculate_index(index.span());
      if (value_index < num_elements()) {
        absl::StrAppend(&result, values_[value_index]);
      }

      // Emit comma if it isn't the last element
      if (index[index.size - 1] < sizes_[sizes_.size - 1] - 1) {
        absl::StrAppend(&result, ", ");
      }

      // Emit closing square brackets
      for (int64_t i = sizes_.size - 1; i >= 0; --i) {
        if (index[i] < sizes_[i] - 1) {
          break;
        }
        absl::StrAppend(&result, "]");
        if (i != 0 && index[i - 1] < sizes_[i - 1] - 1) {
          absl::StrAppend(&result, ",\n");
        }
      }
    } while (next_index(&index));
    return result;
  }

 private:
  struct default_init_t {};
  struct no_default_init_t {};
  // A fixed sized dynamically allocated buffer to replace std::vector usage. It
  // saves one word for storing capacity which is always the same as size and it
  // provides the ability to leave its elements uninitialized if the element
  // type is trivially destructible.
  template <typename D>
  struct OwnedBuffer {
    explicit OwnedBuffer(size_t size)
        : data(std::is_trivially_destructible_v<D> ? new D[size]
                                                   : new D[size]()),
          size(size) {}
    explicit OwnedBuffer(size_t size, default_init_t)
        : data(new D[size]()), size(size) {}

    explicit OwnedBuffer(size_t size, D init) : OwnedBuffer(size) {
      std::fill(data.get(), data.get() + size, init);
    }

    OwnedBuffer(OwnedBuffer&& other)
        : data(std::move(other.data)), size(other.size) {
      other.size = 0;
    }

    OwnedBuffer& operator=(OwnedBuffer&& other) {
      data = std::move(other.data);
      size = other.size;
      other.size = 0;
      return *this;
    }

    OwnedBuffer Clone() const {
      OwnedBuffer clone(size);
      std::memcpy(clone.data.get(), data.get(), size * sizeof(D));
      return clone;
    }

    D& operator[](int64_t index) { return data[index]; }
    const D& operator[](int64_t index) const { return data[index]; }

    absl::Span<const D> span() const {
      return absl::MakeConstSpan(data.get(), size);
    }

    std::unique_ptr<D[]> data;
    size_t size;
  };

  explicit Array(absl::Span<const int64_t> sizes, no_default_init_t)
      : sizes_(sizes.size()), values_(calculate_elements(sizes)) {
    std::memcpy(sizes_.data.get(), sizes.data(),
                sizeof(int64_t) * sizes.size());
  }

  // Extracts the dimensions of an initializer_list to an array type int64_t.
  // Used by the initializer list based constructors to convert the size type
  // into int64_t to be passed to the size based constructor.
  template <typename D>
  static std::array<int64_t, 1> ToInt64Array(const InitializerList1D<D>& data) {
    return std::array<int64_t, 1>{static_cast<int64_t>(data.size())};
  }

  template <typename D>
  static std::array<int64_t, 2> ToInt64Array(const InitializerList2D<D>& data) {
    return std::array<int64_t, 2>{static_cast<int64_t>(data.size()),
                                  static_cast<int64_t>(data.begin()->size())};
  }

  template <typename D>
  static std::array<int64_t, 3> ToInt64Array(const InitializerList3D<D>& data) {
    return std::array<int64_t, 3>{
        static_cast<int64_t>(data.size()),
        static_cast<int64_t>(data.begin()->size()),
        static_cast<int64_t>(data.begin()->begin()->size())};
  }

  template <typename D>
  static std::array<int64_t, 4> ToInt64Array(const InitializerList4D<D>& data) {
    return std::array<int64_t, 4>{
        static_cast<int64_t>(data.size()),
        static_cast<int64_t>(data.begin()->size()),
        static_cast<int64_t>(data.begin()->begin()->size()),
        static_cast<int64_t>(data.begin()->begin()->begin()->size())};
  }

  // Returns the linear index from the list of per-dimension indexes. Function
  // is templated so can be used with an std::array from operator() to avoid
  // memory allocation.
  // The returned value may be larger than or equal to the number of elements if
  // the indexes exceed the array's corresponding dimension size.
  int64_t calculate_index(absl::Span<const int64_t> indexes) const {
    DCHECK_EQ(sizes_.size, indexes.size());
    int64_t index = 0;
    for (int64_t i = 0; i < sizes_.size; ++i) {
      index *= sizes_[i];
      index += indexes[i];
    }
    return index;
  }

  // Advances the specified set of indexes and returns true if we haven't
  // wrapped around (i.e. result isn't {0, 0, ...}).
  bool next_index(OwnedBuffer<int64_t>* index) const {
    DCHECK_EQ(index->size, sizes_.size);
    for (int64_t i = sizes_.size - 1; i >= 0; --i) {
      (*index)[i]++;
      if ((*index)[i] < sizes_[i]) {
        return true;
      }
      (*index)[i] = 0;
    }
    return false;
  }

  static size_t calculate_elements(absl::Span<const int64_t> sizes) {
    return std::accumulate(sizes.begin(), sizes.end(), 1LL,
                           std::multiplies<int64_t>());
  }

  OwnedBuffer<int64_t> sizes_;
  OwnedBuffer<T> values_;
};

// Specialization of FillRandom() method for complex64 type. Uses real part of
// the stddev parameter as the standard deviation value.
template <>
void Array<complex64>::FillRandom(const complex64& stddev, double mean,
                                  int seed);

}  // namespace xla

#endif  // XLA_ARRAY_H_
