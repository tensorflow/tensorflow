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

#ifndef TENSORFLOW_COMPILER_XLA_ARRAY_H_
#define TENSORFLOW_COMPILER_XLA_ARRAY_H_

#include <algorithm>
#include <array>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace array_impl {

// conjunction
//
// Performs a compile-time logical AND operation on the passed types (which
// must have  `::value` members convertible to `bool`. Short-circuits if it
// encounters any `false` members (and does not compare the `::value` members
// of any remaining arguments).
//
// This metafunction is designed to be a drop-in replacement for the C++17
// `std::conjunction` metafunction.
template <typename... Ts>
struct conjunction;

template <typename T, typename... Ts>
struct conjunction<T, Ts...>
    : std::conditional<T::value, conjunction<Ts...>, T>::type {};

template <>
struct conjunction<> : std::true_type {};

// A type trait that is valid when all elements in a parameter pack are of
// integral type. Not using an alias template to work around MSVC 14.00 bug.
template <typename... Ts>
struct pack_is_integral : conjunction<std::is_integral<Ts>...> {};

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
  using InitializerList1D = std::initializer_list<T>;
  using InitializerList2D = std::initializer_list<InitializerList1D>;
  using InitializerList3D = std::initializer_list<InitializerList2D>;
  using InitializerList4D = std::initializer_list<InitializerList3D>;

  using value_type = T;

  // Creates a new array with the specified dimensions and initialized elements.
  explicit Array(absl::Span<const int64_t> sizes)
      : sizes_(sizes.begin(), sizes.end()), values_(new T[num_elements()]()) {}

  // Creates a new array with the specified dimensions and specified value for
  // every cell.
  Array(absl::Span<const int64_t> sizes, T value)
      : sizes_(sizes.begin(), sizes.end()), values_(new T[num_elements()]) {
    Fill(value);
  }

  // Creates a 2D array from the given nested initializer list. The outer
  // initializer list is the first dimension, the inner is the second dimension.
  // For example, {{1, 2, 3}, {4, 5, 6}} results in an array with n1=2 and n2=3.
  Array(InitializerList2D values)
      : Array(ToInt64Vector({values.size(), values.begin()->size()})) {
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
  template <typename T2, typename = typename std::enable_if<
                             (std::is_same<T, Eigen::half>::value ||
                              std::is_same<T, bfloat16>::value ||
                              std::is_same<T, float>::value ||
                              std::is_same<T, double>::value) &&
                             std::is_same<T2, float>::value>::type>
  Array(std::initializer_list<T2> values)
      : Array(ToInt64Vector({values.size()})) {
    int64_t idx = 0;
    for (const auto& it1 : values) {
      values_[idx] = static_cast<T>(it1);
      ++idx;
    }
    CHECK(idx == num_elements());
  }

  // Creates a 2D array of a floating-point type (half, bfloat16, float,
  // or double) from an initializer list of float values.
  template <typename T2, typename = typename std::enable_if<
                             (std::is_same<T, Eigen::half>::value ||
                              std::is_same<T, bfloat16>::value ||
                              std::is_same<T, float>::value ||
                              std::is_same<T, double>::value) &&
                             std::is_same<T2, float>::value>::type>
  Array(std::initializer_list<std::initializer_list<T2>> values)
      : Array(ToInt64Vector({values.size(), values.begin()->size()})) {
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
  Array(InitializerList3D values)
      : Array(ToInt64Vector({values.size(), values.begin()->size(),
                             values.begin()->begin()->size()})) {
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
  template <typename T2, typename = typename std::enable_if<
                             (std::is_same<T, Eigen::half>::value ||
                              std::is_same<T, bfloat16>::value ||
                              std::is_same<T, float>::value ||
                              std::is_same<T, double>::value) &&
                             std::is_same<T2, float>::value>::type>
  Array(std::initializer_list<std::initializer_list<std::initializer_list<T2>>>
            values)
      : Array(ToInt64Vector({values.size(), values.begin()->size(),
                             values.begin()->begin()->size()})) {
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
  Array(InitializerList4D values)
      : Array(ToInt64Vector({values.size(), values.begin()->size(),
                             values.begin()->begin()->size(),
                             values.begin()->begin()->begin()->size()})) {
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
  template <typename T2, typename = typename std::enable_if<
                             (std::is_same<T, Eigen::half>::value ||
                              std::is_same<T, bfloat16>::value ||
                              std::is_same<T, float>::value ||
                              std::is_same<T, double>::value) &&
                             std::is_same<T2, float>::value>::type>
  Array(std::initializer_list<
        std::initializer_list<std::initializer_list<std::initializer_list<T2>>>>
            values)
      : Array(ToInt64Vector({values.size(), values.begin()->size(),
                             values.begin()->begin()->size(),
                             values.begin()->begin()->begin()->size()})) {
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
      : sizes_(other.sizes_), values_(new T[num_elements()]) {
    std::copy(&other.values_[0], &other.values_[0] + num_elements(),
              &values_[0]);
  }

  Array(Array<T>&& other)
      : sizes_(std::move(other.sizes_)), values_(std::move(other.values_)) {}

  Array<T>& operator=(const Array<T>& other) {
    sizes_ = other.sizes_;
    values_.reset(new T[num_elements()]);
    std::copy(&other.values_[0], &other.values_[0] + num_elements(),
              &values_[0]);
    return *this;
  }

  Array<T>& operator=(Array<T>&& other) {
    sizes_ = std::move(other.sizes_);
    values_ = std::move(other.values_);
    return *this;
  }

  // Fills the array with the specified value.
  void Fill(const T& value) {
    std::fill(&values_[0], &values_[0] + num_elements(), value);
  }

  // Fills the array with sequentially increasing values.
  void FillIota(const T& value) {
    std::iota(&values_[0], &values_[0] + num_elements(), value);
  }

  // Fills the array with a repeating sequence:
  //   [value, value + 1, ..., value + length - 1, value, ... ]
  void FillRepeatedIota(const T& value, int64_t length) {
    for (int64_t i = 0; i < num_elements(); i += length) {
      std::iota(&values_[i], &values_[std::min(i + length, num_elements())],
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

  // Sets all the values in the array to values specified in the container.
  template <typename Container = std::initializer_list<T>>
  void SetValues(const Container& container) {
    CHECK_EQ(std::distance(std::begin(container), std::end(container)),
             num_elements());
    std::copy(std::begin(container), std::end(container), &values_[0]);
  }

  // Invokes a callback with the (indices, value_ptr) for each cell in the
  // array.
  void Each(std::function<void(absl::Span<const int64_t>, T*)> f) {
    std::vector<int64_t> index(sizes_.size());
    for (int64_t i = 0; i < num_elements(); ++i, next_index(&index)) {
      f(index, &values_[i]);
    }
  }

  // Invokes a callback with the (indices, value) for each cell in the array.
  void Each(std::function<void(absl::Span<const int64_t>, T)> f) const {
    std::vector<int64_t> index(sizes_.size());
    for (int64_t i = 0; i < num_elements(); ++i, next_index(&index)) {
      f(index, values_[i]);
    }
  }

  // Invokes a callback with the (indices, value_ptr) for each cell in the
  // array. If a callback returns a non-OK status, returns that else returns
  // Status::OK().
  Status EachStatus(std::function<Status(absl::Span<const int64_t>, T*)> f) {
    std::vector<int64_t> index(sizes_.size());
    for (int64_t i = 0; i < num_elements(); ++i, next_index(&index)) {
      Status s = f(index, &values_[i]);
      if (!s.ok()) {
        return s;
      }
    }
    return OkStatus();
  }

  // Invokes a callback with the (indices, value) for each cell in the array.
  // If a callback returns a non-OK status, returns that else returns
  // Status::OK().
  Status EachStatus(
      std::function<Status(absl::Span<const int64_t>, T)> f) const {
    std::vector<int64_t> index(sizes_.size());
    for (int64_t i = 0; i < num_elements(); ++i, next_index(&index)) {
      Status s = f(index, values_[i]);
      if (!s.ok()) {
        return s;
      }
    }
    return OkStatus();
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
    // We are using a std::array to avoid having to allocate memory in this
    // function for performance reasons.
    std::array<int64_t, sizeof...(dims)> indexes{
        {static_cast<int64_t>(dims)...}};
    return values_[calculate_index(indexes)];
  }

  // Returns the value at the cell specified by the indexes. The number of
  // arguments have to match with the number of dimensions for the array.
  const T& operator()(absl::Span<const int64_t> indexes) const {
    return values_[calculate_index(indexes)];
  }

  // Returns the value at the cell specified by the indexes. The number of
  // arguments have to match with the number of dimensions for the array.
  T& operator()(absl::Span<const int64_t> indexes) {
    return values_[calculate_index(indexes)];
  }

  // Low-level accessor for stuff like memcmp, handle with care. Returns pointer
  // to the underlying storage of the array (similarly to std::vector::data()).
  T* data() const {
    // TODO(tberghammer): Get rid of the const_cast. Currently it is needed
    // because the Eigen backend needs a non-const pointers even for reading
    // from the array.
    return const_cast<Array*>(this)->values_.get();
  }

  // Returns the size of the dimension at the given index.
  int64_t dim(int64_t n) const {
    const int64_t sizes_size = sizes_.size();
    CHECK(n < sizes_size);
    return sizes_[n];
  }

  // Returns a vector containing the dimensions of the array.
  const std::vector<int64_t>& dimensions() const { return sizes_; }

  int64_t num_dimensions() const { return sizes_.size(); }

  // Returns the total number of elements in the array.
  int64_t num_elements() const {
    return std::accumulate(sizes_.begin(), sizes_.end(), 1LL,
                           std::multiplies<int64_t>());
  }

  const T* begin() const { return &values_[0]; }
  T* begin() { return &values_[0]; }
  const T* end() const { return &values_[num_elements()]; }
  T* end() { return &values_[num_elements()]; }

  bool operator==(const Array<T>& other) const {
    if (sizes_.size() != other.sizes_.size()) {
      return false;
    }
    for (int64_t i = 0, end = sizes_.size(); i < end; ++i) {
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
                 absl::Span<const int64_t> limits) const {
    CHECK_EQ(starts.size(), num_dimensions());
    CHECK_EQ(limits.size(), num_dimensions());

    std::vector<int64_t> sizes;
    std::transform(starts.begin(), starts.end(), limits.begin(),
                   std::back_inserter(sizes),
                   [](int64_t start, int64_t limit) { return limit - start; });
    Array<T> result(sizes);

    std::vector<int64_t> index(sizes_.size());
    int64_t slice_i = 0;
    for (int64_t i = 0; i < num_elements(); ++i, next_index(&index)) {
      if (array_impl::all_inside_range(index, starts, limits)) {
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
    std::vector<int64_t> limit_indices;
    std::transform(start_indices.begin(), start_indices.end(),
                   from.dimensions().begin(), std::back_inserter(limit_indices),
                   std::plus<int64_t>{});
    std::vector<int64_t> index(sizes_.size());
    int64_t from_i = 0;
    for (int64_t i = 0; i < num_elements(); ++i, next_index(&index)) {
      if (array_impl::all_inside_range(index, start_indices, limit_indices)) {
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
    int64_t old_num_elements = num_elements();
    sizes_ = std::vector<int64_t>(new_dimensions.begin(), new_dimensions.end());
    CHECK_EQ(num_elements(), old_num_elements);
  }

  // Performs a permutation of dimensions.
  void TransposeDimensions(absl::Span<const int64_t> permutation) {
    std::vector<int64_t> permuted_dims(permutation.size());
    for (int64_t i = 0; i < permutation.size(); ++i) {
      permuted_dims[i] = this->dim(permutation[i]);
    }
    Array<T> permuted(permuted_dims);
    std::vector<int64_t> src_indices(sizes_.size(), -1);
    permuted.Each([&](absl::Span<const int64_t> indices, int64_t* value) {
      CHECK_EQ(sizes_.size(), indices.size());
      for (int64_t i = 0; i < sizes_.size(); ++i) {
        src_indices[permutation[i]] = indices[i];
      }
      *value = (*this)(src_indices);
    });
    *this = std::move(permuted);
  }

  template <typename H>
  friend H AbslHashValue(H h, const Array& array) {
    return H::combine(std::move(h), absl::MakeSpan(array.begin(), array.end()),
                      array.dimensions());
  }

  // Returns a string representation of the array suitable for debugging.
  std::string ToString() const {
    if (sizes_.empty()) {
      return "";
    }
    std::vector<std::string> pieces;
    std::vector<int64_t> index(sizes_.size());
    do {
      // Emit leading spaces and opening square brackets
      if (index.back() == 0) {
        for (int64_t i = sizes_.size() - 1; i >= 0; --i) {
          if (i == 0 || index[i - 1] != 0) {
            for (int64_t j = 0; j < sizes_.size(); ++j) {
              pieces.push_back(j < i ? " " : "[");
            }
            break;
          }
        }
      }
      int value_index = calculate_index(index);
      if (value_index < num_elements()) {
        pieces.push_back(absl::StrCat(values_[value_index]));
      }

      // Emit comma if it isn't the last element
      if (index.back() < sizes_.back() - 1) {
        pieces.push_back(", ");
      }

      // Emit closing square brackets
      for (int64_t i = sizes_.size() - 1; i >= 0; --i) {
        if (index[i] < sizes_[i] - 1) {
          break;
        }
        pieces.push_back("]");
        if (i != 0 && index[i - 1] < sizes_[i - 1] - 1) {
          pieces.push_back(",\n");
        }
      }
    } while (next_index(&index));
    return absl::StrJoin(pieces, "");
  }

 private:
  // Converts an initializer_list of type U to a vector of type int64_t. Used by
  // the initializer list based constructors to convert the size type into
  // int64_t to be passed to the size based constructor.
  template <typename U>
  static std::vector<int64_t> ToInt64Vector(
      const std::initializer_list<U>& data) {
    return std::vector<int64_t>(data.begin(), data.end());
  }

  // Returns the linear index from the list of per-dimension indexes. Function
  // is templated so can be used with an std::array from operator() to avoid
  // memory allocation.
  // The returned value may be larger than or equal to the number of elements if
  // the indexes exceed the array's corresponding dimension size.
  template <typename U>
  int64_t calculate_index(const U& indexes) const {
    CHECK_EQ(sizes_.size(), indexes.size());
    int64_t index = 0;
    for (int64_t i = 0; i < sizes_.size(); ++i) {
      index *= sizes_[i];
      index += indexes[i];
    }
    return index;
  }

  // Advances the specified set of indexes and returns true if we haven't
  // wrapped around (i.e. result isn't {0, 0, ...}).
  bool next_index(std::vector<int64_t>* index) const {
    CHECK_EQ(index->size(), sizes_.size());
    for (int64_t i = sizes_.size() - 1; i >= 0; --i) {
      (*index)[i]++;
      if ((*index)[i] < sizes_[i]) {
        return true;
      }
      (*index)[i] = 0;
    }
    return false;
  }

  std::vector<int64_t> sizes_;
  std::unique_ptr<T[]> values_;
};

// Specialization of FillRandom() method for complex64 type. Uses real part of
// the stddev parameter as the standard deviation value.
template <>
void Array<complex64>::FillRandom(const complex64& stddev, const double mean,
                                  const int seed);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_ARRAY_H_
