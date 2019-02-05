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
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

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
// integral type.
template <typename... T>
using pack_is_integral = conjunction<std::is_integral<T>...>;

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

  // Creates a new array with the specified dimensions.
  explicit Array(absl::Span<const int64> sizes) : Array(sizes, T()) {}

  // Creates a new array with the specified dimensions and specified value for
  // every cell.
  Array(absl::Span<const int64> sizes, T value)
      : sizes_(sizes.begin(), sizes.end()), values_(new T[num_elements()]) {
    Fill(value);
  }

  // Creates a 2D array from the given nested initializer list. The outer
  // initializer list is the first dimension, the inner is the second dimension.
  // For example, {{1, 2, 3}, {4, 5, 6}} results in an array with n1=2 and n2=3.
  Array(InitializerList2D values)
      : Array(ToInt64Vector({values.size(), values.begin()->size()})) {
    int64 idx = 0;
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
    int64 idx = 0;
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
    int64 idx = 0;
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
    int64 idx = 0;
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
    int64 idx = 0;
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
    int64 idx = 0;
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
    int64 idx = 0;
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

  Array<T>& operator=(const Array<T>& other) {
    sizes_ = other.sizes_;
    values_.reset(new T[num_elements()]);
    std::copy(&other.values_[0], &other.values_[0] + num_elements(),
              &values_[0]);
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
  void FillRepeatedIota(const T& value, int64 length) {
    for (int64 i = 0; i < num_elements(); i += length) {
      std::iota(&values_[i], &values_[std::min(i + length, num_elements())],
                value);
    }
  }

  // Fills the array with the sequence i*multiplier for i=0,1,...
  void FillWithMultiples(const T& multiplier) {
    for (int64 i = 0; i < num_elements(); ++i) {
      values_[i] = static_cast<T>(i) * multiplier;
    }
  }

  // Fills the array with random normal variables with the specified mean.
  void FillRandom(const T& stddev, const double mean = 0.0,
                  const int seed = 12345) {
    std::mt19937 g(seed);
    std::normal_distribution<double> distribution(mean,
                                                  static_cast<double>(stddev));
    for (int64 i = 0; i < num_elements(); ++i) {
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
  void Each(std::function<void(absl::Span<const int64>, T*)> f) {
    std::vector<int64> index(sizes_.size());
    for (int64 i = 0; i < num_elements(); ++i, next_index(&index)) {
      f(index, &values_[i]);
    }
  }

  // Invokes a callback with the (indices, value) for each cell in the array.
  void Each(std::function<void(absl::Span<const int64>, T)> f) const {
    std::vector<int64> index(sizes_.size());
    for (int64 i = 0; i < num_elements(); ++i, next_index(&index)) {
      f(index, values_[i]);
    }
  }

  // Invokes a callback with the (indices, value_ptr) for each cell in the
  // array. If a callback returns a non-OK status, returns that else returns
  // Status::OK().
  Status EachStatus(std::function<Status(absl::Span<const int64>, T*)> f) {
    std::vector<int64> index(sizes_.size());
    for (int64 i = 0; i < num_elements(); ++i, next_index(&index)) {
      Status s = f(index, &values_[i]);
      if (!s.ok()) {
        return s;
      }
    }
    return Status::OK();
  }

  // Invokes a callback with the (indices, value) for each cell in the array.
  // If a callback returns a non-OK status, returns that else returns
  // Status::OK().
  Status EachStatus(std::function<Status(absl::Span<const int64>, T)> f) const {
    std::vector<int64> index(sizes_.size());
    for (int64 i = 0; i < num_elements(); ++i, next_index(&index)) {
      Status s = f(index, values_[i]);
      if (!s.ok()) {
        return s;
      }
    }
    return Status::OK();
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
    std::array<int64, sizeof...(dims)> indexes{{static_cast<int64>(dims)...}};
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
    std::array<int64, sizeof...(dims)> indexes{{static_cast<int64>(dims)...}};
    return values_[calculate_index(indexes)];
  }

  // Returns the value at the cell specified by the indexes. The number of
  // arguments have to match with the number of dimensions for the array.
  const T& operator()(absl::Span<const int64> indexes) const {
    return values_[calculate_index(indexes)];
  }

  // Returns the value at the cell specified by the indexes. The number of
  // arguments have to match with the number of dimensions for the array.
  T& operator()(absl::Span<const int64> indexes) {
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
  int64 dim(int64 n) const {
    CHECK(n < sizes_.size());
    return sizes_[n];
  }

  // Returns a vector containing the dimensions of the array.
  const std::vector<int64>& dimensions() const { return sizes_; }

  int64 num_dimensions() const { return sizes_.size(); }

  // Returns the total number of elements in the array.
  int64 num_elements() const {
    return std::accumulate(sizes_.begin(), sizes_.end(), 1LL,
                           std::multiplies<int64>());
  }

  const T* begin() const { return &values_[0]; }
  T* begin() { return &values_[0]; }
  const T* end() const { return &values_[num_elements()]; }
  T* end() { return &values_[num_elements()]; }

  bool operator==(const Array<T>& other) const {
    if (sizes_.size() != other.sizes_.size()) {
      return false;
    }
    for (int64 i = 0; i < sizes_.size(); ++i) {
      if (sizes_[i] != other.sizes_[i]) {
        return false;
      }
    }
    for (int64 i = 0; i < num_elements(); ++i) {
      if (values_[i] != other.values_[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const Array<T>& other) const { return !(*this == other); }

  // Performs the equivalent of a slice operation on this array.
  Array<T> Slice(absl::Span<const int64> starts,
                 absl::Span<const int64> limits) const {
    CHECK_EQ(starts.size(), num_dimensions());
    CHECK_EQ(limits.size(), num_dimensions());

    std::vector<int64> sizes;
    std::transform(starts.begin(), starts.end(), limits.begin(),
                   std::back_inserter(sizes),
                   [](int64 start, int64 limit) { return limit - start; });
    Array<T> result(sizes);

    std::vector<int64> index(sizes_.size());
    int64 slice_i = 0;
    for (int64 i = 0; i < num_elements(); ++i, next_index(&index)) {
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
                   absl::Span<const int64> start_indices) {
    CHECK_EQ(from.num_dimensions(), num_dimensions());
    std::vector<int64> limit_indices;
    std::transform(start_indices.begin(), start_indices.end(),
                   from.dimensions().begin(), std::back_inserter(limit_indices),
                   std::plus<int64>{});
    std::vector<int64> index(sizes_.size());
    int64 from_i = 0;
    for (int64 i = 0; i < num_elements(); ++i, next_index(&index)) {
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
  void Reshape(absl::Span<const int64> new_dimensions) {
    int64 old_num_elements = num_elements();
    sizes_ = std::vector<int64>(new_dimensions.begin(), new_dimensions.end());
    CHECK_EQ(num_elements(), old_num_elements);
  }

  // Returns a string representation of the array suitable for debugging.
  string ToString() const {
    std::vector<string> pieces;
    std::vector<int64> index(sizes_.size());
    do {
      // Emit leading spaces and opening square brackets
      if (index.back() == 0) {
        for (int64 i = sizes_.size() - 1; i >= 0; --i) {
          if (i == 0 || index[i - 1] != 0) {
            for (int64 j = 0; j < sizes_.size(); ++j) {
              pieces.push_back(j < i ? " " : "[");
            }
            break;
          }
        }
      }

      pieces.push_back(absl::StrCat(values_[calculate_index(index)]));

      // Emit comma if it isn't the last element
      if (index.back() != sizes_.back() - 1) {
        pieces.push_back(", ");
      }

      // Emit closing square brackets
      for (int64 i = sizes_.size() - 1; i >= 0; --i) {
        if (index[i] != sizes_[i] - 1) {
          break;
        }
        pieces.push_back("]");
        if (i != 0 && index[i - 1] != sizes_[i - 1] - 1) {
          pieces.push_back(",\n");
        }
      }
    } while (next_index(&index));
    return absl::StrJoin(pieces, "");
  }

 private:
  // Converts an initializer_list of type U to a vector of type int64. Used by
  // the initializer list based constructors to convert the size type into int64
  // to be passed to the size based constructor.
  template <typename U>
  static std::vector<int64> ToInt64Vector(
      const std::initializer_list<U>& data) {
    return std::vector<int64>(data.begin(), data.end());
  }

  // Returns the linear index from the list of per-dimension indexes. Function
  // is templated so can be used with an std::array from operator() to avoid
  // memory allocation.
  template <typename U>
  int64 calculate_index(const U& indexes) const {
    CHECK_EQ(sizes_.size(), indexes.size());
    int64 index = 0;
    for (int64 i = 0; i < sizes_.size(); ++i) {
      index *= sizes_[i];
      index += indexes[i];
    }
    return index;
  }

  // Advances the specified set of indexes and returns true if we haven't
  // wrapped around (i.e. result isn't {0, 0, ...}).
  bool next_index(std::vector<int64>* index) const {
    CHECK_EQ(index->size(), sizes_.size());
    for (int64 i = sizes_.size() - 1; i >= 0; --i) {
      (*index)[i]++;
      if ((*index)[i] < sizes_[i]) {
        return true;
      }
      (*index)[i] = 0;
    }
    return false;
  }

  std::vector<int64> sizes_;
  std::unique_ptr<T[]> values_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_ARRAY_H_
