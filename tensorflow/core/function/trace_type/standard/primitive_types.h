/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");;
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_FUNCTION_TRACE_TYPE_STANDARD_PRIMITIVE_TYPES_H_
#define TENSORFLOW_CORE_FUNCTION_TRACE_TYPE_STANDARD_PRIMITIVE_TYPES_H_
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/core/function/trace_type/standard/trace_type.h"

namespace tensorflow {
namespace trace_type {

// Represents cases where the value is not defined.
class None : public TraceType {
 public:
  explicit None();
  std::unique_ptr<TraceType> clone() const override;

  bool is_subtype_of(const TraceType& other) const override;
  std::unique_ptr<TraceType> most_specific_common_supertype(
      const std::vector<const TraceType*>& others) const override;

  std::string to_string() const override;
  std::size_t hash() const override;

  bool operator==(const TraceType& other) const override;
};

// Represents type hierarchies that have a generic top type.
class Any : public TraceType {
 public:
  // Passing in absl::nullopt instantiates the top type.
  explicit Any(absl::optional<std::unique_ptr<TraceType>> base);
  std::unique_ptr<TraceType> clone() const override;

  absl::optional<const TraceType*> base() const;

  bool is_subtype_of(const TraceType& other) const override;
  std::unique_ptr<TraceType> most_specific_common_supertype(
      const std::vector<const TraceType*>& others) const override;

  std::string to_string() const override;
  std::size_t hash() const override;

  bool operator==(const TraceType& other) const override;

 private:
  absl::optional<std::unique_ptr<TraceType>> base_;
};

// TODO(b/231340870): Add support for other types such as tf.dtype.
template <typename T>
class Literal : public TraceType {
 public:
  explicit Literal(T value) : value_(value) {}

  std::unique_ptr<TraceType> clone() const override {
    return std::unique_ptr<TraceType>(new Literal(value_));
  }

  const T& value() const { return value_; }

  bool is_subtype_of(const TraceType& other) const override {
    return *this == other;
  }

  std::unique_ptr<TraceType> most_specific_common_supertype(
      const std::vector<const TraceType*>& others) const override {
    for (const auto& other : others) {
      if (*this != *other) return nullptr;
    }
    return std::unique_ptr<TraceType>(new Literal<T>(value_));
  }

  std::string to_string() const override;

  std::size_t hash() const override { return std::hash<T>()(value_); }

  bool operator==(const TraceType& other) const override {
    const Literal<T>* casted_other = dynamic_cast<const Literal<T>*>(&other);
    if (casted_other == nullptr) return false;
    return casted_other->value() == value_;
  }

 private:
  T value_;
};

template <>
inline std::string Literal<int>::to_string() const {
  return "Int<" + std::to_string(value_) + ">";
}

template <>
inline std::string Literal<bool>::to_string() const {
  return "Bool<" + std::string(value_ ? "True" : "False") + ">";
}

template <>
inline std::string Literal<std::string>::to_string() const {
  return "String<" + value_ + ">";
}

// TODO(b/232114163): Reconsider flexibility of structural subtyping.
// Represents a fixed collection of types.
class Product : public TraceType {
 public:
  explicit Product(std::vector<std::unique_ptr<TraceType>> elements);
  std::unique_ptr<TraceType> clone() const override;

  const std::vector<const TraceType*> elements() const;

  bool is_subtype_of(const TraceType& other) const override;
  std::unique_ptr<TraceType> most_specific_common_supertype(
      const std::vector<const TraceType*>& others) const override;

  std::string to_string() const override;
  std::size_t hash() const override;

  bool operator==(const TraceType& other) const override;

 private:
  std::vector<std::unique_ptr<TraceType>> elements_;
};

struct TraceTypeHashByRef {
  std::size_t operator()(const TraceType* const& t) const noexcept {
    return t->hash();
  }
};

struct TraceTypeEqByRef {
  bool operator()(const TraceType* lhs, const TraceType* rhs) const {
    return *lhs == *rhs;
  }
};

using RecordMap = absl::flat_hash_map<const TraceType*, const TraceType*,
                                      TraceTypeHashByRef, TraceTypeEqByRef>;

// TODO(b/232114163): Reconsider flexibility of structural subtyping.
// Represents a set of fields consisting of a key type and a value type.
// Key type must be an invariant type (a.is_subtype_of(b) implies a == b).
class Record : public TraceType {
 public:
  explicit Record(std::vector<std::unique_ptr<TraceType>> keys,
                  std::vector<std::unique_ptr<TraceType>> values);

  std::unique_ptr<TraceType> clone() const override;

  const RecordMap& fields() const;

  bool is_subtype_of(const TraceType& other) const override;
  std::unique_ptr<TraceType> most_specific_common_supertype(
      const std::vector<const TraceType*>& others) const override;

  std::string to_string() const override;
  std::size_t hash() const override;

  bool operator==(const TraceType& other) const override;

 private:
  std::vector<std::unique_ptr<TraceType>> owned_keys_;
  std::vector<std::unique_ptr<TraceType>> owned_values_;
  RecordMap fields_;
};

}  // namespace trace_type
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FUNCTION_TRACE_TYPE_STANDARD_PRIMITIVE_TYPES_H_
