/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
/// @file value.h
/// @brief The TaggedValue struct that supports Python-like behavior in C++.
///
/// The TaggedValue struct implements a tagged union data structure
/// (https://en.wikipedia.org/wiki/Tagged_union) in the TensorFlow C++ API. It
/// contains a `Type` enum (sometimes referred to as a "tag")
/// and a `Data` union for holding values.

#ifndef TENSORFLOW_CC_EXPERIMENTAL_LIBTF_VALUE_H_
#define TENSORFLOW_CC_EXPERIMENTAL_LIBTF_VALUE_H_

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/core/platform/intrusive_ptr.h"
#include "tensorflow/core/platform/statusor.h"

// TODO(b/195578409): Move all value objects into `impl`. Currently only values
// that do not reference TaggedValue are there.
#include "tensorflow/cc/experimental/libtf/impl/none.h"
#include "tensorflow/cc/experimental/libtf/impl/scalars.h"
#include "tensorflow/cc/experimental/libtf/impl/string.h"
#include "tensorflow/cc/experimental/libtf/impl/tensor_spec.h"

namespace tf {
namespace libtf {
namespace impl {
// Necessary forward declares.
class TaggedValue;
class Tuple;
template <class T>
// TODO(ccrusius): Use absl::Hash specializations instead.
class TaggedValueHash;
using List = std::vector<TaggedValue>;
using ListPtr = std::shared_ptr<List>;
using Dict =
    absl::flat_hash_map<TaggedValue, TaggedValue, TaggedValueHash<TaggedValue>>;
using DictPtr = std::shared_ptr<Dict>;
using TuplePtr = std::shared_ptr<Tuple>;
using Func =
    std::function<absl::StatusOr<TaggedValue>(TaggedValue, TaggedValue)>;
// A capsule holds a pointer and a destructor for the pointer (i.e. a generic
// shared_ptr to void with a custom deleter).
using Capsule = std::shared_ptr<void>;
using TaggedValueTensor =
    tensorflow::core::IntrusivePtr<tensorflow::AbstractTensorHandle>;

// Declare hash types so they can be instantiated below.

/// @brief TaggedValue hashing infrastructure, which uses absl::hash.
///
/// Hashable TaggedValues overload `AbslHashValue`. Non-hashable structures
/// return 0.
template <>
struct TaggedValueHash<TaggedValue> {
  size_t operator()(const TaggedValue& v) const;
};

/// @brief Hash implementation for TaggedValue Tuples.
template <>
struct TaggedValueHash<Tuple> {
  size_t operator()(const Tuple& t) const;
};

/// @brief The basic `TaggedValue` tagged union type.
///
/// A `TaggedValue` contains a `Type` (or "tag") as an enum and a `Value` union.
/// Values include tensors, primitive values, lists, tuples, and dictionaries.
/// In the future we might also want to have representation of python objects in
/// the form of PyObject*.
class TaggedValue final {
 public:
  /// @brief Enum that describes the possible types a `TaggedValue` can be.
  ///
  /// A `TaggedValue` must be one of the following types: NONE, INT64, FLOAT32,
  /// STRING, FUNC, DICT, LIST, TUPLE, TENSOR, TENSOR_SPEC, CAPSULE.
  enum Type {
    NONE = 0,
    INT64 = 1,
    FLOAT32 = 2,
    STRING = 3,
    FUNC = 4,
    DICT = 5,
    LIST = 6,
    TUPLE = 7,
    TENSOR = 8,
    TENSOR_SPEC = 9,
    CAPSULE = 10,
  };
  TaggedValue() : type_(NONE), data_() {}

  /// Move assignment operator.
  TaggedValue& operator=(TaggedValue&& v) {
    destroy();
    MoveIntoUnion(std::move(v));
    return *this;
  }
  /// Move constructor.
  TaggedValue(TaggedValue&& v) : type_(NONE) { MoveIntoUnion(std::move(v)); }
  /// Copy constructor.
  TaggedValue(const TaggedValue& v) : type_(NONE) { CopyIntoUnion(v); }
  /// Copy assignment operator.
  TaggedValue& operator=(const TaggedValue& v) {
    destroy();
    CopyIntoUnion(v);
    return *this;
  }
  /// TaggedValue constructor for type TENSOR.
  explicit TaggedValue(TaggedValueTensor tensor)
      : type_(TENSOR), data_(std::move(tensor)) {}
  /// TaggedValue constructor for type TENSOR_SPEC.
  explicit TaggedValue(tensorflow::PartialTensorShape shape,
                       tensorflow::DataType dtype)
      : type_(TENSOR_SPEC), data_(shape, dtype) {}
  /// TaggedValue constructor for type FUNC.
  explicit TaggedValue(Func f32) : type_(FUNC), data_(f32) {}
  /// TaggedValue constructor for type FLOAT32.
  explicit TaggedValue(float f32) : type_(FLOAT32), data_(Float32(f32)) {}
  /// TaggedValue constructor for type INT64.
  explicit TaggedValue(int64_t i64) : type_(INT64), data_(Int64(i64)) {}
  /// TaggedValue constructor for type FLOAT32.
  explicit TaggedValue(Float32 f32) : type_(FLOAT32), data_(f32) {}
  /// TaggedValue constructor for type INT64.
  explicit TaggedValue(Int64 i64) : type_(INT64), data_(i64) {}
  /// TaggedValue constructor for type STRING.
  explicit TaggedValue(const char* s) : type_(STRING), data_(s) {}
  /// Constructs a TaggedValue with type NONE.
  static TaggedValue None() {
    TaggedValue v;
    v.type_ = NONE;
    return v;
  }
  /// Constructs a TaggedValue with type LIST.
  static TaggedValue List() {
    TaggedValue v;
    v.type_ = LIST;
    using T = decltype(v.data_.list);
    new (&v.data_.list) T(std::make_shared<T::element_type>());
    return v;
  }
  /// Constructs a TaggedValue with type TUPLE.
  static TaggedValue Tuple() {
    TaggedValue v;
    v.type_ = TUPLE;
    using T = decltype(v.data_.tuple);
    new (&v.data_.tuple) T(std::make_shared<T::element_type>());
    return v;
  }
  /// Constructs a TaggedValue with type DICT.
  static TaggedValue Dict() {
    TaggedValue v;
    v.type_ = DICT;
    using T = decltype(v.data_.dict);
    new (&v.data_.dict) T(std::make_shared<T::element_type>());
    return v;
  }
  /// Constructs a TaggedValue with type TENSOR.
  static TaggedValue Tensor(tensorflow::AbstractTensorHandle* raw_ptr) {
    TaggedValue v;
    v.type_ = TENSOR;
    using T = decltype(v.data_.tensor);
    new (&v.data_.tensor) T(raw_ptr, /*add_ref=*/false);
    return v;
  }

  /// Constructs a TaggedValue with type CAPSULE with a default destructor.
  template <class T>
  static TaggedValue Capsule(T* data) {
    return Capsule(static_cast<void*>(data),
                   [](void* x) { delete static_cast<T*>(x); });
  }
  /// Constructs a TaggedValue with type CAPSULE with a custom destructor.
  static TaggedValue Capsule(void* data, void (*deleter)(void*)) {
    TaggedValue v;
    v.type_ = CAPSULE;
    using T = decltype(v.data_.capsule);
    new (&v.data_.capsule) T(data, deleter);
    return v;
  }
  /// Destroys TaggedValue. Shared pointers in unions must be explicitly
  /// deleted.
  void destroy() {
    if (type_ != NONE) {
      // Explicitly run the destructor on the correct type.
      visit<void>([](auto& x) {
        using T = typename std::decay<decltype(x)>::type;
        x.~T();
      });
      // Make the type None, whenever we destroy so we always have an
      // initialized value.
      type_ = NONE;
    }
  }
  ~TaggedValue() { destroy(); }

  /// @brief Get the underlying value based on type.
  ///
  /// @tparam T The desired return type.
  /// @return The unwrapped value. If this `TaggedValue` type does not currently
  ///         contain a value of type `T`, the program terminates via a call to
  ///         `assert`.
  template <typename T>
  T& get() {
    assert(type_ == EnumValueOf<T>::value);
    return UnionAccess<T>::unsafe_reference(*this);
  }

  /// @brief Get the underlying value based on type.
  ///
  /// @tparam T The desired return type.
  /// @return The unwrapped value. If this `TaggedValue` type does not currently
  ///         contain a value of type `T`, the program terminates via a call to
  ///         `assert`.
  template <typename T>
  const T& get() const {
    assert(type_ == EnumValueOf<T>::value);
    return UnionAccess<T>::unsafe_reference(*this);
  }

  /// Retrieves underlying value from a TaggedValue with type INT64.
  const Int64& i64() const { return get<impl::Int64>(); }

  /// Retrieves underlying value from a TaggedValue with type FLOAT32.
  const Float32& f32() const { return get<impl::Float32>(); }

  /// Retrieves underlying value from a TaggedValue with type STRING.
  const char* s() const { return get<impl::String>().str().c_str(); }

  /// Retrieves underlying value from a TaggedValue with type LIST.
  impl::List& list() { return *get<impl::ListPtr>(); }
  /// Retrieves underlying value from a TaggedValue with type LIST.
  const impl::List& list() const { return *get<impl::ListPtr>(); }

  /// Retrieves underlying value from a TaggedValue with type TUPLE.
  impl::Tuple& tuple() { return *get<impl::TuplePtr>(); }
  /// Retrieves underlying value from TaggedValues with type TUPLE.
  const impl::Tuple& tuple() const { return *get<impl::TuplePtr>(); }

  /// Retrieves underlying value from a TaggedValue with type DICT.
  impl::Dict& dict() { return *get<impl::DictPtr>(); }
  /// Retrieves underlying value from TaggedValues with type DICT.
  const impl::Dict& dict() const { return *get<impl::DictPtr>(); }

  /// Retrieves underlying value from a TaggedValue with type FUNC.
  impl::Func func() const { return get<impl::Func>(); }

  // TODO(danielellis): make const-only if possible, once the API allows for it
  /// Retrieves underlying value from a TaggedValue with type TENSOR.
  TaggedValueTensor& tensor() { return get<TaggedValueTensor>(); }
  /// Retrieves underlying value from a TaggedValue with type TENSOR.
  const TaggedValueTensor& tensor() const { return get<TaggedValueTensor>(); }

  /// Retrieves underlying value from a TaggedValue with type TENSOR_SPEC.
  const TensorSpec& tensor_spec() const { return get<TensorSpec>(); }

  /// Retrieves underlying value from a TaggedValue with type CAPSULE.
  void* capsule() const { return get<impl::Capsule>().get(); }

  /// Retrieves type of TaggedValue.
  Type type() const { return type_; }

  /// @brief Implements equality operator for TaggedValue.
  bool operator==(const TaggedValue& o) const {
    if (type_ != o.type_) return false;
    switch (type_) {
      case LIST:
        return data_.list == o.data_.list;
        break;
      case TUPLE:
        return data_.tuple == o.data_.tuple;
        break;
      case DICT:
        return data_.dict == o.data_.dict;
        break;
      case FUNC:
        // TODO(b/187536093):  This is definitely wrong because the exact ptr of
        // the function pointer is almost always different, because we hold
        // it by value. Two tagged values that hold the same std::function
        // will have different std::function ptrs. operator== is not defined
        // for std::function's so we need a better solution here, or these
        // are not comparable which seems bad.
        return &data_.func == &o.data_.func;
        break;
      case FLOAT32:
        return data_.f32 == o.data_.f32;
        break;
      case INT64:
        return data_.i64 == o.data_.i64;
        break;
      case STRING:
        return data_.s == o.data_.s;
        break;
      case TENSOR:
        return data_.tensor == o.data_.tensor;
      case TENSOR_SPEC:
        return data_.tensor_spec == o.data_.tensor_spec;
      case CAPSULE:
        return data_.capsule.get() == o.data_.capsule.get();
      case NONE:
        return true;
    }
  }

  /// @brief Implements visitor pattern for doing type-based dispatch.
  ///
  /// @tparam R The desired return type.
  /// @tparam Visitor The visitor class which has a callable operator.
  /// @return The `visitor` called on the correct value.
  template <class R, class Visitor>
  R visit(Visitor visitor) {
    switch (type_) {
      case LIST:
        return visitor(data_.list);
      case TUPLE:
        return visitor(data_.tuple);
      case DICT:
        return visitor(data_.dict);
      case FUNC:
        return visitor(data_.func);
      case FLOAT32:
        return visitor(data_.f32);
      case INT64:
        return visitor(data_.i64);
      case STRING:
        return visitor(data_.s);
      case TENSOR:
        return visitor(data_.tensor);
      case TENSOR_SPEC:
        return visitor(data_.tensor_spec);
      case CAPSULE:
        return visitor(data_.capsule);
      case NONE:
        return visitor(impl::None::GetInstance());
    }
  }

  /// @brief Implements visitor pattern for doing type-based dispatch.
  ///
  /// @tparam R The desired return type.
  /// @tparam Visitor The visitor class which has a callable operator.
  /// @return The `visitor` called on the correct value.
  template <class R, class Visitor>
  R visit(Visitor visitor) const {
    switch (type_) {
      case LIST:
        return visitor(data_.list);
      case TUPLE:
        return visitor(data_.tuple);
      case DICT:
        return visitor(data_.dict);
      case FUNC:
        return visitor(data_.func);
      case FLOAT32:
        return visitor(data_.f32);
      case INT64:
        return visitor(data_.i64);
      case STRING:
        return visitor(data_.s);
      case TENSOR:
        return visitor(data_.tensor);
      case TENSOR_SPEC:
        return visitor(data_.tensor_spec);
      case CAPSULE:
        return visitor(data_.capsule);
      case NONE:
        return visitor(impl::None::GetInstance());
    }
  }

 private:
  /// @brief A utility class for mapping C++ types to Type values.
  template <typename T>
  struct EnumValueOf;

  /// @brief A utility class for accessing the `Data` union members.
  template <typename T>
  struct UnionAccess;

  // Unsafe Move, because it assumes the union has already been destroyed
  // or is new!
  void MoveIntoUnion(TaggedValue&& v) {
    assert(type_ == NONE);
    type_ = v.type_;
    if (type_ != NONE) {
      visit<void>([&v](auto& left) -> void {
        using T = typename std::decay<decltype(left)>::type;
        new (&left) T(std::move(UnionAccess<T>::unsafe_reference(v)));
      });
    }
    // Destroy the source r-value reference (making it None)
    v.destroy();
  }

  // Unsafe Move, because it assumes the union has already been destroyed
  // or is new!
  void CopyIntoUnion(const TaggedValue& v) {
    assert(type_ == NONE);
    type_ = v.type_;
    if (type_ != NONE) {
      visit<void>([&v](auto& left) -> void {
        using T = typename std::decay<decltype(left)>::type;
        new (&left) T(UnionAccess<T>::unsafe_reference(v));
      });
    }
  }

  /// @brief The type of the TaggedValue, i.e. the "tag" of a tagged union.
  ///
  /// In principle this could be incorporated into the union
  /// for pointer types and non-64bit values, but then int64 and float64 values
  /// would need to be indirected.  This means that we are aiming for a total
  /// data type size of <=16 bytes, comprised of one pointer (8 bytes) and
  /// one type (<=8bytes).
  Type type_;

  // we use an explicit union here because we want to avoid C++17's
  // variant structures due to c++14 compatibility requirements.
  // TODO(b/183980966): Compare against absl::variant.
  union Data {
    explicit Data() {}
    explicit Data(Float32 f32) : f32(f32) {}
    explicit Data(Int64 i64) : i64(i64) {}
    explicit Data(const char* s) : s(String(s)) {}
    explicit Data(Func fn) : func(fn) {}
    explicit Data(TaggedValueTensor tensor_in) {
      new (&tensor) TaggedValueTensor(std::move(tensor_in));
    }
    explicit Data(tensorflow::PartialTensorShape shape,
                  tensorflow::DataType dtype)
        : tensor_spec({shape, dtype}) {}
    ~Data() {}
    Float32 f32;
    Int64 i64;
    String s;
    Func func;
    // TODO(aselle): look at tensorflow thing
    std::shared_ptr<impl::Dict> dict;
    std::shared_ptr<impl::List> list;
    std::shared_ptr<impl::Tuple> tuple;
    impl::Capsule capsule;
    TaggedValueTensor tensor;
    TensorSpec tensor_spec;
  } data_;
  friend std::ostream& operator<<(std::ostream& o, const TaggedValue& v);
  friend TaggedValueHash<TaggedValue>;
};

#define TF_ENUM_VALUE_OF(TYPE, ENUM)      \
  template <>                             \
  struct TaggedValue::EnumValueOf<TYPE> { \
    static constexpr Type value = ENUM;   \
  };

TF_ENUM_VALUE_OF(impl::Capsule, CAPSULE);
TF_ENUM_VALUE_OF(impl::Float32, FLOAT32);
TF_ENUM_VALUE_OF(impl::Int64, INT64);
TF_ENUM_VALUE_OF(impl::List, LIST);
TF_ENUM_VALUE_OF(impl::ListPtr, LIST);
TF_ENUM_VALUE_OF(impl::Tuple, TUPLE);
TF_ENUM_VALUE_OF(impl::TuplePtr, TUPLE);
TF_ENUM_VALUE_OF(impl::Dict, DICT);
TF_ENUM_VALUE_OF(impl::DictPtr, DICT);
TF_ENUM_VALUE_OF(impl::None, NONE);
TF_ENUM_VALUE_OF(impl::Func, FUNC);
TF_ENUM_VALUE_OF(impl::String, STRING);
TF_ENUM_VALUE_OF(impl::TaggedValueTensor, TENSOR);
TF_ENUM_VALUE_OF(impl::TensorSpec, TENSOR_SPEC);
#undef TF_ENUM_VALUE_OF

#define TF_UNION_ACCESS_INSTANCE(TYPE, MEMBER)                               \
  template <>                                                                \
  struct TaggedValue::UnionAccess<TYPE> {                                    \
    static TYPE& unsafe_reference(TaggedValue& t) { return t.data_.MEMBER; } \
    static const TYPE& unsafe_reference(const TaggedValue& t) {              \
      return t.data_.MEMBER;                                                 \
    }                                                                        \
  };

TF_UNION_ACCESS_INSTANCE(impl::Capsule, capsule);
TF_UNION_ACCESS_INSTANCE(impl::Float32, f32);
TF_UNION_ACCESS_INSTANCE(impl::Int64, i64);
TF_UNION_ACCESS_INSTANCE(impl::ListPtr, list);
TF_UNION_ACCESS_INSTANCE(impl::TuplePtr, tuple);
TF_UNION_ACCESS_INSTANCE(impl::DictPtr, dict);
TF_UNION_ACCESS_INSTANCE(impl::Func, func);
TF_UNION_ACCESS_INSTANCE(impl::String, s);
TF_UNION_ACCESS_INSTANCE(impl::TaggedValueTensor, tensor);
TF_UNION_ACCESS_INSTANCE(impl::TensorSpec, tensor_spec);
#undef TF_UNION_ACCESS_INSTANCE

/// The union accessor for `NoneType`.
template <>
struct TaggedValue::UnionAccess<impl::None> {
  static impl::None& unsafe_reference(TaggedValue& t) {
    return None::GetInstance();
  }
  static const impl::None& unsafe_reference(const TaggedValue& t) {
    return None::GetInstance();
  }
};

/// @brief The Tuple class for holding tuples of TaggedValues.
/// TODO: Need to wrap vector in Tuple otherwise variant has duplicate types.
class Tuple {
  using TU = std::vector<TaggedValue>;
  using value_type = TU::value_type;
  using iterator = TU::iterator;
  using const_iterator = TU::const_iterator;
  TU values_;

 public:
  TU::iterator begin() { return values_.begin(); }
  TU::iterator end() { return values_.end(); }
  TU::const_iterator begin() const { return values_.begin(); }
  TU::const_iterator end() const { return values_.end(); }
  const TU::value_type& operator[](size_t i) const { return values_[i]; }
  TU::value_type& operator[](size_t i) { return values_[i]; }
  size_t size() const { return values_.size(); }
  void emplace_back(TaggedValue v) { values_.emplace_back(std::move(v)); }
  void push_back(const TaggedValue& v) { values_.push_back(v); }
};

/// Hashing infrastructure for Tuple.
inline size_t TaggedValueHash<Tuple>::operator()(const Tuple& t) const {
  std::size_t hash = 0;
  for (auto& i : t) {
    hash ^= TaggedValueHash<TaggedValue>()(i);
  }
  return hash;
}

/// @brief The TaggedValueHashVisitor class for doing type-based hashing
/// of TaggedValues.
class TaggedValueHashVisitor {
 public:
  size_t operator()(const TaggedValueTensor& v) {
    assert(false);
    return 0;
  }
  size_t operator()(const ListPtr& v) {
    assert(false);
    return 0;
  }
  size_t operator()(const DictPtr& v) {
    assert(false);
    return 0;
  }
  size_t operator()(const Capsule& t) { return std::hash<Capsule>()(t); }
  size_t operator()(const Func& t) {
    assert(false);
    return 0;
  }
  size_t operator()(const TuplePtr& t) {
    std::size_t hash = 0;
    for (auto it = t->begin(); it != t->end(); ++it) {
      hash ^= TaggedValueHash<TaggedValue>()(*it);
    }
    return hash;
  }
  template <class T>
  size_t operator()(const T& t) {
    return absl::Hash<T>()(t);
  }
};

/// Hashing infrastructure for TaggedValues. Hashable TaggedValues overload
/// `AbslHashValue`. Non-hashable structures return 0, since we have no easy
/// way to abort.
inline size_t TaggedValueHash<TaggedValue>::operator()(
    const TaggedValue& v) const {
  return v.visit<size_t>(TaggedValueHashVisitor());
}

}  // namespace impl
}  // namespace libtf
}  // namespace tf

#endif  // TENSORFLOW_CC_EXPERIMENTAL_LIBTF_VALUE_H_
