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
// Tagged union for holding values in the TensorFlow Lite C++ API.
#ifndef TENSORFLOW_CC_EXPERIMENTAL_LIBTF_VALUE_H_
#define TENSORFLOW_CC_EXPERIMENTAL_LIBTF_VALUE_H_

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/intrusive_ptr.h"
#include "tensorflow/core/platform/statusor.h"

// TODO(ccrusius): Move all value objects into `impl`. Currently only values
// that do not reference TaggedValue are there.
#include "tensorflow/cc/experimental/libtf/impl/scalars.h"
#include "tensorflow/cc/experimental/libtf/impl/string.h"

namespace tf {
namespace libtf {
namespace impl {
// Necessary forward declares.
class None {};
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
    std::function<tensorflow::StatusOr<TaggedValue>(TaggedValue, TaggedValue)>;
struct TensorSpec {
  tensorflow::PartialTensorShape shape;
  tensorflow::DataType dtype;
  bool operator==(const TensorSpec& o) const {
    return dtype == o.dtype && shape.IsIdenticalTo(o.shape);
  }
};
// A capsule holds a pointer and a destructor for the pointer (i.e. a generic
// shared_ptr to void with a custom deleter).
using Capsule = std::shared_ptr<void>;
using TaggedValueTensor =
    tensorflow::core::IntrusivePtr<tensorflow::AbstractTensorHandle>;

// Declare hash types so they can be instantiated below.
template <>
struct TaggedValueHash<TaggedValue> {
  size_t operator()(const TaggedValue& v) const;
};
template <>
struct TaggedValueHash<Tuple> {
  size_t operator()(const Tuple& t) const;
};

// Basic tagged union value type. This will include all values that we
// wish to represent. Notably tensors, primitive values, lists, tuples,
// dictionaries. The future we might also want to have representation
// of python objects in the form of PyObject*.
class TaggedValue final {
 public:
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

  TaggedValue& operator=(TaggedValue&& v) {
    destroy();
    MoveIntoUnion(std::move(v));
    return *this;
  }

  TaggedValue(TaggedValue&& v) : type_(NONE) { MoveIntoUnion(std::move(v)); }
  TaggedValue(const TaggedValue& v) : type_(NONE) { CopyIntoUnion(v); }
  TaggedValue& operator=(const TaggedValue& v) {
    destroy();
    CopyIntoUnion(v);
    return *this;
  }
  // Construction
  explicit TaggedValue(TaggedValueTensor tensor)
      : type_(TENSOR), data_(std::move(tensor)) {}
  explicit TaggedValue(tensorflow::PartialTensorShape shape,
                       tensorflow::DataType dtype)
      : type_(TENSOR_SPEC), data_(shape, dtype) {}
  explicit TaggedValue(Func f32) : type_(FUNC), data_(f32) {}
  explicit TaggedValue(float f32) : type_(FLOAT32), data_(Float32(f32)) {}
  explicit TaggedValue(int64_t i64) : type_(INT64), data_(Int64(i64)) {}
  explicit TaggedValue(Float32 f32) : type_(FLOAT32), data_(f32) {}
  explicit TaggedValue(Int64 i64) : type_(INT64), data_(i64) {}
  explicit TaggedValue(const char* s) : type_(STRING), data_(s) {}
  static TaggedValue None() {
    TaggedValue v;
    new (&v.data_.none) impl::None();
    v.type_ = NONE;
    return v;
  }
  static TaggedValue List() {
    TaggedValue v;
    v.type_ = LIST;
    using T = decltype(v.data_.list);
    new (&v.data_.list) T(std::make_shared<T::element_type>());
    return v;
  }
  static TaggedValue Tuple() {
    TaggedValue v;
    v.type_ = TUPLE;
    using T = decltype(v.data_.tuple);
    new (&v.data_.tuple) T(std::make_shared<T::element_type>());
    return v;
  }
  static TaggedValue Dict() {
    TaggedValue v;
    v.type_ = DICT;
    using T = decltype(v.data_.dict);
    new (&v.data_.dict) T(std::make_shared<T::element_type>());
    return v;
  }
  static TaggedValue Tensor(tensorflow::AbstractTensorHandle* raw_ptr) {
    TaggedValue v;
    v.type_ = TENSOR;
    using T = decltype(v.data_.tensor);
    new (&v.data_.tensor) T(raw_ptr, /*add_ref=*/false);
    return v;
  }

  // Construct a capsule with a default destructor.
  template <class T>
  static TaggedValue Capsule(T* data) {
    return Capsule(static_cast<void*>(data),
                   [](void* x) { delete static_cast<T*>(x); });
  }
  // Construct a capsule with a custom destructor.
  static TaggedValue Capsule(void* data, void (*deleter)(void*)) {
    TaggedValue v;
    v.type_ = CAPSULE;
    using T = decltype(v.data_.capsule);
    new (&v.data_.capsule) T(data, deleter);
    return v;
  }
  // Destroy tagged union properly. Shared pointers in unions must be explicitly
  // deleted.
  void destroy() {
    // Explicitly run the destructor on the correct type.
    visit<void>([](auto& x) {
      using T = typename std::decay<decltype(x)>::type;
      x.~T();
    });
    // Make the type None, whenever we destroy so we always have an initialized
    // value.
    type_ = NONE;
    new (&data_.none) impl::None();
  }
  ~TaggedValue() { destroy(); }

  /// @brief Get the underlying value based on type.
  ///
  /// @tparam T The desired return type.
  /// @return The unrwapped value. If this `TaggedValue` type does not currently
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
  /// @return The unrwapped value. If this `TaggedValue` type does not currently
  ///         contain a value of type `T`, the program terminates via a call to
  ///         `assert`.
  template <typename T>
  const T& get() const {
    assert(type_ == EnumValueOf<T>::value);
    return UnionAccess<T>::unsafe_reference(*this);
  }

  const Int64& i64() const { return get<impl::Int64>(); }

  const Float32& f32() const { return get<impl::Float32>(); }

  const char* s() const { return get<impl::String>().str().c_str(); }

  impl::List& list() { return *get<impl::ListPtr>(); }
  const impl::List& list() const { return *get<impl::ListPtr>(); }

  impl::Tuple& tuple() { return *get<impl::TuplePtr>(); }
  const impl::Tuple& tuple() const { return *get<impl::TuplePtr>(); }

  impl::Dict& dict() { return *get<impl::DictPtr>(); }
  const impl::Dict& dict() const { return *get<impl::DictPtr>(); }

  impl::Func func() const { return get<impl::Func>(); }

  // TODO(danielellis): make const-only if possible, once the API allows for it
  TaggedValueTensor& tensor() { return get<TaggedValueTensor>(); }
  const TaggedValueTensor& tensor() const { return get<TaggedValueTensor>(); }

  const TensorSpec& tensor_spec() const { return get<TensorSpec>(); }

  void* capsule() const { return get<impl::Capsule>().get(); }

  Type type() const { return type_; }

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
        // TODO(b/187536093):  This is definitely wrong beacuse the exact ptr of
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
        return visitor(data_.none);
    }
  }

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
        return visitor(data_.none);
    }
  }

 private:
  /// @brief An utility class for mapping C++ types to Type values.
  template <typename T>
  struct EnumValueOf;

  /// @brief An utility class for accessing the `Data` union members.
  template <typename T>
  struct UnionAccess;

  // Unsafe Move, because it assumes the union has already been destroyed
  // or is new!
  void MoveIntoUnion(TaggedValue&& v) {
    assert(type_ == NONE);
    type_ = v.type_;
    visit<void>([&v](auto& left) -> void {
      using T = typename std::decay<decltype(left)>::type;
      new (&left) T(std::move(UnionAccess<T>::unsafe_reference(v)));
    });
    // Destroy the source r-value reference (making it None)
    v.destroy();
  }

  // Unsafe Move, because it assumes the union has already been destroyed
  // or is new!
  void CopyIntoUnion(const TaggedValue& v) {
    assert(type_ == NONE);
    type_ = v.type_;
    visit<void>([&v](auto& left) -> void {
      using T = typename std::decay<decltype(left)>::type;
      new (&left) T(UnionAccess<T>::unsafe_reference(v));
    });
  }

  // the union type. in principle this could be incorporated into the union
  // for pointer types and non-64bit values, but then int64 and float64 values
  // would need to be indirected.  This means that we are aiming for a total
  // data type size of <=16 bytes. One pointer and one type.
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
    impl::None none;
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
TF_UNION_ACCESS_INSTANCE(impl::None, none);
TF_UNION_ACCESS_INSTANCE(impl::Func, func);
TF_UNION_ACCESS_INSTANCE(impl::String, s);
TF_UNION_ACCESS_INSTANCE(impl::TaggedValueTensor, tensor);
TF_UNION_ACCESS_INSTANCE(impl::TensorSpec, tensor_spec);
#undef TF_UNION_ACCESS_INSTANCE

// Need to wrap vector in Tuple otherwise variant has duplicate types.
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

// Hashing infrastructure
inline size_t TaggedValueHash<Tuple>::operator()(const Tuple& t) const {
  std::size_t hash = 0;
  for (auto& i : t) {
    hash ^= TaggedValueHash<TaggedValue>()(i);
  }
  return hash;
}

class TaggedValueHashVisitor {
 public:
  size_t operator()(const None& v) { return 38383827; }
  size_t operator()(const TaggedValueTensor& v) {
    assert(false);
    return 0;
  }
  size_t operator()(const TensorSpec& v) {
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

// Hashing infrastructure, considering AbslHash. non hashable structures return
// 0, since we have no easy way to abort.
inline size_t TaggedValueHash<TaggedValue>::operator()(
    const TaggedValue& v) const {
  return v.visit<size_t>(TaggedValueHashVisitor());
}

}  // namespace impl
}  // namespace libtf
}  // namespace tf

#endif  // TENSORFLOW_CC_EXPERIMENTAL_LIBTF_VALUE_H_
