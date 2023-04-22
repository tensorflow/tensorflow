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
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/intrusive_ptr.h"
#include "tensorflow/core/platform/statusor.h"

namespace tf {
namespace libtf {
namespace impl {
// Necessary forward declares.
class None {};
class TaggedValue;
class Tuple;
template <class T>
class TaggedValueHash;
using Float32 = float;
using Int64 = int64_t;
using String = const char*;
using List = std::vector<TaggedValue>;
using ListPtr = std::shared_ptr<List>;
using Dict =
    std::unordered_map<TaggedValue, TaggedValue, TaggedValueHash<TaggedValue>>;
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
using WeakCapsule = std::weak_ptr<void>;
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

// Intern a string `s` into char* that is unique per sequence of characters.
const char* InternString(const char* s);

// Identity template
template <class T>
using IdentityHelper = T;

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
    WEAK_CAPSULE = 11,
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
  static TaggedValue WeakCapsule(TaggedValue capsule) {
    TaggedValue v;
    v.type_ = WEAK_CAPSULE;
    using T = decltype(v.data_.weak_capsule);
    switch (capsule.type_) {
      case CAPSULE:
        new (&v.data_.weak_capsule) T(capsule.data_.capsule);
        break;
      case WEAK_CAPSULE:
        new (&v.data_.weak_capsule) T(capsule.data_.weak_capsule);
        break;
      default:
        return v = TaggedValue::None();
    }
    return v;
  }
  // Destroy tagged union properly. Shared pointers in unions must be explicitly
  // deleted.
  void destroy() {
    // Explicitly run the destructor on the correct type.
    visit<void>([](auto& x) {
      using T =
          IdentityHelper<typename std::remove_reference<decltype(x)>::type>;
      x.~T();
    });
    // Make the type None, whenever we destroy so we always have an initialized
    // value.
    type_ = NONE;
    new (&data_.none) impl::None();
  }
  ~TaggedValue() { destroy(); }

  Int64& i64() {
    assert(type_ == INT64);
    return data_.i64;
  }
  const Int64& i64() const {
    assert(type_ == INT64);
    return data_.i64;
  }
  Float32& f32() {
    assert(type_ == FLOAT32);
    return data_.f32;
  }
  const Float32& f32() const {
    assert(type_ == FLOAT32);
    return data_.f32;
  }
  const char* s() const {
    assert(type_ == STRING);
    return data_.s;
  }
  impl::List& list() {
    assert(type_ == LIST);
    return *data_.list;
  }
  const impl::List& list() const {
    assert(type_ == LIST);
    return *data_.list;
  }
  impl::Tuple& tuple() {
    assert(type_ == TUPLE);
    return *data_.tuple;
  }
  const impl::Tuple& tuple() const {
    assert(type_ == TUPLE);
    return *data_.tuple;
  }
  impl::Dict& dict() {
    assert(type_ == DICT);
    return *data_.dict;
  }
  const impl::Dict& dict() const {
    assert(type_ == DICT);
    return *data_.dict;
  }
  impl::Func func() const {
    assert(type_ == FUNC);
    return data_.func;
  }
  // TODO(danielellis): make const-only if possible, once the API allows for it
  TaggedValueTensor& tensor() {
    assert(type_ == TENSOR);
    return data_.tensor;
  }
  const TaggedValueTensor& tensor() const {
    assert(type_ == TENSOR);
    return data_.tensor;
  }
  const TensorSpec& tensor_spec() const {
    assert(type_ == TENSOR_SPEC);
    return data_.tensor_spec;
  }
  void* capsule() const {
    assert(type_ == CAPSULE);
    return data_.capsule.get();
  }
  std::shared_ptr<void> weak_capsule() const {
    // Instead of making this function, allow the user to access the weak
    // capsule by constructing a strong capsule again with a static constructor
    // by taking the weak capsule as a tagged value.
    assert(type_ == WEAK_CAPSULE);
    return data_.weak_capsule.lock();
  }

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
      case WEAK_CAPSULE:
        return data_.weak_capsule.lock() == o.data_.weak_capsule.lock();
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
      case WEAK_CAPSULE:
        return visitor(data_.weak_capsule);
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
      case WEAK_CAPSULE:
        return visitor(data_.weak_capsule);
      case NONE:
        return visitor(data_.none);
    }
  }

  // get tagged value by type. Returns null if the type does not match.
  template <class R>
  R* get_if();
  template <class R>
  const R* get_if() const;

 private:
  // Unsafe Move, because it assumes the union has already been destroyed
  // or is new!
  void MoveIntoUnion(TaggedValue&& v) {
    assert(type_ == NONE);
    type_ = v.type_;
    visit<void>([&v](auto& left) {
      using T = typename std::remove_reference<decltype(left)>::type;
      new (&left) IdentityHelper<T>(std::move(*v.get_if<T>()));
    });
    // Destroy the source r-value reference (making it None)
    v.destroy();
  }

  // Unsafe Move, because it assumes the union has already been destroyed
  // or is new!
  void CopyIntoUnion(const TaggedValue& v) {
    assert(type_ == NONE);
    type_ = v.type_;
    visit<void>([&v](auto& left) {
      using T = typename std::remove_reference<decltype(left)>::type;
      new (&left) IdentityHelper<T>(*v.get_if<T>());
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
    explicit Data(const char* s) : s(InternString(s)) {}
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
    const char* s;
    Func func;
    // TODO(aselle): look at tensorflow thing
    std::shared_ptr<impl::Dict> dict;
    std::shared_ptr<impl::List> list;
    std::shared_ptr<impl::Tuple> tuple;
    impl::Capsule capsule;
    impl::WeakCapsule weak_capsule;
    TaggedValueTensor tensor;
    impl::None none;
    TensorSpec tensor_spec;
  } data_;
  friend std::ostream& operator<<(std::ostream& o, const TaggedValue& v);
  friend TaggedValueHash<TaggedValue>;
  template <class T>
  friend class GetHelper;
};

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
  size_t operator()(const char* v) { return reinterpret_cast<size_t>(v); }
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
  size_t operator()(const WeakCapsule& t) {
    return std::hash<Capsule>()(t.lock());
  }
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
    return std::hash<T>()(t);
  }
};

// Hashing infrastructure, considering AbslHash. non hashable structures return
// 0, since we have no easy way to abort.
inline size_t TaggedValueHash<TaggedValue>::operator()(
    const TaggedValue& v) const {
  return v.visit<size_t>(TaggedValueHashVisitor());
}

#define TF_TAG_MATCH(cpp_type, enum_type, member)        \
  template <>                                            \
  class GetHelper<cpp_type> {                            \
   public:                                               \
    cpp_type* operator()(TaggedValue& v) {               \
      return v.type_ == enum_type ? &v.member : nullptr; \
    }                                                    \
    const cpp_type* operator()(const TaggedValue& v) {   \
      return v.type_ == enum_type ? &v.member : nullptr; \
    }                                                    \
  };

template <class T>
class GetHelper {};
TF_TAG_MATCH(impl::Capsule, TaggedValue::CAPSULE, data_.capsule);
TF_TAG_MATCH(impl::WeakCapsule, TaggedValue::WEAK_CAPSULE, data_.weak_capsule);
TF_TAG_MATCH(impl::Float32, TaggedValue::FLOAT32, data_.f32);
TF_TAG_MATCH(impl::Int64, TaggedValue::INT64, data_.i64);
TF_TAG_MATCH(impl::ListPtr, TaggedValue::LIST, data_.list);
TF_TAG_MATCH(impl::TuplePtr, TaggedValue::TUPLE, data_.tuple);
TF_TAG_MATCH(impl::DictPtr, TaggedValue::DICT, data_.dict);
TF_TAG_MATCH(impl::None, TaggedValue::NONE, data_.none);
TF_TAG_MATCH(impl::Func, TaggedValue::FUNC, data_.func);
TF_TAG_MATCH(impl::String, TaggedValue::STRING, data_.s);
TF_TAG_MATCH(impl::TaggedValueTensor, TaggedValue::TENSOR, data_.tensor);
TF_TAG_MATCH(impl::TensorSpec, TaggedValue::TENSOR_SPEC, data_.tensor_spec);

template <class R>
R* TaggedValue::get_if() {
  return GetHelper<R>()(*this);
}
template <class R>
const R* TaggedValue::get_if() const {
  return GetHelper<R>()(*this);
}

}  // namespace impl
}  // namespace libtf
}  // namespace tf

#endif  // TENSORFLOW_CC_EXPERIMENTAL_LIBTF_VALUE_H_
