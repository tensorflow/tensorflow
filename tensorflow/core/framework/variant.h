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

#ifndef TENSORFLOW_FRAMEWORK_VARIANT_H_
#define TENSORFLOW_FRAMEWORK_VARIANT_H_

#include <functional>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

template <typename T>
string TypeNameVariant(const T& value);

template <typename T>
string DebugStringVariant(const T& value);

template <typename T>
void EncodeVariant(const T& value, VariantTensorData* data);

template <typename T>
bool DecodeVariant(const VariantTensorData& data, T* value);

template <typename T>
void EncodeVariant(const T& value, string* buf);

template <typename T>
bool DecodeVariant(const string& buf, T* value);

// This is an implementation of a type-erased container that can store an
// object of any type. The implementation is very similar to std::any, but has
// restrictions on the types of objects that can be stored, and eschews some of
// the fancier constructors available for std::any. An object of
// tensorflow::Variant is intended to be used as the value that will be stored
// in a tensorflow::Tensor object when its type is DT_VARIANT.
//
// tensorflow::Variant can store an object of a class that satisfies the
// following constraints:
//
// * The class is CopyConstructible.
// * The class has a default constructor.
// * It's either a protocol buffer, a tensorflow::Tensor, or defines the
// following functions:
//
//   string TypeName() const;
//   void Encode(VariantTensorData* data) const;
//   void Decode(const VariantTensorData& data);
//
// Simple POD types can elide the Encode/Decode functions, they are provided by
// helper methods.
// Here are some typical usage patterns:
//
//   Variant x = 10;
//   EXPECT_EQ(*x.MaybeDecodeAndGet<int>(), 10);
//
//   Tensor t(DT_FLOAT, TensorShape({}));
//   t.flat<float>()(0) = 42.0f;
//   Variant x = t;
//   EXPECT_EQ(x.MaybeDecodeAndGet<Tensor>()->flat<float>()(0), 42.0f);
//
// Accessing the stored object:
//
// The MaybeDecodeAndGet<T> function is the main mechanism to access the object
// stored in the container. It is type-safe, that is, calling
// MaybeDecodeAndGet<T> when the stored object's type is not T, returns a
// nullptr. A raw pointer to the stored object can be obtained by calling
// MaybeDecodeAndGet<void>().
//
// Serializing/deserializing Variant object:
//
// The Variant class delegates serializing and deserializing operations to the
// contained object. Helper functions to do these operations are provided for
// POD data types, tensorflow::Tensor, and protocol buffer objects. However,
// other classes have to provide Encode/Decode functions to handle
// serialization.
//
// Objects stored in a Variant object often contain references to other
// tensorflow::Tensors of primitive types (Eg., a list of tensorflow::Tensors).
// To efficiently support those use cases, a structure is imposed on the
// serialization format. Namely, classes should serialize their contents into a
// VariantTensorData object:
//
//   struct VariantTensorData {
//     string type_name;
//     string metadata;
//     std::vector<Tensor> tensors;
//   };
//
// Objects with references to other Tensors can simply store those tensors in
// the `tensors` field, and serialize other metadata content in to the
// `metadata` field.
//
// Serialization example:
//
//   Foo f = Foo {...};
//   Variant x = f;
//   string serialized_f;
//   x.Encode(&serialized_f);
//
//   Variant y = Foo(); // default constructed Foo.
//   y.Decode(&serialized_f);
//   EXPECT_EQ(*x.MaybeDecodeAndGet<Foo>(), *y.MaybeDecodeAndGet<Foo>());
//
//
// A Variant storing serialized Variant data (a value of type
// VariantTensorDataProto) has different behavior from a standard Variant.
// Namely, its TypeName matches the TypeName of the original Variant;
// and its non-const get method performs lazy deserialization.
//
// Serialization with lazy decoding example:
//
//   Foo f = Foo {...};
//   Variant x = f;
//
//   VariantTensorData serialized_data_f;
//   VariantTensorDataProto serialized_proto_f;
//   x.Encode(&serialized_data_f);
//   serialized_data_f.ToProto(&serialized_proto_f);
//
//   Variant y_type_unknown = serialized_proto_f;  // Store serialized Variant.
//
//   EXPECT_EQ(x.TypeName(), y_type_unknown.TypeName());  // Looks like Foo.
//   EXPECT_EQ(MakeTypeIndex<VariantTensorDataProto>(),
//             y_type_unknown.TypeId());
//   // Decode and get y_type_unknown; compare to value in x.
//   EXPECT_EQ(*x.MaybeDecodeAndGet<Foo>(),
//   *y_type_unknown.MaybeDecodeAndGet<Foo>());
//
// The deserializing MaybeDecodeAndGet() call updates the internal
// representation:
//
//   EXPECT_EQ(MakeTypeIndex<Foo>(), y_type_unknown.TypeId());
//
class Variant {
 public:
  Variant() noexcept {}

  Variant(const Variant& other) {
    mutex_lock other_lock(other.mu_);
    if (other.IsEmptyLocked()) {
      value_ = std::unique_ptr<ValueInterface>();
    } else {
      value_ = other.value_->Clone();
    }
  }

  Variant(Variant&& other) noexcept {
    mutex_lock other_lock(other.mu_);
    value_ = std::move(other.value_);
  }

  // Make sure that the type is CopyConstructible and not a tensorflow::Variant
  // object itself. We want the copy constructor to be chosen for the
  // tensorflow::Variant case.
  template <typename T, typename VT = typename std::decay<T>::type,
            typename std::enable_if<!std::is_same<Variant, VT>::value &&
                                        std::is_copy_constructible<VT>::value,
                                    void>::type* = nullptr>
  Variant(T&& value)  // NOLINT
      : value_(new Value<VT>(in_place, std::forward<T>(value))) {}

  Variant& operator=(const Variant& rhs) {
    Variant(rhs).swap(*this);
    return *this;
  }

  Variant& operator=(Variant&& rhs) noexcept {
    Variant(std::move(rhs)).swap(*this);
    return *this;
  }

  bool is_empty() const {
    mutex_lock lock(mu_);
    return IsEmptyLocked();
  }

  void clear() noexcept {
    mutex_lock lock(mu_);
    value_.reset();
  }

  void swap(Variant& other) noexcept NO_THREAD_SAFETY_ANALYSIS {
    if (this == &other) return;
    mutex_lock lock0(this < &other ? mu_ : other.mu_);
    mutex_lock lock1(this < &other ? other.mu_ : mu_);
    value_.swap(other.value_);
  }

  // Note, unlike TypeName(), TypeId() does not return the TypeIndex
  // of the original type when a TensorValueDataProto is stored as the
  // value.  In this case, it returns the TypeIndex of TensorValueDataProto.
  TypeIndex TypeId() const {
    mutex_lock lock(mu_);
    return TypeIdLocked();
  }

  string DebugString() const {
    mutex_lock lock(mu_);
    return strings::StrCat("Variant<type: ", TypeNameLocked(),
                           " value: ", value_->DebugString(), ">");
  }

  // Returns a pointer to the stored value if it is type T, or nullptr
  // otherwise.
  //
  // In the special case that a serialized Variant is stored (the
  // value is a VariantTensorDataProto),
  // MaybeDecodeAndGet<VariantTensorDataProto>() is disallowed.  Instead, the
  // original pre-serialized type must be used to access the data:
  // MaybeDecodeAndGet<ORIGINAL_TYPE>().  This in turn performs a decode of the
  // serialized data and mutates the variant to hold the deserialized form.  A
  // pointer to this instance is returned.
  template <typename T>
  T* MaybeDecodeAndGet() {
    static_assert(
        !std::is_same<typename std::decay<T>::type,
                      VariantTensorDataProto>::value,
        "MaybeDecodeAndGet<VariantTensorDataProto> is disallowed due to the "
        "possibility of race conditions when other threads call a "
        "mutating MaybeDecodeAndGet<ORIGINAL_TYPE>.  Please access the the "
        "value via MaybeDecodeAndGet<ORIGINAL_TYPE>.");
    mutex_lock lock(mu_);
    return GetLocked<T>();
  }

  // Returns a pointer to the stored value if it is type T, or nullptr
  // otherwise.
  //
  // In the special case that a serialized Variant is stored (the
  // value is a VariantTensorDataProto), access to the data is not
  // permitted.
  template <typename T>
  const T* MaybeDecodeAndGet() const {
    static_assert(
        !std::is_same<typename std::decay<T>::type,
                      VariantTensorDataProto>::value,
        "MaybeDecodeAndGet<VariantTensorDataProto> is disallowed due to the "
        "possibility of race conditions when other threads call a "
        "mutating MaybeDecodeAndGet<ORIGINAL_TYPE>.  Please access the the "
        "value via MaybeDecodeAndGet<ORIGINAL_TYPE>.");
    mutex_lock lock(mu_);
    if (IsEmptyLocked()) {
      return nullptr;
    }
    const TypeIndex TTypeIndex = MakeTypeIndex<T>();
    if (TTypeIndex != TypeIdLocked()) {
      CHECK(TypeIdLocked() != MakeTypeIndex<VariantTensorDataProto>())
          << ": Cannot call MaybeDecodeAndGet on const Variant holding "
             "serialized data. Access a non-const version of this object if "
             "you wish to access data stored inside it.";
      return nullptr;
    }
    return std::addressof(
        static_cast<const Variant::Value<T>*>(value_.get())->value);
  }

  // Returns TypeNameVariant(value).
  //
  // In the special case that a serialized Variant is stored (value
  // is a VariantTensorDataProto), returns value.TypeName(), the
  // TypeName field stored in the VariantTensorDataProto buffer.
  string TypeName() const {
    mutex_lock lock(mu_);
    return TypeNameLocked();
  }

  // Serialize the contents of the stored object into `data`.
  void Encode(VariantTensorData* data) const {
    mutex_lock lock(mu_);
    if (!IsEmptyLocked()) {
      value_->Encode(data);
    }
  }

  // Deserialize `data` and update the stored object.
  bool Decode(const VariantTensorData& data) {
    mutex_lock lock(mu_);
    return DecodeLocked(data);
  }

  // Helper methods to directly serialize/deserialize from strings.
  void Encode(string* buf) const {
    mutex_lock lock(mu_);
    if (!IsEmptyLocked()) {
      value_->Encode(buf);
    }
  }
  bool Decode(const string& buf) {
    mutex_lock lock(mu_);
    if (!IsEmptyLocked()) {
      return value_->Decode(buf);
    }
    return true;
  }

 private:
  bool IsEmptyLocked() const EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return value_ == nullptr;
  }

  TypeIndex TypeIdLocked() const EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    const TypeIndex VoidTypeIndex = MakeTypeIndex<void>();
    if (IsEmptyLocked()) {
      return VoidTypeIndex;
    }
    return value_->TypeId();
  }

  template <typename T, typename VT = typename std::decay<T>::type>
  T* GetLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (IsEmptyLocked()) {
      return nullptr;
    }
    const TypeIndex TTypeIndex = MakeTypeIndex<T>();
    if (TTypeIndex != TypeIdLocked()) {
      if (TypeIdLocked() == MakeTypeIndex<VariantTensorDataProto>()) {
        if (!OverrideAssignDecodeFromTensorDataLocked<VT>()) return nullptr;
      } else {
        return nullptr;
      }
    }
    return std::addressof(static_cast<Variant::Value<T>*>(value_.get())->value);
  }

  string TypeNameLocked() const EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (IsEmptyLocked()) {
      return "";
    }
    return value_->TypeName();
  }

  bool DecodeLocked(const VariantTensorData& data)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (!IsEmptyLocked()) {
      return value_->Decode(data);
    }
    return true;
  }

  // Accesses the VariantTensorDataProto object stored in value_.
  // If TypeNameVariant(VT()) == value_.TypeName() then attempts to
  // decode the VariantTensorDataProto and store the result in value_.
  // If the Decode succeeds, returns true.  Otherwise returns false.
  // If TypeNameVariant(VT()) != value_.TypeName() returns false.
  //
  // Requires:
  //   value_ is not empty and contains a VariantTensorDataProto.
  template <typename VT>
  bool OverrideAssignDecodeFromTensorDataLocked()
      EXCLUSIVE_LOCKS_REQUIRED(mu_) TF_MUST_USE_RESULT;

  struct in_place_t {};
  static constexpr in_place_t in_place{};

  struct ValueInterface {
    virtual ~ValueInterface() = default;
    virtual TypeIndex TypeId() const = 0;
    virtual void* RawPtr() = 0;
    virtual const void* RawPtr() const = 0;
    virtual std::unique_ptr<ValueInterface> Clone() const = 0;
    virtual string TypeName() const = 0;
    virtual string DebugString() const = 0;
    virtual void Encode(VariantTensorData* data) const = 0;
    virtual bool Decode(const VariantTensorData& data) = 0;
    virtual void Encode(string* buf) const = 0;
    virtual bool Decode(const string& data) = 0;
  };

  template <typename T>
  struct Value : ValueInterface {
    template <class... Args>
    explicit Value(in_place_t /*tag*/, Args&&... args)
        : value(std::forward<Args>(args)...) {}

    TypeIndex TypeId() const override {
      const TypeIndex value_type_index =
          MakeTypeIndex<typename std::decay<T>::type>();
      return value_type_index;
    }

    void* RawPtr() override { return &value; }

    const void* RawPtr() const override { return &value; }

    std::unique_ptr<ValueInterface> Clone() const override {
      return std::unique_ptr<ValueInterface>(new Value(in_place, value));
    }

    string TypeName() const override { return TypeNameVariant(value); }

    string DebugString() const override { return DebugStringVariant(value); }

    void Encode(VariantTensorData* data) const override {
      EncodeVariant(value, data);
    }

    bool Decode(const VariantTensorData& data) override {
      return DecodeVariant(data, &value);
    }

    void Encode(string* buf) const override { EncodeVariant(value, buf); }

    bool Decode(const string& buf) override {
      return DecodeVariant(buf, &value);
    }

    T value;
  };

  mutable mutex mu_;

  // value_ can point to any type T as wrapped by a ValueInterface.
  // The only real requirement is that T is default-constructible.
  // Note: if T is a VariantTensorDataProto, then the behavior of
  // Variant is slightly altered (see discussion at the top).
  std::unique_ptr<ValueInterface> value_ GUARDED_BY(mu_);
};

template <>
void* Variant::MaybeDecodeAndGet();

template <>
const void* Variant::MaybeDecodeAndGet() const;

template <typename VT>
bool Variant::OverrideAssignDecodeFromTensorDataLocked() {
  // Attempt to decode the value.
  const VariantTensorDataProto* data_proto =
      GetLocked<VariantTensorDataProto>();
  CHECK_NOTNULL(data_proto);
  const VariantTensorData data(*data_proto);
  std::unique_ptr<ValueInterface> candidate_value(
      new Value<VT>(in_place, VT()));
  if (value_->TypeName() != TypeNameLocked()) {
    return false;
  }
  if (!candidate_value->Decode(data)) {
    return false;
  }
  // This deletes the pointer to the original VariantTensorDataProto
  // and is the reason users are not allowed to call
  // MaybeDecodeAndGet<VariantTensorDataProto> directly (as it may result in a
  // race condition where they are left with a hanging pointer).
  value_ = std::move(candidate_value);
  return true;
}

}  // end namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_VARIANT_H_
