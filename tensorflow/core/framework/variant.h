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

#include "tensorflow/core/framework/tensor.pb.h"  // TODO(b/62899350): Remove
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
//   EXPECT_EQ(*x.get<int>(), 10);
//
//   Tensor t(DT_FLOAT, TensorShape({}));
//   t.flat<float>()(0) = 42.0f;
//   Variant x = t;
//   EXPECT_EQ(x.get<Tensor>()->flat<float>()(0), 42.0f);
//
// Accessing the stored object:
//
// The get<T> function is the main mechanism to access the object
// stored in the container. It is type-safe, that is, calling
// get<T> when the stored object's type is not T, returns a
// nullptr. A raw pointer to the stored object can be obtained by calling
// get<void>().
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
//   EXPECT_EQ(*x.get<Foo>(), *y.get<Foo>());
//
//
// A Variant storing serialized Variant data (a value of type
// VariantTensorDataProto) has different behavior from a standard Variant.
// Namely, its TypeName matches the TypeName of the original Variant;
// and its non-const get method performs lazy deserialization.
//
// Decode and copy example:
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
//   Foo f_decoded;
//   EXPECT_TRUE(x.MaybeDecodeAndCopy(&f_decoded));
//   EXPECT_EQ(f_decoded, f);
//
class Variant {
 public:
  constexpr Variant() noexcept = default;

  Variant(const Variant& other)
      : value_(other.is_empty() ? std::unique_ptr<ValueInterface>()
                                : other.value_->Clone()) {}

  Variant(Variant&& other) noexcept = default;

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

  bool is_empty() const { return value_ == nullptr; }

  void clear() noexcept { value_.reset(); }

  void swap(Variant& other) noexcept { value_.swap(other.value_); }

  // Note, unlike TypeName(), TypeId() does not return the TypeIndex
  // of the original type when a TensorValueDataProto is stored as the
  // value.  In this case, it returns the TypeIndex of TensorValueDataProto.
  TypeIndex TypeId() const {
    const TypeIndex VoidTypeIndex = MakeTypeIndex<void>();
    if (is_empty()) {
      return VoidTypeIndex;
    }
    return value_->TypeId();
  }

  string DebugString() const {
    return strings::StrCat("Variant<type: ", TypeName(),
                           " value: ", value_->DebugString(), ">");
  }

  // Returns a pointer to the stored value if it is type T, or nullptr
  // otherwise.
  template <typename T>
  T* get() {
    const TypeIndex TTypeIndex = MakeTypeIndex<T>();
    if (is_empty() || (TTypeIndex != TypeId())) return nullptr;
    return std::addressof(static_cast<Variant::Value<T>*>(value_.get())->value);
  }

  // Returns a pointer to the stored value if it is type T, or nullptr
  // otherwise.
  template <typename T>
  const T* get() const {
    const TypeIndex TTypeIndex = MakeTypeIndex<T>();
    if (is_empty() || (TTypeIndex != TypeId())) return nullptr;
    return std::addressof(
        static_cast<const Variant::Value<T>*>(value_.get())->value);
  }

  // Returns TypeNameVariant(value).
  //
  // In the special case that a serialized Variant is stored (value
  // is a VariantTensorDataProto), returns value.TypeName(), the
  // TypeName field stored in the VariantTensorDataProto buffer.
  string TypeName() const {
    if (is_empty()) {
      return "";
    }
    return value_->TypeName();
  }

  // Serialize the contents of the stored object into `data`.
  void Encode(VariantTensorData* data) const {
    if (!is_empty()) {
      value_->Encode(data);
    }
  }

  // Deserialize `data` and update the stored object.
  bool Decode(const VariantTensorData& data) {
    if (!is_empty()) {
      return value_->Decode(data);
    }
    return true;
  }

  // Helper methods to directly serialize/deserialize from strings.
  void Encode(string* buf) const {
    if (!is_empty()) {
      value_->Encode(buf);
    }
  }
  bool Decode(const string& buf) {
    if (!is_empty()) {
      return value_->Decode(buf);
    }
    return true;
  }

  template <typename T>
  bool MaybeDecodeAndCopy(T* out) const {
    const T* ret = get<T>();
    if (ret != nullptr) {
      *out = std::move(*ret);
      return true;
    };
    Variant decoded = T();
    if (!TryDecode(&decoded)) return false;
    T* decoded_ret = decoded.get<T>();
    CHECK_NOTNULL(decoded_ret);
    *out = std::move(*decoded_ret);
    return true;
  }

 private:
  bool TryDecode(Variant* out) const;

 private:
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

  // value_ can point to any type T as wrapped by a ValueInterface.
  // The only real requirement is that T is default-constructible.
  std::unique_ptr<ValueInterface> value_;
};

template <>
void* Variant::get();

template <>
const void* Variant::get() const;

}  // end namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_VARIANT_H_
