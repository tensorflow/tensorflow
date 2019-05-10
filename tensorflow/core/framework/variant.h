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

#ifndef TENSORFLOW_CORE_FRAMEWORK_VARIANT_H_
#define TENSORFLOW_CORE_FRAMEWORK_VARIANT_H_

#include <functional>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "absl/memory/memory.h"
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

// Allows for specializations of Variant Decoding.  `data` may be modified in
// the process of decoding to `value`.
template <typename T>
bool DecodeVariant(VariantTensorData* data, T* value);

template <typename T>
bool DecodeVariant(string* buf, T* value);

template <typename T>
void EncodeVariant(const T& value, VariantTensorData* data);

template <typename T>
void EncodeVariant(const T& value, string* buf);

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
//   bool Decode(VariantTensorData data);
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
//   y.Decode(std::move(serialized_f));
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
//
class Variant {
 public:
  Variant() noexcept : is_inline_(false) {}

  ~Variant();

  Variant(const Variant& other);
  Variant(Variant&& other) noexcept;

  // Make sure that the type is CopyConstructible and not a
  // tensorflow::Variant object itself. We want the copy constructor to be
  // chosen for the tensorflow::Variant case.
  template <typename T, typename VT = typename std::decay<T>::type,
            typename std::enable_if<!std::is_same<Variant, VT>::value &&
                                        std::is_move_constructible<VT>::value,
                                    void>::type* = nullptr>
  Variant(T&& value);

  template <typename T, typename VT = typename std::decay<T>::type,
            typename std::enable_if<!std::is_same<Variant, VT>::value &&
                                        std::is_copy_constructible<VT>::value,
                                    void>::type* = nullptr>
  Variant(const T& value);

  template <typename T, typename VT = typename std::decay<T>::type,
            typename std::enable_if<!std::is_same<Variant, VT>::value &&
                                        std::is_copy_constructible<VT>::value,
                                    void>::type* = nullptr>
  Variant& operator=(const T& value);

  template <typename T, typename VT = typename std::decay<T>::type,
            typename std::enable_if<!std::is_same<Variant, VT>::value &&
                                        std::is_move_constructible<VT>::value,
                                    void>::type* = nullptr>
  Variant& operator=(T&& value);

  Variant& operator=(const Variant& rhs) {
    if (&rhs == this) return *this;
    Variant(rhs).swap(*this);
    return *this;
  }

  Variant& operator=(Variant&& rhs) noexcept {
    if (&rhs == this) return *this;
    Variant(std::move(rhs)).swap(*this);
    return *this;
  }

  bool is_empty() const { return GetValue() == nullptr; }

  void clear() noexcept;

  void swap(Variant& other) noexcept;

  // Note, unlike TypeName(), TypeId() does not return the TypeIndex
  // of the original type when a TensorValueDataProto is stored as the
  // value.  In this case, it returns the TypeIndex of TensorValueDataProto.
  TypeIndex TypeId() const {
    const TypeIndex VoidTypeIndex = MakeTypeIndex<void>();
    if (is_empty()) {
      return VoidTypeIndex;
    }
    return GetValue()->TypeId();
  }

  string DebugString() const {
    return strings::StrCat(
        "Variant<type: ", TypeName(),
        " value: ", is_empty() ? "[empty]" : GetValue()->DebugString(), ">");
  }

  // Returns a pointer to the stored value if it is type T, or nullptr
  // otherwise.
  template <typename T>
  T* get() {
    const TypeIndex TTypeIndex = MakeTypeIndex<T>();
    if (is_empty() || (TTypeIndex != TypeId())) return nullptr;
    return std::addressof(static_cast<Variant::Value<T>*>(GetValue())->value);
  }

  // Returns a pointer to the stored value if it is type T, or nullptr
  // otherwise.
  template <typename T>
  const T* get() const {
    const TypeIndex TTypeIndex = MakeTypeIndex<T>();
    if (is_empty() || (TTypeIndex != TypeId())) return nullptr;
    return std::addressof(
        static_cast<const Variant::Value<T>*>(GetValue())->value);
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
    return GetValue()->TypeName();
  }

  // Serialize the contents of the stored object into `data`.
  void Encode(VariantTensorData* data) const {
    if (!is_empty()) {
      GetValue()->Encode(data);
    }
  }

  // Deserialize `data` and update the stored object.
  bool Decode(VariantTensorData data);

  // Helper methods to directly serialize/deserialize from strings.
  void Encode(string* buf) const {
    if (!is_empty()) {
      GetValue()->Encode(buf);
    }
  }
  bool Decode(string buf) {
    if (!is_empty()) {
      return GetValue()->Decode(std::move(buf));
    }
    return true;
  }

  template <typename VT>
  static constexpr bool CanInlineType() {
    return ((sizeof(Value<VT>) <= InlineValue::kMaxValueSize) &&
            (alignof(Value<VT>) <= kMaxInlineValueAlignSize));
  }

 private:
  struct in_place_t {};
  static constexpr in_place_t kInPlace{};

  struct ValueInterface {
    virtual ~ValueInterface() = default;
    virtual TypeIndex TypeId() const = 0;
    virtual void* RawPtr() = 0;
    virtual const void* RawPtr() const = 0;
    virtual ValueInterface* Clone() const = 0;
    virtual void CloneInto(ValueInterface* memory) const = 0;
    virtual void Swap(ValueInterface* memory) = 0;
    virtual void MoveAssign(ValueInterface* memory) = 0;
    virtual void MoveInto(ValueInterface* memory) = 0;
    virtual string TypeName() const = 0;
    virtual string DebugString() const = 0;
    virtual void Encode(VariantTensorData* data) const = 0;
    virtual bool Decode(VariantTensorData data) = 0;
    virtual void Encode(string* buf) const = 0;
    virtual bool Decode(string data) = 0;
  };

  template <typename T>
  struct Value : ValueInterface {
    template <class... Args>
    explicit Value(in_place_t /*tag*/, Args&&... args)
        : value(std::forward<Args>(args)...) {}

    // NOTE(ebrevdo): Destructor must be explicitly defined for CUDA to happily
    // build `alignof(Variant<void*>)`.
    ~Value() final = default;

    TypeIndex TypeId() const override {
      const TypeIndex value_type_index =
          MakeTypeIndex<typename std::decay<T>::type>();
      return value_type_index;
    }

    void* RawPtr() override { return &value; }

    const void* RawPtr() const override { return &value; }

    ValueInterface* Clone() const override {
      // NOTE: Use placement new here because we override `operator delete`,
      // and need to match the call to `port::Free()` with a call to
      // `port::Malloc()`.
      auto* clone = static_cast<Value*>(port::Malloc(sizeof(Value)));
      new (clone) Value(kInPlace, value);
      return clone;
    }

    void MoveAssign(ValueInterface* memory) override {
      CHECK(TypeId() == memory->TypeId())
          << TypeId().name() << " vs. " << memory->TypeId().name();
      static_cast<Value*>(memory)->value = std::move(value);
    }

    void CloneInto(ValueInterface* memory) const override {
      new (memory) Value(kInPlace, value);
    }

    void MoveInto(ValueInterface* memory) override {
      new (memory) Value(kInPlace, std::move(value));
    }

    void Swap(ValueInterface* memory) override {
      CHECK(TypeId() == memory->TypeId())
          << TypeId().name() << " vs. " << memory->TypeId().name();
      std::swap(value, static_cast<Value*>(memory)->value);
    }

    string TypeName() const override { return TypeNameVariant(value); }

    string DebugString() const override { return DebugStringVariant(value); }

    void Encode(VariantTensorData* data) const override {
      EncodeVariant(value, data);
    }

    bool Decode(VariantTensorData data) override {
      return DecodeVariant(&data, &value);
    }

    void Encode(string* buf) const override { EncodeVariant(value, buf); }

    bool Decode(string buf) override { return DecodeVariant(&buf, &value); }

    // We override operator delete in order to selectively free memory
    // depending on if Value<VT> is stored inline or on the heap:
    //
    // Value<VT> is stored inline if its size <= InlineValue::kMaxValueSize and
    // its alignment <= kMaxInlineValueAlignSize.  This check is performed by
    // CanInlineType<VT>().
    //
    // We only need to call its destructor in this case and then overwrite
    // the inline memory with zeros.  Variant::clear() does this.
    // Thus, in the inline case, the delete operator does nothing (calling
    // delete on the memory location calls the destructor only).
    //
    // If !CanInlineType<VT>(), then it is stored as a pointer inside HeapValue.
    // The memory buffer it resides in on the heap was allocated with
    // port::Malloc, and it should be deallocated via port::Free.
    //
    // operator delete is stored in the vtable since ~ValueInterface is a
    // virtual destructor; furthermore it has access to VT and can calculate
    // CanInlineType<VT>().
    static void operator delete(void* ptr);

    static void operator delete(void*, void*) {
      // Some compilers require an overridden class-specific deallocation
      // function, which will be called if placement `new` throws an
      // exception.
    }

    T value;
  };
  static constexpr int kMaxInlineValueAlignSize = alignof(Value<void*>);

  using HeapValue = std::unique_ptr<ValueInterface>;

  struct InlineValue {
    // We try to size InlineValue so that sizeof(Variant) <= 64 and it can fit
    // into the aligned space of a TensorBuffer.
    static constexpr int kMaxValueSize = (64 - /*some extra padding=*/16);

    typedef char ValueDataArray[kMaxValueSize];
    alignas(kMaxInlineValueAlignSize) ValueDataArray value_data;
    bool has_value = false;

    explicit InlineValue() {}

    InlineValue(const InlineValue& other) noexcept
        : has_value(other.has_value) {
      if (other.has_value) {
        other.AsValueInterface()->CloneInto(AsValueInterface());
      }
    }

    InlineValue(InlineValue&& other) noexcept : has_value(other.has_value) {
      if (other.has_value) {
        other.AsValueInterface()->MoveInto(AsValueInterface());
        other.Cleanup();
      }
    }

    void Cleanup() {
      // **NOTE** This must be a no-op if the memory representation of
      // InlineValue is all zeros, in order to properly interact with
      // HeapOrInline::ResetMemory().
      if (has_value) {
        // This doesn't actually delete anything on the heap; the delete
        // operator of Value<VT> is overridden to do nothing for inline
        // values; the side-effect of delete is that the virtual destructor is
        // called.
        //
        // We leave it to callers to overwrite the data buffer in value_data
        // with new objects.
        delete AsValueInterface();
      }
      has_value = false;
    }

    InlineValue& operator=(const InlineValue& other) {
      if (&other == this) return *this;
      Cleanup();
      if (other.has_value) {
        other.AsValueInterface()->CloneInto(AsValueInterface());
      }
      has_value = other.has_value;
      return *this;
    }

    InlineValue& operator=(InlineValue&& other) {
      if (&other == this) return *this;
      if (other.has_value) {
        if (has_value && AsValueInterface()->TypeId() ==
                             other.AsValueInterface()->TypeId()) {
          other.AsValueInterface()->Swap(AsValueInterface());
        } else {
          if (has_value) {
            if (AsValueInterface()->TypeId() !=
                other.AsValueInterface()->TypeId()) {
              Cleanup();
              other.AsValueInterface()->MoveInto(AsValueInterface());
            } else {
              other.AsValueInterface()->MoveAssign(AsValueInterface());
            }
          } else {
            other.AsValueInterface()->MoveInto(AsValueInterface());
          }
          other.Cleanup();
          has_value = true;
        }
      } else {
        Cleanup();
      }
      return *this;
    }

    ValueInterface* AsValueInterface() {
      return reinterpret_cast<ValueInterface*>(value_data);
    }

    const ValueInterface* AsValueInterface() const {
      return reinterpret_cast<const ValueInterface*>(value_data);
    }

    // **WARNING** This must be a no-op when the byte-representation of
    // InlineValue is all zeros.
    ~InlineValue() { Cleanup(); }
  };

  // value_ can point to any type T as wrapped by a ValueInterface.
  // The only real requirement is that T is default-constructible.
  union HeapOrInline {
    HeapOrInline() { ResetMemory(); }
    explicit HeapOrInline(HeapValue&& v) : heap_value(std::move(v)) {}
    explicit HeapOrInline(InlineValue&& v) : inline_value(std::move(v)) {}
    ~HeapOrInline() {}  // Taken care of by owner.

    // This must be called when modifying which element of HeapOrInline is
    // being used, because the destructor of the new class may be called
    // while the memory is still a representation of the old class.
    // **WARNING** This code assumes that the destructors of HeapValue and
    // InlineValue are no-ops when the internal representation is zeros.
    //
    // Example of when this is needed:
    //   value.heap_value = HeapValue(...);
    //   // Segfault.  This calls InlineValue::Cleanup on value.inline_value
    //   // but the internal memory representation is that of HeapValue.
    //   value.inline_value = InlineValue();
    //
    //   The correct way to do this:
    //   value.heap_value = HeapValue(...);
    //   value.ResetMemory();
    //   value.inline_value = InlineValue();
    void ResetMemory();

    HeapValue heap_value;
    InlineValue inline_value;
  } value_;
  bool is_inline_;

  bool IsInlineValue() const { return is_inline_; }

  ValueInterface* GetValue() {
    if (IsInlineValue()) {
      return value_.inline_value.AsValueInterface();
    } else {
      return value_.heap_value.get();
    }
  }

  const ValueInterface* GetValue() const {
    if (IsInlineValue()) {
      return value_.inline_value.AsValueInterface();
    } else {
      return value_.heap_value.get();
    }
  }

  // PRECONDITION: Called on construction or clear() has been called before
  // this method.
  template <typename T, typename VT>
  void InsertValueMove(T&& value) {
    if (is_inline_) {
      Value<VT>* inline_value_data =
          reinterpret_cast<Value<VT>*>(value_.inline_value.value_data);
      new (inline_value_data) Value<VT>(kInPlace, std::forward<T>(value));
      value_.inline_value.has_value = true;
    } else {
      auto* moved = static_cast<Value<VT>*>(port::Malloc(sizeof(Value<VT>)));
      new (moved) Value<VT>(kInPlace, std::forward<T>(value));
      value_.heap_value = HeapValue(moved);
    }
  }

  // PRECONDITION: Called on construction or clear() has been called before
  // this method.
  template <typename T, typename VT>
  void InsertValueCopy(const T& value) {
    if (is_inline_) {
      Value<VT>* inline_value_data =
          reinterpret_cast<Value<VT>*>(value_.inline_value.value_data);
      new (inline_value_data) Value<VT>(kInPlace, value);
      value_.inline_value.has_value = true;
    } else {
      auto* moved = static_cast<Value<VT>*>(port::Malloc(sizeof(Value<VT>)));
      new (moved) Value<VT>(kInPlace, value);
      value_.heap_value = HeapValue(moved);
    }
  }
};

// Make sure that a Variant object can reside in a 64-byte aligned Tensor
// buffer.
static_assert(sizeof(Variant) <= 64,
              "Expected internal representation to be 64 bytes.");

inline Variant::Variant(const Variant& other) : is_inline_(other.is_inline_) {
  if (!other.is_empty()) {
    if (other.IsInlineValue()) {
      value_.inline_value = InlineValue();
      other.GetValue()->CloneInto(GetValue());
      value_.inline_value.has_value = true;
    } else {
      value_.heap_value = HeapValue(other.GetValue()->Clone());
      is_inline_ = false;
    }
  }
}

inline Variant::Variant(Variant&& other) noexcept
    : is_inline_(other.is_inline_) {
  if (!other.is_empty()) {
    if (other.IsInlineValue()) {
      value_.inline_value = InlineValue();
      other.GetValue()->MoveInto(GetValue());
      value_.inline_value.has_value = true;
    } else {
      value_.heap_value = std::move(other.value_.heap_value);
      is_inline_ = false;
    }
  }
}

template <typename VT>
void Variant::Value<VT>::operator delete(void* ptr) {
  if (!CanInlineType<VT>()) port::Free(ptr);
}

template <typename T, typename VT,
          typename std::enable_if<!std::is_same<Variant, VT>::value &&
                                      std::is_move_constructible<VT>::value,
                                  void>::type*>
inline Variant::Variant(T&& value) : is_inline_(CanInlineType<VT>()) {
  InsertValueMove<T, VT>(std::forward<T>(value));
}

template <typename T, typename VT,
          typename std::enable_if<!std::is_same<Variant, VT>::value &&
                                      std::is_copy_constructible<VT>::value,
                                  void>::type*>
inline Variant::Variant(const T& value) : is_inline_(CanInlineType<VT>()) {
  InsertValueCopy<T, VT>(value);
}

template <typename T, typename VT,
          typename std::enable_if<!std::is_same<Variant, VT>::value &&
                                      std::is_move_constructible<VT>::value,
                                  void>::type*>
inline Variant& Variant::operator=(T&& value) {
  clear();
  is_inline_ = CanInlineType<VT>();
  InsertValueMove<T, VT>(std::forward<T>(value));
  return *this;
}

template <typename T, typename VT,
          typename std::enable_if<!std::is_same<Variant, VT>::value &&
                                      std::is_copy_constructible<VT>::value,
                                  void>::type*>
inline Variant& Variant::operator=(const T& value) {
  clear();
  is_inline_ = CanInlineType<VT>();
  InsertValueCopy<T, VT>(value);
  return *this;
}

inline void Variant::HeapOrInline::ResetMemory() {
  memset(  // NOLINT: not TriviallyCopyable
      this, 0, sizeof(Variant::HeapOrInline));
}

inline void Variant::clear() noexcept {
  if (!is_empty()) {
    if (IsInlineValue()) {
      value_.inline_value.~InlineValue();
    } else {
      value_.heap_value.~HeapValue();
    }
    value_.ResetMemory();
  }
  is_inline_ = false;
}

inline void Variant::swap(Variant& other) noexcept {
  if (is_empty()) {
    if (other.IsInlineValue()) {
      value_.ResetMemory();
      value_.inline_value = std::move(other.value_.inline_value);
      other.value_.ResetMemory();
      other.value_.heap_value = HeapValue();
      is_inline_ = true;
      other.is_inline_ = false;
    } else {
      value_.ResetMemory();
      value_.heap_value = std::move(other.value_.heap_value);
      other.value_.ResetMemory();
      other.value_.heap_value = HeapValue();
      is_inline_ = false;
      other.is_inline_ = false;
    }
  } else if (other.is_empty()) {
    if (IsInlineValue()) {
      other.value_.ResetMemory();
      other.value_.inline_value = std::move(value_.inline_value);
      value_.ResetMemory();
      value_.heap_value = HeapValue();
      other.is_inline_ = true;
      is_inline_ = false;
    } else {
      other.value_.ResetMemory();
      other.value_.heap_value = std::move(value_.heap_value);
      value_.ResetMemory();
      value_.heap_value = HeapValue();
      other.is_inline_ = false;
      is_inline_ = false;
    }
  } else {  // Both Variants have values.
    if (other.IsInlineValue() && IsInlineValue()) {
      std::swap(value_.inline_value, other.value_.inline_value);
    } else if (!other.IsInlineValue() && !IsInlineValue()) {
      std::swap(value_.heap_value, other.value_.heap_value);
    } else if (other.IsInlineValue() && !IsInlineValue()) {
      HeapValue v = std::move(value_.heap_value);
      value_.ResetMemory();
      value_.inline_value = std::move(other.value_.inline_value);
      other.value_.ResetMemory();
      other.value_.heap_value = std::move(v);
      is_inline_ = true;
      other.is_inline_ = false;
    } else {  // !other.IsInlineValue() && IsInlineValue()
      HeapValue v = std::move(other.value_.heap_value);
      other.value_.ResetMemory();
      other.value_.inline_value = std::move(value_.inline_value);
      value_.ResetMemory();
      value_.heap_value = std::move(v);
      is_inline_ = false;
      other.is_inline_ = true;
    }
  }
}

template <>
void* Variant::get();

template <>
const void* Variant::get() const;

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_VARIANT_H_
