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
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {

template <typename T>
std::string TypeNameVariant(const T& value);

template <typename T>
std::string DebugStringVariant(const T& value);

// Allows for specializations of Variant Decoding.  `data` may be modified in
// the process of decoding to `value`.
template <typename T>
bool DecodeVariant(VariantTensorData* data, T* value);

template <typename T>
bool DecodeVariant(std::string* buf, T* value);

template <typename T>
void EncodeVariant(const T& value, VariantTensorData* data);

template <typename T>
void EncodeVariant(const T& value, std::string* buf);

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
//   EXPECT_EQ(TypeIndex::Make<VariantTensorDataProto>(),
//             y_type_unknown.TypeId());
//
class Variant {
 public:
  // Constructs a Variant holding no value (aka `is_empty()`).
  //
  // This is done by pointing at nullptr via the heap value.
  Variant() noexcept : heap_value_(/*pointer=*/nullptr), is_inline_(false) {}

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

  // Constructs a value of type T with the given args in-place in this Variant.
  // Returns a reference to the newly constructed value.
  // The signature is based on std::variant<Types...>::emplace() in C++17.
  template <typename T, class... Args>
  T& emplace(Args&&... args) {
    ResetMemory();
    is_inline_ = CanInlineType<T>();
    if (is_inline_) {
      new (&inline_value_)
          InlineValue(InlineValue::Tag<T>{}, std::forward<Args>(args)...);
      return static_cast<Variant::Value<T>*>(inline_value_.AsValueInterface())
          ->value;
    } else {
      new (&heap_value_) HeapValue(
          absl::make_unique<Value<T>>(InPlace(), std::forward<Args>(args)...));
      return static_cast<Variant::Value<T>*>(heap_value_.get())->value;
    }
  }

  bool is_empty() const { return GetValue() == nullptr; }

  void clear() noexcept;

  void swap(Variant& other) noexcept;

  // Note, unlike TypeName(), TypeId() does not return the TypeIndex
  // of the original type when a TensorValueDataProto is stored as the
  // value.  In this case, it returns the TypeIndex of TensorValueDataProto.
  TypeIndex TypeId() const {
    const TypeIndex VoidTypeIndex = TypeIndex::Make<void>();
    if (is_empty()) {
      return VoidTypeIndex;
    }
    return GetValue()->TypeId();
  }

  std::string DebugString() const {
    return strings::StrCat("Variant<type: ", TypeName(),
                           " value: ", SummarizeValue(), ">");
  }

  std::string SummarizeValue() const {
    return is_empty() ? "[empty]" : GetValue()->DebugString();
  }

  // Returns a pointer to the stored value if it is type T, or nullptr
  // otherwise.
  template <typename T>
  T* get() {
    const TypeIndex TTypeIndex = TypeIndex::Make<T>();
    if (is_empty() || (TTypeIndex != TypeId())) return nullptr;
    return std::addressof(static_cast<Variant::Value<T>*>(GetValue())->value);
  }

  // Returns a pointer to the stored value if it is type T, or nullptr
  // otherwise.
  template <typename T>
  const T* get() const {
    const TypeIndex TTypeIndex = TypeIndex::Make<T>();
    if (is_empty() || (TTypeIndex != TypeId())) return nullptr;
    return std::addressof(
        static_cast<const Variant::Value<T>*>(GetValue())->value);
  }

  // Returns TypeNameVariant(value).
  //
  // In the special case that a serialized Variant is stored (value
  // is a VariantTensorDataProto), returns value.TypeName(), the
  // TypeName field stored in the VariantTensorDataProto buffer.
  std::string TypeName() const {
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
  void Encode(std::string* buf) const {
    if (!is_empty()) {
      GetValue()->Encode(buf);
    }
  }
  bool Decode(std::string buf) {
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
  static constexpr in_place_t InPlace() { return in_place_t{}; }

  struct ValueInterface {
    virtual ~ValueInterface() = default;
    virtual TypeIndex TypeId() const = 0;
    virtual void* RawPtr() = 0;
    virtual const void* RawPtr() const = 0;
    virtual std::unique_ptr<ValueInterface> Clone() const = 0;
    virtual void CloneInto(ValueInterface* memory) const = 0;
    virtual void MoveAssign(ValueInterface* memory) = 0;
    virtual void MoveInto(ValueInterface* memory) = 0;
    virtual std::string TypeName() const = 0;
    virtual std::string DebugString() const = 0;
    virtual void Encode(VariantTensorData* data) const = 0;
    virtual bool Decode(VariantTensorData data) = 0;
    virtual void Encode(std::string* buf) const = 0;
    virtual bool Decode(std::string data) = 0;
  };

  template <typename T>
  struct Value final : ValueInterface {
    template <class... Args>
    explicit Value(in_place_t /*tag*/, Args&&... args)
        : value(std::forward<Args>(args)...) {}

    // NOTE(ebrevdo): Destructor must be explicitly defined for CUDA to happily
    // build `alignof(Variant<void*>)`.
    ~Value() final = default;

    TypeIndex TypeId() const final {
      const TypeIndex value_type_index =
          TypeIndex::Make<typename std::decay<T>::type>();
      return value_type_index;
    }

    void* RawPtr() final { return &value; }

    const void* RawPtr() const final { return &value; }

    std::unique_ptr<ValueInterface> Clone() const final {
      return absl::make_unique<Value>(InPlace(), value);
    }

    void MoveAssign(ValueInterface* memory) final {
      CHECK(TypeId() == memory->TypeId())
          << TypeId().name() << " vs. " << memory->TypeId().name();
      static_cast<Value*>(memory)->value = std::move(value);
    }

    void CloneInto(ValueInterface* memory) const final {
      new (memory) Value(InPlace(), value);
    }

    void MoveInto(ValueInterface* memory) final {
      new (memory) Value(InPlace(), std::move(value));
    }

    std::string TypeName() const final { return TypeNameVariant(value); }

    std::string DebugString() const final { return DebugStringVariant(value); }

    void Encode(VariantTensorData* data) const final {
      EncodeVariant(value, data);
    }

    bool Decode(VariantTensorData data) final {
      return DecodeVariant(&data, &value);
    }

    void Encode(std::string* buf) const final { EncodeVariant(value, buf); }

    bool Decode(std::string buf) final { return DecodeVariant(&buf, &value); }

    T value;
  };
  static constexpr int kMaxInlineValueAlignSize = alignof(Value<void*>);

  using HeapValue = std::unique_ptr<ValueInterface>;

  struct InlineValue {
    // We try to size InlineValue so that sizeof(Variant) <= 64 and it can fit
    // into the aligned space of a TensorBuffer.
    static constexpr int kMaxValueSize = (64 - /*some extra padding=*/8);

    typedef char ValueDataArray[kMaxValueSize];
    alignas(kMaxInlineValueAlignSize) ValueDataArray value_data;

    // Tag is used for deducing the right type when constructing a Value in
    // place.
    template <typename VT>
    struct Tag {};

    template <typename VT, class... Args>
    explicit InlineValue(Tag<VT> /*tag*/, Args&&... args) noexcept {
      Value<VT>* inline_value_data = reinterpret_cast<Value<VT>*>(value_data);
      new (inline_value_data) Value<VT>(InPlace(), std::forward<Args>(args)...);
    }

    InlineValue(const InlineValue& other) noexcept {
      other.AsValueInterface()->CloneInto(AsValueInterface());
    }

    InlineValue(InlineValue&& other) noexcept {
      other.AsValueInterface()->MoveInto(AsValueInterface());
    }

    void ResetMemory() { AsValueInterface()->~ValueInterface(); }

    InlineValue& operator=(const InlineValue& other) {
      if (&other == this) return *this;
      ResetMemory();
      other.AsValueInterface()->CloneInto(AsValueInterface());
      return *this;
    }

    InlineValue& operator=(InlineValue&& other) {
      if (&other == this) return *this;
      if (AsValueInterface()->TypeId() == other.AsValueInterface()->TypeId()) {
        other.AsValueInterface()->MoveAssign(AsValueInterface());
      } else {
        ResetMemory();
        other.AsValueInterface()->MoveInto(AsValueInterface());
      }
      return *this;
    }

    ValueInterface* AsValueInterface() {
      return reinterpret_cast<ValueInterface*>(value_data);
    }

    const ValueInterface* AsValueInterface() const {
      return reinterpret_cast<const ValueInterface*>(value_data);
    }

    ~InlineValue() { ResetMemory(); }
  };

  union {
    HeapValue heap_value_;
    InlineValue inline_value_;
  };
  // is_inline_ provides discrimination between which member of the prior union
  // is currently within it's lifetime. To switch from one member to the other,
  // the destructor must be called on the currently alive member before calling
  // the constructor on the other member. In effect, a member is expected to be
  // live at any given time and that member is tracked via this boolean.
  bool is_inline_;

  bool IsInlineValue() const { return is_inline_; }

  // ResetMemory causes the destructor of the currently active member of the
  // union to be run. This must be follwed with a placement new call on the
  // member whose lifetime is to start. Additionally, is_inline_ needs to be set
  // accordingly. ResetAndSetInline and ResetAndSetHeap are simple helper
  // functions for performing the actions that are required to follow.
  void ResetMemory() {
    if (IsInlineValue()) {
      inline_value_.~InlineValue();
    } else {
      heap_value_.~HeapValue();
    }
  }

  // ResetAndSetInline clears the current state and then constructs a new value
  // inline with the provided arguments.
  template <typename... Args>
  void ResetAndSetInline(Args&&... args) noexcept {
    ResetMemory();
    new (&inline_value_) InlineValue(std::forward<Args>(args)...);
    is_inline_ = true;
  }

  // ResetAndSetHeap clears the current state then constructs a new value on the
  // heap with the provided arguments.
  template <typename... Args>
  void ResetAndSetHeap(Args&&... args) noexcept {
    ResetMemory();
    new (&heap_value_) HeapValue(std::forward<Args>(args)...);
    is_inline_ = false;
  }

  ValueInterface* GetValue() {
    if (IsInlineValue()) {
      return inline_value_.AsValueInterface();
    } else {
      return heap_value_.get();
    }
  }

  const ValueInterface* GetValue() const {
    if (IsInlineValue()) {
      return inline_value_.AsValueInterface();
    } else {
      return heap_value_.get();
    }
  }

  // PRECONDITION: Called on construction or ResetMemory() has been called
  // before this method.
  template <typename VT, typename T>
  void InsertValue(T&& value) {
    if (IsInlineValue()) {
      new (&inline_value_)
          InlineValue(InlineValue::Tag<VT>{}, std::forward<T>(value));
    } else {
      new (&heap_value_) HeapValue(
          absl::make_unique<Value<VT>>(InPlace(), std::forward<T>(value)));
    }
  }
};

// Make sure that a Variant object can reside in a 64-byte aligned Tensor
// buffer.
static_assert(sizeof(Variant) <= 64,
              "Expected internal representation to be 64 bytes.");

inline Variant::Variant(const Variant& other)
    : is_inline_(other.IsInlineValue()) {
  if (IsInlineValue()) {
    new (&inline_value_) InlineValue(other.inline_value_);
  } else {
    new (&heap_value_)
        HeapValue(other.heap_value_ ? other.heap_value_->Clone() : nullptr);
  }
}

inline Variant::Variant(Variant&& other) noexcept
    : is_inline_(other.IsInlineValue()) {
  if (IsInlineValue()) {
    new (&inline_value_) InlineValue(std::move(other.inline_value_));
  } else {
    new (&heap_value_) HeapValue(std::move(other.heap_value_));
  }
}

template <typename T, typename VT,
          typename std::enable_if<!std::is_same<Variant, VT>::value &&
                                      std::is_move_constructible<VT>::value,
                                  void>::type*>
inline Variant::Variant(T&& value) : is_inline_(CanInlineType<VT>()) {
  InsertValue<VT>(std::forward<T>(value));
}

template <typename T, typename VT,
          typename std::enable_if<!std::is_same<Variant, VT>::value &&
                                      std::is_copy_constructible<VT>::value,
                                  void>::type*>
inline Variant::Variant(const T& value) : is_inline_(CanInlineType<VT>()) {
  InsertValue<VT>(value);
}

template <typename T, typename VT,
          typename std::enable_if<!std::is_same<Variant, VT>::value &&
                                      std::is_move_constructible<VT>::value,
                                  void>::type*>
inline Variant& Variant::operator=(T&& value) {
  ResetMemory();
  is_inline_ = CanInlineType<VT>();
  InsertValue<VT>(std::forward<T>(value));
  return *this;
}

template <typename T, typename VT,
          typename std::enable_if<!std::is_same<Variant, VT>::value &&
                                      std::is_copy_constructible<VT>::value,
                                  void>::type*>
inline Variant& Variant::operator=(const T& value) {
  ResetMemory();
  is_inline_ = CanInlineType<VT>();
  InsertValue<VT>(value);
  return *this;
}

inline void Variant::clear() noexcept {
  // We set the internal unique_ptr to nullptr so that we preserve the
  // invariant that one of the two states must be set at all times. nullptr
  // indicates that the variant is empty.
  ResetAndSetHeap(/*pointer=*/nullptr);
}

inline void Variant::swap(Variant& other) noexcept {
  if (is_empty()) {
    if (other.IsInlineValue()) {
      ResetAndSetInline(std::move(other.inline_value_));
    } else {
      ResetAndSetHeap(std::move(other.heap_value_));
    }
    other.clear();
  } else if (other.is_empty()) {
    if (IsInlineValue()) {
      other.ResetAndSetInline(std::move(inline_value_));
    } else {
      other.ResetAndSetHeap(std::move(heap_value_));
    }
    clear();
  } else {  // Both Variants have values.
    if (other.IsInlineValue() && IsInlineValue()) {
      std::swap(inline_value_, other.inline_value_);
    } else if (!other.IsInlineValue() && !IsInlineValue()) {
      std::swap(heap_value_, other.heap_value_);
    } else if (other.IsInlineValue() && !IsInlineValue()) {
      HeapValue v = std::move(heap_value_);
      ResetAndSetInline(std::move(other.inline_value_));
      other.ResetAndSetHeap(std::move(v));
    } else {  // !other.IsInlineValue() && IsInlineValue()
      HeapValue v = std::move(other.heap_value_);
      other.ResetAndSetInline(std::move(inline_value_));
      ResetAndSetHeap(std::move(v));
    }
  }
}

template <>
void* Variant::get();

template <>
const void* Variant::get() const;

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_VARIANT_H_
