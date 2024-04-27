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
#ifndef TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_VALUE_H_
#define TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_VALUE_H_

#include <new>
#include <type_traits>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"

namespace mlrt {

class Value;

namespace value_internal {

struct InPlaceStorageT {
  // Many tensor implementations like tensorflow::Tensor requires multiple
  // words, and we'd like to keep these values inplace.
  //
  // TODO(chky): Consider a better size for inplace storage.
  alignas(8) char data[56];
};

template <typename T>
using IsInPlaceStorage =
    std::integral_constant<bool, sizeof(T) <= sizeof(InPlaceStorageT) &&
                                     alignof(T) <= alignof(InPlaceStorageT) &&
                                     std::is_move_constructible<T>::value>;

// Since we type-erase the value to be put in class Value, we need to an enum
// value to select the operation that should be applied on the type-erased
// value.
enum class Action {
  kDestroy = 0,  // Destructor
  kCopy,         // Copy constructor/assignment
  kMove,         // Move constructor/assignment
  kError,        // Error handler
  kTypeInfo      // Get type info
};

struct TypeInfo {};

using HandlerFuncPtr = TypeInfo* (*)(Action, Value*, Value*);

template <typename T>
class InPlaceHandler;
template <typename T>
class OutOfPlaceHandler;

template <typename T>
using Handler = std::conditional_t<IsInPlaceStorage<T>::value,
                                   InPlaceHandler<T>, OutOfPlaceHandler<T>>;

template <typename T, typename Enable = void>
struct HasHandleError : std::false_type {};

template <typename T>
struct HasHandleError<
    T, std::void_t<decltype(std::declval<T>().HandleError(nullptr))>>
    : std::true_type {};

}  // namespace value_internal

// A container for type-erased value. The value should be at least copy
// constructable to be put into this container. This container has both move and
// copy semantics, but if the concrete value does not support copy, calling the
// copy operations on this class will result in undefined behavior.
class alignas(64) Value {
 public:
  // Value is default constructible. The payload is unset in the default
  // constructed Value.
  Value() = default;

  Value(const Value&);
  Value& operator=(const Value&);
  Value(Value&&) noexcept;
  Value& operator=(Value&&) noexcept;

  // Construct Value and store `t` as the payload.
  template <typename T,
            typename std::enable_if<!std::is_same_v<std::decay_t<T>, Value>,
                                    int>::type = 0>
  explicit Value(T&& t);

  template <typename T,
            typename std::enable_if<!std::is_same_v<std::decay_t<T>, Value>,
                                    int>::type = 0>
  Value& operator=(T&& value) {
    Set(std::forward<T>(value));
    return *this;
  }

  ~Value();

  // Get() function returns the payload of the Value object in the requested
  // type.
  //
  // Dynamic type checking is performed in the debug mode.
  template <typename T>
  T& Get();

  template <typename T>
  const T& Get() const;

  // Emplace() constructs the payload object of type T in place with the given
  // args. If the value is already initialized, the original value will be
  // destroyed.
  template <typename T, typename... Args>
  void Emplace(Args&&... args);

  // Construct() constructs the payload object of type T in place with the given
  // args. The value should be uninitialized before calling this method.
  // Otherwise the behavior is undefined.
  template <typename T, typename... Args>
  void Construct(Args&&... args);

  // Destroy() destroys the payload object of type T. The value must be already
  // initialized with a value of type T. Otherwise the behavior is undefined.
  template <typename T>
  void Destroy();

  // Set() stores the argument `t` as the payload of Value.
  template <typename T>
  void Set(T&& t);

  // Reset the Value object to empty.
  void Reset();

  // Call T::HandleError() method on the underlying value of type T. If T does
  // not have a HandleError() method, this method does nothing.
  void HandleError(Value& arg);

  // Check if Value contains a payload.
  bool HasValue() const { return handler_ != nullptr; }

  // Check if Value contains object of type T.
  template <typename T>
  bool IsType() const;

  // Check if object of type T is stored in place.
  template <typename T>
  static constexpr bool IsInPlace() {
    return value_internal::IsInPlaceStorage<T>::value;
  }

 private:
  union {
    value_internal::InPlaceStorageT storage_{};
    void* value_;
  };
  value_internal::HandlerFuncPtr handler_ = nullptr;

  template <typename>
  friend class value_internal::InPlaceHandler;
  template <typename>
  friend class value_internal::OutOfPlaceHandler;
};

// We only optimize the code for 64-bit architectures for now.
static_assert(sizeof(Value) == 64 || sizeof(void*) != 8);

// -----------------------------------------------------------
// Implementation details.

namespace value_internal {

template <typename T>
TypeInfo* GetTypeInfo();

template <typename T,
          typename std::enable_if_t<HasHandleError<T>::value, int> = 0>
void HandleErrorInternal(Value* self, Value* arg) {
  std::move(self->Get<T>()).HandleError(arg);
}

template <typename T,
          typename std::enable_if_t<!HasHandleError<T>::value, int> = 0>
static void HandleErrorInternal(Value* self, Value* arg) {}

template <class T>
struct InPlaceHandler {
  template <typename... Args>
  static void Construct(Value* self, Args&&... args) {
    new (&self->storage_) T(std::forward<Args>(args)...);
    self->handler_ = &Handle;
  }

  static TypeInfo* Handle(Action action, Value* self, Value* other) {
    switch (action) {
      case Action::kDestroy:
        Destroy(self);
        return nullptr;
      case Action::kCopy:
        Copy(self, other);
        return nullptr;
      case Action::kMove:
        Move(self, other);
        return nullptr;
      case Action::kError:
        HandleError(self, other);
        return nullptr;
      case Action::kTypeInfo:
        return GetTypeInfo<T>();
    }
  }

  static void Destroy(Value* self) {
    DCHECK(self->HasValue());
    auto* p = std::launder(reinterpret_cast<T*>(&self->storage_));
    p->~T();
    self->handler_ = nullptr;
  }

  template <typename V, typename std::enable_if_t<
                            std::is_copy_constructible<V>::value, int> = 0>
  static void CopyInternal(Value* self, Value* dest) {
    DCHECK(self->HasValue() && !dest->HasValue());
    Construct(dest, *std::launder(reinterpret_cast<const V*>(&self->storage_)));
  }

  template <typename V, typename std::enable_if_t<
                            !std::is_copy_constructible<V>::value, int> = 0>
  static void CopyInternal(Value* self, Value* dest) {
    LOG(FATAL) << "Copying a mlrt::Value whose underlying type is "  // Crash Ok
                  "not copyable is a runtime error.";
  }

  static void Copy(Value* self, Value* dest) { CopyInternal<T>(self, dest); }

  static void Move(Value* self, Value* dest) {
    DCHECK(self->HasValue() && !dest->HasValue());
    Construct(dest,
              std::move(*std::launder(reinterpret_cast<T*>(&self->storage_))));
    Destroy(self);
  }

  static void HandleError(Value* self, Value* arg) {
    HandleErrorInternal<T>(self, arg);
  }
};

template <class T>
struct OutOfPlaceHandler {
  template <typename... Args>
  static void Construct(Value* self, Args&&... args) {
    self->value_ = new T(std::forward<Args>(args)...);
    self->handler_ = &Handle;
  }

  static TypeInfo* Handle(Action action, Value* self, Value* other) {
    switch (action) {
      case Action::kDestroy:
        Destroy(self);
        return nullptr;
      case Action::kCopy:
        Copy(self, other);
        return nullptr;
      case Action::kMove:
        Move(self, other);
        return nullptr;
      case Action::kError:
        HandleError(self, other);
        return nullptr;
      case Action::kTypeInfo:
        return GetTypeInfo<T>();
    }
  }

  static void Destroy(Value* self) {
    DCHECK(self->HasValue());
    delete static_cast<T*>(self->value_);
    self->handler_ = nullptr;
  }

  template <typename V, typename std::enable_if_t<
                            std::is_copy_constructible<V>::value, int> = 0>
  static void CopyInternal(Value* self, Value* dest) {
    DCHECK(self->HasValue() && !dest->HasValue());
    Construct(dest, *static_cast<const V*>(self->value_));
  }

  template <typename V, typename std::enable_if_t<
                            !std::is_copy_constructible<V>::value, int> = 0>
  static void CopyInternal(Value* self, Value* dest) {
    LOG(FATAL) << "Copying a mlrt::Value whose underlying type is "  // Crash Ok
                  "not copyable is a runtime error.";
  }

  static void Copy(Value* self, Value* dest) { CopyInternal<T>(self, dest); }

  static void Move(Value* self, Value* dest) {
    DCHECK(self->HasValue() && !dest->HasValue());
    dest->value_ = self->value_;
    dest->handler_ = &Handle;
    self->handler_ = nullptr;
  }

  static void HandleError(Value* self, Value* arg) {
    HandleErrorInternal<T>(self, arg);
  }
};

template <typename T>
__attribute__((noinline)) TypeInfo* GetTypeInfo() {
  static TypeInfo kTypeInfo;
  return &kTypeInfo;
}

}  // namespace value_internal

template <typename T, typename std::enable_if<
                          !std::is_same_v<std::decay_t<T>, Value>, int>::type>
Value::Value(T&& t) {
  Construct<std::decay_t<T>>(std::forward<T>(t));
}

inline Value::Value(const Value& v) {
  if (v.HasValue())
    v.handler_(value_internal::Action::kCopy, const_cast<Value*>(&v), this);
}

inline Value& Value::operator=(const Value& v) {
  Reset();
  if (v.HasValue())
    v.handler_(value_internal::Action::kCopy, const_cast<Value*>(&v), this);
  return *this;
}

inline Value::Value(Value&& v) noexcept {
  if (v.HasValue()) v.handler_(value_internal::Action::kMove, &v, this);
}

inline Value& Value::operator=(Value&& v) noexcept {
  Reset();
  if (v.HasValue()) v.handler_(value_internal::Action::kMove, &v, this);
  return *this;
}

inline void Value::HandleError(Value& arg) {
  if (HasValue()) handler_(value_internal::Action::kError, this, &arg);
}

inline Value::~Value() { Reset(); }

template <typename T>
T& Value::Get() {
  return const_cast<T&>(static_cast<const Value*>(this)->Get<T>());
}

template <typename T>
const T& Value::Get() const {
  DCHECK(IsType<T>());

  if constexpr (IsInPlace<T>()) {
    return *std::launder(reinterpret_cast<const T*>(&storage_));
  }

  return *static_cast<const T*>(value_);
}

// Emplace() constructs the payload object of type T in place with the given
// args.
template <typename T, typename... Args>
void Value::Emplace(Args&&... args) {
  Reset();
  Construct<std::decay_t<T>>(std::forward<Args>(args)...);
}

// Set() stores the argument `t` as the payload of Value.
template <typename T>
void Value::Set(T&& t) {
  Emplace<T>(std::forward<T>(t));
}

template <typename T, typename... Args>
void Value::Construct(Args&&... args) {
  DCHECK(!HasValue());
  static_assert(!std::is_same_v<T, Value>);
  value_internal::Handler<T>::Construct(this, std::forward<Args>(args)...);
}

template <typename T>
void Value::Destroy() {
  DCHECK(HasValue());
  DCHECK(IsType<T>());
  static_assert(!std::is_same_v<T, Value>);
  value_internal::Handler<T>::Destroy(this);
}

// Reset the Value object to empty.
inline void Value::Reset() {
  if (handler_ == nullptr) return;
  handler_(value_internal::Action::kDestroy, this, nullptr);
}

template <typename T>
bool Value::IsType() const {
  return handler_(value_internal::Action::kTypeInfo, const_cast<Value*>(this),
                  nullptr) == value_internal::GetTypeInfo<T>();
}

}  // namespace mlrt

#endif  // TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_VALUE_H_
