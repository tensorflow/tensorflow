/* Copyright 2016 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TFRT_UTILS_ANY_PTR_H_
#define TENSORFLOW_CORE_TFRT_UTILS_ANY_PTR_H_

#include <cstddef>
#include <memory>

namespace tfrt {

/// A (sort of) type-safe void*. Appears as null if a caller attempts to use it
/// as the wrong type.
///
///  Example use:
///
///    // A function that returns an AnyPtr:
///    AnyPtr StringOrInt() {
///      if (use_string) {
///        return AnyPtr(&some_string);
///      } else {
///        return AnyPtr(&some_int);
///      }
///    }
///
///    // Use an AnyPtr at the correct type:
///    AnyPtr ptr = StringOrInt();
///    if (ptr.get<int>() != nullptr) {
///      DoSomethingWithInt(*ptr.get<int>());
///    } else if (ptr.get<string>() != nullptr) {
///      DoSomethingWithString(*ptr.get<string>());
///    } else {
///      // Handle error.
///    }
///
/// Typical best practice for this class is to use it when two disjoint pieces
/// of code must agree on type, but intermediate code is type agnostic. Large
/// chains of conditionals that handle a multitude of types is discouraged as an
/// anti-pattern.
///
/// Note that this will appear null even if T is somewhere on the underlying
/// type's inheritance hierarchy, if you must use the object at some other type
/// you must do so explicitly when constructing an AnyPtr, like so:
///
///   SomeObject object;
///   AnyPtr any_ptr(static_cast<SomeInterface*>(&object));
///   SomeInterface* interface = any_ptr.get<SomeInterface>();
///
/// This class is a value type; It can be copied or assigned. It performs no
/// internal allocations and should be relatively cheap to copy or return by
/// value.
class AnyPtr {
 public:
  /// AnyPtr is void and null by default.
  AnyPtr() : type_id_(FastTypeId<void>()), ptr_(nullptr) {}

  /// Implicit construction from nullptr.
  AnyPtr(std::nullptr_t) : AnyPtr() {}  // NOLINT

  /// Construct from a pointer to any type.
  template <typename T>
  AnyPtr(T* ptr)  // NOLINT
      : type_id_(FastTypeId<T>()),
        // We need a double cast here, first to drop the type, and second to
        // drop constness. We always cast back to the appropriate type and
        // constness in get<>(), since FastTypeId is different for a const and
        // non-const T.
        ptr_(const_cast<void*>(reinterpret_cast<const void*>(ptr))) {}

  /// Accessor for the underlying pointer if it is of type T, otherwise null.
  template <typename T>
  T* get() const {
    if (type_id_ != FastTypeId<T>()) {
      return nullptr;
    }
    return reinterpret_cast<T*>(ptr_);
  }

 private:
  template <typename Type>
  static size_t FastTypeId() {
    // Use a static variable to get a unique per-type address.
    static int dummy;
    return reinterpret_cast<std::size_t>(&dummy);
  }

  // The code for the type of 'ptr_'.
  std::size_t type_id_;

  // The underlying pointer.
  void* ptr_;
};

/// Like AnyPtr, but owns the pointed-to object (calls delete upon destruction).
/// This class is move-only, like std::unique_ptr.
class UniqueAnyPtr {
 public:
  /// UniqueAnyPtr is void and null by default.
  UniqueAnyPtr() = default;
  UniqueAnyPtr(std::nullptr_t) : UniqueAnyPtr() {}  // NOLINT

  /// Construct from a unique pointer to any type.
  template <typename T>
  explicit UniqueAnyPtr(std::unique_ptr<T> ptr)
      : ptr_(ptr.release()), deleter_(DeleterForType<T>()) {}

  ~UniqueAnyPtr() { deleter_(ptr_); }

  // Disable copy.
  UniqueAnyPtr(const UniqueAnyPtr& other) = delete;
  UniqueAnyPtr& operator=(const UniqueAnyPtr& other) = delete;

  // Allow move.
  UniqueAnyPtr(UniqueAnyPtr&& other) noexcept { swap(other); }

  UniqueAnyPtr& operator=(UniqueAnyPtr&& other) noexcept {
    swap(other);
    return *this;
  }

  /// Accessor for the underlying pointer if it is of type T, otherwise null.
  template <typename T>
  T* get() const {
    return ptr_.get<T>();
  }

  /// Accessor for the underlying pointer as an AnyPtr.
  const AnyPtr& as_any_ptr() const { return ptr_; }

  void swap(UniqueAnyPtr& other) noexcept {
    using ::std::swap;
    swap(ptr_, other.ptr_);
    swap(deleter_, other.deleter_);
  }

 private:
  // We use a raw function pointer. This eliminates the copy and calling
  // overhead of std::function.
  using Deleter = void (*)(AnyPtr ptr);

  // Returns a 'Deleter' that will delete it's argument as an instance of 'T'.
  // Always returns the same value for the same 'T'.
  template <typename T>
  static Deleter DeleterForType() {
    return [](AnyPtr ptr) { delete ptr.get<T>(); };
  }

  static Deleter NoOpDeleter() {
    return [](AnyPtr ptr) {};
  }

  AnyPtr ptr_ = nullptr;
  Deleter deleter_ = NoOpDeleter();
};

}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_UTILS_ANY_PTR_H_
