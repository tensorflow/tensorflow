/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_TSL_UTIL_TIED_REF_H_
#define XLA_TSL_UTIL_TIED_REF_H_

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace tsl {

template <typename T>
class Tied;

// A weak reference to a value stored in a Tied<T> container.
//
// Tied<T> is a container for values of type T whose lifetime is tied to the
// container object. Callers receive TiedRef<T> handles that can be used to
// check if the value (and the container it is tied to) is still alive.
//
// When the Tied<T> container is destroyed, all outstanding TiedRefs safely
// expire (Lock returns nullptr). Additionally, when a TiedRef is destroyed
// while the Tied<T> container is still alive, the tied value is eagerly
// released — there is no need to wait for the container to be destroyed.
//
// Example:
//
//   class Session : private Tied<Connection> {
//    public:
//     TiedRef<Connection> Connect() {
//       return Tie(std::make_unique<Connection>(*this));
//     }
//   };
//
//   // Caller does not control Session's lifetime.
//   void DoWork(Session* session) {
//     auto conn_ref = session->Connect();
//     // ... later, possibly after `session` was replaced or shut down:
//     if (auto conn = conn_ref.Lock()) {
//       conn->Execute();  // Safe: session is still alive.
//     }
//   }
//
// It is the caller's responsibility to ensure that the Tied<T> container stays
// alive for as long as the locked shared_ptr is in use. Lock() guarantees that
// the value was alive at the time of the call, but does not prevent the
// container from being destroyed concurrently. In practice, the caller
// typically receives the container by pointer (e.g. Session*) with an
// external guarantee that it outlives the current scope — so locking and
// using the value within that scope is safe. Callers may also maintain their
// own cache of TiedRefs and use Lock() to detect which ones are still valid
// after the underlying session has been replaced.
//
template <typename T>
class TiedRef {
 public:
  TiedRef() = default;
  ~TiedRef();

  TiedRef(TiedRef&& other) noexcept;
  TiedRef& operator=(TiedRef&& other) noexcept;

  // Returns a shared_ptr to the tied value if both the TiedRef and the Tied<T>
  // container are still alive, or nullptr otherwise.
  std::shared_ptr<T> Lock() const;

  // Returns true if the Tied<T> container has been destroyed.
  bool Expired() const;

 private:
  friend class Tied<T>;
  explicit TiedRef(std::weak_ptr<std::shared_ptr<T>> ptr);

  std::weak_ptr<std::shared_ptr<T>> ptr_;
};

// A container for values of type T whose lifetime is tied to *this object.
// When *this is destroyed, all tied values are destroyed. Callers receive
// TiedRef<T> handles to check if the values are still alive.
// See TiedRef<T> for full documentation and usage examples.
template <typename T>
class Tied {
 public:
  Tied() = default;
  ~Tied() = default;

  Tied(Tied&&) = default;
  Tied& operator=(Tied&&) = default;

  // Ties ownership of `value` to *this and returns a TiedRef to it. The
  // value is released when the TiedRef is destroyed or when *this is
  // destroyed, whichever comes first. A shared_ptr obtained from Lock()
  // independently extends the value's lifetime.
  TiedRef<T> Tie(std::unique_ptr<T> value);

 private:
  std::vector<std::shared_ptr<std::shared_ptr<T>>> entries_;
};

//===----------------------------------------------------------------------===//
// TiedRef<T> and Tied<T> implementation detail.
//===----------------------------------------------------------------------===//

template <typename T>
TiedRef<T>::TiedRef(std::weak_ptr<std::shared_ptr<T>> ptr)
    : ptr_(std::move(ptr)) {}

template <typename T>
TiedRef<T>::~TiedRef() {
  if (auto locked = ptr_.lock()) {
    locked->reset();
  }
}

template <typename T>
TiedRef<T>::TiedRef(TiedRef&& other) noexcept : ptr_(std::move(other.ptr_)) {}

template <typename T>
TiedRef<T>& TiedRef<T>::operator=(TiedRef&& other) noexcept {
  if (this != &other) {
    if (auto locked = ptr_.lock()) {
      locked->reset();
    }
    ptr_ = std::move(other.ptr_);
  }
  return *this;
}

template <typename T>
std::shared_ptr<T> TiedRef<T>::Lock() const {
  if (auto locked = ptr_.lock()) {
    return *locked;
  }
  return nullptr;
}

template <typename T>
bool TiedRef<T>::Expired() const {
  return ptr_.expired();
}

template <typename T>
TiedRef<T> Tied<T>::Tie(std::unique_ptr<T> value) {
  // Lazy garbage collection: erase entries whose inner shared_ptr was reset
  // by a TiedRef destructor.
  entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                [](const auto& entry) { return !*entry; }),
                 entries_.end());

  if (!value) {
    return TiedRef<T>{};
  }

  auto& entry = entries_.emplace_back(std::make_shared<std::shared_ptr<T>>(
      std::shared_ptr<T>(value.release())));
  return TiedRef<T>{std::weak_ptr<std::shared_ptr<T>>(entry)};
}

}  // namespace tsl

#endif  // XLA_TSL_UTIL_TIED_REF_H_
