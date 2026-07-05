/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_RUNTIME_OBJECT_POOL_H_
#define XLA_RUNTIME_OBJECT_POOL_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {

// A pool of heap-owned objects of type `T`. Objects are created lazily by
// invoking the user-provided `builder`. `T` must be movable; use a movable
// wrapper for non-movable resources. `Args...` are the argument types forwarded
// from `GetOrCreate` to the builder when a new object has to be created.
template <typename T, typename... Args>
class ObjectPool {
  // Returns a borrowed object to its parent pool.
  struct ReturnToPool {
    void operator()(T* object) const;
    ObjectPool* parent = nullptr;
  };

 public:
  explicit ObjectPool(absl::AnyInvocable<absl::StatusOr<T>(Args...)> builder);
  ~ObjectPool();

  // Creates a pool from a builder that cannot fail and eagerly creates
  // `preallocate` available objects by calling the builder with `args`.
  // Later GetOrCreate calls use the arguments passed to GetOrCreate.
  ObjectPool(absl::AnyInvocable<T(Args...)> builder, size_t preallocate,
             Args... args);

  // A move-only RAII handle for an object borrowed from the pool. The
  // borrowed object is returned to the pool when this handle is destroyed or
  // overwritten by move assignment.
  using BorrowedObject = std::unique_ptr<T, ReturnToPool>;

  // Creates `num_objects` additional objects with the builder and returns them
  // to the pool as available objects. If the builder fails, returns the first
  // error without adding any object from this call to the pool.
  absl::Status Preallocate(size_t num_objects, Args... args);

  // Borrows an object that is currently available in the pool. Does not invoke
  // the builder; returns ResourceExhausted if the pool has no available object.
  absl::StatusOr<BorrowedObject> Get();

  // Borrows an available object from the pool, or creates a new object with the
  // builder if the pool is empty. Returns any error produced by the builder.
  absl::StatusOr<BorrowedObject> GetOrCreate(Args... args);

  // Returns the total number of objects created by the builder. This is
  // `num_available()` plus the number of objects currently borrowed from the
  // pool.
  size_t num_created() const;

  // Returns the number of objects that `Get()` can borrow without creating.
  size_t num_available() const;

 private:
  absl::StatusOr<std::unique_ptr<T>> CreateObject(Args... args);

  std::unique_ptr<T> PopObject();
  void PushObject(std::unique_ptr<T> object);

  absl::AnyInvocable<absl::StatusOr<T>(Args...)> builder_;

  mutable absl::Mutex mu_;
  std::vector<std::unique_ptr<T>> available_ ABSL_GUARDED_BY(mu_);
  size_t num_created_ ABSL_GUARDED_BY(mu_);
};

//===----------------------------------------------------------------------===//
// ObjectPool implementation.
//===----------------------------------------------------------------------===//

template <typename T, typename... Args>
ObjectPool<T, Args...>::ObjectPool(
    absl::AnyInvocable<absl::StatusOr<T>(Args...)> builder)
    : builder_(std::move(builder)), num_created_(0) {}

template <typename T, typename... Args>
ObjectPool<T, Args...>::ObjectPool(absl::AnyInvocable<T(Args...)> builder,
                                   size_t preallocate, Args... args)
    : ObjectPool([b = std::move(builder)](Args... args) mutable {
        return b(std::forward<Args>(args)...);
      }) {
  Preallocate(preallocate, std::forward<Args>(args)...).IgnoreError();
}

template <typename T, typename... Args>
ObjectPool<T, Args...>::~ObjectPool() {
  absl::MutexLock lock(mu_);
  CHECK_EQ(available_.size(), num_created_);  // Crash OK
}

template <typename T, typename... Args>
size_t ObjectPool<T, Args...>::num_created() const {
  absl::MutexLock lock(mu_);
  return num_created_;
}

template <typename T, typename... Args>
size_t ObjectPool<T, Args...>::num_available() const {
  absl::MutexLock lock(mu_);
  return available_.size();
}

template <typename T, typename... Args>
auto ObjectPool<T, Args...>::CreateObject(Args... args)
    -> absl::StatusOr<std::unique_ptr<T>> {
  DCHECK(builder_) << "ObjectPool builder is not initialized";
  ASSIGN_OR_RETURN(T object, builder_(std::forward<Args>(args)...));
  return std::make_unique<T>(std::move(object));
}

template <typename T, typename... Args>
auto ObjectPool<T, Args...>::PopObject() -> std::unique_ptr<T> {
  absl::MutexLock lock(mu_);
  if (available_.empty()) {
    return nullptr;
  }
  std::unique_ptr<T> object = std::move(available_.back());
  available_.pop_back();
  return object;
}

template <typename T, typename... Args>
void ObjectPool<T, Args...>::PushObject(std::unique_ptr<T> object) {
  absl::MutexLock lock(mu_);
  // Make sure that all followup `PushObject` calls will have space in a vector.
  available_.reserve(num_created_);
  available_.push_back(std::move(object));
}

template <typename T, typename... Args>
void ObjectPool<T, Args...>::ReturnToPool::operator()(T* object) const {
  parent->PushObject(std::unique_ptr<T>(object));
}

template <typename T, typename... Args>
absl::Status ObjectPool<T, Args...>::Preallocate(size_t num_objects,
                                                 Args... args) {
  std::vector<std::unique_ptr<T>> objects(num_objects);
  for (size_t i = 0; i < num_objects; ++i) {
    ASSIGN_OR_RETURN(objects[i], CreateObject(args...));
  }

  absl::MutexLock lock(mu_);
  for (std::unique_ptr<T>& object : objects) {
    available_.push_back(std::move(object));
  }
  num_created_ += objects.size();

  return absl::OkStatus();
}

template <typename T, typename... Args>
auto ObjectPool<T, Args...>::Get() -> absl::StatusOr<BorrowedObject> {
  if (std::unique_ptr<T> object = PopObject(); ABSL_PREDICT_TRUE(object)) {
    return BorrowedObject(object.release(), ReturnToPool{this});
  }
  return absl::ResourceExhaustedError("Object pool has no available object");
}

template <typename T, typename... Args>
auto ObjectPool<T, Args...>::GetOrCreate(Args... args)
    -> absl::StatusOr<BorrowedObject> {
  if (std::unique_ptr<T> object = PopObject(); ABSL_PREDICT_TRUE(object)) {
    return BorrowedObject(object.release(), ReturnToPool{this});
  }
  ASSIGN_OR_RETURN(auto object, CreateObject(std::forward<Args>(args)...));

  absl::MutexLock lock(mu_);
  num_created_++;
  return BorrowedObject(object.release(), ReturnToPool{this});
}

}  // namespace xla

#endif  // XLA_RUNTIME_OBJECT_POOL_H_
