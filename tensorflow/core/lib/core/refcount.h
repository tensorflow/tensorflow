/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LIB_CORE_REFCOUNT_H_
#define TENSORFLOW_LIB_CORE_REFCOUNT_H_

#include <atomic>
#include <memory>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace core {

class RefCounted {
 public:
  // Initial reference count is one.
  RefCounted();

  // Increments reference count by one.
  void Ref() const;

  // Decrements reference count by one.  If the count remains
  // positive, returns false.  When the count reaches zero, returns
  // true and deletes this, in which case the caller must not access
  // the object afterward.
  bool Unref() const;

  // Return whether the reference count is one.
  // If the reference count is used in the conventional way, a
  // reference count of 1 implies that the current thread owns the
  // reference and no other thread shares it.
  // This call performs the test for a reference count of one, and
  // performs the memory barrier needed for the owning thread
  // to act on the object, knowing that it has exclusive access to the
  // object.
  bool RefCountIsOne() const;

 protected:
  // Make destructor protected so that RefCounted objects cannot
  // be instantiated directly. Only subclasses can be instantiated.
  virtual ~RefCounted();

 private:
  mutable std::atomic_int_fast32_t ref_;

  RefCounted(const RefCounted&) = delete;
  void operator=(const RefCounted&) = delete;
};

// A deleter class to form a std::unique_ptr that unrefs objects.
struct RefCountDeleter {
  void operator()(tensorflow::core::RefCounted* o) const { o->Unref(); }
};

// A unique_ptr that unrefs the owned object on destruction.
template <typename T>
using RefCountPtr = std::unique_ptr<T, RefCountDeleter>;

// Helper class to unref an object when out-of-scope.
class ScopedUnref {
 public:
  explicit ScopedUnref(RefCounted* o) : obj_(o) {}
  ~ScopedUnref() {
    if (obj_) obj_->Unref();
  }

 private:
  RefCounted* obj_;

  ScopedUnref(const ScopedUnref&) = delete;
  void operator=(const ScopedUnref&) = delete;
};

// Inlined routines, since these are performance critical
inline RefCounted::RefCounted() : ref_(1) {}

inline RefCounted::~RefCounted() { DCHECK_EQ(ref_.load(), 0); }

inline void RefCounted::Ref() const {
  DCHECK_GE(ref_.load(), 1);
  ref_.fetch_add(1, std::memory_order_relaxed);
}

inline bool RefCounted::Unref() const {
  DCHECK_GT(ref_.load(), 0);
  // If ref_==1, this object is owned only by the caller. Bypass a locked op
  // in that case.
  if (RefCountIsOne() || ref_.fetch_sub(1) == 1) {
    // Make DCHECK in ~RefCounted happy
    DCHECK((ref_.store(0), true));
    delete this;
    return true;
  } else {
    return false;
  }
}

inline bool RefCounted::RefCountIsOne() const {
  return (ref_.load(std::memory_order_acquire) == 1);
}

}  // namespace core
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_CORE_REFCOUNT_H_
