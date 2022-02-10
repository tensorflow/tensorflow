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

#ifndef TENSORFLOW_CORE_PLATFORM_REFCOUNT_H_
#define TENSORFLOW_CORE_PLATFORM_REFCOUNT_H_

#include <atomic>
#include <map>
#include <memory>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

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

  // Gets the current reference count.
  int_fast32_t RefCount() const;

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

  // Increments reference count by one if the object is not being destructed.
  // This function is used by WeakRefCounted for securely acquiring a
  // strong reference. It is only safe to call this as part of the weak
  // reference implementation.
  bool TryRef() const;

 private:
  mutable std::atomic_int_fast32_t ref_;

  RefCounted(const RefCounted&) = delete;
  void operator=(const RefCounted&) = delete;
};

// A deleter class to form a std::unique_ptr that unrefs objects.
struct RefCountDeleter {
  void operator()(const RefCounted* o) const { o->Unref(); }
};

// A unique_ptr that unrefs the owned object on destruction.
template <typename T>
using RefCountPtr = std::unique_ptr<T, RefCountDeleter>;

// Helper class to unref an object when out-of-scope.
class ScopedUnref {
 public:
  explicit ScopedUnref(const RefCounted* o) : obj_(o) {}
  ~ScopedUnref() {
    if (obj_) obj_->Unref();
  }

 private:
  const RefCounted* obj_;

  ScopedUnref(const ScopedUnref&) = delete;
  void operator=(const ScopedUnref&) = delete;
};

// Forward declaration for friend class of WeakRefCounted.
template <typename T>
class WeakPtr;

// A WeakNotifyFn is called when the weakly referred object is being destroyed.
// The object may already be destructed when the call occurs. A WeakNotifyFn
// can be passed into WeakPtr at construction.
using WeakNotifyFn = std::function<void()>;

// A base class for RefCounted objects that allow weak references by WeakPtr.
// WeakRefCounted and every WeakPtr to it, each holds a strong reference to a
// WeakRefData.
//
// If the WeakRefCounted is valid, WeakPtr::GetNewRef() returns a new strong
// reference to the WeakRefCounted.
// If the WeakRefCounted is being destructed, `WeakRefCounted::ref_ == 0`;
// if the WeakRefcounted is already destructed,`WeakRefData::ptr == nullptr`.
// In either case, WeakPtr::GetNewRef() returns a nullptr.
class WeakRefCounted : public RefCounted {
 public:
  int WeakRefCount() const {
    // Each weak ref owns one ref to data_, and *this owns the last one.
    return data_->RefCount() - 1;
  }

 protected:
  ~WeakRefCounted() override { data_->Notify(); }

 private:
  struct WeakRefData : public RefCounted {
    explicit WeakRefData(WeakRefCounted* ptr) : ptr(ptr), next_notifier_id(1) {}

    mutable mutex mu;
    WeakRefCounted* ptr TF_GUARDED_BY(mu);
    std::map<int, WeakNotifyFn> notifiers;
    int next_notifier_id;

    // Notifies WeakPtr instansces that this object is being destructed.
    void Notify() {
      mutex_lock ml(mu);

      while (!notifiers.empty()) {
        auto iter = notifiers.begin();
        WeakNotifyFn notify_fn = std::move(iter->second);
        notifiers.erase(iter);

        mu.unlock();
        notify_fn();
        mu.lock();
      }
      ptr = nullptr;
    }

    WeakRefCounted* GetNewRef() {
      mutex_lock ml(mu);
      if (ptr != nullptr && ptr->TryRef()) {
        return ptr;
      }
      return nullptr;
    }

    // Inserts notify_fn and returns a non-zero id.
    // Returns 0 if insertion fails due to the object is being destroyed.
    // 0 is also used by WeakPtr to represent "no notify_fn".
    int AddNotifier(WeakNotifyFn notify_fn) {
      mutex_lock ml(mu);
      if (ptr == nullptr) {
        return 0;
      }
      int notifier_id = next_notifier_id++;
      notifiers.emplace(notifier_id, std::move(notify_fn));
      return notifier_id;
    }

    void RemoveNotifier(int notifier_id) {
      mutex_lock ml(mu);
      notifiers.erase(notifier_id);
    }
  };

  RefCountPtr<WeakRefData> data_{new WeakRefData(this)};

  template <typename T>
  friend class WeakPtr;
  // MSVC14 workaround: access permission of a nested class member is not
  // treated as an ordinary member in MSVC14.
  friend struct WeakRefData;
};

// A weak reference to a WeakRefCounted object. Refer to WeakRefCounted.
template <typename T>
class WeakPtr {
 public:
  // Creates a weak reference.
  // When the object is being destroyed, notify_fn is called.
  explicit WeakPtr(WeakRefCounted* ptr, WeakNotifyFn notify_fn = nullptr)
      : data_(nullptr), notifier_id_(0) {
    if (ptr != nullptr) {
      ptr->data_->Ref();
      data_.reset(ptr->data_.get());
      if (notify_fn) {
        notifier_id_ = data_->AddNotifier(notify_fn);
      }
    }
  }

  ~WeakPtr() {
    if (data_ != nullptr && notifier_id_ != 0) {
      data_->RemoveNotifier(notifier_id_);
    }
  }

  // NOTE(feyu): change data_ to a IntrusivePtr to make WeakPtr copyable.
  WeakPtr(const WeakPtr& other) = delete;
  WeakPtr& operator=(const WeakPtr& other) = delete;

  WeakPtr(WeakPtr&& other) {
    data_ = std::move(other.data_);
    notifier_id_ = other.notifier_id_;
    other.notifier_id_ = 0;
  }

  WeakPtr& operator=(WeakPtr&& other) {
    if (this != &other) {
      if (data_ != nullptr && notifier_id_ != 0) {
        data_->RemoveNotifier(notifier_id_);
      }
      data_ = std::move(other.data_);
      notifier_id_ = other.notifier_id_;
      other.notifier_id_ = 0;
    }
    return *this;
  }

  // Returns a new strong reference to the referred object, or nullptr if the
  // object is in an invalid state (being destructed or already destructed).
  RefCountPtr<T> GetNewRef() const {
    RefCountPtr<T> ref;
    if (data_ != nullptr) {
      WeakRefCounted* ptr = data_->GetNewRef();
      ref.reset(static_cast<T*>(ptr));
    }
    return std::move(ref);
  }

 private:
  RefCountPtr<WeakRefCounted::WeakRefData> data_;
  int notifier_id_;
};

// Inlined routines, since these are performance critical
inline RefCounted::RefCounted() : ref_(1) {}

inline RefCounted::~RefCounted() {
  // A destructing object has ref_ == 0.
  // It is a bug if the object is resurrected (ref_ > 0) before delete is
  // called by Unref().
  DCHECK_EQ(ref_.load(), 0);
}

inline void RefCounted::Ref() const {
  // Ref() uses relaxed order because it is never called with old_ref == 0.
  // When old_ref >= 1, no actions depend on the new value of ref.
  int_fast32_t old_ref = ref_.fetch_add(1, std::memory_order_relaxed);
  DCHECK_GT(old_ref, 0);
}

inline bool RefCounted::TryRef() const {
  // This is not on a hot path.
  // Be conservative and use seq_cst to prevent racing with Unref() when
  // old_ref == 0, as done in LLVM libstdc++.
  int_fast32_t old_ref = ref_.load();
  while (old_ref != 0) {
    if (ref_.compare_exchange_weak(old_ref, old_ref + 1)) {
      return true;
    }
  }
  // Already destructing, cannot increase ref.
  return false;
}

inline bool RefCounted::Unref() const {
  DCHECK_GT(ref_.load(), 0);
  // acq_rel is used to prevent reordering introduces object access after
  // destruction.

  // Using release alone is a bug on systems where acq_rel differs from release.
  // (e.g. arm), according to Herb Sutter's 2012 talk on "Atomic<> Weapons".
  if (ref_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    delete this;
    return true;
  }
  return false;
}

inline int_fast32_t RefCounted::RefCount() const {
  return ref_.load(std::memory_order_acquire);
}

inline bool RefCounted::RefCountIsOne() const {
  return (ref_.load(std::memory_order_acquire) == 1);
}

}  // namespace core
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_REFCOUNT_H_
