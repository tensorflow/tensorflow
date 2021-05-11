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
#ifndef TENSORFLOW_CORE_PLATFORM_REFCOUNTED_SHARED_PTR_H_
#define TENSORFLOW_CORE_PLATFORM_REFCOUNTED_SHARED_PTR_H_

namespace tensorflow {
namespace core {

// A utility for managing the lifetime of ref-counted objects.
//
// Generally used for objects that derive from `tensorflow::RefCounted`.
template <class T>
class IntrusivePtr {
 public:
  // add_ref=false indicates that IntrusivePtr owns the underlying pointer.
  //
  // In most cases, we expect this to be called with add_ref=false, except in
  // special circumstances where the lifetime of the underlying RefCounted
  // object needs to be externally managed.
  IntrusivePtr(T* h, bool add_ref) { reset(h, add_ref); }
  IntrusivePtr(const IntrusivePtr& o) { reset(o.handle_, /*add_ref=*/true); }
  IntrusivePtr(IntrusivePtr&& o) {
    reset(o.handle_, /*add_ref=*/false);
    o.handle_ = nullptr;
  }
  IntrusivePtr() {}
  void reset(T* h, bool add_ref) {
    if (handle_) handle_->Unref();
    handle_ = h;
    if (add_ref && handle_) handle_->Ref();
  }
  IntrusivePtr& operator=(const IntrusivePtr& o) {
    reset(o.handle_, /*add_ref=*/true);
    return *this;
  }
  IntrusivePtr& operator=(IntrusivePtr&& o) {
    reset(o.handle_, /*add_ref=*/false);
    o.handle_ = nullptr;
    return *this;
  }
  bool operator==(const IntrusivePtr& o) const { return handle_ == o.handle_; }
  T* operator->() const { return handle_; }
  T& operator*() const { return *handle_; }
  T* get() const { return handle_; }

  ~IntrusivePtr() {
    if (handle_) handle_->Unref();
  }

 private:
  T* handle_ = nullptr;
};

}  // namespace core
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_REFCOUNTED_SHARED_PTR_H_
