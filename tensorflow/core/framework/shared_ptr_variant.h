/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_SHARED_PTR_VARIANT_H_
#define TENSORFLOW_CORE_FRAMEWORK_SHARED_PTR_VARIANT_H_

#include <memory>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

template <typename T>
struct SharedPtrVariant {
  std::shared_ptr<T> shared_ptr;

  SharedPtrVariant() : shared_ptr() {}

  explicit SharedPtrVariant(std::shared_ptr<T>&& ptr)
      : shared_ptr(std::forward<decltype(ptr)>(ptr)) {
    VLOG(3) << "Creating shared_ptr of " << shared_ptr.get()
            << " count is: " << shared_ptr.use_count();
  }

  SharedPtrVariant(SharedPtrVariant&& rhs)
      : shared_ptr(std::move(rhs.shared_ptr)) {
    VLOG(3) << "Moving SharedPtrVariant of " << shared_ptr.get()
            << " count is: " << shared_ptr.use_count();
  }

  SharedPtrVariant& operator=(const SharedPtrVariant& rhs) = delete;

  SharedPtrVariant& operator=(SharedPtrVariant&& rhs) {
    if (&rhs == this) return *this;
    std::swap(shared_ptr, rhs.shared_ptr);
    VLOG(3) << "Move-assign of SharedPtrVariant of " << shared_ptr.get()
            << " count is: " << shared_ptr.use_count();
    return *this;
  }

  SharedPtrVariant(const SharedPtrVariant& rhs) : shared_ptr(rhs.shared_ptr) {
    VLOG(3) << "Copying SharedPtrVariant of " << shared_ptr.get()
            << " count is: " << shared_ptr.use_count();
  }

  ~SharedPtrVariant() {
    VLOG(3) << "Destroying SharedPtrVariant of " << shared_ptr.get()
            << " count is: " << shared_ptr.use_count();
  }

  void Encode(VariantTensorData*) const {
    // Not supported.
  }

  bool Decode(const VariantTensorData&) {
    return false;  // Not supported.
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_SHARED_PTR_VARIANT_H_
