/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace core {

RefCounted::RefCounted() : ref_(1) {}

RefCounted::~RefCounted() { DCHECK_EQ(ref_.load(), 0); }

void RefCounted::Ref() const {
  DCHECK_GE(ref_.load(), 1);
  ref_.fetch_add(1, std::memory_order_relaxed);
}

bool RefCounted::Unref() const {
  DCHECK_GT(ref_.load(), 0);
  // If ref_==1, this object is owned only by the caller. Bypass a locked op
  // in that case.
  if (ref_.load(std::memory_order_acquire) == 1 || ref_.fetch_sub(1) == 1) {
    // Make DCHECK in ~RefCounted happy
    DCHECK((ref_.store(0), true));
    delete this;
    return true;
  } else {
    return false;
  }
}

bool RefCounted::RefCountIsOne() const {
  return (ref_.load(std::memory_order_acquire) == 1);
}

}  // namespace core
}  // namespace tensorflow
