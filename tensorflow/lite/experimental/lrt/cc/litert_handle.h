// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITERT_HANDLE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITERT_HANDLE_H_

#include <algorithm>

namespace litert {
namespace internal {

// This class is used to wrap and manage the lifetime of opaque handles from the
// C API into an equivalent C++ object. The class is similar to an
// std::unique_ptr<> with a deleter, which is introduced in C++23.
template <typename H>
class Handle {
 public:
  Handle() = default;
  Handle(H h, void (*deleter)(H)) : h_(h), deleter_(deleter) {}

  ~Handle() {
    if (deleter_ && h_) {
      deleter_(h_);
    }
  }

  Handle(Handle&& other) {
    std::swap(h_, other.h_);
    std::swap(deleter_, other.deleter_);
  }

  Handle& operator=(Handle&& other) {
    std::swap(h_, other.h_);
    std::swap(deleter_, other.deleter_);
    return *this;
  }

  Handle(const Handle&) = delete;
  Handle& operator=(const Handle& other) = delete;

  // Return true if the underlying handle is valid.
  bool IsValid() const { return h_ != nullptr; }

  H Get() {
    assert(h_);
    return h_;
  }

  H Get() const {
    assert(h_);
    return h_;
  }

  H Release() {
    assert(h_);
    deleter_ = nullptr;
    return h_;
  }

 private:
  H h_ = nullptr;
  void (*deleter_)(H) = nullptr;
};

}  // namespace internal
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITERT_HANDLE_H_
