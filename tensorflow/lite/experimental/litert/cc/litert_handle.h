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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_HANDLE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_HANDLE_H_

#include <memory>
#include <type_traits>

namespace litert {
namespace internal {

template <typename H>
inline void DummyDeleter(H) {}

// This class is used to wrap and manage the lifetime of opaque handles from the
// C API into an equivalent C++ object. The class is a wrapper on
// std::unique_ptr<> that has a default constructor and doesn't crash if the
// deleter is null.
template <typename H, void (*deleter)(H)>
class Handle {
 public:
  Handle() = default;
  explicit Handle(H handle, bool owned) noexcept
      : ptr_(handle, owned ? deleter : DummyDeleter<H>) {}

  Handle(Handle&& other) noexcept { *this = std::move(other); }

  Handle& operator=(Handle&& other) noexcept {
    std::swap(ptr_, other.ptr_);
    return *this;
  }

  // Return true if the underlying LiteRtTensorBuffer handle is valid.
  explicit operator bool() const noexcept { return static_cast<bool>(ptr_); }

  // Return the underlying LiteRtTensorBuffer handle.
  H Get() const noexcept { return ptr_.get(); }

  H Release() noexcept { return ptr_.release(); }

  bool IsOwned() const noexcept {
    return ptr_.get_deleter() != DummyDeleter<H>;
  }

 private:
  std::unique_ptr<std::remove_pointer_t<H>, void (*)(H)> ptr_ = {nullptr,
                                                                 DummyDeleter};
};

// This class is similar to Handle, but the managed opaque handle is not owned
// (i.e., it will not be destroyed).
template <typename H>
class NonOwnedHandle : public Handle<H, DummyDeleter<H>> {
 public:
  explicit NonOwnedHandle(H handle) noexcept
      : Handle<H, DummyDeleter<H>>(handle, /*owned=*/false) {}
};

}  // namespace internal
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_HANDLE_H_
