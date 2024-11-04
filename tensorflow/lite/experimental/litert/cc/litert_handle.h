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

namespace litert {
namespace internal {

// This class is used to wrap and manage the lifetime of opaque handles from the
// C API into an equivalent C++ object. The class is a wrapper on
// std::unique_ptr<> that has a default constructor and doesn't crash if the
// deleter is null.
template <typename T>
class Handle : public std::unique_ptr<T, void (*)(T*)> {
 public:
  Handle() : std::unique_ptr<T, void (*)(T*)>(nullptr, DummyDeleter) {}
  Handle(T* ptr, void (*deleter)(T*))
      : std::unique_ptr<T, void (*)(T*)>(ptr,
                                         deleter ? deleter : DummyDeleter) {}

 private:
  static void DummyDeleter(T*) {}
};

}  // namespace internal
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_HANDLE_H_
