// Copyright 2025 Google LLC.
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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_ACCELERATOR_REGISTRY_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_ACCELERATOR_REGISTRY_H_

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_shared_library.h"
#include "tensorflow/lite/experimental/litert/runtime/accelerator.h"

namespace litert::internal {

// Holds a list of accelerators.
//
// This is a helper class for the LiteRT environment that manages the
// accelerators (and their resources) that are registered with it.
class AcceleratorRegistry {
 public:
  struct Deleter {
    void operator()(LiteRtAcceleratorT* accelerator) {
      DestroyAccelerator(accelerator);
    }
  };

  // Wraps a pointer for LiteRtAcceleratorT with a custom deleter that handles
  // cleaning up the accelerator internal data.
  using Ptr = std::unique_ptr<::LiteRtAcceleratorT, Deleter>;

  // Internal implementation for the C API.
  [[nodiscard]]
  static Ptr CreateEmptyAccelerator() {
    return Ptr(new LiteRtAcceleratorT());
  }

  // Internal implementation for the C API.
  static void DestroyAccelerator(::LiteRtAcceleratorT* accelerator);

  // Registers an accelerator.
  Expected<LiteRtAcceleratorT*> RegisterAccelerator(Ptr accelerator);

  // Returns the idx-th accelerator that was registered.
  [[nodiscard]]
  Expected<LiteRtAcceleratorT*> Get(LiteRtParamIndex idx);

  // Goes through accelerators and find the index of the given one.
  Expected<LiteRtParamIndex> FindAcceleratorIndex(
      LiteRtAcceleratorT* accelerator);

  // Gives ownership of the shared library to the registry.
  //
  // This should be called when an accelerator is loaded from a shared library
  // to tie the library lifetime to the registry.
  //
  // The library will be closed when the registry is destroyed.
  void TakeOwnershipOfSharedLibrary(SharedLibrary library);

  // Returns the number of accelerators that have been registered.
  size_t size() const { return accelerators_.size(); }
  auto begin() const { return accelerators_.begin(); }
  auto begin() { return accelerators_.begin(); }
  auto end() const { return accelerators_.end(); }
  auto end() { return accelerators_.end(); }

 private:
  std::vector<Ptr> accelerators_;
  // Some accelerators are loaded as shared libraries. This list keeps these
  // libraries loaded while the environment uses them.
  std::vector<SharedLibrary> accelerator_shared_libraries_;
};

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_ACCELERATOR_REGISTRY_H_
