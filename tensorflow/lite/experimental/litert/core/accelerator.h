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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_ACCELERATOR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_ACCELERATOR_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

extern "C" {

// We need to forward declare this to avoid a dependency loop.
struct LiteRtCompiledModelT;

struct LiteRtAcceleratorT {
  // Points to the type-erased accelerator state.
  void* data;

  // NOLINTBEGIN(*-readability-class-member-naming)

  // Releases the the data.
  //
  // This function is used by the framework to clean up the accelerator. It
  // should not be called by client code.
  void (*ReleaseData)(void*);

  // Retrieves the accelerator name.
  LiteRtStatus (*GetName)(LiteRtAcceleratorT* accelerator, const char** name);

  // Retrieves the accelerator version.
  LiteRtStatus (*GetVersion)(LiteRtAcceleratorT* accelerator,
                             LiteRtApiVersion* version);

  // Retrieves the accelerator hardware support.
  LiteRtStatus (*GetHardwareSupport)(
      LiteRtAcceleratorT* accelerator,
      LiteRtHwAcceleratorSet* supported_hardware);

  // Applies the accelerator to a compiled model.
  LiteRtStatus (*ApplyToModel)(LiteRtAcceleratorT* accelerator,
                               LiteRtCompiledModelT* supported_hardware);

  // NOLINTEND(*-readability-class-member-naming)
};

}  // extern "C"

#ifdef __cplusplus

namespace litert::internal {

class AcceleratorRegistry {
 public:
  struct Deleter {
    void operator()(LiteRtAcceleratorT* accelerator) {
      DestroyAccelerator(accelerator);
    }
  };

  // Wraps a pointer for LiteRtAcceleratorT with a custom deleter that handles
  // cleaning up the accelerator internal data.
  using Ptr = std::unique_ptr<LiteRtAcceleratorT, Deleter>;

  // Internal implementation for the C API.
  [[nodiscard]]
  static Ptr CreateEmptyAccelerator() {
    return Ptr(new LiteRtAcceleratorT());
  }

  // Internal implementation for the C API.
  static void DestroyAccelerator(LiteRtAcceleratorT* accelerator);

  // Registers an accelerator.
  Expected<LiteRtAcceleratorT*> RegisterAccelerator(Ptr accelerator);

  // Returns the idx-th accelerator that was registered.
  [[nodiscard]]
  Expected<LiteRtAcceleratorT*> Get(LiteRtParamIndex idx);

  // Goes through accelerators and find the index of the given one.
  Expected<LiteRtParamIndex> FindAcceleratorIndex(
      LiteRtAcceleratorT* accelerator);

  // Returns the number of accelerators that have been registered.
  size_t size() const { return accelerators_.size(); }
  auto begin() const { return accelerators_.begin(); }
  auto begin() { return accelerators_.begin(); }
  auto end() const { return accelerators_.end(); }
  auto end() { return accelerators_.end(); }

 private:
  std::vector<Ptr> accelerators_;
};

}  // namespace litert::internal

#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_ACCELERATOR_H_
