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
struct LiteRtEnvironmentT;

struct LiteRtAcceleratorT {
  // Points to the type-erased accelerator state.
  void* data;

  // Points to the environment that owns this accelerator.
  LiteRtEnvironmentT* env;

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

// This must be the very first field (or base) of every accelerator option
// object in order to make the option objects part of a list.
struct LiteRtAcceleratorCompilationOptionsHeader {
  // Pointer to the next link structure.
  LiteRtAcceleratorCompilationOptionsHeader* next;

  // Identifier for the configuration structure. Used by the accelerator
  // implementation to go through the list and reinterpret the link to its
  // actual type.
  const char* identifier;

  // NOLINTBEGIN(*-readability-class-member-naming)

  // A destructor for this link's data. Releases the memory stored in this link
  // AS WELL AS THE LINK ITSELF.
  //
  // We need this for option objects that may own some of their data. The most
  // common use case here being helper functions that build a path from other
  // program inputs. If the options structure doesn't own the data, then the
  // user must ensure that the string outlives the compiled model, which may be
  // tricky. This let's the user define a function that will be called to clean
  // up the data.
  void (*ReleaseData)(LiteRtAcceleratorCompilationOptionsHeader*);

  // NOLINTEND(*-readability-class-member-naming)

  // The version of the option structure. This allows the consumer code to know
  // the size of the structure and the fields that are accessible.
  //
  // Note: THIS SHOULD BE INCREMENTED EVERY TIME AN OPTION IS ADDED.
  LiteRtApiVersion version;
};

// Sets the destructor options destructor.
//
// We need this for option objects that may own some of their data. The most
// common use case here being helper functions that build a path from other
// program inputs. If the options structure doesn't own the data, then the user
// must ensure that the string outlives the compiled model, which may be tricky.
// This lets the user define a function that will be called to clean up the
// data.
LiteRtStatus LiteRtSetAcceleratorCompilationOptionsDestructor(
    LiteRtAcceleratorCompilationOptionsHeader* options,
    void (*Destructor)(LiteRtAcceleratorCompilationOptionsHeader*));

// Sets the identifier for an acceleration compilation option object.
//
// Warning: The identifier's lifetime is not managed by the object.
LiteRtStatus LiteRtSetAcceleratorCompilationOptionsIdentifier(
    LiteRtAcceleratorCompilationOptionsHeader* options, const char* identifier);

// Sets the version to the accelerator options version.
//
// Note: This should probably be in sync with the accelerator code version.
LiteRtStatus LiteRtSetAcceleratorCompilationOptionsVersion(
    LiteRtAcceleratorCompilationOptionsHeader* options,
    LiteRtApiVersion version);

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
