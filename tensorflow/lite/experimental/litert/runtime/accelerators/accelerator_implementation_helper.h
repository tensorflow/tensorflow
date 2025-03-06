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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_ACCELERATORS_ACCELERATOR_IMPLEMENTATION_HELPER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_ACCELERATORS_ACCELERATOR_IMPLEMENTATION_HELPER_H_

#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_accelerator.h"
#include "tensorflow/lite/experimental/litert/c/litert_accelerator_compilation_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_accelerator_registration.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/runtime/accelerator_model_compilation_data.h"

namespace litert::internal {

struct AcceleratorDestructor {
  void operator()(LiteRtAccelerator accelerator) {
    LiteRtDestroyAccelerator(accelerator);
  }
};

// RAII wrapper for accelerator handles.
using AcceleratorGuard =
    std::unique_ptr<std::pointer_traits<LiteRtAccelerator>::element_type,
                    AcceleratorDestructor>;

// Helps setting up an accelerator handle for accelerators that use the
// `AcceleratorImplementationHelper` template as a base class.
template <class T>
Expected<void> SetAcceleratorBoilerplateFunctions(
    AcceleratorGuard& accelerator) {
  LITERT_RETURN_IF_ERROR(
      LiteRtSetAcceleratorGetName(accelerator.get(), T::GetName));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetAcceleratorGetVersion(accelerator.get(), T::GetVersion));
  LITERT_RETURN_IF_ERROR(LiteRtSetAcceleratorGetHardwareSupport(
      accelerator.get(), T::GetHardwareSupport));
  LITERT_RETURN_IF_ERROR(LiteRtSetDelegateFunction(
      accelerator.get(), T::CreateDelegate, T::DestroyDelegate));
  return {};
}

// Goes through the options in the linked list and returns the model
// compilation data if it exists.
inline static Expected<const litert::internal::ModelCompilationData*>
GetModelCompilationData(LiteRtAcceleratorCompilationOptions options) {
  LiteRtApiVersion payload_version;
  void* payload_data;
  LITERT_RETURN_IF_ERROR(LiteRtFindAcceleratorCompilationOptionsData(
      options, litert::internal::ModelCompilationData::kIdentifier,
      &payload_version, &payload_data));
  return reinterpret_cast<litert::internal::ModelCompilationData*>(
      payload_data);
}

// Helps accelerator implementation by providing a lot of the boilerplate
// needed.
//
// Warning: The provided Ptr assumes that AcceleratorClass instances are
// created using `operator new`.
//
// Warning: `version` should be incremented every time the code of this
// accelerator is updated according to semanting versioning.
template <class AcceleratorClass, const char* name_, LiteRtApiVersion version_,
          LiteRtHwAcceleratorSet hardware_support_>
class AcceleratorImplementationHelper {
 public:
  // The accelerator name returned by `GetName`.
  constexpr static const absl::string_view kName = name_;
  // The accelerator version returned by `GetVersion`.
  constexpr static const LiteRtApiVersion kVersion = version_;
  // The accelerator hardware support returned by `GetHardwareSupport`.
  constexpr static const LiteRtHwAcceleratorSet kHwSupport = hardware_support_;

  struct Deleter {
    void operator()(AcceleratorClass* accelerator_impl) {
      delete accelerator_impl;
    }
  };

  // Owning pointer wrapping the accelerator.
  using Ptr = std::unique_ptr<AcceleratorClass, Deleter>;

  // Creates a new instance of the accelerator implementation.
  template <class... Args>
  static Ptr Allocate(Args&&... args) {
    return Ptr(new AcceleratorClass(std::forward<Args>(args)...));
  }

  // Deletes the accelerator data.
  static void Destroy(void* accelerator_impl) {
    Deleter()(reinterpret_cast<AcceleratorClass*>(accelerator_impl));
  }

  // Returns the accelerator's name by setting `name`.
  static LiteRtStatus GetName(LiteRtAccelerator accelerator,
                              const char** name) {
    LITERT_ENSURE(accelerator != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Accelerator handle is invalid.");
    LITERT_ENSURE(name != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Name pointer is null.");
    *name = kName.data();
    return kLiteRtStatusOk;
  }

  // Returns the accelerator's version by setting `version`.
  static LiteRtStatus GetVersion(LiteRtAccelerator accelerator,
                                 LiteRtApiVersion* version) {
    LITERT_ENSURE(accelerator != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Accelerator handle is invalid.");
    LITERT_ENSURE(version != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Version pointer is null.");
    *version = kVersion;
    return kLiteRtStatusOk;
  }

  // Returns the accelerator's hardware support by setting `hw_set`.
  static LiteRtStatus GetHardwareSupport(LiteRtAccelerator accelerator,
                                         LiteRtHwAcceleratorSet* hw_set) {
    LITERT_ENSURE(accelerator != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Accelerator handle is invalid.");
    LITERT_ENSURE(hw_set != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Hardware support pointer is null.");
    *hw_set = kHwSupport;
    return kLiteRtStatusOk;
  }
};

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_ACCELERATORS_ACCELERATOR_IMPLEMENTATION_HELPER_H_
