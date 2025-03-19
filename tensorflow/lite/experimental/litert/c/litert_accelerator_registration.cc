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

#include "tensorflow/lite/experimental/litert/c/litert_accelerator_registration.h"

#include <memory>
#include <utility>

#include "tensorflow/lite/experimental/litert/c/litert_accelerator.h"
#include "tensorflow/lite/experimental/litert/c/litert_accelerator_compilation_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/core/environment.h"
#include "tensorflow/lite/experimental/litert/runtime/accelerator.h"

LiteRtStatus LiteRtCreateAccelerator(LiteRtAccelerator* accelerator) {
  if (!accelerator) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *accelerator =
      litert::internal::AcceleratorRegistry::CreateEmptyAccelerator().release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDestroyAccelerator(LiteRtAccelerator accelerator) {
  litert::internal::AcceleratorRegistry::DestroyAccelerator(accelerator);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtRegisterAccelerator(LiteRtEnvironment environment,
                                       LiteRtAccelerator accelerator,
                                       void* data, void (*ReleaseData)(void*)) {
  std::unique_ptr<void, void (*)(void*)> data_guard(data, ReleaseData);
  litert::internal::AcceleratorRegistry::Ptr accelerator_guard(accelerator);
  if (!accelerator_guard) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  accelerator_guard->env = environment;
  litert::Expected<LiteRtAccelerator> registered_accelerator =
      environment->GetAcceleratorRegistry().RegisterAccelerator(
          std::move(accelerator_guard));
  if (!registered_accelerator.HasValue()) {
    return registered_accelerator.Error().Status();
  }
  registered_accelerator.Value()->data = data_guard.release();
  registered_accelerator.Value()->ReleaseData = ReleaseData;
  return kLiteRtStatusOk;
}

// Sets the function used to retrieve the accelerator name.
LiteRtStatus LiteRtSetAcceleratorGetName(
    LiteRtAccelerator accelerator,
    LiteRtStatus (*GetName)(LiteRtAccelerator accelerator, const char** name)) {
  if (!accelerator) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  accelerator->GetName = GetName;
  return kLiteRtStatusOk;
}

// Sets the function used to retrieve the accelerator version.
LiteRtStatus LiteRtSetAcceleratorGetVersion(
    LiteRtAccelerator accelerator,
    LiteRtStatus (*GetVersion)(LiteRtAccelerator accelerator,
                               LiteRtApiVersion* version)) {
  if (!accelerator) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  accelerator->GetVersion = GetVersion;
  return kLiteRtStatusOk;
}

// Sets the function used to retrieve the accelerator hardware support.
LiteRtStatus LiteRtSetAcceleratorGetHardwareSupport(
    LiteRtAccelerator accelerator,
    LiteRtStatus (*GetHardwareSupport)(
        LiteRtAccelerator accelerator,
        LiteRtHwAcceleratorSet* supported_hardware)) {
  if (!accelerator) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  accelerator->GetHardwareSupport = GetHardwareSupport;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetDelegateFunction(
    LiteRtAccelerator accelerator,
    LiteRtStatus (*CreateDelegate)(LiteRtAccelerator accelerator,
                                   LiteRtAcceleratorCompilationOptions options,
                                   void** delegate),
    void (*DestroyDelegate)(void* delegate)) {
  if (!accelerator) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  accelerator->CreateDelegate = CreateDelegate;
  accelerator->DestroyDelegate = DestroyDelegate;
  return kLiteRtStatusOk;
}
