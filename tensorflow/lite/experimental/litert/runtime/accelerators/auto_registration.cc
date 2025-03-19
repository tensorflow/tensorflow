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

#include "tensorflow/lite/experimental/litert/runtime/accelerators/auto_registration.h"

#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_shared_library.h"
#include "tensorflow/lite/experimental/litert/core/environment.h"

// Define a function pointer to allow the accelerator registration to be
// overridden by the LiteRT environment. This is to use the GPU accelerator
// statically linked.
extern "C" bool (*LiteRtRegisterStaticLinkedAcceleratorGpu)(
    LiteRtEnvironmentT& environment) = nullptr;

namespace litert {

Expected<void> TriggerAcceleratorAutomaticRegistration(
    LiteRtEnvironmentT& environment) {
  // Register the GPU accelerator.
  if (LiteRtRegisterStaticLinkedAcceleratorGpu != nullptr &&
      LiteRtRegisterStaticLinkedAcceleratorGpu(environment)) {
    LITERT_LOG(LITERT_INFO, "Statically linked GPU accelerator registered.");
    return {};
  }
  auto gpu_registration = RegisterSharedObjectAccelerator(
      environment, /*plugin_path=*/"libLiteRtGpuAccelerator.so",
      /*registration_function_name=*/"LiteRtRegisterAcceleratorGpuOpenCl");
  if (!gpu_registration) {
    LITERT_LOG(LITERT_WARNING,
               "GPU accelerator could not be loaded and registered: %s.",
               gpu_registration.Error().Message().c_str());
  } else {
    LITERT_LOG(LITERT_INFO, "GPU accelerator registered.");
  }
  return {};
};

Expected<void> RegisterSharedObjectAccelerator(
    LiteRtEnvironmentT& environment, absl::string_view plugin_path,
    absl::string_view registration_function_name) {
  auto maybe_lib = SharedLibrary::Load(plugin_path, RtldFlags::Lazy().Local());
  if (!maybe_lib.HasValue()) {
    maybe_lib = SharedLibrary::Load(RtldFlags::kDefault);
  }
  // Note: the Load(kDefault) overload always succeeds, so we are sure that
  // maybe_lib contains a value.
  SharedLibrary lib(std::move(maybe_lib.Value()));
  LITERT_ASSIGN_OR_RETURN(auto registration_function,
                          lib.LookupSymbol<LiteRtStatus (*)(LiteRtEnvironment)>(
                              registration_function_name.data()));
  LITERT_RETURN_IF_ERROR(registration_function(&environment));
  environment.GetAcceleratorRegistry().TakeOwnershipOfSharedLibrary(
      std::move(lib));
  return {};
}

}  // namespace litert
