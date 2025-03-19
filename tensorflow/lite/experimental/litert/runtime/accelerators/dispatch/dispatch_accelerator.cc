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

#include "tensorflow/lite/experimental/litert/runtime/accelerators/dispatch/dispatch_accelerator.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/experimental/litert/c/litert_accelerator.h"
#include "tensorflow/lite/experimental/litert/c/litert_accelerator_compilation_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_accelerator_registration.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_dispatch_delegate.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment.h"
#include "tensorflow/lite/experimental/litert/cc/litert_any.h"
#include "tensorflow/lite/experimental/litert/cc/litert_dispatch_delegate.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/core/environment.h"
#include "tensorflow/lite/experimental/litert/runtime/accelerator_model_compilation_data.h"

namespace litert {

class NpuAccelerator final {
  constexpr static const absl::string_view kName = "NpuAccelerator";
  // Warning: this should be incremented every time the code of this accelerator
  // is updated according to semanting versioning.
  constexpr static const LiteRtApiVersion kVersion{1, 0, 0};
  constexpr static const LiteRtHwAcceleratorSet kHwSupport =
      kLiteRtHwAcceleratorNpu;

 public:
  explicit NpuAccelerator(std::string library_folder)
      : library_folder_(std::move(library_folder)) {}

  struct Deleter {
    void operator()(NpuAccelerator* npu_accelerator) { delete npu_accelerator; }
  };
  using Ptr = std::unique_ptr<NpuAccelerator, Deleter>;

  static Expected<Ptr> Create(std::string library_folder) {
    LITERT_RETURN_IF_ERROR(
        !library_folder.empty(),
        Error(kLiteRtStatusErrorInvalidArgument,
              "Dispatch API implementation library folder was not specified."));
    return Ptr(new NpuAccelerator(std::move(library_folder)));
  }

  // C API

  // Deletes the accelerator data.
  static void Destroy(void* npu_accelerator) {
    Deleter()(reinterpret_cast<NpuAccelerator*>(npu_accelerator));
  }

  // Stores the accelerator's name in `name`.
  static LiteRtStatus GetName(LiteRtAccelerator accelerator,
                              const char** name) {
    LITERT_ENSURE(accelerator != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Accelerator handle is invalid.");
    LITERT_ENSURE(name != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Name pointer is null.");
    *name = kName.data();
    return kLiteRtStatusOk;
  }

  // Stores the accelerator's version in `version`.
  static LiteRtStatus GetVersion(LiteRtAccelerator accelerator,
                                 LiteRtApiVersion* version) {
    LITERT_ENSURE(accelerator != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Accelerator handle is invalid.");
    LITERT_ENSURE(version != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Version pointer is null.");
    *version = kVersion;
    return kLiteRtStatusOk;
  }

  // Stores the accelerator's hardware support in `hw_set`.
  static LiteRtStatus GetHardwareSupport(LiteRtAccelerator accelerator,
                                         LiteRtHwAcceleratorSet* hw_set) {
    LITERT_ENSURE(accelerator != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Accelerator handle is invalid.");
    LITERT_ENSURE(hw_set != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Harware support pointer is null.");
    *hw_set = kHwSupport;
    return kLiteRtStatusOk;
  }

  // Goes through the options in the linked list and returns the model
  // compilation data if it exists.
  static Expected<const litert::internal::ModelCompilationData*>
  GetModelCompilationData(LiteRtAcceleratorCompilationOptions options) {
    LiteRtApiVersion payload_version;
    void* payload_data;
    LITERT_RETURN_IF_ERROR(LiteRtFindAcceleratorCompilationOptionsData(
        options, litert::internal::ModelCompilationData::kIdentifier,
        &payload_version, &payload_data));
    return reinterpret_cast<litert::internal::ModelCompilationData*>(
        payload_data);
  }

  // Creates a Dispatch delegate instance.
  static LiteRtStatus CreateDelegate(
      LiteRtAccelerator accelerator,
      LiteRtAcceleratorCompilationOptions options, void** delegate) {
    LITERT_ENSURE(delegate != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Delegate pointer is null.");
    LITERT_ENSURE(accelerator != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Accelerator handle is invalid.");
    LITERT_ENSURE(accelerator->env != nullptr,
                  kLiteRtStatusErrorInvalidArgument,
                  "Accelerator is not registered to an environment.");

    LITERT_ASSIGN_OR_RETURN(const auto* compilation_data,
                            GetModelCompilationData(options));
    const char* allocation_base = compilation_data->allocation_base;

    LITERT_ENSURE(allocation_base != nullptr, kLiteRtStatusErrorRuntimeFailure,
                  "No model allocation was passed by the runtime.");

    auto dispatch_delegate_options = litert::CreateDispatchDelegateOptionsPtr(
        &accelerator->env->GetOptions());
    LITERT_ENSURE(dispatch_delegate_options != nullptr,
                  kLiteRtStatusErrorRuntimeFailure,
                  "Dispatch delegate options failed to be created.");

    LITERT_ENSURE(
        LiteRtDispatchDelegateAddAllocBaseOption(
            dispatch_delegate_options.get(), allocation_base) == kTfLiteOk,
        kLiteRtStatusErrorRuntimeFailure,
        "Could not add allocation base to dispatch delegate options.");

    auto dispatch_delegate = litert::CreateDispatchDelegatePtr(
        &accelerator->env->GetOptions(), std::move(dispatch_delegate_options));
    LITERT_ENSURE(dispatch_delegate != nullptr,
                  kLiteRtStatusErrorRuntimeFailure,
                  "Dispatch delegate failed to be created.");

    *delegate = dispatch_delegate.release();
    return kLiteRtStatusOk;
  }

  // Destroys a Dispatch delegate instance.
  static void DestroyDelegate(void* delegate) {
    LiteRtDestroyDispatchDelegate(
        reinterpret_cast<TfLiteOpaqueDelegate*>(delegate));
  }

 private:
  // Note: we do not directly use the option structure because we want to copy
  // and own all the option data.

  // Folder to the Dispatch API implementation shared library.
  std::string library_folder_;
};

namespace {

struct AcceleratorDestructor {
  void operator()(LiteRtAccelerator accelerator) {
    LiteRtDestroyAccelerator(accelerator);
  }
};

using AcceleratorGuard =
    std::unique_ptr<std::pointer_traits<LiteRtAccelerator>::element_type,
                    AcceleratorDestructor>;

}  // namespace
}  // namespace litert

extern "C" {

LiteRtStatus LiteRtRegisterNpuAccelerator(
    LiteRtEnvironmentT* environment, LiteRtNpuAcceleratorOptions* options) {
  LITERT_ENSURE(environment != nullptr, kLiteRtStatusErrorInvalidArgument,
                "accelerator handle is invalid");
  LiteRtAccelerator accelerator_handle;
  LITERT_RETURN_IF_ERROR(LiteRtCreateAccelerator(&accelerator_handle));
  litert::AcceleratorGuard accelerator(accelerator_handle);

  LiteRtSetAcceleratorGetName(accelerator.get(),
                              litert::NpuAccelerator::GetName);
  LiteRtSetAcceleratorGetVersion(accelerator.get(),
                                 litert::NpuAccelerator::GetVersion);
  LiteRtSetAcceleratorGetHardwareSupport(
      accelerator.get(), litert::NpuAccelerator::GetHardwareSupport);

  LiteRtSetDelegateFunction(accelerator.get(),
                            litert::NpuAccelerator::CreateDelegate,
                            litert::NpuAccelerator::DestroyDelegate);

  std::string library_folder;
  if (options && options->library_folder) {
    library_folder = options->library_folder;
  }
  // Check the environment options if the library folder wasn't set in the
  // options.
  if (library_folder.empty()) {
    if (auto env_library_folder =
            environment->GetOption(kLiteRtEnvOptionTagDispatchLibraryDir);
        env_library_folder.has_value()) {
      LITERT_ASSIGN_OR_RETURN(
          library_folder, litert::Get<std::string>(env_library_folder.value()));
    }
  }

  LITERT_ASSIGN_OR_RETURN(
      auto accelerator_impl,
      litert::NpuAccelerator::Create(std::move(library_folder)));

  LITERT_RETURN_IF_ERROR(LiteRtRegisterAccelerator(
      environment, accelerator.release(), accelerator_impl.release(),
      litert::NpuAccelerator::Destroy));
  return kLiteRtStatusOk;
}

}  // extern "C"
