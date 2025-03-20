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

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/qnn_manager.h"

#include <stdlib.h>

#include <cstdint>
#include <filesystem>  // NOLINT
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/qairt/latest/include/QNN/HTP/QnnHtpContext.h"
#include "third_party/qairt/latest/include/QNN/HTP/QnnHtpDevice.h"
#include "third_party/qairt/latest/include/QNN/QnnBackend.h"
#include "third_party/qairt/latest/include/QNN/QnnCommon.h"
#include "third_party/qairt/latest/include/QNN/QnnContext.h"
#include "third_party/qairt/latest/include/QNN/QnnDevice.h"
#include "third_party/qairt/latest/include/QNN/QnnInterface.h"
#include "third_party/qairt/latest/include/QNN/QnnLog.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "third_party/qairt/latest/include/QNN/System/QnnSystemCommon.h"
#include "third_party/qairt/latest/include/QNN/System/QnnSystemContext.h"
#include "third_party/qairt/latest/include/QNN/System/QnnSystemInterface.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_shared_library.h"
#include "tensorflow/lite/experimental/litert/core/dynamic_loading.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/common.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/qnn_log.h"

namespace litert::qnn {

namespace {

constexpr char kLibQnnGetProvidersSymbol[] = "QnnInterface_getProviders";

constexpr char kLibQnnSystemGetProvidersSymbol[] =
    "QnnSystemInterface_getProviders";

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(
    const QnnInterface_t*** provider_list, uint32_t* num_providers);

typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(
    const QnnSystemInterface_t***, uint32_t*);

Expected<absl::Span<const QnnInterface_t*>> LoadProvidersFromLib(
    SharedLibrary& lib) {
  QnnInterfaceGetProvidersFn_t get_providers = nullptr;
  LITERT_ASSIGN_OR_RETURN(get_providers,
                          lib.LookupSymbol<QnnInterfaceGetProvidersFn_t>(
                              kLibQnnGetProvidersSymbol));
  const QnnInterface_t** interface_providers = nullptr;
  uint32_t num_providers = 0;
  if (QNN_SUCCESS != get_providers(&interface_providers, &num_providers)) {
    return Error(kLiteRtStatusErrorRuntimeFailure, "Failed to get providers");
  }
  return absl::MakeSpan(interface_providers, num_providers);
}

Expected<absl::Span<const QnnSystemInterface_t*>> LoadSystemProvidersFromLib(
    SharedLibrary& lib) {
  LITERT_ASSIGN_OR_RETURN(QnnSystemInterfaceGetProvidersFn_t get_providers,
                          lib.LookupSymbol<QnnSystemInterfaceGetProvidersFn_t>(
                              kLibQnnSystemGetProvidersSymbol));
  const QnnSystemInterface_t** interface_providers = nullptr;
  uint32_t num_providers = 0;
  if (QNN_SUCCESS != get_providers(&interface_providers, &num_providers)) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to get system providers");
  }
  return absl::MakeSpan(interface_providers, num_providers);
}

}  // namespace

QnnManager::~QnnManager() {
  (void)FreeDevice();
  (void)FreeBackend();
  (void)FreeLogging();
}

LiteRtStatus QnnManager::LoadLib(absl::string_view path) {
  LITERT_LOG(LITERT_INFO, "Loading qnn shared library from \"%s\"",
             path.data());
  LITERT_ASSIGN_OR_RETURN(lib_,
                          SharedLibrary::Load(path, RtldFlags::Default()));
  LITERT_LOG(LITERT_INFO, "Loaded qnn shared library", "");
  return kLiteRtStatusOk;
}

LiteRtStatus QnnManager::LoadSystemLib(absl::string_view path) {
  LITERT_ASSIGN_OR_RETURN(lib_system_,
                          SharedLibrary::Load(path, RtldFlags::Default()));
  return kLiteRtStatusOk;
}

const QnnApi* QnnManager::Api() const {
  if (interface_ == nullptr) {
    return nullptr;
  }
  return &interface_->QNN_INTERFACE_VER_NAME;
}

LiteRtStatus QnnManager::ResolveApi() {
  if (!lib_.Loaded()) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "Cannot resolve functions: libQnn*.so has not been loaded.\n");
    return kLiteRtStatusErrorDynamicLoading;
  }

  LITERT_ASSIGN_OR_RETURN(auto providers, LoadProvidersFromLib(lib_));
  for (const auto& prov : providers) {
    const bool major =
        prov->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR;

    const bool minor =
        prov->apiVersion.coreApiVersion.minor == QNN_API_VERSION_MINOR;

    const bool patch =
        prov->apiVersion.coreApiVersion.patch == QNN_API_VERSION_PATCH;

    if (major && minor && patch) {
      interface_ = prov;
      break;
    }
  }

  if (interface_ == nullptr) {
    LITERT_LOG(LITERT_ERROR, "%s", "No valid interface was provided\n");
    return kLiteRtStatusErrorDynamicLoading;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus QnnManager::ResolveSystemApi() {
  if (!lib_.Loaded()) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "Cannot resolve functions: libQnn*.so has not been loaded.\n");
    return kLiteRtStatusErrorDynamicLoading;
  }

  LITERT_ASSIGN_OR_RETURN(auto system_providers,
                          LoadSystemProvidersFromLib(lib_system_));
  for (const auto& system_prov : system_providers) {
    const bool major =
        system_prov->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR;

    const bool minor =
        system_prov->systemApiVersion.minor == QNN_SYSTEM_API_VERSION_MINOR;

    const bool patch =
        system_prov->systemApiVersion.patch == QNN_SYSTEM_API_VERSION_PATCH;

    if (major && minor && patch) {
      system_interface_ = system_prov;
      break;
    }
  }

  if (system_interface_ == nullptr) {
    LITERT_LOG(LITERT_ERROR, "%s", "No valid system interface was provided\n");
    return kLiteRtStatusErrorDynamicLoading;
  }

  return kLiteRtStatusOk;
}

const QnnSystemApi* QnnManager::SystemApi() const {
  if (system_interface_ == nullptr) {
    return nullptr;
  }
  return &system_interface_->QNN_SYSTEM_INTERFACE_VER_NAME;
}

LiteRtStatus QnnManager::FreeLogging() {
  if (log_handle_ != nullptr) {
    if (QNN_SUCCESS != Api()->logFree(log_handle_)) {
      LITERT_LOG(LITERT_ERROR, "%s", "Failed to free logging\n");
      return kLiteRtStatusErrorNotFound;
    }
  }
  log_handle_ = nullptr;
  return kLiteRtStatusOk;
}

LiteRtStatus QnnManager::FreeBackend() {
  if (backend_handle_ != nullptr) {
    if (QNN_SUCCESS != Api()->backendFree(backend_handle_)) {
      LITERT_LOG(LITERT_ERROR, "%s", "Failed to free backend\n");
      return kLiteRtStatusErrorNotFound;
    }
  }
  backend_handle_ = nullptr;
  return kLiteRtStatusOk;
}

LiteRtStatus QnnManager::FreeDevice() {
  if (device_handle_ != nullptr) {
    if (QNN_SUCCESS != Api()->deviceFree(device_handle_)) {
      LITERT_LOG(LITERT_ERROR, "%s", "Failed to free device\n");
      return kLiteRtStatusErrorNotFound;
    }
  }
  device_handle_ = nullptr;
  return kLiteRtStatusOk;
}

LiteRtStatus QnnManager::GenerateContextBinary(
    Qnn_ContextHandle_t context_handle, std::vector<char>& buffer) {
  Qnn_ContextBinarySize_t bin_size = 0;
  if (QNN_SUCCESS != Api()->contextGetBinarySize(context_handle, &bin_size)) {
    LITERT_LOG(LITERT_ERROR, "%s", "Failed to get context bin size\n");
    return kLiteRtStatusErrorNotFound;
  }
  buffer.clear();
  buffer.resize(bin_size);

  Qnn_ContextBinarySize_t written_bin_size = 0;
  if (QNN_SUCCESS != Api()->contextGetBinary(context_handle, buffer.data(),
                                             buffer.size(),
                                             &written_bin_size)) {
    LITERT_LOG(LITERT_ERROR, "%s", "Failed to generated context binary \n");
    return kLiteRtStatusErrorNotFound;
  }

  LITERT_LOG(LITERT_INFO, "Serialized a context bin of size (bytes): %lu\n",
             written_bin_size);

  return kLiteRtStatusOk;
}

LiteRtStatus QnnManager::ValidateOp(const Qnn_OpConfig_t& op_config) {
  if (Qnn_ErrorHandle_t error =
          Api()->backendValidateOpConfig(BackendHandle(), op_config);
      QNN_SUCCESS != error) {
    LITERT_LOG(LITERT_ERROR, "Failed to validate op %s\n, error: %lld",
               op_config.v1.name, static_cast<long long>(error));
    return kLiteRtStatusErrorInvalidLegalization;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus QnnManager::Init(absl::Span<const QnnBackend_Config_t*> configs,
                              std::optional<std::string> shared_library_dir,
                              std::optional<QnnHtpDevice_Arch_t> soc_model) {
  // If shared_library_dir is provided, add it to the path as it may contain
  // libs to be loaded.
  // TOOD: This should probably be done upstream in litert_dispatch.
  if (shared_library_dir) {
    LITERT_LOG(LITERT_INFO, "Adding shared library dir to path: %s",
               shared_library_dir->c_str());

    static constexpr char kAdsp[] = "ADSP_LIBRARY_PATH";
    if (getenv(kAdsp) == nullptr) {
      setenv(kAdsp, shared_library_dir->data(), /*overwrite=*/1);
    }

    // TODO: Put dynamic loading module in cc or vendor/cc.
    litert::internal::PutLibOnLdPath(shared_library_dir->data(), kLibQnnHtpSo);
  }

  LITERT_RETURN_IF_ERROR(LoadLib(kLibQnnHtpSo));
  LITERT_RETURN_IF_ERROR(ResolveApi());

  LITERT_RETURN_IF_ERROR(LoadSystemLib(kLibQnnSystemSo));
  LITERT_RETURN_IF_ERROR(ResolveSystemApi());

  if (auto status = Api()->logCreate(GetDefaultStdOutLogger(),
                                     QNN_LOG_LEVEL_INFO, &LogHandle());
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to create QNN logger: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  if (auto status =
          Api()->backendCreate(LogHandle(), configs.data(), &BackendHandle());
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to create QNN backend: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  if (soc_model.has_value()) {
    soc_model_ = *soc_model;
    LITERT_LOG(LITERT_INFO,
               "Initializing QNN backend for device architecture %d",
               *soc_model);
    QnnHtpDevice_CustomConfig_t arch_custom_config = {};
    arch_custom_config.option = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
    arch_custom_config.arch.arch = *soc_model;
    arch_custom_config.arch.deviceId = 0;

    QnnDevice_Config_t arch_device_config = {};
    arch_device_config.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
    arch_device_config.customConfig = &arch_custom_config;

    const QnnDevice_Config_t* device_configs[2] = {
        &arch_device_config,
        nullptr,
    };

    if (auto status =
            Api()->deviceCreate(nullptr, device_configs, &DeviceHandle());
        status != QNN_SUCCESS) {
      LITERT_LOG(LITERT_ERROR, "Failed to create QNN device: %d", status);
      return kLiteRtStatusErrorRuntimeFailure;
    }
  }

  return kLiteRtStatusOk;
}

Expected<QnnManager::SystemContextHandle>
QnnManager::CreateSystemContextHandle() {
  QnnSystemContext_Handle_t system_context_handle;
  if (auto status = SystemApi()->systemContextCreate(&system_context_handle);
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to create QNN system context: %d", status);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to create QNN system context");
  }
  auto deleter = SystemApi()->systemContextFree;
  return SystemContextHandle{system_context_handle, deleter};
}

Expected<QnnManager::ContextHandle> QnnManager::CreateContextHandle(
    absl::Span<const QnnContext_Config_t*> configs) {
  Qnn_ContextHandle_t context_handle;
  if (auto status = Api()->contextCreate(BackendHandle(), DeviceHandle(),
                                         configs.data(), &context_handle);
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to create QNN context: %d", status);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to create QNN context");
  }
  auto deleter = Api()->contextFree;
  return ContextHandle{context_handle, /*profile=*/nullptr, deleter};
}

Expected<QnnManager::ContextHandle> QnnManager::CreateContextHandle(
    absl::Span<const QnnContext_Config_t*> configs,
    absl::Span<const uint8_t> bytecode, Qnn_ProfileHandle_t profile_handle) {
  Qnn_ContextHandle_t context_handle;
  if (auto status = Api()->contextCreateFromBinary(
          BackendHandle(), DeviceHandle(), configs.data(), bytecode.data(),
          bytecode.size(), &context_handle, profile_handle);
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to create QNN context: %d", status);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to create QNN context");
  }
  auto deleter = Api()->contextFree;
  return ContextHandle{context_handle, profile_handle, deleter};
}

Expected<QnnManager::Ptr> QnnManager::Create(
    absl::Span<const QnnBackend_Config_t*> configs,
    std::optional<std::string> shared_library_dir,
    std::optional<QnnHtpDevice_Arch_t> soc_model) {
  Ptr qnn_manager(new QnnManager);
  if (auto status = qnn_manager->Init(configs, shared_library_dir, soc_model);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to set up QNN manager");
  }
  return qnn_manager;
}

absl::Span<const QnnBackend_Config_t*> QnnManager::DefaultBackendConfigs() {
  static const QnnBackend_Config_t* configs[] = {nullptr};
  return absl::MakeSpan(configs);
}

absl::Span<const QnnContext_Config_t*> QnnManager::DefaultContextConfigs() {
  static const QnnContext_Config_t* configs[] = {nullptr};
  return absl::MakeSpan(configs);
}

absl::Span<const QnnContext_Config_t*>
QnnManager::WeightSharingContextConfigs() {
  static QnnHtpContext_CustomConfig_t customConfig =
      QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT;
  customConfig.option = QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
  customConfig.weightSharingEnabled = true;
  static QnnContext_Config_t contextConfig = QNN_CONTEXT_CONFIG_INIT;
  contextConfig.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  contextConfig.customConfig = &customConfig;
  static const QnnContext_Config_t* configs[2] = {&contextConfig, nullptr};
  return absl::MakeSpan(configs);
}

};  // namespace litert::qnn
