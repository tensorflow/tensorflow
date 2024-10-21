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

#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/qnn_manager.h"

#include <cstdint>
#include <iostream>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/qairt/latest/include/QNN/QnnCommon.h"
#include "third_party/qairt/latest/include/QNN/QnnInterface.h"
#include "third_party/qairt/latest/include/QNN/QnnLog.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "third_party/qairt/latest/include/QNN/System/QnnSystemCommon.h"
#include "third_party/qairt/latest/include/QNN/System/QnnSystemContext.h"
#include "tensorflow/lite/experimental/lrt/c/litert_common.h"
#include "tensorflow/lite/experimental/lrt/c/litert_logging.h"
#include "tensorflow/lite/experimental/lrt/c/litert_support.h"
#include "tensorflow/lite/experimental/lrt/core/dynamic_loading.h"
#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/common.h"
#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/qnn_log.h"

namespace litert::qnn {

namespace {

constexpr char kLibQnnGetProvidersSymbol[] = "QnnInterface_getProviders";

constexpr char kLibQnnSystemGetProvidersSymbol[] =
    "QnnSystemInterface_getProviders";

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(
    const QnnInterface_t*** provider_list, uint32_t* num_providers);

typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(
    const QnnSystemInterface_t***, uint32_t*);

absl::Span<const QnnInterface_t*> LoadProvidersFromLib(void* lib_so) {
  QnnInterfaceGetProvidersFn_t get_providers = nullptr;
  LITERT_RETURN_VAL_IF_NOT_OK(
      litert::ResolveLibSymbol<QnnInterfaceGetProvidersFn_t>(
          lib_so, kLibQnnGetProvidersSymbol, &get_providers),
      {});

  const QnnInterface_t** interface_providers = nullptr;
  uint32_t num_providers = 0;
  if (QNN_SUCCESS != get_providers(&interface_providers, &num_providers)) {
    LITERT_LOG(LITERT_ERROR, "%s", "Failed to get providers\n");
    return {};
  }

  return absl::MakeSpan(interface_providers, num_providers);
}

absl::Span<const QnnSystemInterface_t*> LoadSystemProvidersFromLib(
    void* lib_so) {
  QnnSystemInterfaceGetProvidersFn_t get_providers = nullptr;
  LITERT_RETURN_VAL_IF_NOT_OK(
      litert::ResolveLibSymbol<QnnSystemInterfaceGetProvidersFn_t>(
          lib_so, kLibQnnSystemGetProvidersSymbol, &get_providers),
      {});

  const QnnSystemInterface_t** interface_providers = nullptr;
  uint32_t num_providers = 0;
  if (QNN_SUCCESS != get_providers(&interface_providers, &num_providers)) {
    LITERT_LOG(LITERT_ERROR, "%s", "Failed to get system providers\n");
    return {};
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
  LITERT_RETURN_STATUS_IF_NOT_OK(litert::OpenLib(path, &lib_so_));
  return kLiteRtStatusOk;
}

LiteRtStatus QnnManager::LoadSystemLib(absl::string_view path) {
  LITERT_RETURN_STATUS_IF_NOT_OK(litert::OpenLib(path, &lib_system_so_));
  return kLiteRtStatusOk;
}

const QnnApi* QnnManager::Api() const {
  if (interface_ == nullptr) {
    return nullptr;
  }
  return &interface_->QNN_INTERFACE_VER_NAME;
}

LiteRtStatus QnnManager::ResolveApi() {
  if (lib_so_ == nullptr) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "Cannot resolve functions: libQnn*.so has not been loaded.\n");
    return kLiteRtStatusErrorDynamicLoading;
  }

  auto providers = LoadProvidersFromLib(lib_so_);
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
  if (lib_so_ == nullptr) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "Cannot resolve functions: libQnn*.so has not been loaded.\n");
    return kLiteRtStatusErrorDynamicLoading;
  }

  auto system_providers = LoadSystemProvidersFromLib(lib_system_so_);
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

LiteRtStatus QnnManager::Init(absl::Span<const QnnBackend_Config_t*> configs,
                              std::optional<std::string> shared_library_dir,
                              std::optional<QnnHtpDevice_Arch_t> soc_model) {
  if (shared_library_dir.has_value()) {
    // We must change the variable environment used to load DSP libraries.
    std::string new_adsp_library_path;
    if (auto* adsp_library_path = getenv("ADSP_LIBRARY_PATH");
        adsp_library_path != nullptr) {
      new_adsp_library_path = absl::StrFormat(
          "%s:%s", shared_library_dir->data(), adsp_library_path);
    } else {
      new_adsp_library_path = shared_library_dir->data();
    }
    LITERT_LOG(LITERT_INFO, "Setting ADSP_LIBRARY_PATH to %s",
               new_adsp_library_path.data());
    setenv("ADSP_LIBRARY_PATH", new_adsp_library_path.data(), /*overwrite=*/1);
  }

  auto lib_qnn_htp_so_path =
      shared_library_dir.has_value()
          ? absl::StrFormat("%s/%s", shared_library_dir->data(), kLibQnnHtpSo)
          : kLibQnnHtpSo;
  LITERT_RETURN_STATUS_IF_NOT_OK(LoadLib(lib_qnn_htp_so_path));
  LITERT_RETURN_STATUS_IF_NOT_OK(ResolveApi());

  auto lib_qnn_system_so_path =
      shared_library_dir.has_value()
          ? absl::StrFormat("%s/%s", shared_library_dir->data(),
                            kLibQnnSystemSo)
          : kLibQnnSystemSo;
  LITERT_RETURN_STATUS_IF_NOT_OK(LoadSystemLib(lib_qnn_system_so_path));
  LITERT_RETURN_STATUS_IF_NOT_OK(ResolveSystemApi());

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

absl::StatusOr<QnnManager::SystemContextHandle>
QnnManager::CreateSystemContextHandle() {
  QnnSystemContext_Handle_t system_context_handle;
  if (auto status = SystemApi()->systemContextCreate(&system_context_handle);
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to create QNN system context: %d", status);
    return absl::InternalError("Failed to create QNN system context");
  }
  auto deleter = SystemApi()->systemContextFree;
  return SystemContextHandle{system_context_handle, deleter};
}

absl::StatusOr<QnnManager::ContextHandle> QnnManager::CreateContextHandle(
    absl::Span<const QnnContext_Config_t*> configs) {
  Qnn_ContextHandle_t context_handle;
  if (auto status = Api()->contextCreate(BackendHandle(), DeviceHandle(),
                                         configs.data(), &context_handle);
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to create QNN context: %d", status);
    return absl::InternalError("Failed to create QNN context");
  }
  auto deleter = Api()->contextFree;
  return ContextHandle{context_handle, /*profile_handle=*/nullptr, deleter};
}

absl::StatusOr<QnnManager::ContextHandle> QnnManager::CreateContextHandle(
    absl::Span<const QnnContext_Config_t*> configs,
    absl::Span<const uint8_t> bytecode, Qnn_ProfileHandle_t profile_handle) {
  Qnn_ContextHandle_t context_handle;
  if (auto status = Api()->contextCreateFromBinary(
          BackendHandle(), DeviceHandle(), configs.data(), bytecode.data(),
          bytecode.size(), &context_handle, profile_handle);
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to create QNN context: %d", status);
    return absl::InternalError("Failed to create QNN context");
  }
  auto deleter = Api()->contextFree;
  return ContextHandle{context_handle, profile_handle, deleter};
}

absl::StatusOr<QnnManager::Ptr> QnnManager::Create(
    absl::Span<const QnnBackend_Config_t*> configs,
    std::optional<std::string> shared_library_dir,
    std::optional<QnnHtpDevice_Arch_t> soc_model) {
  Ptr qnn_manager(new QnnManager);
  if (qnn_manager->Init(configs, shared_library_dir, soc_model) !=
      kLiteRtStatusOk) {
    return absl::InternalError("Failed to set up QNN manager");
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

};  // namespace litert::qnn
