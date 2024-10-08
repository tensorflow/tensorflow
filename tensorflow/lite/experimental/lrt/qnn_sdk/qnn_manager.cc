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

#include "tensorflow/lite/experimental/lrt/qnn_sdk/qnn_manager.h"

#include <cstdint>
#include <iostream>
#include <vector>

#include "absl/types/span.h"
#include "third_party/qairt/include/QNN/QnnCommon.h"
#include "third_party/qairt/include/QNN/QnnInterface.h"
#include "third_party/qairt/include/QNN/QnnLog.h"
#include "third_party/qairt/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/core/dynamic_loading.h"
#include "tensorflow/lite/experimental/lrt/core/logging.h"
#include "tensorflow/lite/experimental/lrt/qnn_sdk/log.h"

namespace qnn {

namespace {

// This is one of two qnns symbol that needs resolution. It is used to populate
// pointers to related available qnn functions.
constexpr char kLibQnnGetProvidersSymbol[] = "QnnInterface_getProviders";

// Type definition for the QnnInterface_getProviders symbol.
typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(
    const QnnInterface_t*** provider_list, uint32_t* num_providers);

absl::Span<const QnnInterface_t*> LoadProvidersFromLib(void* lib_so) {
  QnnInterfaceGetProvidersFn_t get_providers = nullptr;
  LRT_RETURN_VAL_IF_NOT_OK(
      lrt::ResolveLibSymbol<QnnInterfaceGetProvidersFn_t>(
          lib_so, kLibQnnGetProvidersSymbol, &get_providers),
      {});

  const QnnInterface_t** interface_providers = nullptr;
  uint32_t num_providers = 0;
  if (QNN_SUCCESS != get_providers(&interface_providers, &num_providers)) {
    LITE_RT_LOG(LRT_ERROR, "%s", "Failed to get providers\n");
    return {};
  }

  return absl::MakeSpan(interface_providers, num_providers);
}

}  // namespace

LrtStatus QnnManager::LoadLib(absl::string_view path) {
  LRT_RETURN_STATUS_IF_NOT_OK(lrt::OpenLib(path, &lib_so_));
  return kLrtStatusOk;
}

void QnnManager::DumpLibDetails() const {
  if (lib_so_ == nullptr) {
    return;
  }
  lrt::DumpLibInfo(lib_so_);
}

// TODO: Repace QnnManager::Funcs with indirection access operator.
const QnnApi* QnnManager::Api() const {
  if (interface_ == nullptr) {
    return nullptr;
  }
  return &interface_->QNN_INTERFACE_VER_NAME;
}

LrtStatus QnnManager::ResolveApi() {
  if (lib_so_ == nullptr) {
    LITE_RT_LOG(LRT_ERROR, "%s",
                "Cannot resolve functions: libQnn*.so has not been loaded.\n");
    return kLrtStatusDynamicLoadErr;
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
    LITE_RT_LOG(LRT_ERROR, "%s", "No valid interface was provided\n");
    return kLrtStatusDynamicLoadErr;
  }

  return kLrtStatusOk;
}

LrtStatus QnnManager::FreeLogging() {
  if (log_handle_ != nullptr) {
    if (QNN_SUCCESS != Api()->logFree(log_handle_)) {
      LITE_RT_LOG(LRT_ERROR, "%s", "Failed to free logging\n");
      return kLrtStatusErrorNotFound;
    }
  }
  log_handle_ = nullptr;
  return kLrtStatusOk;
}

LrtStatus QnnManager::FreeBackend() {
  if (backend_handle_ != nullptr) {
    if (QNN_SUCCESS != Api()->backendFree(backend_handle_)) {
      LITE_RT_LOG(LRT_ERROR, "%s", "Failed to free backend\n");
      return kLrtStatusErrorNotFound;
    }
  }
  backend_handle_ = nullptr;
  return kLrtStatusOk;
}

LrtStatus QnnManager::FreeDevice() {
  if (device_handle_ != nullptr) {
    if (QNN_SUCCESS != Api()->deviceFree(device_handle_)) {
      LITE_RT_LOG(LRT_ERROR, "%s", "Failed to free device\n");
      return kLrtStatusErrorNotFound;
    }
  }
  device_handle_ = nullptr;
  return kLrtStatusOk;
}

LrtStatus QnnManager::FreeContext() {
  if (context_handle_ != nullptr) {
    if (QNN_SUCCESS != Api()->contextFree(context_handle_, nullptr)) {
      LITE_RT_LOG(LRT_ERROR, "%s", "Failed to free context\n");
      return kLrtStatusErrorNotFound;
    }
  }
  context_handle_ = nullptr;
  return kLrtStatusOk;
}

LrtStatus QnnManager::GenerateContextBin(std::vector<char>& buffer) {
  Qnn_ContextBinarySize_t bin_size = 0;
  if (QNN_SUCCESS != Api()->contextGetBinarySize(ContextHandle(), &bin_size)) {
    LITE_RT_LOG(LRT_ERROR, "%s", "Failed to get context bin size\n");
    return kLrtStatusErrorNotFound;
  }
  buffer.clear();
  buffer.resize(bin_size);

  Qnn_ContextBinarySize_t written_bin_size = 0;
  if (QNN_SUCCESS != Api()->contextGetBinary(ContextHandle(), buffer.data(),
                                             buffer.size(),
                                             &written_bin_size)) {
    LITE_RT_LOG(LRT_ERROR, "%s", "Failed to generated context binary \n");
    return kLrtStatusErrorNotFound;
  }

  LITE_RT_LOG(LRT_INFO, "Serialized a context bin of size (bytes): %lu\n",
              written_bin_size);

  return kLrtStatusOk;
}

LrtStatus SetupAll(std::optional<QnnHtpDevice_Arch_t> soc_model,
                   QnnManager& qnn) {
  LRT_RETURN_STATUS_IF_NOT_OK(qnn.LoadLib(kLibQnnHtpSo));
  qnn.DumpLibDetails();

  LRT_RETURN_STATUS_IF_NOT_OK(qnn.ResolveApi());

  if (auto status = qnn.Api()->logCreate(qnn::log::GetDefaultStdOutLogger(),
                                         QNN_LOG_LEVEL_DEBUG, &qnn.LogHandle());
      status != QNN_SUCCESS) {
    LITE_RT_LOG(LRT_ERROR, "Failed to create QNN logger: %d", status);
    return kLrtStatusErrorRuntimeFailure;
  }

  {
    auto cfg = qnn::config::GetDefaultHtpConfigs();
    if (auto status = qnn.Api()->backendCreate(qnn.LogHandle(), cfg.data(),
                                               &qnn.BackendHandle());
        status != QNN_SUCCESS) {
      LITE_RT_LOG(LRT_ERROR, "Failed to create QNN backend: %d", status);
      return kLrtStatusErrorRuntimeFailure;
    }
  }

  if (soc_model.has_value()) {
    LITE_RT_LOG(LRT_INFO, "Initializing QNN backend for device architecture %d",
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

    if (auto status = qnn.Api()->deviceCreate(nullptr, device_configs,
                                              &qnn.DeviceHandle());
        status != QNN_SUCCESS) {
      LITE_RT_LOG(LRT_ERROR, "Failed to create QNN device: %d", status);
      return kLrtStatusErrorRuntimeFailure;
    }
  }

  {
    auto cfg = qnn::config::GetDefaultContextConfigs();
    auto device = nullptr;
    if (auto status = qnn.Api()->contextCreate(
            qnn.BackendHandle(), device, cfg.data(), &qnn.ContextHandle());
        status != QNN_SUCCESS) {
      LITE_RT_LOG(LRT_ERROR, "Failed to create QNN context: %d", status);
      return kLrtStatusErrorRuntimeFailure;
    }
  }

  return kLrtStatusOk;
}

};  // namespace qnn
