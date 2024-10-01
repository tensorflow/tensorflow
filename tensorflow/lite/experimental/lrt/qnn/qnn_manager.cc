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

#include "tensorflow/lite/experimental/lrt/qnn/qnn_manager.h"

#include <cstdint>
#include <iostream>
#include <vector>

#include "absl/types/span.h"
#include "third_party/qairt/include/QNN/QnnCommon.h"
#include "third_party/qairt/include/QNN/QnnInterface.h"
#include "third_party/qairt/include/QNN/QnnLog.h"
#include "third_party/qairt/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/qnn/load_sdk.h"
#include "tensorflow/lite/experimental/lrt/qnn/log.h"

using ::qnn::load::QnnInterfaceGetProvidersFn_t;

namespace qnn {

namespace {

absl::Span<const QnnInterface_t*> LoadProvidersFromLib(void* lib_so) {
  load::QnnInterfaceGetProvidersFn_t get_providers = nullptr;
  get_providers = load::ResolveQnnSymbol<QnnInterfaceGetProvidersFn_t>(
      lib_so, load::kLibQnnGetProvidersSymbol);
  if (get_providers == nullptr) {
    std::cerr << "Failed to resolve get providers symbol\n";
    return {};
  }

  const QnnInterface_t** interface_providers = nullptr;
  uint32_t num_providers = 0;
  if (QNN_SUCCESS != get_providers(&interface_providers, &num_providers)) {
    std::cerr << "Failed to get providers\n";
    return {};
  }

  return absl::MakeSpan(interface_providers, num_providers);
}

}  // namespace

LrtStatus QnnManager::LoadLibSO(absl::string_view path) {
  lib_so_ = load::LoadSO(path);
  if (lib_so_ == nullptr) {
    return kLrtStatusDynamicLoadErr;
  }
  return kLrtStatusOk;
}

void QnnManager::DumpLibSODetails() const {
  if (lib_so_ == nullptr) {
    return;
  }
  load::DumpDlInfo(lib_so_);
}

// TODO: Repace QnnManager::Funcs with indirection access operator.
const QnnFunctionPointers* QnnManager::API() const {
  if (interface_ == nullptr) {
    return nullptr;
  }
  return &interface_->QNN_INTERFACE_VER_NAME;
}

LrtStatus QnnManager::ResolveFuncs() {
  if (lib_so_ == nullptr) {
    std::cerr << "Cannot resolve functions: libQnn*.so has not been loaded.\n";
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
    std::cerr << "No valid interface was provided\n";
    return kLrtStatusDynamicLoadErr;
  }

  return kLrtStatusOk;
}

void QnnManager::DumpProviderDetails() const {
  if (interface_ == nullptr) {
    return;
  }
  log::DumpInterface(interface_);
}

LrtStatus QnnManager::FreeLogging() {
  if (log_handle_ != nullptr) {
    if (QNN_SUCCESS != API()->logFree(log_handle_)) {
      std::cerr << "Failed to free logging\n";
      return kLrtStatusErrorNotFound;
    }
  }
  log_handle_ = nullptr;
  return kLrtStatusOk;
}

LrtStatus QnnManager::FreeBackend() {
  if (backend_handle_ != nullptr) {
    if (QNN_SUCCESS != API()->backendFree(backend_handle_)) {
      std::cerr << "Failed to free backend\n";
      return kLrtStatusErrorNotFound;
    }
  }
  backend_handle_ = nullptr;
  return kLrtStatusOk;
}

LrtStatus QnnManager::FreeContext() {
  if (context_handle_ != nullptr) {
    if (QNN_SUCCESS != API()->contextFree(context_handle_, nullptr)) {
      std::cerr << "Failed to free context\n";
      return kLrtStatusErrorNotFound;
    }
  }
  context_handle_ = nullptr;
  return kLrtStatusOk;
}

LrtStatus QnnManager::GenerateContextBin(std::vector<char>& buffer) {
  Qnn_ContextBinarySize_t bin_size = 0;
  if (QNN_SUCCESS != API()->contextGetBinarySize(ContextHandle(), &bin_size)) {
    std::cerr << "Failed to get context bin size\n";
    return kLrtStatusErrorNotFound;
  }
  buffer.clear();
  buffer.resize(bin_size);

  Qnn_ContextBinarySize_t written_bin_size = 0;
  if (QNN_SUCCESS != API()->contextGetBinary(ContextHandle(), buffer.data(),
                                             buffer.size(),
                                             &written_bin_size)) {
    std::cerr << "Failed to generated context binary \n";
    return kLrtStatusErrorNotFound;
  }

  std::cerr << "Serialized a context bin of size (bytes): " << written_bin_size
            << "\n";

  return kLrtStatusOk;
}

LrtStatus SetupAll(QnnManager& qnn) {
  LRT_RETURN_STATUS_IF_NOT_OK(qnn.LoadLibSO(kLibQnnHtpSo));
  qnn.DumpLibSODetails();

  LRT_RETURN_STATUS_IF_NOT_OK(qnn.ResolveFuncs());
  qnn.DumpProviderDetails();

  {
    if (QNN_SUCCESS != qnn.API()->logCreate(qnn::log::GetDefaultStdOutLogger(),
                                            QNN_LOG_LEVEL_DEBUG,
                                            &qnn.LogHandle())) {
      return kLrtStatusErrorNotFound;
    }
  }

  {
    auto cfg = qnn::config::GetDefaultHtpConfigs();
    if (QNN_SUCCESS != qnn.API()->backendCreate(qnn.LogHandle(), cfg.data(),
                                                &qnn.BackendHandle())) {
      return kLrtStatusErrorNotFound;
    }
  }

  {
    auto cfg = qnn::config::GetDefaultContextConfigs();
    auto device = nullptr;
    if (QNN_SUCCESS != qnn.API()->contextCreate(qnn.BackendHandle(), device,
                                                cfg.data(),
                                                &qnn.ContextHandle())) {
      return kLrtStatusErrorNotFound;
    }
  }
  return kLrtStatusOk;
}

};  // namespace qnn
