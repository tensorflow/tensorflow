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

#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/qnn.h"

#include <dlfcn.h>

#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "third_party/qairt/include/QNN/QnnCommon.h"
#include "third_party/qairt/include/QNN/QnnInterface.h"
#include "third_party/qairt/include/QNN/QnnLog.h"
#include "third_party/qairt/include/QNN/QnnTypes.h"
#include "third_party/qairt/include/QNN/System/QnnSystemCommon.h"
#include "third_party/qairt/include/QNN/System/QnnSystemInterface.h"

namespace lrt {
namespace qnn {

namespace {

void LogCallback(const char* fmt, QnnLog_Level_t level, uint64_t timestamp,
                 va_list argp) {
  char buffer[256];
  vsnprintf(buffer, sizeof(buffer), fmt, argp);
  buffer[sizeof(buffer) - 1] = 0;

  switch (level) {
    case QNN_LOG_LEVEL_ERROR:
      ABSL_LOG(ERROR) << buffer;
      break;
    case QNN_LOG_LEVEL_WARN:
      ABSL_LOG(WARNING) << buffer;
      break;
    case QNN_LOG_LEVEL_INFO:
      ABSL_LOG(INFO) << buffer;
      break;
    case QNN_LOG_LEVEL_DEBUG:
      ABSL_LOG(INFO) << buffer;
      break;
    case QNN_LOG_LEVEL_VERBOSE:
      ABSL_LOG(INFO) << buffer;
      break;
    case QNN_LOG_LEVEL_MAX:
      ABSL_LOG(ERROR) << "Unexpected log level";
      break;
  }
}

}  // namespace

Qnn::~Qnn() {
  if (log_handle_) {
    if (auto status = qnn_interface().logFree(log_handle_);
        status != QNN_SUCCESS) {
      ABSL_LOG(ERROR) << "Failed to free log handle: " << status;
    }
  }
  if (system_dlib_handle_) {
    ::dlclose(system_dlib_handle_);
  }
  if (qnn_dlib_handle_) {
    ::dlclose(qnn_dlib_handle_);
  }
}

absl::StatusOr<std::unique_ptr<Qnn>> Qnn::Create() {
  std::unique_ptr<Qnn> qnn(new Qnn);
  if (auto status = qnn->LoadSystemSymbols(); !status.ok()) {
    return status;
  }
  if (auto status = qnn->LoadQnnSymbols(); !status.ok()) {
    return status;
  }

  if (qnn->qnn_interface().logCreate(LogCallback, QNN_LOG_LEVEL_INFO,
                                     &qnn->log_handle_) != QNN_SUCCESS) {
    return absl::InternalError("Failed to set logging callback");
  }

  return qnn;
}

absl::Status Qnn::LoadSystemSymbols() {
  system_dlib_handle_ = ::dlopen(kSystemLibPath, RTLD_NOW | RTLD_LOCAL);
  if (!system_dlib_handle_) {
    return absl::InternalError("Failed to load QNN System shared library");
  }

  {
    using QnnSystemInterfaceGetProvidersFn_t =
        Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);
    auto QnnSystemInterface_getProviders =
        reinterpret_cast<QnnSystemInterfaceGetProvidersFn_t>(
            ::dlsym(system_dlib_handle_, "QnnSystemInterface_getProviders"));
    if (!QnnSystemInterface_getProviders) {
      return absl::InternalError("QnnSystemInterface_getProviders not found");
    }

    const QnnSystemInterface_t** interface_providers{nullptr};
    uint32_t num_providers{0};
    if (QnnSystemInterface_getProviders(&interface_providers, &num_providers) !=
        QNN_SUCCESS) {
      return absl::InternalError("Failed to fetch interface providers");
    }

    if (!interface_providers) {
      return absl::InternalError("No interface providers found");
    }

    bool found{false};
    for (auto i = 0; i < num_providers; ++i) {
      auto* interface_provider = interface_providers[i];
      if (interface_provider) {
        if (interface_provider->systemApiVersion.major ==
                QNN_SYSTEM_API_VERSION_MAJOR &&
            interface_provider->systemApiVersion.minor >=
                QNN_SYSTEM_API_VERSION_MINOR &&
            interface_provider->systemApiVersion.patch >=
                QNN_SYSTEM_API_VERSION_PATCH) {
          system_interface_ = interface_provider->QNN_SYSTEM_INTERFACE_VER_NAME;
          found = true;
          ABSL_LOG(INFO) << "Found QNN System API v"
                         << interface_provider->systemApiVersion.major << "."
                         << interface_provider->systemApiVersion.minor << "."
                         << interface_provider->systemApiVersion.patch;
          break;
        }
      }
    }

    if (!found) {
      return absl::InternalError("Did not find a valid QNN interface");
    }
  }

  return {};
}

absl::Status Qnn::LoadQnnSymbols() {
  qnn_dlib_handle_ = ::dlopen(kQnnHtpLibPath, RTLD_NOW | RTLD_LOCAL);
  if (!qnn_dlib_handle_) {
    return absl::InternalError("Failed to load QNN HTP shared library");
  }

  {
    using QnnInterfaceGetProvidersFn_t =
        Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
    auto QnnInterface_getProviders =
        reinterpret_cast<QnnInterfaceGetProvidersFn_t>(
            ::dlsym(qnn_dlib_handle_, "QnnInterface_getProviders"));
    if (!QnnInterface_getProviders) {
      return absl::InternalError("QnnInterface_getProviders not found");
    }

    const QnnInterface_t** interface_providers{nullptr};
    uint32_t num_providers{0};
    if (QnnInterface_getProviders(&interface_providers, &num_providers) !=
        QNN_SUCCESS) {
      return absl::InternalError("Failed to fetch interface providers");
    }

    if (!interface_providers) {
      return absl::InternalError("No interface providers found");
    }

    bool found{false};
    for (auto i = 0; i < num_providers; ++i) {
      auto* interface_provider = interface_providers[i];
      if (interface_provider) {
        if (interface_provider->apiVersion.coreApiVersion.major ==
                QNN_API_VERSION_MAJOR &&
            interface_provider->apiVersion.coreApiVersion.minor >=
                QNN_API_VERSION_MINOR &&
            interface_provider->apiVersion.coreApiVersion.patch >=
                QNN_API_VERSION_PATCH) {
          qnn_interface_ = interface_provider->QNN_INTERFACE_VER_NAME;
          found = true;
          ABSL_LOG(INFO) << "Found QNN API v"
                         << interface_provider->apiVersion.coreApiVersion.major
                         << "."
                         << interface_provider->apiVersion.coreApiVersion.minor
                         << "."
                         << interface_provider->apiVersion.coreApiVersion.patch;
          break;
        }
      }
    }

    if (!found) {
      return absl::InternalError("Did not find a valid QNN interface");
    }
  }

  return {};
}

}  // namespace qnn
}  // namespace lrt
