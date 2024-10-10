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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_QNN_MANAGER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_QNN_MANAGER_H_

#include <optional>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/qairt/include/QNN/HTP/QnnHtpDevice.h"
#include "third_party/qairt/include/QNN/QnnBackend.h"
#include "third_party/qairt/include/QNN/QnnCommon.h"
#include "third_party/qairt/include/QNN/QnnDevice.h"
#include "third_party/qairt/include/QNN/QnnInterface.h"
#include "third_party/qairt/include/QNN/System/QnnSystemInterface.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/common.h"

namespace lrt::qnn {

// Wrapper to manage dynamic loading and lifetimes of QNN SDK objects.
class QnnManager {
 public:
  QnnManager() = default;
  ~QnnManager() = default;

  //
  // Manage libQnn*.so Loading
  //

  // Loads the libQnn*.so at given path.
  LrtStatus LoadLib(absl::string_view path);

  // Loads the libQnnSystem.so at given path.
  LrtStatus LoadSystemLib(absl::string_view path);

  // Dumps dynamic loading info about the loaded libQnn*.so. Does
  // nothing if it has not been loaded yet.
  void DumpLibDetails() const;

  // Dumps dynamic loading info about the loaded libQnnSystem.so. Does
  // nothing if it has not been loaded yet.
  void DumpSystemLibDetails() const;

  //
  // Resolve and Access QNN SDK Functions
  //

  // Resolve all available QNN SDK functions from (already) loaded so. If
  // multiple providers are found, selects the first one with a suitable
  // version. Fails if none can be found.
  LrtStatus ResolveApi();

  // Get resolved function pointers for qnn sdk calls. Nullptr if functions
  // have not been resolved yet.
  const QnnApi* Api() const;

  // Dumps information relevant to the loaded api provider. Does nothing if
  // a successful ResolveFuncs hasn't occurred.
  void DumpApiDetails() const;

  // Resolve all available QNN SDK functions from (already) loaded so. If
  // multiple providers are found, selects the first one with a suitable
  // version. Fails if none can be found.
  LrtStatus ResolveSystemApi();

  // Get resolved function pointers for qnn sdk calls. Nullptr if functions
  // have not been resolved yet.
  const QnnSystemApi* SystemApi() const;

  // Dumps information relevant to the loaded api provider. Does nothing if
  // a successful ResolveFuncs hasn't occurred.
  void DumpSystemApiDetails() const;

  //
  // QNN SDK Objects.
  //

  // Get qnn log handle. Nullptr if logCreate has not been successfully called.
  Qnn_LogHandle_t& LogHandle() { return log_handle_; }

  // Signal QNN SDK to free any memory related to logging. Does nothing
  // if logCreate has not been called.
  LrtStatus FreeLogging();

  // Get qnn backend handle. Nullptr if backendCreate has not been successfully
  // called.
  Qnn_BackendHandle_t& BackendHandle() { return backend_handle_; }

  // Signal QNN SDK to free any memory related to backend. Does nothing
  // if backendCreate has not been called.
  LrtStatus FreeBackend();

  // Get qnn device handle. Nullptr if deviceCreate has not been successfully
  // called.
  Qnn_DeviceHandle_t& DeviceHandle() { return device_handle_; }

  // Signal QNN SDK to free any memory related to the device. Does nothing
  // if deviceCreate has not been called.
  LrtStatus FreeDevice();

  // Get qnn context handle. Nullptr if contextCreate has not been successfully
  // called.
  Qnn_ContextHandle_t& ContextHandle() { return context_handle_; }

  // Signal QNN SDK to free any memory related to context. Does nothing
  // if contextCreate has not been called.
  LrtStatus FreeContext();

  // Get qnn system context handle. Nullptr if systemContextCreate has not been
  // successfully called.
  QnnSystemContext_Handle_t& SystemContextHandle() {
    return system_context_handle_;
  }

  // Signal QNN SDK to free any memory related to system context. Does nothing
  // if systemContextCreate has not been called.
  LrtStatus FreeSystemContext();

  //
  // Context Binary
  //

  // Generates QNN context binary from current context. Writes to given
  // buffer.
  LrtStatus GenerateContextBin(std::vector<char>& buffer);

 private:
  void* lib_so_ = nullptr;

  void* lib_system_so_ = nullptr;

  const QnnInterface_t* interface_ = nullptr;

  const QnnSystemInterface_t* system_interface_ = nullptr;

  Qnn_LogHandle_t log_handle_ = nullptr;

  Qnn_BackendHandle_t backend_handle_ = nullptr;

  Qnn_DeviceHandle_t device_handle_ = nullptr;

  Qnn_ContextHandle_t context_handle_ = nullptr;

  QnnSystemContext_Handle_t system_context_handle_ = nullptr;
};

// Runs alls "setup" methods (LoadLibSO, ResolveFuncs) and aditionally
// instantiates the logging, backend and context.
LrtStatus SetupAll(std::optional<QnnHtpDevice_Arch_t> soc_model,
                   QnnManager& qnn, bool load_system = false,
                   bool load_context = false);

// Default QNN Configurations.
namespace config {

inline absl::Span<const QnnBackend_Config_t*> GetDefaultHtpConfigs() {
  static const QnnBackend_Config_t* configs[] = {nullptr};
  return absl::MakeSpan(configs);
}

inline absl::Span<const QnnContext_Config_t*> GetDefaultContextConfigs() {
  static const QnnContext_Config_t* configs[] = {nullptr};
  return absl::MakeSpan(configs);
}

}  // namespace config

}  // namespace lrt::qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_QNN_MANAGER_H_
