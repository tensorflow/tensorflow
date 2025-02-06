/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CORE_TFRT_FALLBACK_FALLBACK_STATE_H_
#define TENSORFLOW_CORE_TFRT_FALLBACK_FALLBACK_STATE_H_

#include <memory>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/graph_execution_state.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace tfrt_stub {

// FallbackState contains the necessary runtime states (eg. Devices) used in
// current tensorflow. It also provides methods used in current tensorflow.
class FallbackState {
 public:
  // The FunctionDefLibrary is passed in to initialize the
  // ProcessFunctionLibraryRuntime member of this class
  static absl::StatusOr<std::unique_ptr<FallbackState>> Create(
      const SessionOptions &session_options,
      const tensorflow::FunctionDefLibrary &fdef_lib);

  static absl::StatusOr<std::unique_ptr<FallbackState>> CreateWithCpuDevice(
      const SessionOptions &session_options,
      const tensorflow::FunctionDefLibrary &fdef_lib);

  static absl::StatusOr<std::unique_ptr<FallbackState>> CreateWithMockGpuDevice(
      const SessionOptions &session_options,
      const tensorflow::FunctionDefLibrary &fdef_lib);

  static absl::StatusOr<std::unique_ptr<FallbackState>> CreateWithDeviceMgr(
      const SessionOptions &session_options,
      const tensorflow::FunctionDefLibrary &fdef_lib,
      absl::Nonnull<DynamicDeviceMgr *> device_mgr);

  FallbackState(const SessionOptions &session_options,
                std::variant<std::vector<std::unique_ptr<Device>>,
                             absl::Nonnull<DynamicDeviceMgr *>>
                    device_mgr,
                const tensorflow::FunctionDefLibrary &fdef_lib);

  // Create GraphExecutionState from the `graph_def`. The result will contain a
  // preprocessed graph with runtime information such as devices.
  absl::StatusOr<std::unique_ptr<GraphExecutionState>>
  CreateGraphExecutionState(GraphDef graph_def, bool run_placer = true,
                            bool enable_tf2xla_mlir_bridge = true) const;

  // Adds `func_def` to the function library.
  absl::Status AddFunctionDef(const FunctionDef &func_def);

  const SessionOptions &session_options() const { return session_options_; }

  const DeviceMgr &device_manager() const { return *device_manager_ptr_; }

  DeviceMgr &device_manager() { return *device_manager_ptr_; }

  const DeviceSet &device_set() const { return device_set_; }

  const ProcessFunctionLibraryRuntime &process_function_library_runtime()
      const {
    return pflr_;
  }

  const FunctionLibraryDefinition &func_lib_def() const {
    return func_lib_def_;
  }

 private:
  SessionOptions session_options_;
  DynamicDeviceMgr device_manager_;
  absl::Nonnull<DynamicDeviceMgr *> device_manager_ptr_;
  DeviceSet device_set_;
  FunctionLibraryDefinition func_lib_def_;
  ProcessFunctionLibraryRuntime pflr_;
};

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_FALLBACK_FALLBACK_STATE_H_
