/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TFRT_IFRT_IFRT_SERVING_EXECUTABLE_TEST_UTIL_H_
#define TENSORFLOW_CORE_TFRT_IFRT_IFRT_SERVING_EXECUTABLE_TEST_UTIL_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/tsl/framework/test_util/mock_serving_device_selector.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_core_selector.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_executable.h"
#include "tsl/platform/threadpool.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace tensorflow {
namespace ifrt_serving {
namespace test_utils {

// A test helper class to create and IfrtServingExecutable.
class IfrtServingExecutableTestHelper {
 public:
  explicit IfrtServingExecutableTestHelper(
      tsl::test_util::MockServingDeviceSelector* device_selector);

  // Creates an IfrtServingExecutable with the given program id.
  // Note the instance of this class must outlive the returned
  // IfrtServingExecutable.
  std::unique_ptr<IfrtServingExecutable> MakeExecutable(
      int64_t program_id, std::string mlir_module_path);

  IfrtRestoreTensorRegistry* ifrt_restore_tensor_registry() {
    return &ifrt_restore_tensor_registry_;
  }

  int num_cores() const { return client_->addressable_device_count(); }

 private:
  static constexpr int kThreadPoolNumThreads = 16;

  tsl::test_util::MockServingDeviceSelector* device_selector_;  // Not owned.
  std::unique_ptr<IfrtServingCoreSelector> core_selector_;
  std::shared_ptr<xla::ifrt::Client> client_;
  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_;
  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry_;
  IfrtRestoreTensorRegistry ifrt_restore_tensor_registry_;
  std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue_;
  std::unique_ptr<tensorflow::StaticDeviceMgr> device_mgr_;

  mlir::DialectRegistry registry_;
  std::unique_ptr<mlir::MLIRContext> context_;
};

// Returns the path to the MLIR module for the given module name.
std::string GetMlirModulePath(absl::string_view module_name);

}  // namespace test_utils
}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_IFRT_SERVING_EXECUTABLE_TEST_UTIL_H_
