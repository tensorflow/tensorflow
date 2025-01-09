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

#include "tensorflow/core/tfrt/ifrt/ifrt_serving_executable_test_util.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/tsl/framework/test_util/mock_serving_device_selector.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_persistent_compilation_cache.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_core_selector.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_executable.h"
#include "tensorflow/core/tfrt/ifrt/tf_host_callback.h"
#include "tsl/platform/env.h"
#include "tsl/platform/status.h"
#include "tsl/platform/threadpool.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
namespace tensorflow {
namespace ifrt_serving {
namespace test_utils {

inline constexpr absl::string_view kMlirModulePath =
    "tensorflow/core/tfrt/ifrt/testdata/";

std::string GetMlirModulePath(absl::string_view module_name) {
  return tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kMlirModulePath, module_name));
}

IfrtServingExecutableTestHelper::IfrtServingExecutableTestHelper(
    tsl::test_util::MockServingDeviceSelector* device_selector)
    : device_selector_(device_selector) {
  auto client_or = xla::ifrt::test_util::GetClient();
  TF_CHECK_OK(client_or.status());
  client_ = std::move(client_or.value());
  core_selector_ = std::make_unique<IfrtServingCoreSelector>(
      device_selector_, client_->addressable_device_count());

  thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtSharding",
      kThreadPoolNumThreads);
  work_queue_ = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);

  auto device_mgr_or = ifrt_serving::CreateTfDynamicDeviceMgr();
  TF_CHECK_OK(device_mgr_or.status());
  device_mgr_ = std::move(device_mgr_or.value());

  mlir::registerAllDialects(registry_);
  mlir::RegisterAllTensorFlowDialects(registry_);
  context_ = std::make_unique<mlir::MLIRContext>(registry_);
  ifrt_persistent_compilation_cache_ =
      std::make_unique<IfrtPersistentCompilationCache>();
}

std::unique_ptr<IfrtServingExecutable>
IfrtServingExecutableTestHelper::MakeExecutable(int64_t program_id,
                                                std::string mlir_module_path) {
  auto mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, context_.get());
  auto executable_or = IfrtServingExecutable::Create(
      program_id, "test", "main", std::move(mlir_module), client_,
      thread_pool_.get(), &ifrt_loaded_variable_registry_,
      &ifrt_restore_tensor_registry_, work_queue_.get(), device_mgr_.get(),
      tensorflow::IdentityShapeRepresentationFn(), core_selector_.get(),
      /*compilation_environment_proto=*/nullptr, &tf_to_hlo_compiler_,
      ifrt_persistent_compilation_cache_.get());
  TF_CHECK_OK(executable_or.status());
  return std::move(executable_or.value());
}

}  // namespace test_utils
}  // namespace ifrt_serving
}  // namespace tensorflow
