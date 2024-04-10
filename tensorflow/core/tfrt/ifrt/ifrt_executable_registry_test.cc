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

#include "tensorflow/core/tfrt/ifrt/ifrt_executable_registry.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/test_util.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_executable.h"
#include "tensorflow/core/tfrt/ifrt/tf_host_callback.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {
const tsl::thread::ThreadPool& GetThreadPool() {
  constexpr int kMaxParallelism = 16;
  static auto* const thread_pool =
      new tsl::thread::ThreadPool(tsl::Env::Default(), tsl::ThreadOptions(),
                                  "IfrtSharding", kMaxParallelism);
  return *thread_pool;
}

absl::StatusOr<std::unique_ptr<IfrtServingExecutable>>
CreateIfrtServingExecutable(mlir::MLIRContext& context) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/core/tfrt/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/executable.mlir"));

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context);

  if (!mlir_module) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse MLIR file: ", mlir_module_path));
  }

  // Create contexts required for the compiler execution.
  TF_ASSIGN_OR_RETURN(std::shared_ptr<xla::ifrt::Client> client,
                      xla::ifrt::test_util::GetClient());

  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<tensorflow::StaticDeviceMgr> device_mgr,
                      CreateTfStaticDeviceMgr());

  return std::make_unique<IfrtServingExecutable>(
      "test", "main", std::move(mlir_module), client, &GetThreadPool(),
      &ifrt_loaded_variable_registry, device_mgr.get(),
      tensorflow::IdentityShapeRepresentationFn());
}

TEST(IfrtExecutableRegistry, Basic) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);

  mlir::MLIRContext context(registry);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<IfrtServingExecutable> executable,
                          CreateIfrtServingExecutable(context));
  IfrtServingExecutable* raw_ptr = executable.get();

  int64_t program_id = 1234;

  TF_ASSERT_OK_AND_ASSIGN(auto handle, ServingExecutableRegistry::Register(
                                           program_id, std::move(executable)));

  IfrtServingExecutable* executable_ptr =
      ServingExecutableRegistry::Lookup(program_id);
  ASSERT_EQ(executable_ptr, raw_ptr);
}

TEST(IfrtExecutableRegistry, InvalidProgramIdShallReturnNull) {
  int64_t program_id = 1234;

  IfrtServingExecutable* executable_ptr =
      ServingExecutableRegistry::Lookup(program_id);
  ASSERT_EQ(executable_ptr, nullptr);
}

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
