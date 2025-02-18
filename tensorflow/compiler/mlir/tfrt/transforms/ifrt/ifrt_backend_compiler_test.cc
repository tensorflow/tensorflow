/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_backend_compiler.h"

#include <memory>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/tsl/framework/test_util/mock_serving_device_selector.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_executable_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_model_context.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_core_selector.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/saved_model/saved_model_testutil.h"
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tensorflow {
namespace ifrt_serving {

tsl::thread::ThreadPool& GetThreadPool() {
  constexpr int kMaxParallelism = 16;
  static tsl::thread::ThreadPool* thread_pool =
      new tsl::thread::ThreadPool(tsl::Env::Default(), tsl::ThreadOptions(),
                                  "IfrtSharding", kMaxParallelism);
  return *thread_pool;
}

class IfrtBackendCompilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mlir::registerAllDialects(registry_);
    mlir::RegisterAllTensorFlowDialects(registry_);
    context_.appendDialectRegistry(registry_);

    // Create contexts required for the compiler execution.
    TF_ASSERT_OK_AND_ASSIGN(client_, xla::ifrt::test_util::GetClient());

    core_selector_ = std::make_unique<IfrtServingCoreSelector>(
        &mock_serving_device_selector_, client_->addressable_device_count());

    runtime_context_.resource_context().CreateResource<IfrtModelContext>(
        "IfrtModelContext", client_, core_selector_.get(), &GetThreadPool(),
        /*compilation_environment_proto=*/nullptr);
  }

  void verifyModules() {
    absl::MutexLock l(&ServingExecutableRegistry::mu_);
    for (const auto& [_, executable] :
         *ServingExecutableRegistry::executables_) {
      absl::MutexLock l(&executable->mutex_);
      executable->module_->walk([](mlir::func::FuncOp func) {
        ASSERT_FALSE(func->hasAttr("tfrt_ifrt_serving.program_id"));
      });
    }
  }

  mlir::DialectRegistry registry_;
  mlir::MLIRContext context_;
  std::shared_ptr<xla::ifrt::Client> client_;

  std::unique_ptr<tensorflow::tfrt_stub::Runtime> runtime_ =
      tensorflow::tfrt_stub::DefaultTfrtRuntime(/*num_threads=*/1);
  tensorflow::tfrt_stub::GraphExecutionOptions graph_execution_options_ =
      tensorflow::tfrt_stub::GraphExecutionOptions(runtime_.get());
  tfrt::ResourceContext resource_context_;
  tensorflow::tfrt_stub::ModelRuntimeContext runtime_context_ =
      tensorflow::tfrt_stub::ModelRuntimeContext(
          &graph_execution_options_, /*export_dir=*/"", &resource_context_);

  tsl::test_util::MockServingDeviceSelector mock_serving_device_selector_;
  std::unique_ptr<IfrtServingCoreSelector> core_selector_;
  IfrtBackendCompiler compiler_;
};

namespace {
using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

struct IfrtBackendCompilerTestParams {
  std::string mlir_file_name;
};

class IfrtBackendCompilerParameterizedTest
    : public IfrtBackendCompilerTest,
      public ::testing::WithParamInterface<IfrtBackendCompilerTestParams> {};

TEST_P(IfrtBackendCompilerParameterizedTest, CompilesOk) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/", GetParam().mlir_file_name));
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context_);

  ASSERT_TRUE(mlir_module);
  ASSERT_TRUE(mlir_module.get() != nullptr);

  TF_ASSERT_OK(
      compiler_.CompileTensorflow(runtime_context_, mlir_module.get()));
  verifyModules();
}

INSTANTIATE_TEST_SUITE_P(IfrtBackendCompilerParameterizedTest,
                         IfrtBackendCompilerParameterizedTest,
                         ::testing::ValuesIn<IfrtBackendCompilerTestParams>({
                             {.mlir_file_name = "ifrt_cluster.mlir"},
                             {.mlir_file_name = "restore_with_reference.mlir"},
                         }));

TEST_F(IfrtBackendCompilerTest, CompileShallFailAfterModelIsFrozen) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/ifrt_cluster.mlir"));
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context_);

  ASSERT_TRUE(mlir_module);
  ASSERT_TRUE(mlir_module.get() != nullptr);

  TF_ASSERT_OK(
      compiler_.CompileTensorflow(runtime_context_, mlir_module.get()));

  std::optional<IfrtModelContext*> ifrt_model_context =
      runtime_context_.resource_context().GetResource<IfrtModelContext>(
          "IfrtModelContext");
  ASSERT_TRUE(ifrt_model_context.has_value());

  TF_ASSERT_OK((*ifrt_model_context)->Freeze());

  mlir::OwningOpRef<mlir::ModuleOp> another_mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context_);

  EXPECT_THAT(
      compiler_.CompileTensorflow(runtime_context_, another_mlir_module.get()),
      StatusIs(
          absl::StatusCode::kFailedPrecondition,
          HasSubstr("Cannot compile IFRT programs after the model is frozen")));
}

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
