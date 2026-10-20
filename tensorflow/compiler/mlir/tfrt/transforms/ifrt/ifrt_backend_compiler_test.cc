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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/host_runtime/tfrt_ops.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/tsl/framework/test_util/mock_serving_device_selector.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_executable_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_model_context.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_core_selector.h"
#include "tensorflow/core/tfrt/ifrt/sharding_utils.h"
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
    h2d_transfer_executor_factory_ =
        std::make_unique<H2DTransferExecutorFactory>();

    runtime_context_.resource_context().CreateResource<IfrtModelContext>(
        "IfrtModelContext", client_, core_selector_.get(), &GetThreadPool(),
        /*compilation_environment_proto=*/nullptr,
        /*h2d_transfer_executor_factory=*/h2d_transfer_executor_factory_.get());
  }

  void verifyModules() {
    absl::MutexLock l(ServingExecutableRegistry::mu_);
    for (const auto& [_, executable] :
         *ServingExecutableRegistry::executables_) {
      absl::MutexLock l(executable->mutex_);
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
  std::unique_ptr<H2DTransferExecutorFactory> h2d_transfer_executor_factory_;
  IfrtBackendCompiler compiler_;
};

namespace {
using ::testing::ElementsAre;
using ::testing::HasSubstr;

std::vector<uint64_t> CollectIfrtCallProgramIds(mlir::ModuleOp module) {
  std::vector<uint64_t> program_ids;
  module.walk([&](mlir::TF::IfrtCallOp call) {
    program_ids.push_back(call.getProgramId());
  });
  module.walk([&](mlir::TF::AsyncIfrtCallOp call) {
    program_ids.push_back(call.getProgramId());
  });
  return program_ids;
}

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

TEST_F(IfrtBackendCompilerTest,
       ReusesProgramIdForIdenticalClusterAcrossCompilations) {
  constexpr absl::string_view kDataDirectory =
      "tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/ifrt_cluster.mlir"));

  mlir::OwningOpRef<mlir::ModuleOp> first_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context_);
  ASSERT_TRUE(first_module);
  TF_ASSERT_OK(
      compiler_.CompileTensorflow(runtime_context_, first_module.get()));
  std::vector<uint64_t> first_program_ids =
      CollectIfrtCallProgramIds(first_module.get());
  ASSERT_EQ(first_program_ids.size(), 1);

  mlir::OwningOpRef<mlir::ModuleOp> second_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context_);
  ASSERT_TRUE(second_module);
  TF_ASSERT_OK(
      compiler_.CompileTensorflow(runtime_context_, second_module.get()));
  std::vector<uint64_t> second_program_ids =
      CollectIfrtCallProgramIds(second_module.get());

  EXPECT_THAT(second_program_ids, ElementsAre(first_program_ids[0]));
  verifyModules();

  std::optional<IfrtModelContext*> ifrt_model_context =
      runtime_context_.resource_context().GetResource<IfrtModelContext>(
          "IfrtModelContext");
  ASSERT_TRUE(ifrt_model_context.has_value());
  TF_ASSERT_OK((*ifrt_model_context)->Freeze());

  mlir::OwningOpRef<mlir::ModuleOp> frozen_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context_);
  ASSERT_TRUE(frozen_module);
  TF_ASSERT_OK(
      compiler_.CompileTensorflow(runtime_context_, frozen_module.get()));
  std::vector<uint64_t> frozen_program_ids =
      CollectIfrtCallProgramIds(frozen_module.get());
  EXPECT_THAT(frozen_program_ids, ElementsAre(first_program_ids[0]));
}

TEST_F(IfrtBackendCompilerTest, CompileShallFailAfterModelIsFrozen) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/restore_with_reference.mlir"));
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

  std::string unseen_mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/ifrt_cluster.mlir"));
  mlir::OwningOpRef<mlir::ModuleOp> another_mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(unseen_mlir_module_path, &context_);

  EXPECT_THAT(
      compiler_.CompileTensorflow(runtime_context_, another_mlir_module.get()),
      absl_testing::StatusIs(
          absl::StatusCode::kFailedPrecondition,
          HasSubstr("Cannot compile IFRT programs after the model is frozen")));
}

std::vector<std::vector<int>> CollectVariableArgIndices(mlir::ModuleOp module) {
  std::vector<std::vector<int>> all_indices;
  auto collect = [&](auto call) {
    std::vector<int> indices;
    for (mlir::Attribute attr : call.getVariableArgIndices()) {
      indices.push_back(mlir::cast<mlir::IntegerAttr>(attr).getInt());
    }
    all_indices.push_back(std::move(indices));
  };
  module.walk([&](mlir::TF::IfrtCallOp call) { collect(call); });
  module.walk([&](mlir::TF::AsyncIfrtCallOp call) { collect(call); });
  return all_indices;
}

// The same TPU cluster called with different `variable_arg_indices` must not
// share a program id, since the executable binds variables by those indices.
TEST_F(IfrtBackendCompilerTest,
       DoesNotReuseProgramIdForDifferentVariableArgIndices) {
  constexpr absl::string_view kDataDirectory =
      "tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata";
  std::string variable_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/ifrt_cluster_variable_arg.mlir"));
  std::string non_variable_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/ifrt_cluster_non_variable_arg.mlir"));

  mlir::OwningOpRef<mlir::ModuleOp> variable_module =
      mlir::parseSourceFile<mlir::ModuleOp>(variable_path, &context_);
  ASSERT_TRUE(variable_module);
  TF_ASSERT_OK(
      compiler_.CompileTensorflow(runtime_context_, variable_module.get()));
  ASSERT_THAT(CollectVariableArgIndices(variable_module.get()),
              ElementsAre(ElementsAre(0)));
  std::vector<uint64_t> variable_program_ids =
      CollectIfrtCallProgramIds(variable_module.get());
  ASSERT_EQ(variable_program_ids.size(), 1);

  mlir::OwningOpRef<mlir::ModuleOp> non_variable_module =
      mlir::parseSourceFile<mlir::ModuleOp>(non_variable_path, &context_);
  ASSERT_TRUE(non_variable_module);
  TF_ASSERT_OK(
      compiler_.CompileTensorflow(runtime_context_, non_variable_module.get()));
  ASSERT_THAT(CollectVariableArgIndices(non_variable_module.get()),
              ElementsAre(ElementsAre()));
  std::vector<uint64_t> non_variable_program_ids =
      CollectIfrtCallProgramIds(non_variable_module.get());
  ASSERT_EQ(non_variable_program_ids.size(), 1);
  EXPECT_NE(non_variable_program_ids[0], variable_program_ids[0]);

  std::optional<IfrtModelContext*> ifrt_model_context =
      runtime_context_.resource_context().GetResource<IfrtModelContext>(
          "IfrtModelContext");
  ASSERT_TRUE(ifrt_model_context.has_value());
  TF_ASSERT_OK((*ifrt_model_context)->Freeze());

  // After freeze, each call site reuses the program compiled for its indices.
  mlir::OwningOpRef<mlir::ModuleOp> frozen_variable_module =
      mlir::parseSourceFile<mlir::ModuleOp>(variable_path, &context_);
  ASSERT_TRUE(frozen_variable_module);
  TF_ASSERT_OK(compiler_.CompileTensorflow(runtime_context_,
                                           frozen_variable_module.get()));
  EXPECT_THAT(CollectIfrtCallProgramIds(frozen_variable_module.get()),
              ElementsAre(variable_program_ids[0]));

  mlir::OwningOpRef<mlir::ModuleOp> frozen_non_variable_module =
      mlir::parseSourceFile<mlir::ModuleOp>(non_variable_path, &context_);
  ASSERT_TRUE(frozen_non_variable_module);
  TF_ASSERT_OK(compiler_.CompileTensorflow(runtime_context_,
                                           frozen_non_variable_module.get()));
  EXPECT_THAT(CollectIfrtCallProgramIds(frozen_non_variable_module.get()),
              ElementsAre(non_variable_program_ids[0]));
}

// After freeze, a call site whose TPU cluster was compiled during warmup but
// with different `variable_arg_indices` fails with a specific error.
TEST_F(IfrtBackendCompilerTest,
       CompileFailsAfterFreezeForDifferentVariableArgIndices) {
  constexpr absl::string_view kDataDirectory =
      "tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata";
  mlir::OwningOpRef<mlir::ModuleOp> variable_module =
      mlir::parseSourceFile<mlir::ModuleOp>(
          tensorflow::GetDataDependencyFilepath(
              absl::StrCat(kDataDirectory, "/ifrt_cluster_variable_arg.mlir")),
          &context_);
  ASSERT_TRUE(variable_module);
  TF_ASSERT_OK(
      compiler_.CompileTensorflow(runtime_context_, variable_module.get()));

  std::optional<IfrtModelContext*> ifrt_model_context =
      runtime_context_.resource_context().GetResource<IfrtModelContext>(
          "IfrtModelContext");
  ASSERT_TRUE(ifrt_model_context.has_value());
  TF_ASSERT_OK((*ifrt_model_context)->Freeze());

  mlir::OwningOpRef<mlir::ModuleOp> non_variable_module =
      mlir::parseSourceFile<mlir::ModuleOp>(
          tensorflow::GetDataDependencyFilepath(absl::StrCat(
              kDataDirectory, "/ifrt_cluster_non_variable_arg.mlir")),
          &context_);
  ASSERT_TRUE(non_variable_module);
  EXPECT_THAT(
      compiler_.CompileTensorflow(runtime_context_, non_variable_module.get()),
      absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                             HasSubstr("different variable_arg_indices")));
}

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
