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
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
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
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_model_context.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_core_selector.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/saved_model/saved_model_testutil.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tensorflow {
namespace ifrt_serving {
namespace {

tsl::thread::ThreadPool& GetThreadPool() {
  constexpr int kMaxParallelism = 16;
  static tsl::thread::ThreadPool* thread_pool =
      new tsl::thread::ThreadPool(tsl::Env::Default(), tsl::ThreadOptions(),
                                  "IfrtSharding", kMaxParallelism);
  return *thread_pool;
}

TEST(IfrtBackendCompilerTest, Basic) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/ifrt_cluster.mlir"));

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);

  mlir::MLIRContext context(registry);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context);

  ASSERT_TRUE(mlir_module);
  ASSERT_TRUE(mlir_module.get() != nullptr);

  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  std::unique_ptr<tensorflow::tfrt_stub::Runtime> runtime =
      tensorflow::tfrt_stub::DefaultTfrtRuntime(/*num_threads=*/1);
  tensorflow::tfrt_stub::GraphExecutionOptions graph_execution_options(
      runtime.get());
  tfrt::ResourceContext resource_context;
  tensorflow::tfrt_stub::ModelRuntimeContext runtime_context(
      &graph_execution_options, /*export_dir=*/"", &resource_context);

  tsl::test_util::MockServingDeviceSelector mock_serving_device_selector;
  IfrtServingCoreSelector core_selector(&mock_serving_device_selector,
                                        client->addressable_device_count());

  runtime_context.resource_context().CreateResource<IfrtModelContext>(
      "IfrtModelContext", client, &core_selector, &GetThreadPool(),
      /*compilation_environment_proto=*/nullptr);

  IfrtBackendCompiler compiler;
  TF_ASSERT_OK(compiler.CompileTensorflow(runtime_context, mlir_module.get()));
}

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
