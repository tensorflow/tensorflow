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

#include "tensorflow/core/tfrt/saved_model/utils/serialize_utils.h"

#include <cstdlib>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/import_model.h"
#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/saved_model/saved_model_testutil.h"
#include "tensorflow/core/tfrt/saved_model/saved_model_util.h"
#include "tensorflow/core/tfrt/utils/utils.h"
#include "tsl/platform/env.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {
namespace {

TEST(SerializeBEFTest, HandlesCompleteProcess) {
  tfrt::BefBuffer old_bef;

  // Load BEF Buffer Data.

  const std::string saved_model_mlir_path =
      "third_party/tensorflow/compiler/mlir/tfrt/tests/saved_model/testdata/"
      "test.mlir";

  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);
  auto module =
      mlir::parseSourceFile<mlir::ModuleOp>(saved_model_mlir_path, &context);
  ASSERT_TRUE(module);

  std::unique_ptr<Runtime> runtime =
      tensorflow::tfrt_stub::Runtime::Create(/*num_inter_op_threads=*/1);
  tfrt_stub::GraphExecutionOptions options(runtime.get());
  tfrt::ResourceContext resource_context;
  tfrt_stub::ModelRuntimeContext model_context(
      &options, options.compile_options.saved_model_dir, &resource_context);
  TF_ASSERT_OK(ConvertTfMlirToBef(options.compile_options, module.get(),
                                  &old_bef, model_context));

  // Create Filepath for .mlir.bef.
  const std::string filepath =
      io::JoinPath(getenv("TEST_UNDECLARED_OUTPUTS_DIR"),
                   std::string("serialized_bef.mlir.bef"));

  // Serialize BEF Buffer.
  TF_ASSERT_OK(tensorflow::tfrt_stub::SerializeBEF(old_bef, filepath));
  ASSERT_NE(old_bef.size(), 0);

  // Create new empty BEF buffer and deserialize to verify data.

  TF_ASSERT_OK_AND_ASSIGN(const tfrt::BefBuffer bef,
                          DeserializeBEFBuffer(filepath));

  // Check for any data loss during deserialization process.
  ASSERT_TRUE(old_bef.size() == bef.size());

  // Check file creation.
  std::unique_ptr<Runtime> default_runtime =
      DefaultTfrtRuntime(/*num_threads=*/1);
  SavedModel::Options default_options =
      DefaultSavedModelOptions(default_runtime.get());
  TF_EXPECT_OK(tfrt::CreateBefFileFromBefBuffer(
                   *default_options.graph_execution_options.runtime, bef)
                   .status());
}

TEST(SerializeMLRTTest, HandlesSerializeAndDeserializeProcess) {
  mlrt::bc::Buffer old_bytecode;

  // Load MLRT Bytecode Data.

  const std::string saved_model_mlir_path =
      "third_party/tensorflow/compiler/mlir/tfrt/tests/saved_model/testdata/"
      "test.mlir";

  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);
  auto module =
      mlir::parseSourceFile<mlir::ModuleOp>(saved_model_mlir_path, &context);
  ASSERT_TRUE(module);
  mlir::OwningOpRef<mlir::ModuleOp> module_with_op_keys;
  std::unique_ptr<Runtime> runtime =
      tensorflow::tfrt_stub::Runtime::Create(/*num_inter_op_threads=*/1);
  tfrt_stub::GraphExecutionOptions options(runtime.get());
  options.enable_mlrt = true;
  tfrt::ResourceContext resource_context;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tfrt_stub::FallbackState> fallback_state,
      tfrt_stub::FallbackState::Create(SessionOptions(), FunctionDefLibrary()));
  tfrt_stub::ModelRuntimeContext model_context(
      &options, options.compile_options.saved_model_dir, &resource_context);
  TF_ASSERT_OK_AND_ASSIGN(
      old_bytecode, mlrt_compiler::ConvertTfMlirToBytecode(
                        options.compile_options, *fallback_state, module.get(),
                        model_context, &module_with_op_keys));

  // Create Filepath for .mlir.mlrt.
  const std::string aot_package_path =
      GetAotPackagePath(getenv("TEST_UNDECLARED_OUTPUTS_DIR"));
  tsl::Env* env = tsl::Env::Default();
  TF_ASSERT_OK(env->RecursivelyCreateDir(aot_package_path));

  const std::string filepath =
      io::JoinPath(aot_package_path, std::string("serialized_mlrt.mlir.mlrt"));

  // Serialize MLRT Bytecode.
  TF_ASSERT_OK(
      tensorflow::tfrt_stub::SerializeMLRTBytecode(old_bytecode, filepath));
  ASSERT_NE(old_bytecode.size(), 0);

  // Create new MLRT Bytecode and deserialize to verify data.
  mlrt::bc::Buffer bytecode;
  TF_ASSERT_OK_AND_ASSIGN(bytecode, DeserializeMlrtBytecodeBuffer(filepath));

  // Check for any data loss during deserialization process.
  ASSERT_TRUE(old_bytecode.size() == bytecode.size());
  EXPECT_STREQ(old_bytecode.data(), bytecode.data());

  TF_ASSERT_OK_AND_ASSIGN(
      bytecode,
      LoadMlrtAndMlir(options.compile_options, module_with_op_keys.get(),
                      getenv("TEST_UNDECLARED_OUTPUTS_DIR"),
                      fallback_state.get()));

  // Check for any data loss during deserialization process.
  ASSERT_TRUE(old_bytecode.size() == bytecode.size());
  EXPECT_STREQ(old_bytecode.data(), bytecode.data());
}
}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
