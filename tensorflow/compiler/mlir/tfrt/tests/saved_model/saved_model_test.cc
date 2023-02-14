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

#include "tensorflow/compiler/mlir/tfrt/saved_model/saved_model.h"

#include "absl/strings/match.h"
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(SavedModelTest, MapSignatures) {
  std::string saved_model_mlir_path = tensorflow::GetDataDependencyFilepath(
      "tensorflow/compiler/mlir/tfrt/tests/saved_model/testdata/test.mlir");

  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);
  auto module =
      mlir::parseSourceFile<mlir::ModuleOp>(saved_model_mlir_path, &context);
  ASSERT_TRUE(module);

  std::vector<std::string> inputs;
  std::vector<std::pair<tensorflow::DataType, tensorflow::PartialTensorShape>>
      in_specs;
  std::vector<std::string> outputs;
  std::vector<std::pair<tensorflow::DataType, tensorflow::PartialTensorShape>>
      out_specs;
  std::vector<mlir::Operation*> bound_inputs;
  TF_ASSERT_OK(MapFunctionSignaturesFromTFSavedModelMLIR(
      module.get(), [&](const TFRTSavedModelSignatureInfo& sig_info) {
        // Only check the signature of "serving_default".
        if (sig_info.func_name != "serving_default") return;

        transform(sig_info.input_names, std::back_inserter(inputs),
                  [](llvm::StringRef x) { return x.str(); });
        in_specs.assign(sig_info.input_specs.begin(),
                        sig_info.input_specs.end());
        transform(sig_info.output_names, std::back_inserter(outputs),
                  [](llvm::StringRef x) { return x.str(); });
        out_specs.assign(sig_info.output_specs.begin(),
                         sig_info.output_specs.end());
        bound_inputs.assign(sig_info.bound_inputs.begin(),
                            sig_info.bound_inputs.end());
      }));

  ASSERT_EQ(inputs.size(), 1);
  EXPECT_EQ(inputs[0], "x");
  ASSERT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[0], "r");

  ASSERT_EQ(in_specs.size(), 1);
  ASSERT_EQ(in_specs[0].first, tensorflow::DT_INT32);
  ASSERT_TRUE(in_specs[0].second.IsIdenticalTo(PartialTensorShape({1, 3})));

  ASSERT_EQ(out_specs.size(), 1);
  ASSERT_EQ(out_specs[0].first, tensorflow::DT_INT32);
  ASSERT_TRUE(out_specs[0].second.IsIdenticalTo(PartialTensorShape({1, 1})));

  ASSERT_EQ(bound_inputs.size(), 2);

  auto global_tensor =
      llvm::cast<mlir::tf_saved_model::GlobalTensorOp>(bound_inputs[0]);
  auto asset = llvm::cast<mlir::tf_saved_model::AssetOp>(bound_inputs[1]);

  EXPECT_EQ(global_tensor.getSymName(), "y");
  EXPECT_EQ(asset.getSymName(), "z");
}

TEST(SavedModelTest, CompileToBEF) {
  std::string saved_model_mlir_path = tensorflow::GetDataDependencyFilepath(
      "tensorflow/compiler/mlir/tfrt/tests/saved_model/testdata/test.mlir");

  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);
  auto module =
      mlir::parseSourceFile<mlir::ModuleOp>(saved_model_mlir_path, &context);
  ASSERT_TRUE(module);

  tfrt::BefBuffer bef_buffer;
  TfrtCompileOptions options;
  TF_ASSERT_OK(ConvertTfMlirToBef(options, module.get(), &bef_buffer));
}

TEST(SavedModelTest, ConvertTfMlirToBefWithXlaFuncExport) {
  std::string saved_model_mlir_path = tensorflow::GetDataDependencyFilepath(
      "tensorflow/compiler/mlir/tfrt/tests/saved_model/testdata/"
      "xla_launch.mlir");

  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);
  auto module =
      mlir::parseSourceFile<mlir::ModuleOp>(saved_model_mlir_path, &context);
  ASSERT_TRUE(module);

  tfrt::BefBuffer bef_buffer;
  TfrtCompileOptions options;
  options.device_target = TfrtDeviceInfraTarget::kGpu;
  options.use_bridge_for_gpu = true;

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tfrt_stub::FallbackState> fallback_state,
      tfrt_stub::FallbackState::Create(SessionOptions(), FunctionDefLibrary()));
  TF_ASSERT_OK(ConvertTfMlirToBef(options, module.get(), &bef_buffer,
                                  fallback_state.get()));

  // The module contains an XLA function, as well as a while body and a while
  // condition within the XLA function.
  EXPECT_EQ(fallback_state->process_function_library_runtime()
                .GetFunctionLibraryDefinition()
                ->num_functions(),
            3);
}

TEST(SavedModelTest, ConvertTfMlirToBefExportingXlaReduceWindow) {
  std::string saved_model_mlir_path = tensorflow::GetDataDependencyFilepath(
      "tensorflow/compiler/mlir/tfrt/tests/saved_model/testdata/"
      "xla_launch_xla_reduce_window.mlir");

  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);
  auto module =
      mlir::parseSourceFile<mlir::ModuleOp>(saved_model_mlir_path, &context);
  ASSERT_TRUE(module);

  tfrt::BefBuffer bef_buffer;
  TfrtCompileOptions options;
  options.device_target = TfrtDeviceInfraTarget::kGpu;
  options.use_bridge_for_gpu = true;

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tfrt_stub::FallbackState> fallback_state,
      tfrt_stub::FallbackState::Create(SessionOptions(), FunctionDefLibrary()));
  TF_ASSERT_OK(ConvertTfMlirToBef(options, module.get(), &bef_buffer,
                                  fallback_state.get()));

  // The module contains an XLA function, as well as a sum_reducer function
  // referenced by an XlaReduceWindow op.
  EXPECT_EQ(fallback_state->process_function_library_runtime()
                .GetFunctionLibraryDefinition()
                ->num_functions(),
            2);
}

// TODO(b/162442824): Add a SavedModel test that covers the error pass.

}  // namespace
}  // namespace tensorflow
