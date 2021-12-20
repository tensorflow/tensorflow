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

#include <string>
#include <utility>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/ir/importexport/tests/roundtrip/roundtrip.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/tools/tfg_graph_transforms/export.h"
#include "tensorflow/tools/tfg_graph_transforms/import.h"
#include "tensorflow/tools/tfg_graph_transforms/utils.h"

namespace {

void RunRoundTrip(const std::string& input_file) {
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);

  tensorflow::StatusOr<mlir::OwningModuleRef> module_ref_status =
      mlir::tfg::graph_transforms::ImportSavedModel(&context, input_file);

  mlir::OwningModuleRef module_ref = std::move(module_ref_status.ValueOrDie());

  // Generate the Temp file and use it for the export.
  std::string output_file;
  ASSERT_TRUE(tensorflow::Env::Default()->LocalTempFilename(&output_file));
  auto status = mlir::tfg::graph_transforms::ExportTFGToSavedModel(
      *module_ref, input_file, output_file);
  if (!status.ok()) {
    LOG(ERROR) << "Export failed: " << status.ToString();
  }
  ASSERT_TRUE(status.ok());

  tensorflow::SavedModel original_model, final_model;

  status = mlir::tfg::graph_transforms::ReadSavedModelProto(input_file,
                                                            original_model);
  ASSERT_TRUE(status.ok());

  status = mlir::tfg::graph_transforms::ReadSavedModelProto(output_file,
                                                            final_model);
  ASSERT_TRUE(status.ok());

  tensorflow::MetaGraphDef* original_metagraph =
      original_model.mutable_meta_graphs(0);
  tensorflow::MetaGraphDef* final_metagraph =
      final_model.mutable_meta_graphs(0);

  // In order to compare graph defs, make sure that both original and
  // final graph defs are normalized, e.g, control input are alphabetically
  // sorted.
  tensorflow::NormalizeTensorData(*original_metagraph->mutable_graph_def());
  tensorflow::NormalizeTensorData(*final_metagraph->mutable_graph_def());

  if (!tensorflow::protobuf::util::MessageDifferencer::Equivalent(
          original_model, final_model)) {
#if defined(PLATFORM_GOOGLE)
    // Some of the protobuf comparisons are not available in OSS.
    // This will show the diff inline.
    EXPECT_THAT(original_model, ::testing::EquivToProto(final_model));
#else

    // That's the best we could do given there is no good diff functionality.
    LOG(WARNING) << "Saved model has changed after TFG roundtrip";
#endif
  }
}

constexpr char kTestData[] = "tools/tfg_graph_transforms/tests";

TEST(SavedModelRoundTripTest, V1ModelIsIdentity) {
  const std::string input_file =
      tensorflow::io::JoinPath(tensorflow::testing::TensorFlowSrcRoot(),
                               kTestData, "savedmodel_v1/saved_model.pb");

  RunRoundTrip(input_file);
}

TEST(SavedModelRoundTripTest, V2ModelIsIdentity) {
  const std::string input_file =
      tensorflow::io::JoinPath(tensorflow::testing::TensorFlowSrcRoot(),
                               kTestData, "savedmodel_v2/saved_model.pb");

  RunRoundTrip(input_file);
}

}  // namespace
