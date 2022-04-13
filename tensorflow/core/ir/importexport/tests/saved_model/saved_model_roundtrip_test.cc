/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/ir/importexport/export.h"
#include "tensorflow/core/ir/importexport/import.h"
#include "tensorflow/core/ir/importexport/tests/roundtrip/roundtrip.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"

namespace {

tensorflow::Status ReadModelProto(const std::string& input_file,
                                  tensorflow::SavedModel* out) {
  return tensorflow::ReadBinaryProto(tensorflow::Env::Default(), input_file,
                                     out);
}

void RunRoundTrip(const std::string& input_file) {
  mlir::DialectRegistry registry;
  mlir::MLIRContext context(registry);

  tensorflow::SavedModel original_model;
  auto read_result = ReadModelProto(input_file, &original_model);
  ASSERT_TRUE(read_result.ok());

  tensorflow::GraphDebugInfo debug_info;
  tensorflow::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module_ref_status =
      mlir::tfg::ImportSavedModelToMlir(&context, debug_info, original_model);

  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      std::move(module_ref_status.ValueOrDie());

  tensorflow::SavedModel final_model;
  auto status = tensorflow::ExportMlirToSavedModel(*module_ref, original_model,
                                                   &final_model);
  if (!status.ok()) {
    LOG(ERROR) << "Export failed: " << status.ToString();
  }
  ASSERT_TRUE(status.ok()) << status.ToString();

  tensorflow::MetaGraphDef* original_metagraph =
      original_model.mutable_meta_graphs(0);
  tensorflow::MetaGraphDef* final_metagraph =
      final_model.mutable_meta_graphs(0);

  // In order to compare graph defs, make sure that both original and
  // final graph defs are normalized, e.g, control input are alphabetically
  // sorted.
  tensorflow::NormalizeTensorData(*original_metagraph->mutable_graph_def(),
                                  /*add_fulltype=*/true);
  tensorflow::NormalizeTensorData(*final_metagraph->mutable_graph_def(),
                                  /*add_fulltype=*/false);

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

constexpr char kTestData[] = "core/ir/importexport/tests/saved_model";

TEST(SavedModelRoundTripTest, V1ModelIsIdentity) {
  const std::string input_file =
      tensorflow::io::JoinPath(tensorflow::testing::TensorFlowSrcRoot(),
                               kTestData, "savedmodel_v1/saved_model.pb");

  ASSERT_NO_FATAL_FAILURE(RunRoundTrip(input_file));
}

TEST(SavedModelRoundTripTest, V2ModelIsIdentity) {
  const std::string input_file =
      tensorflow::io::JoinPath(tensorflow::testing::TensorFlowSrcRoot(),
                               kTestData, "savedmodel_v2/saved_model.pb");

  ASSERT_NO_FATAL_FAILURE(RunRoundTrip(input_file));
}

}  // namespace
