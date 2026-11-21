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

#include <cstdint>
#include <functional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"     // from @llvm-project
#include "mlir/IR/BuiltinOps.h"   // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"    // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/ir/importexport/savedmodel_export.h"
#include "tensorflow/core/ir/importexport/savedmodel_import.h"
#include "tensorflow/core/ir/importexport/tests/roundtrip/roundtrip.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"

namespace {

absl::Status ReadModelProto(const std::string& input_file,
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
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module_ref_status =
      mlir::tfg::ImportSavedModelToMlir(&context, debug_info, original_model);

  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      std::move(module_ref_status.value());

  tensorflow::SavedModel final_model;
  auto status = mlir::tfg::ExportMlirToSavedModel(*module_ref, original_model,
                                                  &final_model);
  if (!status.ok()) {
    LOG(ERROR) << "Export failed: " << status;
  }
  ASSERT_TRUE(status.ok()) << status;

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

using ::testing::ElementsAre;
using ::testing::HasSubstr;

constexpr char kSplitGraph[] = R"pb(
  node {
    name: "dim"
    op: "Const"
    attr {
      key: "dtype"
      value { type: DT_INT32 }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {}
          int_val: 0
        }
      }
    }
  }
  node {
    name: "x"
    op: "Placeholder"
    attr {
      key: "dtype"
      value { type: DT_FLOAT }
    }
  }
  node {
    name: "split"
    op: "Split"
    input: "dim"
    input: "x"
    attr {
      key: "T"
      value { type: DT_FLOAT }
    }
    attr {
      key: "num_split"
      value { i: 2 }
    }
  }
)pb";

// SdcaOptimizer has a fixed output before its `num_sparse_features * float`
// output. Without a range check, the counts used with it below overflow the
// result type vector instead of allocating memory for 2^32 - 1 results.
constexpr char kSdcaGraph[] = R"pb(
  node {
    name: "sdca"
    op: "SdcaOptimizer"
    attr {
      key: "num_sparse_features"
      value { i: 0 }
    }
    attr {
      key: "num_dense_features"
      value { i: 0 }
    }
  }
)pb";

// A function that splits its input in two and forwards both halves through
// IdentityN. Exporting it names `split:output:1` and `identity_n:output:1`,
// which needs the output segment sizes of Split and IdentityN.
constexpr char kSplitFunctionGraph[] = R"pb(
  library {
    function {
      signature {
        name: "SplitAndForward"
        input_arg { name: "x" type: DT_FLOAT }
        output_arg { name: "y" type: DT_FLOAT }
      }
      node_def {
        name: "dim"
        op: "Const"
        attr {
          key: "dtype"
          value { type: DT_INT32 }
        }
        attr {
          key: "value"
          value {
            tensor {
              dtype: DT_INT32
              tensor_shape {}
              int_val: 0
            }
          }
        }
      }
      node_def {
        name: "split"
        op: "Split"
        input: "dim:output:0"
        input: "x"
        attr {
          key: "T"
          value { type: DT_FLOAT }
        }
        attr {
          key: "num_split"
          value { i: 2 }
        }
      }
      node_def {
        name: "identity_n"
        op: "IdentityN"
        input: "split:output:0"
        input: "split:output:1"
        attr {
          key: "T"
          value { list { type: DT_FLOAT type: DT_FLOAT } }
        }
      }
      ret { key: "y" value: "identity_n:output:1" }
    }
  }
)pb";

// Returns a SavedModel whose only meta graph holds `graph_text`. If
// `number_attr` is set, that attr is set to `value` on every node that has it.
tensorflow::SavedModel MakeSavedModel(const char* graph_text,
                                      const std::string& number_attr = "",
                                      int64_t value = 0) {
  tensorflow::SavedModel saved_model;
  tensorflow::GraphDef* graph_def =
      saved_model.add_meta_graphs()->mutable_graph_def();
  CHECK(
      tensorflow::protobuf::TextFormat::ParseFromString(graph_text, graph_def));
  for (tensorflow::NodeDef& node : *graph_def->mutable_node()) {
    if (node.attr().count(number_attr)) {
      (*node.mutable_attr())[number_attr].set_i(value);
    }
  }
  return saved_model;
}

absl::Status ImportWithNumberAttr(const char* graph_text,
                                  const std::string& number_attr,
                                  int64_t value) {
  mlir::MLIRContext context;
  return mlir::tfg::ImportSavedModelToMlir(
             &context, tensorflow::GraphDebugInfo(),
             MakeSavedModel(graph_text, number_attr, value))
      .status();
}

// Imports `kSplitFunctionGraph`, calls `update` on every imported op and
// exports the module back to `exported`.
absl::Status ExportSplitFunction(
    const std::function<void(mlir::Builder&, mlir::Operation*)>& update,
    tensorflow::SavedModel* exported) {
  mlir::MLIRContext context;
  const tensorflow::SavedModel saved_model =
      MakeSavedModel(kSplitFunctionGraph);
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      mlir::tfg::ImportSavedModelToMlir(&context, tensorflow::GraphDebugInfo(),
                                        saved_model);
  if (!module.ok()) return module.status();
  mlir::Builder builder(&context);
  (*module)->walk([&](mlir::Operation* op) { update(builder, op); });
  return mlir::tfg::ExportMlirToSavedModel(**module, saved_model, exported);
}

TEST(SavedModelNumberAttrTest, ValidSplitImportIsAccepted) {
  mlir::MLIRContext context;
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      mlir::tfg::ImportSavedModelToMlir(&context, tensorflow::GraphDebugInfo(),
                                        MakeSavedModel(kSplitGraph));
  ASSERT_TRUE(module.ok()) << module.status();
  int split_results = -1;
  (*module)->walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == "tfg.Split")
      split_results = op->getNumResults();
  });
  // Two data results and one control result.
  EXPECT_EQ(split_results, 3);
}

TEST(SavedModelNumberAttrTest, ValidSplitAndFunctionExportIsAccepted) {
  tensorflow::SavedModel exported;
  absl::Status status =
      ExportSplitFunction([](mlir::Builder&, mlir::Operation*) {}, &exported);
  ASSERT_TRUE(status.ok()) << status;
  const tensorflow::FunctionDefLibrary& library =
      exported.meta_graphs(0).graph_def().library();
  ASSERT_EQ(library.function_size(), 1);
  const tensorflow::FunctionDef& function = library.function(0);
  EXPECT_EQ(function.ret().at("y"), "identity_n:output:1");
  bool found_identity_n = false;
  for (const tensorflow::NodeDef& node : function.node_def()) {
    if (node.name() != "identity_n") continue;
    found_identity_n = true;
    EXPECT_THAT(node.input(), ElementsAre("split:output:0", "split:output:1"));
  }
  EXPECT_TRUE(found_identity_n);
}

TEST(SavedModelNumberAttrTest, ImportTruncatingNumberAttrIsRejected) {
  // 2^32 + 2 must not be read as 2.
  absl::Status status =
      ImportWithNumberAttr(kSplitGraph, "num_split", (int64_t{1} << 32) + 2);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  EXPECT_THAT(status.message(), HasSubstr("num_split"));
}

TEST(SavedModelNumberAttrTest, ExportNegativeNumberAttrIsRejected) {
  tensorflow::SavedModel exported;
  absl::Status status = ExportSplitFunction(
      [](mlir::Builder& builder, mlir::Operation* op) {
        if (op->getName().getStringRef() == "tfg.Split")
          op->setAttr("num_split", builder.getI64IntegerAttr(-1));
      },
      &exported);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  EXPECT_THAT(status.message(), HasSubstr("num_split"));
}

TEST(SavedModelNumberAttrTest, ExportTruncatingNumberAttrIsRejected) {
  tensorflow::SavedModel exported;
  absl::Status status = ExportSplitFunction(
      [](mlir::Builder& builder, mlir::Operation* op) {
        if (op->getName().getStringRef() == "tfg.Split")
          op->setAttr("num_split",
                      builder.getI64IntegerAttr((int64_t{1} << 32) + 2));
      },
      &exported);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  EXPECT_THAT(status.message(), HasSubstr("num_split"));
}

TEST(SavedModelNumberAttrTest, ImportNegativeNumberAttrIsRejected) {
  absl::Status status =
      ImportWithNumberAttr(kSdcaGraph, "num_sparse_features", -1);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  EXPECT_THAT(status.message(), HasSubstr("num_sparse_features"));
}

TEST(SavedModelNumberAttrTest, ImportTooLargeNumberAttrIsRejected) {
  absl::Status status = ImportWithNumberAttr(kSdcaGraph, "num_sparse_features",
                                             (int64_t{1} << 32) - 1);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  EXPECT_THAT(status.message(), HasSubstr("num_sparse_features"));
}

TEST(SavedModelNumberAttrTest, ExportMissingNumberAttrIsRejected) {
  tensorflow::SavedModel exported;
  absl::Status status = ExportSplitFunction(
      [](mlir::Builder&, mlir::Operation* op) {
        if (op->getName().getStringRef() == "tfg.Split")
          op->removeAttr("num_split");
      },
      &exported);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  EXPECT_THAT(status.message(), HasSubstr("Number attr not found: num_split"));
}

TEST(SavedModelNumberAttrTest, ExportMissingTypeListAttrIsRejected) {
  tensorflow::SavedModel exported;
  absl::Status status = ExportSplitFunction(
      [](mlir::Builder&, mlir::Operation* op) {
        if (op->getName().getStringRef() == "tfg.IdentityN")
          op->removeAttr("T");
      },
      &exported);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  EXPECT_THAT(status.message(), HasSubstr("Type attr not found: T"));
}

}  // namespace
