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
#include <memory>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/tools/tf_mlir_translate_cl.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v2/tf_executor_to_graph.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/client_library.h"
#include "xla/client/compile_only_client.h"
#include "xla/stream_executor/host/host_platform_id.h"
#include "xla/stream_executor/platform_manager.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace tf2xla {
namespace v2 {
namespace testing {

using tsl::Status;

static constexpr char kMlirToGraphCompilationCheckName[] =
    "mlir-to-graph-compilation-check";
// Use CPU arbitrarily in order to check that a graph compiles at all
static constexpr char kArbitraryDeviceName[] = "XLA_CPU_JIT";

static Status CompileGraph(tensorflow::Graph* graph,
                           xla::CompileOnlyClient* client) {
  if (!graph || !client) {
    return Status(absl::StatusCode::kInvalidArgument,
                  "Invalid graph or client");
  }

  tensorflow::FunctionDefLibrary flib;
  auto flib_def = std::make_unique<tensorflow::FunctionLibraryDefinition>(
      tensorflow::OpRegistry::Global(), flib);

  tensorflow::XlaCompiler::Options options;
  options.device_type = tensorflow::DeviceType(kArbitraryDeviceName);
  options.client = client;
  options.flib_def = flib_def.get();
  tensorflow::XlaCompiler compiler(options);

  std::unique_ptr<tensorflow::Graph> graph_copy(
      new tensorflow::Graph(tensorflow::OpRegistry::Global()));
  tensorflow::CopyGraph(*graph, graph_copy.get());

  tensorflow::XlaCompiler::CompileOptions compile_options;
  tensorflow::XlaCompiler::CompilationResult result;
  return compiler.CompileGraph(compile_options,
                               kMlirToGraphCompilationCheckName,
                               std::move(graph_copy), {}, &result);
}

static mlir::OwningOpRef<mlir::ModuleOp> GraphdefToMlirTranslateFunction(
    llvm::StringRef input, mlir::MLIRContext* context) {
  tensorflow::GraphdefToMlirOptions options{
      debug_info_file,        xla_compile_device_type,
      prune_unused_nodes,     convert_legacy_fed_inputs,
      graph_as_function,      upgrade_legacy,
      enable_shape_inference, unconditionally_use_set_output_shapes,
      enable_soft_placement,  set_original_tf_func_name};

  auto module_or = tensorflow::GraphdefToMlirTranslateFunction(
      input, input_arrays, input_dtypes, input_shapes, output_arrays,
      control_output_arrays, options, context);
  if (!module_or.status().ok()) return nullptr;
  return std::move(module_or).value();
}

static mlir::TranslateToMLIRRegistration GraphdefToMlirTranslate(
    "graphdef-to-mlir", "graphdef-to-mlir", GraphdefToMlirTranslateFunction);

static mlir::LogicalResult MlirToGraphTranslateFunction(
    mlir::ModuleOp module, llvm::raw_ostream& output) {
  if (!module) return mlir::failure();

  tensorflow::GraphExportConfig confs;
  confs.export_entry_func_to_flib = export_entry_func_to_flib;
  confs.export_original_tf_func_name = export_original_tf_func_name;

  std::unique_ptr<tensorflow::FunctionLibraryDefinition> flib_def;
  auto graph =
      std::make_unique<tensorflow::Graph>(tensorflow::OpRegistry::Global());
  absl::flat_hash_set<tensorflow::Node*> control_ret_nodes;
  auto status = tensorflow::tf2xla::v2::ConvertTfExecutorToGraph(
      module, confs, &graph, flib_def.get(), &control_ret_nodes);
  if (!status.ok()) {
    LOG(ERROR) << "Export to Graph failed: " << status;
    return mlir::failure();
  }

  // Use Host platform, which should always exist, to make sure graphs compile.
  auto platform = stream_executor::PlatformManager::PlatformWithId(
      stream_executor::host::kHostPlatformId);
  if (!platform.ok()) {
    return mlir::failure();
  }
  auto client =
      xla::ClientLibrary::GetOrCreateCompileOnlyClient(platform.value());

  tensorflow::XlaOpRegistry::RegisterCompilationKernels();

  // Verify that the resulting graph can compile.
  if (client.ok() && !CompileGraph(graph.get(), client.value()).ok()) {
    return mlir::failure();
  }

  auto graphdef = std::make_unique<tensorflow::GraphDef>();
  // Print the graph to the output after going through GraphDef conversion.
  // The DumpGraphToFile would do this anyway so just skip straight to it.
  graph->ToGraphDef(graphdef.get());
  output << tsl::LegacyUnredactedDebugString(*graphdef);

  return mlir::success();
}

static mlir::TranslateFromMLIRRegistration mlir_to_graph_translate(
    /*name=*/"mlir-to-graph", /*description=*/"convert mlir to graph",
    MlirToGraphTranslateFunction, [](mlir::DialectRegistry& registry) {
      mlir::RegisterAllTensorFlowDialects(registry);
    });

static llvm::LogicalResult MlirToGraphdefTranslateFunction(
    mlir::ModuleOp module, llvm::raw_ostream& output) {
  if (!module) return mlir::failure();

  tensorflow::GraphExportConfig confs;
  confs.export_entry_func_to_flib = export_entry_func_to_flib;
  confs.export_original_tf_func_name = export_original_tf_func_name;

  tensorflow::FunctionLibraryDefinition flib_def(
      tensorflow::OpRegistry::Global(), tensorflow::FunctionDefLibrary());
  auto graph =
      std::make_unique<tensorflow::Graph>(tensorflow::OpRegistry::Global());
  absl::flat_hash_set<tensorflow::Node*> control_ret_nodes;

  auto status = tensorflow::tf2xla::v2::ConvertTfExecutorToGraph(
      module, confs, &graph, &flib_def, &control_ret_nodes);
  if (!status.ok()) {
    LOG(ERROR) << "Export to Graph failed: " << status;
    return mlir::failure();
  }

  tensorflow::GraphDef graphdef;
  graph->ToGraphDef(&graphdef);
  output << tsl::LegacyUnredactedDebugString(graphdef);
  return mlir::success();
}

static mlir::TranslateFromMLIRRegistration mlir_to_graphdef_translate(
    "mlir-to-graphdef", "mlir-to-graphdef", MlirToGraphdefTranslateFunction,
    [](mlir::DialectRegistry& registry) {
      mlir::RegisterAllTensorFlowDialects(registry);
    });

}  // namespace testing
}  // namespace v2
}  // namespace tf2xla
}  // namespace tensorflow
