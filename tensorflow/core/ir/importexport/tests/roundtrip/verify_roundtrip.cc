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

#include <gmock/gmock.h>
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/importexport/export.h"
#include "tensorflow/core/ir/importexport/import.h"
#include "tensorflow/core/ir/importexport/load_proto.h"
#include "tensorflow/core/ir/importexport/tests/roundtrip/roundtrip.h"
#include "tensorflow/core/platform/protobuf.h"

using mlir::MLIRContext;
using mlir::tfg::ImportGraphDefToMlir;
using tensorflow::GraphDef;
using tensorflow::LoadProtoFromFile;
using tensorflow::Status;

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  llvm::cl::opt<std::string> input(llvm::cl::Positional, llvm::cl::Required,
                                   llvm::cl::desc("<input file>"));
  tensorflow::InitMlir y(&argc, &argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "GraphDef Roundtrip testing");
  GraphDef graphdef;
  Status status = LoadProtoFromFile({input.data(), input.size()}, &graphdef);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to load input file '" << input << "': " << status;
    return 2;
  }
  tensorflow::GraphDebugInfo debug_info;
  MLIRContext context;
  auto errorOrModule = ImportGraphDefToMlir(&context, debug_info, graphdef);
  if (!errorOrModule.ok()) {
    LOG(ERROR) << errorOrModule.status();
    return 3;
  }
  auto module = std::move(errorOrModule.ValueOrDie());
  if (failed(mlir::verify(*module))) {
    LOG(ERROR) << "Module verification failed\n";
    return 3;
  }
  {
    // Roundtrip the module to text to ensure the custom printers are complete.
    std::string module_txt;
    llvm::raw_string_ostream os(module_txt);
    module->print(os, mlir::OpPrintingFlags().enableDebugInfo());

    auto new_module =
        mlir::parseSourceString<mlir::ModuleOp>(os.str(), module->getContext());
    if (!new_module) {
      llvm::errs() << "Couldn't reparse module: \n" << *module.get() << "\n";
      return 4;
    }
    module = std::move(new_module);
  }
  GraphDef new_graphdef;
  status = tensorflow::ExportMlirToGraphdef(*module, &new_graphdef);
  if (!status.ok()) {
    llvm::errs()
        << "\n\n=========\n=========\n=========\n=========\n=========\n"
        << *module.get() << "=========\n=========\n=========\n=========\n";
    LOG(ERROR) << "Error exporting MLIR module to GraphDef: " << status;
    return 4;
  }
  // Roundtrip the input graphdef to graph to ensure we add the default
  // attributes.
  {
    tensorflow::GraphConstructorOptions options;
    options.allow_internal_ops = true;
    options.add_default_attributes = true;
    tensorflow::Graph graph(tensorflow::OpRegistry::Global());
    tensorflow::GraphDef preprocessed_graphdef(graphdef);
    auto status = ConvertGraphDefToGraph(
        options, std::move(preprocessed_graphdef), &graph);
    if (!status.ok()) {
      LOG(ERROR) << status;
      return 1;
    }
    graph.ToGraphDef(&graphdef);
  }
  NormalizeTensorData(graphdef, /*add_fulltype=*/true);
  NormalizeTensorData(new_graphdef, /*add_fulltype=*/false);
#if defined(PLATFORM_GOOGLE)
  // This compares the protos with some extra tolerance (NaN, ordering, ...).
  if (!Matches(::testing::proto::TreatingNaNsAsEqual(
          ::testing::proto::IgnoringRepeatedFieldOrdering(
              ::testing::EquivToProto(new_graphdef))))(graphdef)) {
    module->dump();
    EXPECT_THAT(new_graphdef,
                ::testing::proto::TreatingNaNsAsEqual(
                    ::testing::proto::IgnoringRepeatedFieldOrdering(
                        ::testing::EquivToProto(graphdef))));
    return 1;
  }
#endif
  // Because we can't depend on gmock in non-test targets we also use
  // the more strict comparison.
  if (!tensorflow::protobuf::util::MessageDifferencer::Equivalent(
          graphdef, new_graphdef)) {
    // This will show the diff inline.
#if defined(PLATFORM_GOOGLE)
    EXPECT_THAT(new_graphdef, ::testing::EquivToProto(graphdef));
#endif
    llvm::errs() << "Not equivalent\n";
    return 2;
  }
}
