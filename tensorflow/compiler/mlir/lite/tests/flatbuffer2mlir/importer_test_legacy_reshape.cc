/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <iostream>
#include <memory>

#include "absl/strings/string_view.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

using llvm::Optional;
using llvm::cl::opt;

// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %p/reshape.mlir -o - \
// RUN:   | %p/importer_test_legacy_reshape - \
// RUN:   | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - \
// RUN:   | FileCheck %s

// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %p/reshape.mlir -o - \
// RUN:   | %p/importer_test_legacy_reshape - \
// RUN:   | flatbuffer_to_string - \
// RUN:   | FileCheck --check-prefix=FB %s

// Tests for verifying the tflite model with single operand reshape can be
// imported correctly.

// NOLINTNEXTLINE
static opt<std::string> inputFileName(llvm::cl::Positional,
                                      llvm::cl::desc("<input file>"),
                                      llvm::cl::init("-"));

namespace mlir {
namespace {
Optional<std::unique_ptr<tflite::ModelT>> RemoveConstantOpInReshape(
    llvm::StringRef buffer) {
  auto model_ptr = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
      buffer.data(), buffer.size());
  if (nullptr == model_ptr) {
    return llvm::None;
  }
  std::unique_ptr<tflite::ModelT> model(model_ptr->GetModel()->UnPack());

  // CHECK: %[[cst:.*]] = "tfl.pseudo_const"() {value = dense<2> : tensor<2xi32>
  // CHECK: %{{.*}} = "tfl.reshape"(%{{.*}}, %[[cst]])

  // FB: subgraphs: [ {
  // FB-NEXT: tensors: [ {
  // FB-NEXT:   shape: [ 2 ],
  // FB-NEXT:   type: INT32,
  // FB-NEXT:   buffer: 1,
  // FB-NEXT:   name: "std.constant",
  // FB-NEXT:   quantization: {
  // FB-EMPTY:
  // FB-NEXT:   }
  // FB-NEXT: }, {
  // FB-NEXT:   shape: [ 4 ],
  // FB-NEXT:   buffer: 2,
  // FB-NEXT:   name: "Const",
  // FB-NEXT:   quantization: {
  // FB-EMPTY:
  // FB-NEXT:   }
  // FB-NEXT: }, {
  // FB-NEXT:   shape: [ 2, 2 ],
  // FB-NEXT:   buffer: 3,
  // FB-NEXT:   name: "reshape",
  // FB-NEXT:   quantization: {
  // FB-EMPTY:
  // FB-NEXT:   }
  // FB-NEXT: } ],
  // FB-NEXT:   outputs: [ 2 ],
  // FB-NEXT:   operators: [ {
  // FB-NEXT:     inputs: [ 1 ],
  // FB-NEXT:     outputs: [ 2 ]
  // FB-NEXT:   } ],
  // FB-NEXT:   name: "main"
  // FB-NEXT: } ],

  // Find the reshape ops and make it single operand.
  for (auto& sub_graph : model->subgraphs) {
    for (auto& op : sub_graph->operators) {
      if (model->operator_codes[op->opcode_index]->builtin_code ==
          tflite::BuiltinOperator_RESHAPE) {
        auto& output_tensor = sub_graph->tensors[op->outputs[0]];
        auto shape = output_tensor->shape;
        bool static_shape = true;
        for (auto dim : shape) {
          if (dim <= 0) static_shape = false;
        }
        // Remove the second operand
        if (static_shape) {
          op->inputs.resize(1);
        }
      }
    }
  }
  return model;
}

}  // namespace
}  // namespace mlir

int main(int argc, char** argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  auto file_or_err = llvm::MemoryBuffer::getFileOrSTDIN(inputFileName.c_str());
  if (std::error_code error = file_or_err.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '" << inputFileName
                 << "': " << error.message() << "\n";
    return 1;
  }
  auto buffer = file_or_err->get();
  auto maybe_module =
      mlir::RemoveConstantOpInReshape(buffer->getBuffer().str());
  if (!maybe_module.hasValue()) {
    return 1;
  }
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::Model> output_model_location =
      tflite::Model::Pack(builder, maybe_module.getValue().get());
  tflite::FinishModelBuffer(builder, output_model_location);
  std::string output_model_content(
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize());
  std::cout << output_model_content << "\n";
  return 0;
}
