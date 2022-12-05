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
#include "tensorflow/lite/schema/schema_utils.h"

using llvm::Optional;
using llvm::cl::opt;

// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s.mlir -o - \
// RUN:   | %p/importer_test_min_max - \
// RUN:   | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - \
// RUN:   | FileCheck %s

// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s.mlir -o - \
// RUN:   | %p/importer_test_min_max - \
// RUN:   | flatbuffer_to_string - \
// RUN:   | FileCheck --check-prefix=FB %s

// Tests for verifying the tflite model with min/max can be imported
// correctly.

// NOLINTNEXTLINE
static opt<std::string> inputFileName(llvm::cl::Positional,
                                      llvm::cl::desc("<input file>"),
                                      llvm::cl::init("-"));

namespace mlir {
namespace {
Optional<std::unique_ptr<tflite::ModelT>> InjectStatsToFullyConnected(
    llvm::StringRef buffer) {
  auto model_ptr = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
      buffer.data(), buffer.size());
  if (nullptr == model_ptr) {
    return llvm::None;
  }
  std::unique_ptr<tflite::ModelT> model(model_ptr->GetModel()->UnPack());

  // FB-LABEL:     name: "arg0",
  // FB-NEXT:      quantization: {
  // FB-NEXT:              min: [ -1.0 ],
  // FB-NEXT:              max: [ 1.0 ]
  // FB-NEXT:      }

  // FB-LABEL:     name: "arg1",
  // FB-NEXT:            quantization: {
  // FB-EMPTY:
  // FB-NEXT:            }

  // FB-LABEL:     name: "tfl.fully_connected",
  // FB-NEXT:      quantization: {
  // FB-NEXT:        min: [ -0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0,
  // FB-SAME:  -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0, -16.0,
  // FB-SAME:  -17.0, -18.0, -19.0, -20.0, -21.0, -22.0, -23.0, -24.0, -25.0,
  // FB-SAME:  -26.0, -27.0, -28.0, -29.0, -30.0, -31.0, -32.0, -33.0, -34.0,
  // FB-SAME:  -35.0, -36.0, -37.0, -38.0, -39.0 ],
  // FB-NEXT:        max: [ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
  // FB-SAME:  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
  // FB-SAME:  21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
  // FB-SAME:  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0 ],
  // FB-NEXT:        quantized_dimension: 1
  // FB-NEXT:      }

  // FB-LABEL:     name: "tfl.fully_connected:1",
  // FB-NEXT:      quantization: {
  // FB-EMPTY:
  // FB-NEXT:      }

  // FB-LABEL:      operators: [ {
  // FB-NEXT:             inputs: [ 0, 1, 2 ],
  // FB-NEXT:             outputs: [ 3, 4 ],
  // FB-NEXT:             builtin_options_type: FullyConnectedOptions,
  // FB-NEXT:             builtin_options: {
  // FB-EMPTY:
  // FB-NEXT:             }
  // FB-NEXT:       } ],

  // CHECK-LABEL: func @main(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>)
  // CHECK-SAME:      -> tensor<40x40xf32>
  // CHECK:         %[[stat:.*]] = "quantfork.stats"(%arg0) {layerStats = dense<
  // CHECK-SAME:      [-1.000000e+00, 1.000000e+00]> : tensor<2xf32>}
  // CHECK-SAME:      : (tensor<40x37xf32>) -> tensor<40x37xf32>
  // CHECK-NEXT:    %[[cst:.*]] = "tfl.pseudo_const"() {value = dense<
  // CHECK-SAME:      1.000000e+00> : tensor<40xf32>} : () -> tensor<40xf32>
  // CHECK-NEXT:    %[[fc:.*]]:2 = "tfl.fully_connected"(%[[stat]], %arg1,
  // CHECK-NEXT:    %[[stat1:.*]] = "quantfork.stats"(%[[fc]]#0)
  // CHECK-SAME:    {axis = 1 : i64,
  // CHECK-SAME:      axisStats = dense<{{\[}}[-0.000000e+00, 0.000000e+00],
  // CHECK-SAME:      [-1.000000e+00, 1.000000e+00],
  // CHECK-SAME:      [-2.000000e+00, 2.000000e+00]
  // CHECK-NEXT:    return %[[stat1]] : tensor<40x40xf32>
  // CHECK-NEXT:  }

  // Find the tensors and inject the min and max to the input and output
  for (auto& sub_graph : model->subgraphs) {
    for (auto& op : sub_graph->operators) {
      if (tflite::GetBuiltinCode(
              model->operator_codes[op->opcode_index].get()) ==
          tflite::BuiltinOperator_FULLY_CONNECTED) {
        // inject min/max to the input and output tensors
        auto& input_tensor = sub_graph->tensors[op->inputs[0]];
        input_tensor->quantization->scale.clear();
        input_tensor->quantization->zero_point.clear();
        input_tensor->quantization->min.push_back(-1.0);
        input_tensor->quantization->max.push_back(1.0);

        auto& output_tensor = sub_graph->tensors[op->outputs[0]];
        auto shape = output_tensor->shape;
        output_tensor->quantization->scale.clear();
        output_tensor->quantization->zero_point.clear();
        for (int i = 0; i < shape.back(); ++i) {
          output_tensor->quantization->min.push_back(-1.0 * i);
          output_tensor->quantization->max.push_back(1.0 * i);
        }
        output_tensor->quantization->quantized_dimension = shape.size() - 1;
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
      mlir::InjectStatsToFullyConnected(buffer->getBuffer().str());
  if (!maybe_module.has_value()) {
    return 1;
  }
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::Model> output_model_location =
      tflite::Model::Pack(builder, maybe_module.value().get());
  tflite::FinishModelBuffer(builder, output_model_location);
  std::string output_model_content(
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize());
  std::cout << output_model_content << "\n";
  return 0;
}
