/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/optimize/quantize_model.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/optimize/subgraph_quantizer.h"

namespace tflite {
namespace optimize {

TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* model, ErrorReporter* error_reporter) {
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    internal::SubgraphQuantizer quantizer(model, subgraph, error_reporter);
    for (int op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      auto status = quantizer.QuantizeOperator(op_idx);
      if (status != kTfLiteOk) {
        OperatorT* op = subgraph->operators[op_idx].get();
        const BuiltinOperator op_code =
            model->operator_codes[op->opcode_index]->builtin_code;
        error_reporter->Report(
            "Failed to quantized operator: %s in subgraph %d, node: %d",
            EnumNameBuiltinOperator(op_code), subgraph_idx, op_idx);
        return kTfLiteError;
      }
    }
  }

  flatbuffers::Offset<Model> output_model_location =
      Model::Pack(*builder, model);
  FinishModelBuffer(*builder, output_model_location);

  return kTfLiteOk;
}

}  // namespace optimize
}  // namespace tflite
