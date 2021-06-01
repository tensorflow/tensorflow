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
#include <memory>
#include <sstream>

#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"

namespace tflite {

void dump_tensors(std::stringstream& out_stream,
                  const flatbuffers::Vector<int32_t>* tensors) {
  for (int i = 0; i < tensors->Length(); ++i) {
    const int tensor_idx = tensors->Get(i);
    out_stream << "T#" << tensor_idx;
    if (i != tensors->Length() - 1) {
      out_stream << ", ";
    }
  }
}

void dump_node(std::stringstream& out_stream, const int node_no,
               const OperatorCode* op_code, const Operator* op,
               const SubGraph* subgraph) {
  auto builtin_code = GetBuiltinCode(op_code);
  if (builtin_code != BuiltinOperator_CUSTOM) {
    out_stream << "Op#" << node_no << " "
               << EnumNameBuiltinOperator(builtin_code);
  } else {
    out_stream << "Op#" << node_no << " " << op_code->custom_code();
  }

  out_stream << "(";
  dump_tensors(out_stream, op->inputs());
  out_stream << ") -> [";
  dump_tensors(out_stream, op->outputs());
  out_stream << "]\n";
}

std::string model_analyzer(const std::string& model_file_path) {
  std::stringstream out_stream;
  auto fb_model = FlatBufferModel::BuildFromFile(model_file_path.c_str());
  if (!fb_model) {
    out_stream << "Failed to mmap model " << model_file_path;
    return out_stream.str();
  }
  const ::tflite::Model* model = fb_model->GetModel();
  auto* subgraphs = model->subgraphs();
  for (int i = 0; i < subgraphs->Length(); ++i) {
    const SubGraph* subgraph = subgraphs->Get(i);
    out_stream << "Subgraph#" << i;
    if (subgraph->name()) {
      out_stream << " " << subgraph->name()->str();
    }
    out_stream << "(";
    dump_tensors(out_stream, subgraph->inputs());
    out_stream << ") -> [";
    dump_tensors(out_stream, subgraph->outputs());
    out_stream << "]\n";
    for (int j = 0; j < subgraph->operators()->Length(); ++j) {
      const Operator* op = subgraph->operators()->Get(j);
      const OperatorCode* op_code =
          model->operator_codes()->Get(op->opcode_index());
      out_stream << "  ";  // indents for operators
      dump_node(out_stream, /*node_no=*/j, op_code, op, subgraph);
    }
  }
  return out_stream.str();
}

}  // namespace tflite
