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

#include "absl/strings/str_join.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/tools/versioning/gpu_compatibility.h"

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

class StreamErrorReporter : public ErrorReporter {
 public:
  explicit StreamErrorReporter(std::stringstream* out_stream)
      : out_stream_(out_stream) {}
  int Report(const char* format, va_list args) override {
    char buffer[1024];
    int size = vsnprintf(buffer, sizeof(buffer), format, args);
    *out_stream_ << buffer;
    return size;
  }

 private:
  std::stringstream* out_stream_;
};

std::string model_analyzer(const std::string& model_file_or_buffer,
                           bool input_is_filepath,
                           bool check_gpu_compatibility) {
  std::stringstream out_stream;
  StreamErrorReporter error_reporter(&out_stream);
  std::unique_ptr<FlatBufferModel> fb_model;
  if (input_is_filepath) {
    fb_model = FlatBufferModel::BuildFromFile(model_file_or_buffer.c_str(),
                                              &error_reporter);
    if (!fb_model) {
      out_stream << "Failed to mmap model " << model_file_or_buffer;
      return out_stream.str();
    }
  } else {
    fb_model = FlatBufferModel::BuildFromBuffer(model_file_or_buffer.c_str(),
                                                model_file_or_buffer.size(),
                                                &error_reporter);
    if (!fb_model) {
      out_stream << "Failed to mmap the given model buffer.";
      return out_stream.str();
    }
  }
  const ::tflite::Model* model = fb_model->GetModel();
  auto* subgraphs = model->subgraphs();
  for (int i = 0; i < subgraphs->Length(); ++i) {
    std::vector<int> gpu_incompatibile_nodes;
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
      if (check_gpu_compatibility) {
        auto status =
            CheckGpuDelegateCompatibility(op_code, op, subgraph, model);
        if (!status.ok()) {
          gpu_incompatibile_nodes.push_back(j);
          out_stream << "GPU COMPATIBILITY WARNING: " << status.message()
                     << "\n";
        }
      }
    }
    if (!gpu_incompatibile_nodes.empty()) {
      out_stream << "\nGPU COMPATIBILITY WARNING: Subgraph#" << i
                 << " has GPU delegate compatibility issues with nodes "
                 << absl::StrJoin(gpu_incompatibile_nodes, ", ") << "\n";
    }
  }
  return out_stream.str();
}

}  // namespace tflite
