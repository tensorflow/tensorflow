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
#include <algorithm>
#include <memory>
#include <sstream>
#include <string>

#include "absl/strings/str_join.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/tools/versioning/gpu_compatibility.h"
#include "tensorflow/lite/version.h"

namespace tflite {

namespace {

const float kThreshold_zero_buffer_ratio = 10.0f;
constexpr char kSectionSplitter[] =
    "---------------------------------------------------------------\n";
const int kMaxContentDumpCnt = 5;

// Returns string representation of the given tensor data up to 5 elements.
const std::string get_tensor_data_str(const tflite::Tensor* tensor,
                                      const tflite::Model* model) {
  std::stringstream ss;
  auto buffer_idx = tensor->buffer();
  if (buffer_idx != 0 && buffer_idx < model->buffers()->size()) {
    auto* buffer = model->buffers()->Get(buffer_idx);
    if (buffer->data() == nullptr) {
      return "";
    }
    ss << "[";
    if (buffer->data()->size() != 0) {
      size_t type_size;
      switch (tensor->type()) {
        case tflite::TensorType_INT32:
        case tflite::TensorType_UINT32:
        case tflite::TensorType_FLOAT32:
          type_size = 4;
          break;
        default:
          type_size = 1;
      }
      int data_cnt = buffer->data()->size() / type_size;
      for (int i = 0; i < std::min(kMaxContentDumpCnt, data_cnt); ++i) {
        switch (tensor->type()) {
          case tflite::TensorType_INT32: {
            auto data =
                reinterpret_cast<const int32_t*>(buffer->data()->data());
            ss << data[i];
          } break;
          case tflite::TensorType_UINT32: {
            auto data =
                reinterpret_cast<const uint32_t*>(buffer->data()->data());
            ss << data[i];
          } break;
          case tflite::TensorType_INT8: {
            auto data = reinterpret_cast<const int8_t*>(buffer->data()->data());
            ss << data[i];
          } break;
          case tflite::TensorType_UINT8: {
            auto data =
                reinterpret_cast<const uint8_t*>(buffer->data()->data());
            ss << data[i];
          } break;
          case tflite::TensorType_FLOAT32: {
            auto data = reinterpret_cast<const float*>(buffer->data()->data());
            ss << data[i];
          } break;
          case tflite::TensorType_STRING: {
            auto data = reinterpret_cast<const char*>(buffer->data()->data());
            ss << data[i];
          } break;
          default:
            ss << "??";
            break;
        }
        if (i != data_cnt - 1) {
          ss << ", ";
        }
      }
      if (data_cnt > kMaxContentDumpCnt) {
        ss << "...";
      }
    }
    ss << "]";
  }
  return ss.str();
}

// Returns string representation of the given tensor of the subgraph.
const std::string tensor_str(const int tensor_idx, const int subgraph_idx,
                             const tflite::Model* model = nullptr) {
  std::stringstream ss;
  if (subgraph_idx != 0 && tensor_idx != -1)
    ss << "T#" << subgraph_idx << "_" << tensor_idx;
  else
    ss << "T#" << tensor_idx;
  if (model && tensor_idx != -1) {
    const SubGraph* subgraph = model->subgraphs()->Get(subgraph_idx);
    if (subgraph) {
      auto tensor = subgraph->tensors()->Get(tensor_idx);
      if (tensor && tensor->type() == tflite::TensorType_INT32) {
        ss << get_tensor_data_str(tensor, model);
      }
    }
  }
  return ss.str();
}

// Returns string representation of the given subgraph.
const std::string subgraph_str(const int subgraph_idx) {
  std::stringstream ss;
  ss << "Subgraph#" << subgraph_idx;
  return ss.str();
}

struct ModelStats {
  // FlatBuffer buffer usage (in bytes) per subgraph.
  std::vector<size_t> buffer_usage;
};

// Dump details of the given tensor.
void dump_tensor_detail(std::stringstream& out_stream,
                        const tflite::Tensor* tensor, const int tensor_idx,
                        const int subgraph_idx, const tflite::Model* model,
                        ModelStats* stats) {
  out_stream << tensor_str(tensor_idx, subgraph_idx);
  out_stream << "(" << tensor->name()->str() << ") ";
  // Prints `shape_signature` instead of `shape` if it's available since it
  // supports dynamic shapes.
  if (tensor->shape_signature()) {
    out_stream << "shape_signature:[";
    for (int i = 0; i < tensor->shape_signature()->Length(); ++i) {
      const int j = tensor->shape_signature()->Get(i);
      out_stream << j;
      if (i != tensor->shape_signature()->Length() - 1) {
        out_stream << ", ";
      }
    }
    out_stream << "]";
  } else if (tensor->shape()) {
    out_stream << "shape:[";
    for (int i = 0; i < tensor->shape()->Length(); ++i) {
      const int j = tensor->shape()->Get(i);
      out_stream << j;
      if (i != tensor->shape()->Length() - 1) {
        out_stream << ", ";
      }
    }
    out_stream << "]";
  } else {
    out_stream << "shape:n/a";
  }
  out_stream << ", type:" << EnumNameTensorType(tensor->type());

  // Dump buffer size of constant tensors.
  auto buffer_idx = tensor->buffer();
  if (buffer_idx != 0 && buffer_idx < model->buffers()->Length()) {
    auto* buffer = model->buffers()->Get(buffer_idx);
    if (buffer->data() && buffer->data()->size() != 0) {
      out_stream << " RO " << buffer->data()->size() << " bytes";
      out_stream << ", buffer: " << buffer_idx;
      out_stream << ", data:" << get_tensor_data_str(tensor, model);
      stats->buffer_usage[subgraph_idx] += buffer->data()->size();
    }
  }
  out_stream << "\n";
}

// Dump list of input or output tensors.
void dump_tensor_list(std::stringstream& out_stream,
                      const flatbuffers::Vector<int32_t>* tensors,
                      const int subgraph_idx,
                      const tflite::Model* model = nullptr,
                      bool verbose = false) {
  if (tensors == nullptr) {
    return;
  }
  for (int i = 0; i < tensors->Length(); ++i) {
    const int tensor_idx = tensors->Get(i);
    if (verbose) {
      out_stream << "tensor #" << tensor_idx;
    } else {
      out_stream << tensor_str(tensor_idx, subgraph_idx, model);
    }
    if (i != tensors->Length() - 1) {
      if (verbose) {
        out_stream << " and ";
      } else {
        out_stream << ", ";
      }
    }
  }
}

// Returns the string representation of the given OperatorCode.
const std::string get_op_name(const OperatorCode* op_code) {
  auto builtin_code = GetBuiltinCode(op_code);
  if (builtin_code != BuiltinOperator_CUSTOM) {
    return EnumNameBuiltinOperator(builtin_code);
  } else {
    return op_code->custom_code()->str();
  }
}

// Dump the given Operator node.
void dump_node(std::stringstream& out_stream, const int node_no,
               const OperatorCode* op_code, const Operator* op,
               const int subgraph_index, const ::tflite::Model* model) {
  out_stream << "Op#" << node_no << " " << get_op_name(op_code);
  out_stream << "(";
  dump_tensor_list(out_stream, op->inputs(), subgraph_index, model);
  if (GetBuiltinCode(op_code) == BuiltinOperator_CALL_ONCE) {
    out_stream << subgraph_str(
        op->builtin_options_as_CallOnceOptions()->init_subgraph_index());
  } else if (GetBuiltinCode(op_code) == BuiltinOperator_IF) {
    out_stream << ", Then: "
               << subgraph_str(op->builtin_options_as_IfOptions()
                                   ->then_subgraph_index());
    out_stream << ", Else: "
               << subgraph_str(op->builtin_options_as_IfOptions()
                                   ->else_subgraph_index());
  } else if (GetBuiltinCode(op_code) == BuiltinOperator_WHILE) {
    out_stream << ", Cond: "
               << subgraph_str(op->builtin_options_as_WhileOptions()
                                   ->cond_subgraph_index());
    out_stream << ", Body: "
               << subgraph_str(op->builtin_options_as_WhileOptions()
                                   ->body_subgraph_index());
  }
  out_stream << ") -> [";
  dump_tensor_list(out_stream, op->outputs(), subgraph_index);
  out_stream << "]\n";
}

// Dump the summary of the given TFLite flatbuffer model. It's printed at the
// beginning of the analyzer output.
void dump_model_summary(std::stringstream& out_stream,
                        const ::tflite::Model* model) {
  auto* subgraphs = model->subgraphs();
  out_stream
      << "Your TFLite model has '" << subgraphs->Length()
      << "' subgraph(s). In the subgraph description below,\nT# represents the "
         "Tensor numbers. ";
  if (subgraphs->Length() > 0 && subgraphs->Get(0)->operators()->Length() > 0) {
    const Operator* first_op = subgraphs->Get(0)->operators()->Get(0);
    const OperatorCode* first_op_code =
        model->operator_codes()->Get(first_op->opcode_index());
    out_stream << "For example, in " << subgraph_str(0) << ", the "
               << get_op_name(first_op_code) << " op takes\n";
    dump_tensor_list(out_stream, first_op->inputs(), 0, nullptr,
                     /*verbose=*/true);
    out_stream << " as input and produces ";
    dump_tensor_list(out_stream, first_op->outputs(), 0, nullptr,
                     /*verbose=*/true);
    out_stream << " as output.\n\n";
  }
}

// Dump the signature definitions of the given TFLite flatbuffer model.
void dump_model_signature_defs(std::stringstream& out_stream,
                               const ::tflite::Model* model) {
  auto* signatures = model->signature_defs();
  if (signatures == nullptr || signatures->Length() == 0) {
    return;
  }
  out_stream << kSectionSplitter;
  out_stream << "Your TFLite model has '" << signatures->Length()
             << "' signature_def(s).\n\n";
  for (int i = 0; i < signatures->Length(); ++i) {
    auto* signature_def = signatures->Get(i);
    out_stream << "Signature#" << i << " key: '"
               << signature_def->signature_key()->str() << "'\n";
    out_stream << "- Subgraph: "
               << subgraph_str(signature_def->subgraph_index()) << "\n";
    out_stream << "- Inputs: \n";
    for (int j = 0; j < signature_def->inputs()->Length(); ++j) {
      auto* input = signature_def->inputs()->Get(j);
      out_stream << "    '" << input->name()->str() << "' : "
                 << tensor_str(input->tensor_index(),
                               signature_def->subgraph_index())
                 << "\n";
    }
    out_stream << "- Outputs: \n";
    for (int j = 0; j < signature_def->outputs()->Length(); ++j) {
      auto* output = signature_def->outputs()->Get(j);
      out_stream << "    '" << output->name()->str() << "' : "
                 << tensor_str(output->tensor_index(),
                               signature_def->subgraph_index())
                 << "\n";
    }
    out_stream << "\n";
  }
}

// Dump the statistics of the given TFLite flatbuffer model. It's printed at the
// end of the analyzer output.
void dump_model_stats(std::stringstream& out_stream,
                      const ::tflite::Model* model, size_t model_size,
                      ModelStats* stats) {
  size_t total_buffer_size = 0;
  size_t total_zero_buffer_size = 0;
  auto* buffers = model->buffers();
  for (int i = 0; i < buffers->size(); ++i) {
    const tflite::Buffer* buffer = buffers->Get(i);
    if (buffer->data() == nullptr) {
      continue;
    }
    bool is_all_zeros = true;
    const unsigned char* data = buffer->data()->data();
    for (int j = 0; j < buffer->data()->size(); ++j) {
      if (data[j] != 0) {
        is_all_zeros = false;
        break;
      }
    }
    if (is_all_zeros) {
      total_zero_buffer_size += buffer->data()->size();
    }
    total_buffer_size += buffer->data()->size();
  }

  out_stream << kSectionSplitter;
  char temp[2048];
  snprintf(temp, sizeof(temp), "%24s: %10zu bytes\n", "Model size", model_size);
  out_stream << temp;
  snprintf(
      temp, sizeof(temp), "%24s: %10zu bytes (%05.2f %%)\n",
      "Non-data buffer size", model_size - total_buffer_size,
      (static_cast<float>(model_size - total_buffer_size) / model_size * 100));
  out_stream << temp;
  snprintf(temp, sizeof(temp), "%24s: %10zu bytes (%05.2f %%)\n",
           "Total data buffer size", total_buffer_size,
           (static_cast<float>(total_buffer_size) / model_size * 100));
  out_stream << temp;
  if (model->subgraphs()->Length() > 1) {
    for (int i = 0; i < model->subgraphs()->Length(); ++i) {
      float subgraph_buffer_ratio =
          static_cast<float>(stats->buffer_usage[i]) / model_size * 100;
      snprintf(temp, sizeof(temp),
               "          - %-12s: %10zu bytes (%05.2f %%)\n",
               subgraph_str(i).c_str(), stats->buffer_usage[i],
               subgraph_buffer_ratio);
      out_stream << temp;
    }
  }
  float zero_buffer_ratio =
      static_cast<float>(total_zero_buffer_size) / model_size * 100;
  snprintf(temp, sizeof(temp), "%24s: %10zu bytes (%05.2f %%)\n",
           "(Zero value buffers)", total_zero_buffer_size, zero_buffer_ratio);
  out_stream << temp;

  out_stream
      << "\n"
      << "* Buffers of TFLite model are mostly used for constant tensors.\n";
  out_stream << "  And zero value buffers are buffers filled with zeros.\n";
  if (zero_buffer_ratio > kThreshold_zero_buffer_ratio) {
    out_stream << "  (Consider use "
                  "`converter._experimental_unfold_large_splat_constant` "
                  "to save the model size.)\n";
  }
  out_stream << "  Non-data buffers area are used to store operators, "
                "subgraphs and etc.\n";
  out_stream << "  You can find more details from "
                "https://github.com/tensorflow/tensorflow/blob/master/"
                "tensorflow/lite/schema/schema.fbs\n";
}

}  // namespace

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
  ModelStats stats;
  stats.buffer_usage.resize(subgraphs->Length());

  dump_model_summary(out_stream, model);

  bool model_is_gpu_compatible = true;
  for (int i = 0; i < subgraphs->Length(); ++i) {
    std::vector<int> gpu_incompatible_nodes;
    const SubGraph* subgraph = subgraphs->Get(i);
    out_stream << subgraph_str(i);
    if (subgraph->name()) {
      out_stream << " " << subgraph->name()->str();
    }
    out_stream << "(";
    dump_tensor_list(out_stream, subgraph->inputs(), i);
    out_stream << ") -> [";
    dump_tensor_list(out_stream, subgraph->outputs(), i);
    out_stream << "]\n";
    for (int j = 0; j < subgraph->operators()->Length(); ++j) {
      const Operator* op = subgraph->operators()->Get(j);
      const OperatorCode* op_code =
          model->operator_codes()->Get(op->opcode_index());
      out_stream << "  ";  // indents for operators
      dump_node(out_stream, /*node_no=*/j, op_code, op, i, model);
      if (check_gpu_compatibility) {
        auto status =
            CheckGpuDelegateCompatibility(op_code, op, subgraph, model);
        if (!status.ok()) {
          gpu_incompatible_nodes.push_back(j);
          out_stream << "GPU COMPATIBILITY WARNING: " << status.message()
                     << "\n";
        }
      }
    }
    if (!gpu_incompatible_nodes.empty()) {
      model_is_gpu_compatible = false;
      out_stream << "\nGPU COMPATIBILITY WARNING: Subgraph#" << i
                 << " has GPU delegate compatibility issues at nodes "
                 << absl::StrJoin(gpu_incompatible_nodes, ", ")
                 << " with TFLite runtime version " << TF_VERSION_STRING
                 << "\n";
    }

    // Dump Subgraph Tensors.
    out_stream << "\nTensors of " << subgraph_str(i) << "\n";
    auto tensors = subgraph->tensors();
    for (int j = 0; j < tensors->Length(); ++j) {
      auto tensor = tensors->Get(j);
      out_stream << "  ";  // indents for tensors
      dump_tensor_detail(out_stream, tensor, j, i, model, &stats);
    }
    out_stream << "\n";
  }
  if (check_gpu_compatibility && model_is_gpu_compatible) {
    out_stream
        << "\nYour model looks compatible with GPU delegate"
        << " with TFLite runtime version " << TF_VERSION_STRING
        << ".\nBut it doesn't guarantee that your model works well with GPU "
           "delegate.\nThere could be some runtime incompatibililty happen.\n";
  }

  dump_model_signature_defs(out_stream, model);
  dump_model_stats(out_stream, model, fb_model->allocation()->bytes(), &stats);

  return out_stream.str();
}

}  // namespace tflite
