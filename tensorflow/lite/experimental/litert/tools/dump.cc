// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/tools/dump.h"

#include <dlfcn.h>

#ifndef __ANDROID__
#include <link.h>
#endif

#include <cstdint>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/compiler/plugin/compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"

namespace litert::internal {

namespace {

void DumpNode(const LiteRtTensorT& tensor, std::ostream& out) {
  switch (tensor.type_id) {
    case kLiteRtRankedTensorType:
      Dump(tensor.type_detail.ranked_tensor_type, out);
      break;
    case kLiteRtUnrankedTensorType:
      Dump(tensor.type_detail.unranked_tensor_type.element_type, out);
      break;
    default:
      out << "UKNOWN_TENSOR_TYPE" << tensor.type_id;
  }
  Dump(std::make_pair(tensor.q_type_id, tensor.q_type_detail), out);
}

void DumpNode(const LiteRtOpT& op, std::ostream& out) { Dump(op.op_code, out); }

void DumpSignature(const std::vector<LiteRtTensor>& ins,
                   const std::vector<LiteRtTensor>& outs, std::ostream& out) {
  out << "(";
  for (auto it = ins.begin(); it < ins.end(); ++it) {
    DumpNode(**it, out);
    if (it != ins.end() - 1) {
      out << ", ";
    }
  }
  out << ")";

  out << " -> ";
  const bool paren_outs = outs.size() != 1;
  if (paren_outs) {
    out << "(";
  }
  for (auto it = outs.begin(); it < outs.end(); ++it) {
    DumpNode(**it, out);
    if (it != outs.end() - 1) {
      out << ", ";
    }
  }
  if (paren_outs) {
    out << ")";
  }
}

}  // namespace

void Dump(LiteRtOpCode code, std::ostream& out) {
  switch (code) {
    case kLiteRtOpCodeTflAdd:
      out << "TFL_ADD";
      break;
    case kLiteRtOpCodeTflMul:
      out << "TFL_MUL";
      break;
    case kLiteRtOpCodeTflCustom:
      out << "TFL_CUSTOM_OP";
      break;
    case kLiteRtOpCodeTflSlice:
      out << "TFL_SLICE";
      break;
    case kLiteRtOpCodeTflDiv:
      out << "TFL_DIV";
      break;
    case kLiteRtOpCodeTflRsqrt:
      out << "TFL_RSQRT";
      break;
    case kLiteRtOpCodeTflTanh:
      out << "TFL_TANH";
      break;
    case kLiteRtOpCodeTflSub:
      out << "TFL_SUB";
      break;
    case kLiteRtOpCodeTflReshape:
      out << "TFL_RESHAPE";
      break;
    case kLiteRtOpCodeTflBatchMatmul:
      out << "TFL_BATCH_MATMUL";
      break;
    case kLiteRtOpCodeTflSum:
      out << "TFL_SUM";
      break;
    case kLiteRtOpCodeTflConcatenation:
      out << "TFL_CONCATENATION";
      break;
    case kLiteRtOpCodeTflSoftmax:
      out << "TFL_SOFTMAX";
      break;
    case kLiteRtOpCodeTflCast:
      out << "TFL_CAST";
      break;
    case kLiteRtOpCodeTflTranspose:
      out << "TFL_TRANSPOSE";
      break;
    case kLiteRtOpCodeTflSin:
      out << "TFL_SIN";
      break;
    case kLiteRtOpCodeTflCos:
      out << "TFL_COS";
      break;
    case kLiteRtOpCodeTflSelect:
      out << "TFL_SELECT";
      break;
    case kLiteRtOpCodeTflSelectV2:
      out << "TFL_SELECT_V2";
      break;
    case kLiteRtOpCodeTflFullyConnected:
      out << "TFL_FULLY_CONNECTED";
      break;
    case kLiteRtOpCodeTflEmbeddingLookup:
      out << "TFL_EMBEDDING_LOOKUP";
      break;
    case kLiteRtOpCodeTflLogicalAnd:
      out << "TFL_LOGICAL_AND";
      break;
    case kLiteRtOpCodeTflLess:
      out << "TFL_LESS";
      break;
    default:
      out << "UKNOWN_OP_CODE: " << code;
      break;
  }
};

// Dump details about the given LiteRtElementType to the given stream.
void Dump(LiteRtElementType type, std::ostream& out) {
  switch (type) {
    case kLiteRtElementTypeFloat32:
      out << "f32";
      break;
    case kLiteRtElementTypeInt32:
      out << "i32";
      break;
    case kLiteRtElementTypeFloat64:
      out << "f64";
      break;
    case kLiteRtElementTypeInt64:
      out << "i64";
      break;
    case kLiteRtElementTypeFloat16:
      out << "f16";
      break;
    case kLiteRtElementTypeInt16:
      out << "i16";
      break;
    case kLiteRtElementTypeInt8:
      out << "i8";
      break;
    case kLiteRtElementTypeUInt8:
      out << "ui8";
      break;
    case kLiteRtElementTypeBool:
      out << "i1";
      break;
    default:
      out << "UKNNOWN_ELEMENT_TYPE: " << type;
  }
}

void Dump(const LiteRtRankedTensorType& type, std::ostream& out) {
  out << "<";
  for (int i = 0; i < type.layout.rank; ++i) {
    out << type.layout.dimensions[i] << "x";
  }
  Dump(type.element_type, out);
  out << ">";
}

void Dump(const LiteRtTensorT& tensor, std::ostream& out) {
  out << "LiteRtTensor : ";
  DumpNode(tensor, out);
  out << " [ ";
  if (tensor.defining_op == nullptr) {
    out << "*";
  } else {
    DumpNode(*tensor.defining_op, out);
  }
  out << " ] ";

  out << "(";
  for (auto it = tensor.users.begin(); it < tensor.users.end(); ++it) {
    DumpNode(**it, out);
    if (it != tensor.users.end() - 1) {
      out << ", ";
    }
  }
  out << ")";
  out << "\n";
}

void Dump(const LiteRtOpT& op, std::ostream& out) {
  out << "LiteRtOp : [ ";
  DumpNode(op, out);
  out << " ] ";
  DumpSignature(op.inputs, op.outputs, out);
  out << "\n";
}

void Dump(const LiteRtSubgraphT& subgraph, std::ostream& out) {
  constexpr absl::string_view kSubgraphTpl =
      "LiteRtSubgraph : [ #ops=%d #tensors=%d ] ";
  out << absl::StreamFormat(kSubgraphTpl, subgraph.ops.size(),
                            subgraph.tensors.size());
  DumpSignature(subgraph.inputs, subgraph.outputs, out);
  out << "\n";
}

void Dump(const CompilerPlugin& plugin, std::ostream& out) {
  constexpr absl::string_view kPluginDumpTpl =
      "SocManufacturer: %s\nSocModels: { ";
  out << absl::StreamFormat(kPluginDumpTpl, plugin.SocManufacturer());

  for (auto it = plugin.SocModels().begin(); it < plugin.SocModels().end();
       ++it) {
    out << *it;
    if (it != plugin.SocModels().end() - 1) {
      out << ",";
    }
    out << " ";
  }

  out << "}\n";
}

void Dump(void* lib_handle, std::ostream& out) {
#ifndef __ANDROID__
  out << "\n--- Lib Info ---\n";
  if (lib_handle == nullptr) {
    out << "Handle is nullptr\n";
    return;
  }

  Lmid_t dl_ns_idx;
  if (0 != ::dlinfo(lib_handle, RTLD_DI_LMID, &dl_ns_idx)) {
    return;
  }

  std::string dl_origin;
  dl_origin.resize(512);
  if (0 != ::dlinfo(lib_handle, RTLD_DI_ORIGIN, dl_origin.data())) {
    return;
  }

  link_map* lm;
  if (0 != ::dlinfo(lib_handle, RTLD_DI_LINKMAP, &lm)) {
    return;
  }

  out << "Lib Namespace: " << dl_ns_idx << "\n";
  out << "Lib Origin: " << dl_origin << "\n";

  out << "loaded objects:\n";

  auto* forward = lm->l_next;
  auto* backward = lm->l_prev;

  while (forward != nullptr) {
    out << "  " << forward->l_name << "\n";
    forward = forward->l_next;
  }

  out << "***" << lm->l_name << "\n";

  while (backward != nullptr) {
    out << "  " << backward->l_name << "\n";
    backward = backward->l_prev;
  }

  out << "\n";
#endif
}

void Dump(const LiteRtModelT& model, std::ostream& out) {
  out << absl::StreamFormat("LiteRtModel : [ #subgraphs=%d ]\n",
                            model.subgraphs.size());
}

void DumpOptions(const LiteRtOpT& op, std::ostream& out) {
  if (op.option.value == nullptr) {
    out << "null options\n";
    return;
  }
  switch (op.op_code) {
    case kLiteRtOpCodeTflAdd:
      out << "fused_activation_function: "
          << op.option.AsAddOptions()->fused_activation_function << "\n";
      break;
    case kLiteRtOpCodeTflMul:
      out << "fused_activation_function: "
          << op.option.AsMulOptions()->fused_activation_function << "\n";
      break;
    case kLiteRtOpCodeTflBatchMatmul:
      out << "adj_x: " << op.option.AsBatchMatMulOptions()->adj_x << "\n";
      out << "adj_y: " << op.option.AsBatchMatMulOptions()->adj_y << "\n";
      out << "asymmetric_quantize_input: "
          << op.option.AsBatchMatMulOptions()->asymmetric_quantize_inputs
          << "\n";
      break;
    case kLiteRtOpCodeTflConcatenation:
      out << "axis: " << op.option.AsConcatenationOptions()->axis << "\n";
      out << "fused_activation_function: "
          << op.option.AsConcatenationOptions()->fused_activation_function
          << "\n";
      break;
    case kLiteRtOpCodeTflDiv:
      out << "fused_activation_function: "
          << op.option.AsDivOptions()->fused_activation_function << "\n";
      break;
    case kLiteRtOpCodeTflFullyConnected:
      out << "weights_format: "
          << op.option.AsFullyConnectedOptions()->weights_format << "\n";
      out << "keep_num_dims: "
          << op.option.AsFullyConnectedOptions()->keep_num_dims << "\n";
      out << "quantized_bias_type: "
          << op.option.AsFullyConnectedOptions()->quantized_bias_type << "\n";
      out << "asymmetric_quantize_input: "
          << op.option.AsFullyConnectedOptions()->asymmetric_quantize_inputs
          << "\n";
      out << "fused_activation_function: "
          << op.option.AsFullyConnectedOptions()->fused_activation_function
          << "\n";
      break;
    case kLiteRtOpCodeTflSoftmax:
      out << "beta: " << op.option.AsSoftmaxOptions()->beta << "\n";
      break;
    case kLiteRtOpCodeTflStridedSlice:
      out << "begin_mask: " << op.option.AsStridedSliceOptions()->begin_mask
          << "\n";
      out << "end_mask: " << op.option.AsStridedSliceOptions()->end_mask
          << "\n";
      out << "ellipsis_mask: "
          << op.option.AsStridedSliceOptions()->ellipsis_mask << "\n";
      out << "new_axis_mask: "
          << op.option.AsStridedSliceOptions()->new_axis_mask << "\n";
      out << "shrink_axis_mask: "
          << op.option.AsStridedSliceOptions()->shrink_axis_mask << "\n";
      out << "offset: " << op.option.AsStridedSliceOptions()->offset << "\n";
      break;
    case kLiteRtOpCodeTflSub:
      out << "fused_activation_function: "
          << op.option.AsSubOptions()->fused_activation_function << "\n";
      break;
    case kLiteRtOpCodeTflReshape:
      out << "new_shape: ";
      if (op.option.AsReshapeOptions() != nullptr) {
        const int32_t* new_shape =
            op.option.AsReshapeOptions()->new_shape.data();
        int32_t new_shape_size = op.option.AsReshapeOptions()->new_shape.size();
        for (int i = 0; i < new_shape_size; ++i) {
          out << new_shape[i] << " ";
        }
      }
      break;
    case kLiteRtOpCodeTflSum:
      out << "keepdims: " << op.option.AsReducerOptions()->keep_dims << "\n";
      break;
    default:
      out << "No options for op code: " << op.op_code;
      break;
  }
}

void Dump(Quantization quantization, std::ostream& out) {
  switch (quantization.first) {
    case kLiteRtQuantizationNone:
      return;
    case kLiteRtQuantizationPerTensor:
      out << absl::StreamFormat(" <q PerTensor [ .z = %ld, .s = %f ]>",
                                quantization.second.per_tensor.zero_point,
                                quantization.second.per_tensor.scale);
      return;
    default:
      out << " <q UNKNOWN>";
      return;
  }
}

}  // namespace litert::internal
