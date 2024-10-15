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

#include "tensorflow/lite/experimental/lrt/tools/dump.h"

#include <dlfcn.h>

#ifndef __ANDROID__
#include <link.h>
#endif

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_op_code.h"
#include "tensorflow/lite/experimental/lrt/core/compiler_plugin/compiler_plugin.h"
#include "tensorflow/lite/experimental/lrt/core/model.h"

namespace lrt::internal {

namespace {

void DumpNode(const LrtTensorT& tensor, std::ostream& out) {
  switch (tensor.type_id) {
    case kLrtRankedTensorType:
      Dump(tensor.type_detail.ranked_tensor_type, out);
      break;
    case kLrtUnrankedTensorType:
      Dump(tensor.type_detail.unranked_tensor_type.element_type, out);
      break;
    default:
      out << "UKNOWN_TENSOR_TYPE" << tensor.type_id;
  }
}

void DumpNode(const LrtOpT& op, std::ostream& out) { Dump(op.op_code, out); }

void DumpSignature(const std::vector<LrtTensor>& ins,
                   const std::vector<LrtTensor>& outs, std::ostream& out) {
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

void Dump(LrtOpCode code, std::ostream& out) {
  switch (code) {
    case kLrtOpCodeTflAdd:
      out << "TFL_ADD";
      break;
    case kLrtOpCodeTflMul:
      out << "TFL_MUL";
      break;
    case kLrtOpCodeTflCustom:
      out << "TFL_CUSTOM_OP";
      break;
    default:
      out << "UKNOWN_OP_CODE: " << code;
      break;
  }
};

// Dump details about the given LrtElementType to the given stream.
void Dump(LrtElementType type, std::ostream& out) {
  switch (type) {
    case kLrtElementTypeFloat32:
      out << "f32";
      break;
    case kLrtElementTypeInt32:
      out << "i32";
      break;
    case kLrtElementTypeFloat64:
      out << "f64";
      break;
    case kLrtElementTypeInt64:
      out << "i64";
      break;
    case kLrtElementTypeFloat16:
      out << "f16";
      break;
    case kLrtElementTypeInt16:
      out << "i16";
      break;
    case kLrtElementTypeInt8:
      out << "i8";
      break;
    case kLrtElementTypeUInt8:
      out << "ui8";
      break;
    case kLrtElementTypeBool:
      out << "i1";
      break;
    default:
      out << "UKNNOWN_ELEMENT_TYPE: " << type;
  }
}

void Dump(const LrtRankedTensorType& type, std::ostream& out) {
  out << "<";
  for (int i = 0; i < type.layout.rank; ++i) {
    out << type.layout.dimensions[i] << "x";
  }
  Dump(type.element_type, out);
  out << ">";
}

void Dump(const LrtTensorT& tensor, std::ostream& out) {
  out << "LrtTensor : ";
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

void Dump(const LrtOpT& op, std::ostream& out) {
  out << "LrtOp : [ ";
  DumpNode(op, out);
  out << " ] ";
  DumpSignature(op.inputs, op.outputs, out);
  out << "\n";
}

void Dump(const LrtSubgraphT& subgraph, std::ostream& out) {
  constexpr absl::string_view kSubgraphTpl =
      "LrtSubgraph : [ #ops=%d #tensors=%d ] ";
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

void DumpOptions(const LrtOpT& op, std::ostream& out) {
  switch (op.op_code) {
    case kLrtOpCodeTflAdd:
      out << "fused_activation_function: "
          << op.option.AsAddOptions()->fused_activation_function << "\n";
      break;
    case kLrtOpCodeTflMul:
      out << "fused_activation_function: "
          << op.option.AsMulOptions()->fused_activation_function << "\n";
      break;
    case kLrtOpCodeTflBatchMatmul:
      out << "adj_x: " << op.option.AsBatchMatMulOptions()->adj_x << "\n";
      out << "adj_y: " << op.option.AsBatchMatMulOptions()->adj_y << "\n";
      out << "asymmetric_quantize_input: "
          << op.option.AsBatchMatMulOptions()->asymmetric_quantize_inputs
          << "\n";
      break;
    case kLrtOpCodeTflConcatenation:
      out << "fused_activation_function: "
          << op.option.AsConcatenationOptions()->fused_activation_function
          << "\n";
      out << "axis: " << op.option.AsConcatenationOptions()->axis << "\n";
      break;
    case kLrtOpCodeTflDiv:
      out << "fused_activation_function: "
          << op.option.AsDivOptions()->fused_activation_function << "\n";
      break;
    case kLrtOpCodeTflFullyConnected:
      out << "fused_activation_function: "
          << op.option.AsFullyConnectedOptions()->fused_activation_function
          << "\n";
      out << "weights_format: "
          << op.option.AsFullyConnectedOptions()->weights_format << "\n";
      out << "keep_num_dims: "
          << op.option.AsFullyConnectedOptions()->keep_num_dims << "\n";
      out << "quantized_bias_type: "
          << op.option.AsFullyConnectedOptions()->quantized_bias_type << "\n";
      out << "asymmetric_quantize_input: "
          << op.option.AsFullyConnectedOptions()->asymmetric_quantize_inputs
          << "\n";
      break;
    case kLrtOpCodeTflSoftmax:
      out << "beta: " << op.option.AsSoftmaxOptions()->beta << "\n";
      break;
    case kLrtOpCodeTflStridedSlice:
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
    case kLrtOpCodeTflSub:
      out << "fused_activation_function: "
          << op.option.AsSubOptions()->fused_activation_function << "\n";
      break;
    case kLrtOpCodeTflReshape:
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
    default:
      out << "UKNOWN_OP_CODE: " << op.op_code;
      break;
  }
}
}  // namespace lrt::internal
