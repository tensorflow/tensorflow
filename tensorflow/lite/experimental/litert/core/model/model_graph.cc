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

#include "tensorflow/lite/experimental/litert/core/model/model_graph.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"

namespace litert::internal {

namespace {

bool IsOpDead(const LiteRtOpT& op) {
  return op.Inputs().empty() && op.Outputs().empty();
}

bool IsTensorDead(const LiteRtTensorT& tensor) {
  return tensor.DefiningOp() == nullptr && tensor.NumUses() == 0;
}

}  // namespace

void CloneTo(const LiteRtTensorT& src, LiteRtTensorT& dest) {
  dest.SetName({src.Name().cbegin(), src.Name().cend()});
  dest.SetQarams(src.Qparams());
  dest.SetType(src.Type());

  // Manully copy per-channel quantization params,quant array is owned by
  // tensor.
  if (src.Qparams().first == kLiteRtQuantizationPerChannel) {
    std::vector<float> scales(
        src.Qparams().second.per_channel.scales,
        src.Qparams().second.per_channel.scales +
            src.Qparams().second.per_channel.num_channels);
    std::vector<int64_t> zero_points(
        src.Qparams().second.per_channel.zero_points,
        src.Qparams().second.per_channel.zero_points +
            src.Qparams().second.per_channel.num_channels);
    Quantization dest_qparams = MakePerChannelQuantization(
        scales, zero_points,
        src.Qparams().second.per_channel.quantized_dimension,
        [&dest](auto s) { return dest.RequestScratchBuffer(s); });
    dest.SetQarams(std::move(dest_qparams));
  }

  // Move weight buffer from src to dest.
  const auto& src_weights = src.Weights();
  auto& dest_weights = dest.Weights();

  const auto same_manager =
      src_weights.GetBufferManager() == dest_weights.GetBufferManager();

  if (same_manager) {
    dest_weights.SetBufferId(src_weights.GetBufferId());
  } else {
    OwningBufferRef<uint8_t> weights_buffer(src_weights.Buffer().Data(),
                                            src_weights.Buffer().Size());
    SetWeightsFromOwnedBuffer(dest_weights, std::move(weights_buffer));
  }
}

void CloneTo(const LiteRtOpT& src, LiteRtOpT& dest) {
  dest.SetCustomOptions(src.CustomOptions().Data(), src.CustomOptions().Size());
  litert::internal::SetTflOptions(dest, litert::internal::GetTflOptions(src));
  litert::internal::SetTflOpCodeInd(dest,
                                    litert::internal::GetTflOpCodeInd(src));
  dest.SetOpCode(src.OpCode());
}

LiteRtTensorT& MakeClone(LiteRtSubgraphT& parent, const LiteRtTensorT& src) {
  auto& new_tensor = parent.EmplaceTensor();
  CloneTo(src, new_tensor);
  return new_tensor;
}

LiteRtOpT& MakeClone(LiteRtSubgraphT& parent, const LiteRtOpT& src) {
  auto& new_op = parent.EmplaceOp();
  CloneTo(src, new_op);
  return new_op;
}

std::optional<LiteRtParamIndex> FindInput(const LiteRtOpT& op,
                                          const LiteRtTensorT& tensor) {
  return FindInd(op.Inputs().cbegin(), op.Inputs().cend(), &tensor);
}

std::optional<LiteRtParamIndex> FindOutput(const LiteRtOpT& op,
                                           const LiteRtTensorT& tensor) {
  return FindInd(op.Outputs().cbegin(), op.Outputs().cend(), &tensor);
}

std::optional<LiteRtParamIndex> FindInput(const LiteRtSubgraphT& subgraph,
                                          const LiteRtTensorT& tensor) {
  return FindInd(subgraph.Inputs().cbegin(), subgraph.Inputs().cend(), &tensor);
}

std::optional<LiteRtParamIndex> FindOutput(const LiteRtSubgraphT& subgraph,
                                           const LiteRtTensorT& tensor) {
  return FindInd(subgraph.Outputs().cbegin(), subgraph.Outputs().cend(),
                 &tensor);
}

UseIndices FindUseInds(const LiteRtTensorT& tensor, const LiteRtOpT& op) {
  UseIndices res;
  for (auto i = 0; i < tensor.NumUses(); ++i) {
    if (tensor.Users().at(i) == &op) {
      res.push_back(i);
    }
  }
  return res;
}

bool IsConstant(const LiteRtTensorT& tensor) {
  bool is_zero_sized = false;
  auto layout = tensor.Type().second.ranked_tensor_type.layout;
  if (layout.rank == 1) {
    if (layout.dimensions[0] == 0) {
      is_zero_sized = true;
    }
  }
  const auto is_const = tensor.Weights().Buffer().Size() > 0 || is_zero_sized;
  ABSL_DCHECK(!is_const || tensor.DefiningOp() == nullptr)
      << "Constant tensors should not be defined by an op";
  return is_const;
}

void AttachInput(LiteRtTensor tensor, LiteRtOpT& op) {
  op.Inputs().push_back(tensor);
  tensor->Users().push_back(&op);
  tensor->UserArgInds().push_back(op.Inputs().size() - 1);
}

void AttachOutput(LiteRtTensor tensor, LiteRtOpT& op) {
  ABSL_DCHECK(tensor->DefiningOp() == nullptr)
      << "Cannot add an already defined tensor as op output";
  op.Outputs().push_back(tensor);
  tensor->SetDefiningOp(op, op.Outputs().size() - 1);
}

LiteRtTensor DisconnectInput(LiteRtOpT& op, LiteRtParamIndex input_ind) {
  ABSL_DCHECK(input_ind < op.Inputs().size()) << "Removing tensor index oob";
  auto& input = op.Input(input_ind);

  // Find the index of the use for the given in edge.
  auto target_use_ind = -1;
  for (auto i = 0; i < input.NumUses(); ++i) {
    if (input.Users().at(i) == &op && input.UserArgInds().at(i) == input_ind) {
      target_use_ind = i;
    }
  }
  ABSL_DCHECK_GE(target_use_ind, 0) << "Malformed graph";

  // Slide latter input use arg inds to the left.
  for (auto i = input_ind + 1; i < op.Inputs().size(); ++i) {
    auto& r_in = op.Input(i);
    for (auto u = 0; u < r_in.NumUses(); ++u) {
      auto& r_arg_ind = r_in.UserArgInds().at(u);
      if (r_in.Users().at(u) == &op && r_arg_ind > input_ind) {
        r_arg_ind -= 1;
      }
    }
  }

  // Update the edges.
  input.RemoveUse(target_use_ind);
  op.RemoveInput(input_ind);

  return &input;
}

bool IsIO(const LiteRtSubgraphT& subgraph, const LiteRtTensorT& tensor) {
  return FindInput(subgraph, tensor) || FindOutput(subgraph, tensor);
}

LiteRtTensor DisconnectOutput(LiteRtOpT& op, LiteRtParamIndex output_ind) {
  ABSL_DCHECK(output_ind < op.Outputs().size()) << "Removing tensor index oob";
  auto& output = op.Output(output_ind);
  output.ClearDefiningOp();
  op.RemoveOutput(output_ind);
  return &output;
}

void Drop(LiteRtOpT& litert_op) {
  while (!litert_op.Inputs().empty()) {
    DisconnectInput(litert_op, 0);
  }
  while (!litert_op.Outputs().empty()) {
    DisconnectOutput(litert_op, 0);
  }
}

bool DCE(LiteRtSubgraphT& subgraph) {
  const auto ops_removed = subgraph.RemoveOpIf(IsOpDead);

  auto rm_tensor = [&subgraph = std::as_const(subgraph)](const auto& t) {
    return IsTensorDead(t) && !IsIO(subgraph, t);
  };
  const auto tensors_removed = subgraph.RemoveTensorIf(rm_tensor);
  LITERT_LOG(LITERT_INFO, "Removed %d ops, %d tensors", ops_removed,
             tensors_removed);

  return (ops_removed + tensors_removed) > 0;
}

}  // namespace litert::internal
