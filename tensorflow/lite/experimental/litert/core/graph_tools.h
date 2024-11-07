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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_GRAPH_TOOLS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_GRAPH_TOOLS_H_

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_support.h"
#include "tensorflow/lite/experimental/litert/core/util/buffer_ref.h"

#define MATCH_TRUE(v)       \
  {                         \
    if (!(v)) return false; \
  }

#define MATCH_EQ(lhs, rhs)            \
  {                                   \
    if ((lhs) != (rhs)) return false; \
  }

namespace litert::internal {

using RankedTypeInfo = std::tuple<LiteRtElementType, llvm::ArrayRef<int32_t>>;

using TensorUseInfo = std::tuple<LiteRtOp, LiteRtParamIndex>;
using ::litert::BufferRef;

//===----------------------------------------------------------------------===//
//                               Getters                                      //
//===----------------------------------------------------------------------===//

// TODO: b/365299994 - Switch llvm container types for absl.

// Get the ops that reference given tensor.
inline LiteRtResult<llvm::SmallVector<TensorUseInfo>> GetTensorUses(
    LiteRtTensor tensor) {
  LiteRtParamIndex num_uses;
  LiteRtParamIndex* use_user_arg_ind;
  LiteRtOpArray users = nullptr;

  LITERT_RETURN_RESULT_IF_NOT_OK(
      LiteRtGetTensorUses(tensor, &num_uses, &users, &use_user_arg_ind),
      llvm::SmallVector<TensorUseInfo>);

  llvm::ArrayRef<LiteRtOp> users_arr(users, num_uses);
  llvm::ArrayRef<LiteRtParamIndex> user_arg_ind_arr(use_user_arg_ind, num_uses);

  auto results = llvm::zip(users_arr, user_arg_ind_arr);
  llvm::SmallVector<TensorUseInfo> results_vec(results.begin(), results.end());

  return LiteRtResult<llvm::SmallVector<TensorUseInfo>>::FromValue(results_vec);
}

// Get the only user of given tensor, bad status if tensor doesn't have
// exactly one user.
inline LiteRtResult<TensorUseInfo> GetTensorOnlyUse(LiteRtTensor tensor) {
  LITERT_ASSIGN_OR_RETURN_RESULT(auto uses, GetTensorUses(tensor),
                                 TensorUseInfo);
  if (uses.size() != 1) {
    return LiteRtResult<TensorUseInfo>::FromStatus(
        kLiteRtStatusErrorInvalidGraphInvariant);
  }
  return LiteRtResult<TensorUseInfo>::FromValue(uses[0]);
}

// Get tensor inputs to given op.
inline LiteRtResult<llvm::ArrayRef<LiteRtTensor>> GetOpIns(LiteRtOp op) {
  LiteRtParamIndex num_inputs;
  LiteRtTensorArray inputs = nullptr;

  LITERT_RETURN_RESULT_IF_NOT_OK(LiteRtGetOpInputs(op, &num_inputs, &inputs),
                                 llvm::ArrayRef<LiteRtTensor>);

  return LiteRtResult<llvm::ArrayRef<LiteRtTensor>>::FromValue(
      llvm::ArrayRef<LiteRtTensor>(inputs, num_inputs));
}

// Get the only tensor input to given op, bad status if op doesn't have
// exacty one input.
inline LiteRtResult<LiteRtTensor> GetOnlyOpIn(LiteRtOp op) {
  LITERT_ASSIGN_OR_RETURN_RESULT(auto ins, GetOpIns(op), LiteRtTensor);
  if (ins.size() != 1) {
    return LiteRtResult<LiteRtTensor>::FromStatus(
        kLiteRtStatusErrorInvalidGraphInvariant);
  }
  return LiteRtResult<LiteRtTensor>::FromValue(ins[0]);
}

// Get tensors outputs to given op.
inline LiteRtResult<llvm::ArrayRef<LiteRtTensor>> GetOpOuts(LiteRtOp op) {
  LiteRtParamIndex num_outputs;
  LiteRtTensorArray outputs = nullptr;

  LITERT_RETURN_RESULT_IF_NOT_OK(LiteRtGetOpOutputs(op, &num_outputs, &outputs),
                                 llvm::ArrayRef<LiteRtTensor>);

  return LiteRtResult<llvm::ArrayRef<LiteRtTensor>>::FromValue(
      llvm::ArrayRef<LiteRtTensor>(outputs, num_outputs));
}

// Get the only tensor output to given op, bad status if op doesn't have
// exactly one output.
inline LiteRtResult<LiteRtTensor> GetOnlyOpOut(LiteRtOp op) {
  LITERT_ASSIGN_OR_RETURN_RESULT(auto outs, GetOpOuts(op), LiteRtTensor);
  if (outs.size() != 1) {
    return LiteRtResult<LiteRtTensor>::FromStatus(
        kLiteRtStatusErrorInvalidGraphInvariant);
  }
  return LiteRtResult<LiteRtTensor>::FromValue(outs[0]);
}

// Get all ops in given subgraph in topological order.
inline LiteRtResult<llvm::ArrayRef<LiteRtOp>> GetSubgraphOps(
    LiteRtSubgraph subgraph) {
  LiteRtParamIndex num_ops;
  LiteRtOpArray ops = nullptr;
  LITERT_RETURN_RESULT_IF_NOT_OK(LiteRtGetSubgraphOps(subgraph, &num_ops, &ops),
                                 llvm::ArrayRef<LiteRtOp>);

  return LiteRtResult<llvm::ArrayRef<LiteRtOp>>::FromValue(
      llvm::ArrayRef<LiteRtOp>(ops, num_ops));
}

// Get tensor inputs to given subgraph.
inline LiteRtResult<llvm::ArrayRef<LiteRtTensor>> GetSubgraphInputs(
    LiteRtSubgraph subgraph) {
  LiteRtParamIndex num_inputs;
  LiteRtTensorArray inputs = nullptr;
  LITERT_RETURN_RESULT_IF_NOT_OK(
      LiteRtGetSubgraphInputs(subgraph, &num_inputs, &inputs),
      llvm::ArrayRef<LiteRtTensor>);

  return LiteRtResult<llvm::ArrayRef<LiteRtTensor>>::FromValue(
      llvm::ArrayRef<LiteRtTensor>(inputs, num_inputs));
}

// Get tensor outputs to given subgraph.
inline LiteRtResult<llvm::ArrayRef<LiteRtTensor>> GetSubgraphOutputs(
    LiteRtSubgraph subgraph) {
  LiteRtParamIndex num_outputs;
  LiteRtTensorArray outputs = nullptr;
  LITERT_RETURN_RESULT_IF_NOT_OK(
      LiteRtGetSubgraphOutputs(subgraph, &num_outputs, &outputs),
      llvm::ArrayRef<LiteRtTensor>);

  return LiteRtResult<llvm::ArrayRef<LiteRtTensor>>::FromValue(
      llvm::ArrayRef<LiteRtTensor>(outputs, num_outputs));
}

// Get only subgraph in given model, bad status if model doesn't have exactly
// one subgraph.
// TODO: b/365299994 - Add multi-subgraph getters for graph tools.
inline LiteRtResult<LiteRtSubgraph> GetSubgraph(LiteRtModel model) {
  LiteRtParamIndex num_subgraphs;
  LITERT_RETURN_RESULT_IF_NOT_OK(
      LiteRtGetNumModelSubgraphs(model, &num_subgraphs), LiteRtSubgraph);

  if (num_subgraphs != 1) {
    return LiteRtResult<LiteRtSubgraph>::FromStatus(
        kLiteRtStatusErrorUnsupported);
  }

  LiteRtSubgraph subgraph = nullptr;
  LITERT_RETURN_RESULT_IF_NOT_OK(LiteRtGetModelSubgraph(model, 0, &subgraph),
                                 LiteRtSubgraph);

  return LiteRtResult<LiteRtSubgraph>::FromValue(subgraph);
}

// Get raw metadata buffer from model if it exists.
inline LiteRtResult<BufferRef<uint8_t>> GetMetadata(
    LiteRtModel model, const absl::string_view key) {
  using ResT = LiteRtResult<BufferRef<uint8_t>>;
  const uint8_t* buf;
  size_t size;
  LITERT_RETURN_RESULT_IF_NOT_OK(
      LiteRtGetModelMetadata(model, key.data(),
                             reinterpret_cast<const void**>(&buf), &size),
      BufferRef<uint8_t>);
  return ResT::FromValue(BufferRef(buf, size));
}

//===----------------------------------------------------------------------===//
//                               Matchers                                     //
//===----------------------------------------------------------------------===//

// Matches tensor type id, shape and element type for given tensor.
inline bool MatchRankedTensorType(LiteRtTensor tensor,
                                  LiteRtElementType element_type,
                                  llvm::ArrayRef<int32_t> shape) {
  LiteRtTensorTypeId type_id;
  LITERT_RETURN_VAL_IF_NOT_OK(LiteRtGetTensorTypeId(tensor, &type_id), false);
  MATCH_EQ(type_id, kLiteRtRankedTensorType);

  LiteRtRankedTensorType ranked_tensor_type;
  LITERT_RETURN_VAL_IF_NOT_OK(
      LiteRtGetRankedTensorType(tensor, &ranked_tensor_type), false);
  MATCH_EQ(ranked_tensor_type.element_type, element_type);
  MATCH_EQ(ranked_tensor_type.layout.rank, shape.size());

  for (int i = 0; i < shape.size(); ++i) {
    MATCH_EQ(shape[i], ranked_tensor_type.layout.dimensions[i]);
  }

  return true;
}

// Matches users of given tensor (ordering doesn't matter). If strict is true,
// `use_info` must have same number of elements as tensor has uses. If not,
// it must be a subset.
inline bool MatchTensorHasUses(LiteRtTensor tensor,
                               llvm::ArrayRef<TensorUseInfo> use_info,
                               bool strict = true) {
  // uses are unique so this is sufficient to check for equality.
  LITERT_ASSIGN_OR_RETURN_VAL(auto uses, GetTensorUses(tensor), false);
  MATCH_TRUE(!strict || (uses.size() == use_info.size()));

  llvm::SetVector<TensorUseInfo> unique_uses(uses.begin(), uses.end());

  return llvm::all_of(use_info,
                      [&](auto use) { return unique_uses.contains(use); });
}

// Matches a tensor with no uses.
inline bool MatchTensorNoUses(LiteRtTensor tensor) {
  LiteRtParamIndex num_uses;
  LiteRtParamIndex* use_user_arg_ind;
  LiteRtOpArray users = nullptr;

  LITERT_RETURN_VAL_IF_NOT_OK(
      LiteRtGetTensorUses(tensor, &num_uses, &users, &use_user_arg_ind), false);

  return num_uses == 0;
}

// Matches a tensors defining op and output indice.
inline bool MatchTensorDefiningOp(LiteRtTensor tensor,
                                  LiteRtParamIndex expected_defining_op_out_ind,
                                  LiteRtOp expected_defining_op) {
  bool has_defining_op;
  LiteRtTensorDefiningOp defining_op;
  LITERT_RETURN_VAL_IF_NOT_OK(
      LiteRtGetTensorDefiningOp(tensor, &has_defining_op, &defining_op), false);
  MATCH_EQ(has_defining_op, expected_defining_op != nullptr);

  return expected_defining_op == nullptr ||
         (expected_defining_op == defining_op.op &&
          expected_defining_op_out_ind == defining_op.op_output_index);
}

// Matches a tensor that is not the output of an op (subgraph inputs/consts).
inline bool MatchTensorNoDefiningOp(LiteRtTensor tensor) {
  return MatchTensorDefiningOp(tensor, 0, nullptr);
}

// Matches the op code and types of given ops inputs and outputs.
inline bool MatchOpType(LiteRtOp op,
                        llvm::ArrayRef<RankedTypeInfo> input_type_info,
                        llvm::ArrayRef<RankedTypeInfo> output_type_info,
                        LiteRtOpCode code) {
  LiteRtOpCode actual_code;
  LITERT_RETURN_VAL_IF_NOT_OK(LiteRtGetOpCode(op, &actual_code), false);
  MATCH_EQ(actual_code, code);

  const auto exptected_num_inputs = input_type_info.size();

  LITERT_ASSIGN_OR_RETURN_VAL(auto inputs, GetOpIns(op), false);
  for (int i = 0; i < exptected_num_inputs; ++i) {
    const auto& [type, shape] = input_type_info[i];
    MATCH_TRUE(MatchRankedTensorType(inputs[i], type, shape));
  }

  const auto expected_num_outputs = output_type_info.size();

  LITERT_ASSIGN_OR_RETURN_VAL(auto outputs, GetOpOuts(op), false);
  for (int i = 0; i < expected_num_outputs; ++i) {
    const auto& [type, shape] = output_type_info[i];
    MATCH_TRUE(MatchRankedTensorType(outputs[i], type, shape));
  }

  return true;
}

// Checks that doubly linked structure of ops <-> tensors is valid.
inline bool ValidateTopology(llvm::ArrayRef<LiteRtOp> ops) {
  for (auto& op : ops) {
    LITERT_ASSIGN_OR_RETURN_VAL(auto inputs, GetOpIns(op), false);
    for (auto [input_ind, input] : llvm::enumerate(inputs)) {
      MATCH_TRUE(MatchTensorHasUses(input, {{op, input_ind}}, false));
    }

    LITERT_ASSIGN_OR_RETURN_VAL(auto outputs, GetOpOuts(op), false);
    for (auto [output_ind, output] : llvm::enumerate(outputs)) {
      MATCH_TRUE(MatchTensorDefiningOp(output, output_ind, op));
    }
  }
  return true;
}

// Get weights behind given tensor.
template <typename T>
inline LiteRtResult<llvm::ArrayRef<T>> GetWeights(LiteRtTensor tensor) {
  LiteRtWeights weights = nullptr;
  LITERT_RETURN_RESULT_IF_NOT_OK(LiteRtGetTensorWeights(tensor, &weights),
                                 llvm::ArrayRef<T>);
  size_t size;
  const void* data = nullptr;
  LITERT_RETURN_RESULT_IF_NOT_OK(LiteRtGetWeightsBytes(weights, &data, &size),
                                 llvm::ArrayRef<T>);
  return LiteRtResult<llvm::ArrayRef<T>>::FromValue(
      llvm::ArrayRef<T>(static_cast<const T*>(data), size));
}

// Match weights behind given tensor contains data.
template <typename T>
inline bool MatchWeights(LiteRtTensor tensor, absl::Span<T> expected_data) {
  LiteRtWeights weights = nullptr;
  LITERT_RETURN_VAL_IF_NOT_OK(LiteRtGetTensorWeights(tensor, &weights), false);
  MATCH_TRUE(weights != nullptr);

  size_t size;
  const void* data = nullptr;
  LITERT_RETURN_VAL_IF_NOT_OK(LiteRtGetWeightsBytes(weights, &data, &size),
                              false);
  MATCH_TRUE(data != nullptr);

  MATCH_EQ(size, expected_data.size() * sizeof(T));
  return absl::MakeConstSpan(static_cast<const T*>(data),
                             expected_data.size()) == expected_data;
}

// Match given tensor having no (empty) weights.
inline bool MatchNoWeights(LiteRtTensor tensor) {
  LiteRtWeights weights = nullptr;
  LITERT_RETURN_VAL_IF_NOT_OK(LiteRtGetTensorWeights(tensor, &weights), false);
  MATCH_TRUE(weights != nullptr);

  size_t size;
  const void* data = nullptr;
  LITERT_RETURN_VAL_IF_NOT_OK(LiteRtGetWeightsBytes(weights, &data, &size),
                              false);

  return size == 0;
}

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_GRAPH_TOOLS_H_
