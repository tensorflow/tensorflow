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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_CORE_GRAPH_TOOLS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_CORE_GRAPH_TOOLS_H_

#include <cstddef>
#include <cstdint>
#include <tuple>

#ifndef NDEBUG
#include <iostream>
#endif

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_op_code.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/cc/lite_rt_support.h"

#define _D_MATCH_TRUE(v)                                               \
  {                                                                    \
    std::cerr << "failed match true " << __FILE__ << __LINE__ << "\n"; \
    if (!(v)) return false;                                            \
  }

#define _D_MATCH_EQ(lhs, rhs)                                        \
  {                                                                  \
    std::cerr << "failed match eq " << __FILE__ << __LINE__ << "\n"; \
    if (lhs != rhs) return false;                                    \
  }

#define _MATCH_EQ(lhs, rhs)       \
  {                               \
    if (lhs != rhs) return false; \
  }

#define _MATCH_TRUE(v)      \
  {                         \
    if (!(v)) return false; \
  }

#ifndef NDEBUG
#define MATCH_EQ(lhs, rhs) _D_MATCH_EQ(lhs, rhs)
#define MATCH_TRUE(v) _D_MATCH_TRUE(v)
#else
#define MATCH_EQ(lhs, rhs) _MATCH_EQ(lhs, rhs)
#define MATCH_TRUE(v) _MATCH_TRUE(v)
#endif

namespace graph_tools {

using RankedTypeInfo = std::tuple<LrtElementType, llvm::ArrayRef<int32_t>>;

using TensorUseInfo = std::tuple<LrtOp, lrt_param_index_t>;

//===----------------------------------------------------------------------===//
//                               Getters                                      //
//===----------------------------------------------------------------------===//

// TODO: b/365299994 - Switch llvm container types for mobile friendly ones.
// Likely will need to define them.

// Get the ops that reference given tensor.
inline LrtResult<llvm::SmallVector<TensorUseInfo>> GetTensorUses(
    LrtTensor tensor) {
  lrt_param_index_t num_uses;
  lrt_param_index_t* use_user_arg_ind;
  LrtOpArray users = nullptr;

  LRT_RETURN_RESULT_IF_NOT_OK(
      GetTensorUses(tensor, &num_uses, &users, &use_user_arg_ind),
      llvm::SmallVector<TensorUseInfo>);

  llvm::ArrayRef<LrtOp> users_arr(users, num_uses);
  llvm::ArrayRef<lrt_param_index_t> user_arg_ind_arr(use_user_arg_ind,
                                                     num_uses);

  auto results = llvm::zip(users_arr, user_arg_ind_arr);
  llvm::SmallVector<TensorUseInfo> results_vec(results.begin(), results.end());

  return LrtResult<llvm::SmallVector<TensorUseInfo>>::FromValue(results_vec);
}

// Get the only user of given tensor, bad status if tensor doesn't have
// exactly one user.
inline LrtResult<TensorUseInfo> GetTensorOnlyUse(LrtTensor tensor) {
  LRT_ASSIGN_OR_RETURN_RESULT(auto uses, GetTensorUses(tensor), TensorUseInfo);
  if (uses.size() != 1) {
    return LrtResult<TensorUseInfo>::FromCode(kLrtStatusGraphInvariantError);
  }
  return LrtResult<TensorUseInfo>::FromValue(uses[0]);
}

// Get tensor inputs to given op.
inline LrtResult<llvm::ArrayRef<LrtTensor>> GetOpIns(LrtOp op) {
  lrt_param_index_t num_inputs;
  LrtTensorArray inputs = nullptr;

  LRT_RETURN_RESULT_IF_NOT_OK(GetOpInputs(op, &num_inputs, &inputs),
                              llvm::ArrayRef<LrtTensor>);

  return LrtResult<llvm::ArrayRef<LrtTensor>>::FromValue(
      llvm::ArrayRef<LrtTensor>(inputs, num_inputs));
}

// Get the only tensor input to given op, bad status if op doesn't have
// exacty one input.
inline LrtResult<LrtTensor> GetOnlyOpIn(LrtOp op) {
  LRT_ASSIGN_OR_RETURN_RESULT(auto ins, GetOpIns(op), LrtTensor);
  if (ins.size() != 1) {
    return LrtResult<LrtTensor>::FromCode(kLrtStatusGraphInvariantError);
  }
  return LrtResult<LrtTensor>::FromValue(ins[0]);
}

// Get tensors outputs to given op.
inline LrtResult<llvm::ArrayRef<LrtTensor>> GetOpOuts(LrtOp op) {
  lrt_param_index_t num_outputs;
  LrtTensorArray outputs = nullptr;

  LRT_RETURN_RESULT_IF_NOT_OK(GetOpOutputs(op, &num_outputs, &outputs),
                              llvm::ArrayRef<LrtTensor>);

  return LrtResult<llvm::ArrayRef<LrtTensor>>::FromValue(
      llvm::ArrayRef<LrtTensor>(outputs, num_outputs));
}

// Get the only tensor output to given op, bad status if op doesn't have
// exacty one output.
inline LrtResult<LrtTensor> GetOnlyOpOut(LrtOp op) {
  LRT_ASSIGN_OR_RETURN_RESULT(auto outs, GetOpOuts(op), LrtTensor);
  if (outs.size() != 1) {
    return LrtResult<LrtTensor>::FromCode(kLrtStatusGraphInvariantError);
  }
  return LrtResult<LrtTensor>::FromValue(outs[0]);
}

// Get all ops in given subgraph in topological order.
inline LrtResult<llvm::ArrayRef<LrtOp>> GetSubgraphOps(LrtSubgraph subgraph) {
  lrt_param_index_t num_ops;
  LrtOpArray ops = nullptr;
  LRT_RETURN_RESULT_IF_NOT_OK(GetSubgraphOps(subgraph, &num_ops, &ops),
                              llvm::ArrayRef<LrtOp>);

  return LrtResult<llvm::ArrayRef<LrtOp>>::FromValue(
      llvm::ArrayRef<LrtOp>(ops, num_ops));
}

// Get tensor inputs to given subgraph.
inline LrtResult<llvm::ArrayRef<LrtTensor>> GetSubgraphInputs(
    LrtSubgraph subgraph) {
  lrt_param_index_t num_inputs;
  LrtTensorArray inputs = nullptr;
  LRT_RETURN_RESULT_IF_NOT_OK(GetSubgraphInputs(subgraph, &num_inputs, &inputs),
                              llvm::ArrayRef<LrtTensor>);

  return LrtResult<llvm::ArrayRef<LrtTensor>>::FromValue(
      llvm::ArrayRef<LrtTensor>(inputs, num_inputs));
}

// Get tensor outputs to given subgraph.
inline LrtResult<llvm::ArrayRef<LrtTensor>> GetSubgraphOutputs(
    LrtSubgraph subgraph) {
  lrt_param_index_t num_outputs;
  LrtTensorArray outputs = nullptr;
  LRT_RETURN_RESULT_IF_NOT_OK(
      GetSubgraphOutputs(subgraph, &num_outputs, &outputs),
      llvm::ArrayRef<LrtTensor>);

  return LrtResult<llvm::ArrayRef<LrtTensor>>::FromValue(
      llvm::ArrayRef<LrtTensor>(outputs, num_outputs));
}

// Get only subgraph in given model, bad status if model doens't have exactly
// one subgraph.
// TODO: b/365299994 - Add multi-subgraph getters for graph tools.
inline LrtResult<LrtSubgraph> GetSubgraph(LrtModel model) {
  lrt_param_index_t num_subgraphs;
  LRT_RETURN_RESULT_IF_NOT_OK(GetModelNumSubgraphs(model, &num_subgraphs),
                              LrtSubgraph);

  if (num_subgraphs != 1) {
    return LrtResult<LrtSubgraph>::FromCode(kLrtStatusErrorUnsupported);
  }

  LrtSubgraph subgraph = nullptr;
  LRT_RETURN_RESULT_IF_NOT_OK(GetModelSubgraph(model, 0, &subgraph),
                              LrtSubgraph);

  return LrtResult<LrtSubgraph>::FromValue(subgraph);
}

//===----------------------------------------------------------------------===//
//                               Matchers                                     //
//===----------------------------------------------------------------------===//

// Matches tensor type id, shape and element type for given tensor.
inline bool MatchRankedTensorType(LrtTensor tensor, LrtElementType element_type,
                                  llvm::ArrayRef<int32_t> shape) {
  LrtTensorTypeId type_id;
  LRT_RETURN_VAL_IF_NOT_OK(GetTensorTypeId(tensor, &type_id), false);
  MATCH_EQ(type_id, kLrtRankedTensorType);

  LrtRankedTensorType ranked_tensor_type;
  LRT_RETURN_VAL_IF_NOT_OK(GetRankedTensorType(tensor, &ranked_tensor_type),
                           false);
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
inline bool MatchTensorHasUses(LrtTensor tensor,
                               llvm::ArrayRef<TensorUseInfo> use_info,
                               bool strict = true) {
  // uses are unique so this is sufficient to check for equality.
  LRT_ASSIGN_OR_RETURN_VAL(auto uses, GetTensorUses(tensor), false);
  MATCH_TRUE(!strict || (uses.size() == use_info.size()));

  llvm::SetVector<TensorUseInfo> unique_uses(uses.begin(), uses.end());

  return llvm::all_of(use_info,
                      [&](auto use) { return unique_uses.contains(use); });
}

// Matches a tensor with no uses.
inline bool MatchkTensorNoUses(LrtTensor tensor) {
  lrt_param_index_t num_uses;
  lrt_param_index_t* use_user_arg_ind;
  LrtOpArray users = nullptr;

  LRT_RETURN_VAL_IF_NOT_OK(
      GetTensorUses(tensor, &num_uses, &users, &use_user_arg_ind), false);

  return num_uses == 0;
}

// Matches a tensors defining op and output indice.
inline bool MatchTensorDefiningOp(
    LrtTensor tensor, lrt_param_index_t expected_defining_op_out_ind,
    LrtOp expected_defining_op) {
  LrtOp defining_op = nullptr;
  lrt_param_index_t defining_op_out_ind;

  LRT_RETURN_VAL_IF_NOT_OK(
      GetTensorDefiningOp(tensor, &defining_op, &defining_op_out_ind), false);
  MATCH_EQ(defining_op, expected_defining_op);

  return expected_defining_op == nullptr ||
         expected_defining_op_out_ind == defining_op_out_ind;
}

// Matches a tensor that is not the output of an op (subgraph inputs/consts).
inline bool MatchTensorNoDefiningOp(LrtTensor tensor) {
  return MatchTensorDefiningOp(tensor, 0, nullptr);
}

// Matches the op code and types of given ops inputs and outputs.
inline bool MatchOpType(LrtOp op,
                        llvm::ArrayRef<RankedTypeInfo> input_type_info,
                        llvm::ArrayRef<RankedTypeInfo> output_type_info,
                        LrtOpCode code) {
  LrtOpCode actual_code;
  LRT_RETURN_VAL_IF_NOT_OK(GetOpCode(op, &actual_code), false);
  MATCH_EQ(actual_code, code);

  const auto exptected_num_inputs = input_type_info.size();

  LRT_ASSIGN_OR_RETURN_VAL(auto inputs, GetOpIns(op), false);
  for (int i = 0; i < exptected_num_inputs; ++i) {
    const auto& [type, shape] = input_type_info[i];
    MATCH_TRUE(MatchRankedTensorType(inputs[i], type, shape));
  }

  const auto expected_num_outputs = output_type_info.size();

  LRT_ASSIGN_OR_RETURN_VAL(auto outputs, GetOpOuts(op), false);
  for (int i = 0; i < expected_num_outputs; ++i) {
    const auto& [type, shape] = output_type_info[i];
    MATCH_TRUE(MatchRankedTensorType(outputs[i], type, shape));
  }

  return true;
}

// Checks that doubly linked structure of ops <-> tensors is valid.
inline bool ValidateTopology(llvm::ArrayRef<LrtOp> ops) {
  for (auto& op : ops) {
    LRT_ASSIGN_OR_RETURN_VAL(auto inputs, GetOpIns(op), false);
    for (auto [input_ind, input] : llvm::enumerate(inputs)) {
      MATCH_TRUE(MatchTensorHasUses(input, {{op, input_ind}}, false));
    }

    LRT_ASSIGN_OR_RETURN_VAL(auto outputs, GetOpOuts(op), false);
    for (auto [output_ind, output] : llvm::enumerate(outputs)) {
      MATCH_TRUE(MatchTensorDefiningOp(output, output_ind, op));
    }
  }
  return true;
}

// Match buffer behind given tensor contains data.
template <typename T>
inline bool MatchBuffer(LrtTensor tensor, llvm::ArrayRef<T> expected_data) {
  LrtBuffer buffer = nullptr;
  LRT_RETURN_VAL_IF_NOT_OK(GetTensorBuffer(tensor, &buffer), false);
  MATCH_TRUE(buffer != nullptr);

  size_t size;
  const void* data = nullptr;
  LRT_RETURN_VAL_IF_NOT_OK(GetBufferInfo(buffer, &size, &data), false);
  MATCH_TRUE(data != nullptr);

  MATCH_EQ(size, expected_data.size() * sizeof(T));
  return llvm::ArrayRef<T>(static_cast<const T*>(data), expected_data.size()) ==
         expected_data;
}

// Match given tensor having no (empty) buffer.
inline bool MatchNoBuffer(LrtTensor tensor) {
  LrtBuffer buffer = nullptr;
  LRT_RETURN_VAL_IF_NOT_OK(GetTensorBuffer(tensor, &buffer), false);
  MATCH_TRUE(buffer != nullptr);

  size_t size;
  const void* data = nullptr;
  LRT_RETURN_VAL_IF_NOT_OK(GetBufferInfo(buffer, &size, &data), false);

  return size == 0;
}
}  // namespace graph_tools

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_CORE_GRAPH_TOOLS_H_
