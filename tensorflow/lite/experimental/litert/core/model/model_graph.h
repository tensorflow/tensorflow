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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_MODEL_GRAPH_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_MODEL_GRAPH_H_

#include <functional>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_consts.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"

namespace litert::internal {

// using IrMapping = absl::flat_hash_map<LiteRtTensor, LiteRtTensor>;

// CLONING

// Clones the basic data between tensors (like name and data) but not
// things related to incoming/outgoing edges (users, defining op) or weights.
void CloneTo(const LiteRtTensorT& src, LiteRtTensorT& dest);

// Clones the basic data between ops (like op code and options) but
// things related to incoming/outgoing edges (input/output tensors).
void CloneTo(const LiteRtOpT& src, LiteRtOpT& dest);

// Same as clone to, but allocates a the dest tensor into given subgraph.
LiteRtTensorT& MakeClone(LiteRtSubgraphT& parent, const LiteRtTensorT& src);

// Same as clone to, but allocates a the dest op into given subgraph.
LiteRtOpT& MakeClone(LiteRtSubgraphT& parent, const LiteRtOpT& src);

// OBSERVERS

// Checks if tensor is input to given op, return its index if so.
std::optional<LiteRtParamIndex> FindInput(const LiteRtOpT& op,
                                          const LiteRtTensorT& tensor);

// Checks if tensor is output to given op, return its index if so.
std::optional<LiteRtParamIndex> FindOutput(const LiteRtOpT& op,
                                           const LiteRtTensorT& tensor);

// Checks if tensor is input to given subgraph, return its index if so.
std::optional<LiteRtParamIndex> FindInput(const LiteRtSubgraphT& subgraph,
                                          const LiteRtTensorT& tensor);

// Checks if tensor is output to given subgraph, return its index if so.
std::optional<LiteRtParamIndex> FindOutput(const LiteRtSubgraphT& subgraph,
                                           const LiteRtTensorT& tensor);

// Check if tensor is part of subgraph IO.
bool IsIO(const LiteRtSubgraphT& subgraph, const LiteRtTensorT& tensor);

using UseIndices =
    absl::InlinedVector<LiteRtParamIndex, kExpectedMaxNumOfTensorUses>;

// Checks if tensor is used by op, return the use inds for each use of tensor by
// op (there may be multiple). These are the indexes to call
// LiteRtTensorT::GetUse with.
UseIndices FindUseInds(const LiteRtTensorT& tensor, const LiteRtOpT& op);

// Is this tensor a constant tensor?
bool IsConstant(const LiteRtTensorT& tensor);

// MUTATORS

// Attaches the pre-allocated tensor to be an input of given op.
void AttachInput(LiteRtTensor tensor, LiteRtOpT& op);

// Attaches the pre-allocated tensor to be an output of given op.
void AttachOutput(LiteRtTensor tensor, LiteRtOpT& op);

// Remove the input edge from an op. Return the disconnected tensor.
LiteRtTensor DisconnectInput(LiteRtOpT& op, LiteRtParamIndex input_ind);

// Remove an output edge from an op. Return the disconnected tensor.
LiteRtTensor DisconnectOutput(LiteRtOpT& op, LiteRtParamIndex output_ind);

// Remove all incoming and outgoing edges from this op. This can prep nodes
// for removal in DCE.
void Drop(LiteRtOpT& litert_op);

// Run very naive dead code elimination. Removes only ops/tensors that have no
// in/out edges. Ops are handled first. Ignores subgraph IO. Not recursive and
// does only one pass. Returns if the graph was modified.
// NOTE: This de-allocates removed objects, only use when references to these
// objects will not be used.
// TODO: Update this with complete work-list based approach.
bool DCE(LiteRtSubgraphT& subgraph);

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_MODEL_GRAPH_H_
