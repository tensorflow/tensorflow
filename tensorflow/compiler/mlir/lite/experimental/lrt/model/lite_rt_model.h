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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_MODEL_LITE_RT_MODEL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_MODEL_LITE_RT_MODEL_H_

#include <list>
#include <vector>

#include "tensorflow/compiler/mlir/lite/experimental/lrt/api/lite_rt_model_api.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/lite_rt_common.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/lite_rt_op_code.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/lite_rt_support.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/schema/schema_generated.h"

//
// Tensor
//

struct LrtBufferT {
  std::unique_ptr<tflite::BufferT> fb_buffer = nullptr;
};

typedef union {
  LrtUnrankedTensorType unranked_tensor_type;
  LrtRankedTensorType ranked_tensor_type;
} LrtTypeDetail;

struct LrtTensorT {
  // Empty if subgraph output. This is a reference.
  std::vector<LrtOp> users;

  // Which arg number for user i.
  std::vector<lrt_param_index_t> user_arg_inds;

  // Null if subgraph input or constant. This is a reference.
  LrtOp defining_op = nullptr;

  // Which output ind from defining op made this tensor.
  lrt_param_index_t defining_op_out_ind;

  // Not a reference.
  LrtBufferT buffer;

  LrtTensorTypeId type_id;

  LrtTypeDetail type_detail;
};

//
// Op
//

struct LrtOpT {
  // These are references.
  std::vector<LrtTensor> inputs;

  // These are references.
  std::vector<LrtTensor> outputs;

  LrtOpCode op_code;

  // TODO: b/365299994 - Add support for op options.
};

//
// Subgraph
//

struct LrtSubgraphT {
  // Storage and views of tensors. Clients are only shown views. Facilitates
  // efficient topological mutation.
  std::list<LrtTensorT> tensors_storage;
  std::vector<LrtTensor> tensors;

  // Storage and vies of ops.
  std::list<LrtOpT> ops_storage;
  std::vector<LrtOp> ops;

  // Shared view of initial flatbuffer data.
  std::shared_ptr<tflite::SubGraphT> flatbuffer_subgraph;

  // These are references and a subset of `tensors`.
  std::vector<LrtTensor> inputs;

  // These are references and a subset of `tensors`.
  std::vector<LrtTensor> outputs;
};

//
// Model
//

// A (partial) unpacking of the flatbuffer model into a list of subgraphs.
// Keeps a reference to the flatbuffer model. Lifetimes of all storage
// are linked to the containing model.
struct LrtModelT {
  // Subgraphs that have been unpacked into usable types.
  std::vector<LrtSubgraphT> subgraphs;

  // Shared views of remaining unpacked flatbuffer data.
  std::vector<std::shared_ptr<tflite::SubGraphT>> flatbuffer_subgraphs;

  // Initial flatbuffer loaded in. "Subgraphs" field has been invalidated.
  std::unique_ptr<tflite::ModelT> flatbuffer_model;
};

//
// Utils
//

// Used for communicating selections of ops.
struct LrtOpListT {
  std::vector<LrtOp> ops;
};

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_MODEL_LITE_RT_MODEL_H_
