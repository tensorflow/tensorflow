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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_MODEL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_MODEL_H_

#include <list>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_op_code.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/lite/schema/schema_generated.h"

//
// Tensor
//

struct LrtWeightsT {
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
  LrtWeightsT weights;

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

  // This is a placeholder to be used by just custom ops for now.
  std::string custom_options;

  tflite::BuiltinOptionsUnion option;
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

  // TODO: b/365299994 - Delete this.
  // Shared views of remaining unpacked flatbuffer data.
  std::vector<std::shared_ptr<tflite::SubGraphT>> flatbuffer_subgraphs;

  // Initial flatbuffer loaded in. "Subgraphs" field has been invalidated.
  std::unique_ptr<tflite::ModelT> flatbuffer_model;

  // Custom code associated with all customs ops emitted during
  // re-serialization.
  std::string custom_op_code;

  // Look up metadata by key, getting a view of its buffer as a string
  // if it exists.
  LrtResult<FbBufferT> FindMetadata(absl::string_view key) const;
};

//
// Utils
//

// Used for communicating selections of ops.
class LrtOpListT {
 public:
  void Push(LrtOp op) { ops_.push_back(op); }

  std::vector<LrtOp> Vec() const {
    std::vector<LrtOp> res;
    res.reserve(ops_.size());
    res.assign(ops_.begin(), ops_.end());
    return res;
  }

 private:
  // NOTE: This was originally a vector. Was encountering really odd
  // segfaults when freeing after code on another side of a compilation boundary
  // was doing pushes that resized. A list+copy to vector is not optimimal,
  // revisit if bottleneck.
  std::list<LrtOp> ops_;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_MODEL_H_
