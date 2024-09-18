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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_CORE_MODEL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_CORE_MODEL_H_

#include <sstream>
#ifndef NDEBUG
#include <cstdio>
#include <iostream>
#endif

#include <list>
#include <vector>

#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_op_code.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/cc/lite_rt_support.h"
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

  // This is a placeholder to be usd by just custom ops for now.
  std::string custom_options;

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

  // TODO: b/365299994 - Delete this.
  // Shared views of remaining unpacked flatbuffer data.
  std::vector<std::shared_ptr<tflite::SubGraphT>> flatbuffer_subgraphs;

  // Initial flatbuffer loaded in. "Subgraphs" field has been invalidated.
  std::unique_ptr<tflite::ModelT> flatbuffer_model;

  // Custom code associated with all customs ops emitted during
  // re-serialization.
  std::string custom_op_code;
};

//
// Utils
//

// Used for communicating selections of ops.
struct LrtOpListT {
  std::vector<LrtOp> ops;
};

namespace debug {

// TODO: b/365299994 - Flesh out printing api and move elsewhere.
inline void DumpOp(const LrtOpT& op) {
#ifndef NDEBUG
  using DumpInfo = std::pair<std::vector<std::string>, std::string>;

  auto op_name = [&](const LrtOpT& op) -> std::string {
    std::stringstream result;
    switch (op.op_code) {
      case kLrtOpCodeTflAdd:
        result << "TFL_ADD";
        break;
      case kLrtOpCodeTflMul:
        result << "TFL_MUL";
        break;
      case kLrtOpCodeTflCustom:
        result << "TFL_CUSTOM_OP";
        break;
      default:
        result << "UKNOWN_OP_CODE: " << op.op_code;
        break;
    }
    result << " " << &op;
    return result.str();
  };

  // TODO: b/365299994 - Pull tensor dump into separate functiona nd
  // only dump relevant topology when called in DumpOp.
  auto tensor_dump = [&](const LrtTensorT& tensor) -> DumpInfo {
    DumpInfo result;

    for (int i = 0; i < tensor.users.size(); ++i) {
      auto& user = result.first.emplace_back();
      char* s;
      asprintf(&s, "%s [%lu], ", op_name(*tensor.users[i]).c_str(),
               tensor.user_arg_inds[i]);
      user.assign(s);
      free(s);
    }

    if (tensor.defining_op != nullptr) {
      char* s;
      asprintf(&s, "%s [%lu], ", op_name(*tensor.defining_op).c_str(),
               tensor.defining_op_out_ind);
      result.second.assign(s);
      free(s);
    } else {
      result.second = "NO DEF OP";
    }

    return result;
  };

  auto validate_tensor = [](const LrtTensorT& tensor) -> void {
    if (tensor.users.size() != tensor.user_arg_inds.size()) {
      LRT_FATAL("Invalid tensor.");
    }
  };

  auto print_users = [](const DumpInfo& info) {
    for (const auto& user : info.first) {
      std::cerr << "    USER: " << user << "\n";
    }
  };

  auto print_def = [](const DumpInfo& info) {
    std::cerr << "    DEFINING OP: " << info.second << "\n";
  };

  std::cerr << op_name(op) << " {\n";

  for (const auto& inp : op.inputs) {
    validate_tensor(*inp);
    std::cerr << "  INPUT: " << &inp << "\n";
    print_def(tensor_dump(*inp));
    std::cerr << "\n";
  }

  for (const auto& out : op.outputs) {
    validate_tensor(*out);
    std::cerr << "  OUTPUT: " << &out << "\n";
    print_users(tensor_dump(*out));
    if (out != op.outputs.back()) {
      std::cerr << "\n";
    }
  }

  std::cerr << "}\n";
#endif
}

}  // namespace debug

// TODO: b/365299994 - Make dumping a generic streamable.
#define LRT_DUMP_OP(op) \
  _LRT_D_MSG("");       \
  debug::DumpOp(op);

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_CORE_MODEL_H_
