/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/int32_fulltype.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tsl/protobuf/error_codes.pb.h"

namespace tensorflow {

Status Int32FulltypePass::Int32FullTypeForTensor(DataType dtype,
                                                 FullTypeDef* tensor_t,
                                                 bool set_only_int32,
                                                 Node* node, int output_idx) {
  if (tensor_t->type_id() == TFT_TENSOR) {
    if (tensor_t->args_size() != 1) {
      if (node != nullptr) {
        return Status(
            absl::StatusCode::kInvalidArgument,
            absl::StrCat("Full type for node='", node->name(), "' (op='",
                         node->op_def().name(), "') in '", debug_location_,
                         "' has TFT_TENSOR output ", output_idx, " which has ",
                         tensor_t->args_size(), " args instead of 1.\n got:\n",
                         tensor_t->DebugString()));
      } else {
        return Status(absl::StatusCode::kInvalidArgument,
                      absl::StrCat("TFT_TENSOR has ", tensor_t->args_size(),
                                   " args instead of 1.\n got:\n",
                                   tensor_t->DebugString()));
      }
    }
    if (tensor_t->args(0).type_id() == TFT_INT32) {
      tensor_t->set_type_id(TFT_SHAPE_TENSOR);
    }
  } else if ((tensor_t->type_id() == TFT_UNSET) &&
             ((dtype == DT_INT32) || !set_only_int32)) {
    FullTypeDef data_t;
    map_dtype_to_tensor(dtype, data_t);
    tensor_t->set_type_id(TFT_SHAPE_TENSOR);
    (*tensor_t->add_args()) = data_t;
  }
  return absl::OkStatus();
}

static bool is_host_memory_int32(MemoryType mtype, DataType dtype) {
  return (mtype == HOST_MEMORY) && (dtype == DT_INT32);
}

Status Int32FulltypePass::ProcessGraph(Graph* graph, bool ints_on_device) {
  for (Node* n : graph->op_nodes()) {
    auto output_types = n->output_types();
    bool needs_annotation = false;
    for (const auto& output_type : output_types) {
      MemoryType mtype = ints_on_device
                             ? MTypeFromDTypeIntsOnDevice(output_type)
                             : MTypeFromDType(output_type);
      if (is_host_memory_int32(mtype, output_type)) {
        needs_annotation = true;
      }
    }
    if (!needs_annotation) {
      continue;
    }
    if (n->def().has_experimental_type()) {
      FullTypeDef* node_t = n->mutable_def()->mutable_experimental_type();
      if (node_t->type_id() != TFT_PRODUCT) {
        return Status(
            absl::StatusCode::kInvalidArgument,
            absl::StrCat("Full type for node='", n->name(), "' (op='",
                         n->op_def().name(),
                         "') does not start with TFT_PRODUCT.\n got:\n",
                         node_t->DebugString()));
      }
      if (node_t->args_size() != output_types.size()) {
        return Status(
            absl::StatusCode::kInvalidArgument,
            absl::StrCat("Full type for node='", n->name(), "' (op='",
                         n->op_def().name(), "') has ", node_t->args_size(),
                         " outputs but output_types has ", output_types.size(),
                         " outputs.\n got:\n", node_t->DebugString()));
      }
      for (int i = 0; i < node_t->args_size(); ++i) {
        if (MTypeFromDType(output_types[i]) == HOST_MEMORY) {
          TF_RETURN_IF_ERROR(
              Int32FullTypeForTensor(output_types[i], node_t->mutable_args(i),
                                     /*set_only_int32=*/true, n, i));
        }
      }
      VLOG(2) << "Full type information in node '" << n->name() << "' (op='"
              << n->op_def().name()
              << "') modified to use TFT_SHAPE_TENSOR for int32.\n"
              << node_t->DebugString();
    } else {
      FullTypeDef t;
      t.set_type_id(TFT_PRODUCT);
      for (const auto& output_type : output_types) {
        MemoryType mtype = ints_on_device
                               ? MTypeFromDTypeIntsOnDevice(output_type)
                               : MTypeFromDType(output_type);

        if (is_host_memory_int32(mtype, output_type)) {
          FullTypeDef data_t;
          map_dtype_to_tensor(output_type, data_t);
          FullTypeDef out_t;
          out_t.set_type_id(TFT_SHAPE_TENSOR);
          (*out_t.add_args()) = data_t;
          (*t.add_args()) = out_t;
        } else {
          t.add_args();  // Add TFT_UNSET non-HOST_MEMORY outputs
        }
      }
      (*n->mutable_def()->mutable_experimental_type()) = t;
      VLOG(2) << "Full type information with TFT_SHAPE_TENSOR for int32 added "
                 "to node '"
              << n->name() << "' (op='" << n->op_def().name() << "').\n"
              << t.DebugString();
    }
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
