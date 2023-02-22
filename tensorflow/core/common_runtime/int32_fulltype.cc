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
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/tsl/protobuf/error_codes.pb.h"

namespace tensorflow {

Int32Fulltype::Int32Fulltype(Graph* graph) : graph_(graph) {}

Int32Fulltype::~Int32Fulltype() = default;

Status Int32Fulltype::Run() {
  GraphOptimizationPassOptions options;
  // options.debug_filename_prefix, which is used to create graph dump files,
  // will be an empty string.
  return Run(options);
}

Status Int32Fulltype::Run(const GraphOptimizationPassOptions& options) {
  for (Node* n : graph_->op_nodes()) {
    auto output_types = n->output_types();
    bool needs_annotation = false;
    for (const auto& output_type : output_types) {
      // TODO(b/258849883) for the general case, is handling
      // MTypeFromDTypeIntsOnDevice needed? (This is not needed for the
      // mechanism in function.cc.)
      //
      // MemoryType mtype = ints_on_device
      //                       ? MTypeFromDTypeIntsOnDevice(output_type)
      //                      : MTypeFromDType(output_type);
      MemoryType mtype = MTypeFromDType(output_type);
      if (mtype == HOST_MEMORY) {
        needs_annotation = true;
      }
    }
    if (!needs_annotation) {
      continue;
    }
    if (n->def().has_experimental_type()) {
      FullTypeDef* t = n->mutable_def()->mutable_experimental_type();
      if (t->type_id() != TFT_PRODUCT) {
        return Status(
            error::INVALID_ARGUMENT,
            absl::StrCat("Full type for node='", n->name(), "' (op='",
                         n->op_def().name(),
                         "') does not start with TFT_PRODUCT.\n got:\n",
                         t->DebugString()));
      }
      if (t->args_size() != output_types.size()) {
        return Status(
            error::INVALID_ARGUMENT,
            absl::StrCat("Full type for node='", n->name(), "' (op='",
                         n->op_def().name(), "') has ", t->args_size(),
                         " outputs but output_types has ", output_types.size(),
                         " outputs.\n got:\n", t->DebugString()));
      }
      for (int i = 0; i < t->args_size(); ++i) {
        if (MTypeFromDType(output_types[i]) == HOST_MEMORY) {
          if (t->args(i).type_id() == TFT_TENSOR) {
            if (t->args(i).args_size() != 1) {
              return Status(
                  error::INVALID_ARGUMENT,
                  absl::StrCat("Full type for node='", n->name(), "' (op='",
                               n->op_def().name(), "') has TFT_TENSOR output ",
                               i, " which has ", t->args(i).args_size(),
                               " args instead of 1.\n got:\n",
                               t->DebugString()));
            }
            if (t->args(i).args(0).type_id() == TFT_INT32) {
              t->mutable_args(i)->set_type_id(TFT_SHAPE_TENSOR);
            }
          }
        }
      }
      VLOG(2) << "Full type information in node '" << n->name() << "' (op='"
              << n->op_def().name()
              << "') modified to use TFT_SHAPE_TENSOR for int32.\n"
              << t->DebugString();
    } else {
      FullTypeDef t;
      t.set_type_id(TFT_PRODUCT);
      for (const auto& output_type : output_types) {
        if (MTypeFromDType(output_type) == HOST_MEMORY) {
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

  if (VLOG_IS_ON(3)) {
    DumpGraphToFile(
        strings::StrCat(options.debug_filename_prefix, "int32_fulltype"),
        *graph_, nullptr);
  }
  return OkStatus();
}

}  // namespace tensorflow
