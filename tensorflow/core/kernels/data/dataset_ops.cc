/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/data/dataset_ops.h"

#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset_stateful_op_whitelist.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/grappler/graph_topology_view.h"
#include "tensorflow/core/grappler/utils/traversal.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace data {
namespace {
Status FindStatefulOps(const GraphDef& graph_def,
                       std::vector<string>* stateful_op_names) {
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), graph_def.library());

  // Iterate over all nodes in the graph.
  for (const auto& node : graph_def.node()) {
    // Each Dataset graph has a _Retval op in the end which is marked stateful
    if (node.op() == FunctionLibraryDefinition::kRetOp) continue;
    if (!IsNodeStateful(lib_def, node).ok()) {
      stateful_op_names->push_back(node.op());
    }
  }

  // Iterate over all functions.
  for (const auto& fdef : graph_def.library().function()) {
    if (!fdef.signature().is_stateful()) continue;
    for (const auto& node : fdef.node_def()) {
      if (!IsNodeStateful(lib_def, node).ok()) {
        stateful_op_names->push_back(
            absl::StrCat(node.op(), " in function: ", fdef.signature().name()));
      }
    }
  }

  return Status::OK();
}
}  // namespace

/* static */ constexpr const char* const DatasetToGraphOp::kAllowStateful;
/* static */ constexpr const char* const
    DatasetToGraphOp::kStripDeviceAssignment;
/* static */ constexpr const char* const DatasetFromGraphOp::kGraphDef;
/* static */ constexpr const char* const DatasetFromGraphOp::kHandle;

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.
DatasetToGraphOp::DatasetToGraphOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  if (ctx->HasAttr(kAllowStateful)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kAllowStateful, &allow_stateful_ops_));
  }
  if (ctx->HasAttr(kStripDeviceAssignment)) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr(kStripDeviceAssignment, &strip_device_assignment_));
  }
}

void DatasetToGraphOp::Compute(OpKernelContext* ctx) {
  DatasetBase* dataset;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));
  SerializationContext::Params params;
  params.check_external_state = !allow_stateful_ops_;
  GraphDef graph_def;
  OP_REQUIRES_OK(
      ctx, AsGraphDef(ctx, dataset, SerializationContext(params), &graph_def));
  // In case we allow stateful ops, we walk the graph and find all the stateful
  // ops in the Graph. We then log a warning indicating what ops' state we are
  // going to throw away.
  if (allow_stateful_ops_) {
    std::vector<string> stateful_op_names;
    OP_REQUIRES_OK(ctx, FindStatefulOps(graph_def, &stateful_op_names));
    if (!stateful_op_names.empty()) {
      LOG(WARNING)
          << "We found the following stateful ops in the dataset "
             "construction graph whose state would not be serialized and might "
             "cause subtle bugs: "
          << absl::StrJoin(stateful_op_names, ", ");
    }
  }

  if (strip_device_assignment_) {
    auto library = graph_def.mutable_library();
    for (auto& function : (*library->mutable_function())) {
      for (auto& node : (*function.mutable_node_def())) {
        if (!node.device().empty()) {
          *node.mutable_device() = DeviceNameUtils::LocalName(node.device());
        }
      }
    }
  }

  Tensor* result;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &result));
  result->scalar<tstring>()() = graph_def.SerializeAsString();
}

void DatasetCardinalityOp::Compute(OpKernelContext* ctx) {
  DatasetBase* dataset;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));
  Tensor* result;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &result));
  result->scalar<int64>()() = dataset->Cardinality();
}

void DatasetFromGraphOp::Compute(OpKernelContext* ctx) {
  tstring graph_def_string;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument(ctx, kGraphDef, &graph_def_string));
  GraphDef graph_def;
  OP_REQUIRES(ctx, graph_def.ParseFromString(graph_def_string),
              errors::InvalidArgument("Could not parse GraphDef"));
  string output_node;
  for (const auto& node : graph_def.node()) {
    if (node.op() == FunctionLibraryDefinition::kRetOp) {
      output_node = node.input(0);
    }
  }
  Graph graph(OpRegistry::Global());
  OP_REQUIRES_OK(ctx, ImportGraphDef({}, graph_def, &graph, nullptr));

  FunctionLibraryRuntime* flr;
  std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(nullptr);
  OP_REQUIRES_OK(ctx,
                 ctx->function_library()->Clone(&flib_def, &pflr, &flr, true));

  // Some function names may be duplicated (for example, if the serialized
  // graph has an optimized function that retains its original name). We
  // override functions in flib_def in the event of conflict. It is
  // safe to assume that any node in the serialized graph is referring to the
  // serialized function when there is a conflict.
  OP_REQUIRES_OK(ctx,
                 AddToFunctionLibrary(flib_def.get(), graph_def.library()));

  std::vector<Tensor> outputs;
  GraphRunner graph_runner(flr->device());
  OP_REQUIRES_OK(ctx,
                 graph_runner.Run(&graph, flr, {}, {output_node}, &outputs));
  OP_REQUIRES_OK(ctx, ctx->set_output(kHandle, outputs[0]));
}

REGISTER_KERNEL_BUILDER(Name("DatasetToGraph").Device(DEVICE_CPU),
                        DatasetToGraphOp);

REGISTER_KERNEL_BUILDER(Name("DatasetCardinality").Device(DEVICE_CPU),
                        DatasetCardinalityOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalDatasetCardinality").Device(DEVICE_CPU),
    DatasetCardinalityOp);

REGISTER_KERNEL_BUILDER(Name("DatasetFromGraph").Device(DEVICE_CPU),
                        DatasetFromGraphOp);

}  // namespace data
}  // namespace tensorflow
