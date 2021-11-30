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

// On mobile we do not provide this functionality because not all of its
// dependencies are available there.
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset_stateful_op_allowlist.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/grappler/graph_topology_view.h"
#include "tensorflow/core/grappler/utils/traversal.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace data {

/* static */ constexpr const char* const DatasetToGraphOp::kAllowStateful;
/* static */ constexpr const char* const
    DatasetToGraphOp::kStripDeviceAssignment;
/* static */ constexpr const char* const DatasetToGraphOp::kExternalStatePolicy;
/* static */ constexpr const char* const DatasetToGraphOp::kDatasetToGraph;
/* static */ constexpr const char* const DatasetFromGraphOp::kGraphDef;
/* static */ constexpr const char* const DatasetFromGraphOp::kHandle;

namespace {
constexpr char kPyFunc[] = "PyFunc";
}  // namespace

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.
DatasetToGraphOp::DatasetToGraphOp(OpKernelConstruction* ctx)
    : OpKernel(ctx), op_version_(ctx->def().op() == kDatasetToGraph ? 1 : 2) {
  if (op_version_ == 2) {
    if (ctx->HasAttr(kExternalStatePolicy)) {
      int64_t state_change_option;
      OP_REQUIRES_OK(ctx,
                     ctx->GetAttr(kExternalStatePolicy, &state_change_option));
      external_state_policy_ =
          SerializationContext::ExternalStatePolicy(state_change_option);
    }
  } else {
    if (ctx->HasAttr(kAllowStateful)) {
      bool allow_stateful;
      OP_REQUIRES_OK(ctx, ctx->GetAttr(kAllowStateful, &allow_stateful));
      if (allow_stateful) {
        external_state_policy_ =
            SerializationContext::ExternalStatePolicy::kWarn;
      } else {
        external_state_policy_ =
            SerializationContext::ExternalStatePolicy::kFail;
      }
    }
  }

  if (ctx->HasAttr(kStripDeviceAssignment)) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr(kStripDeviceAssignment, &strip_device_assignment_));
  }
}

void DatasetToGraphOp::Compute(OpKernelContext* ctx) {
  DatasetBase* dataset;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));
  if (dataset->options().optional_external_state_policy_case() ==
      Options::kExternalStatePolicy) {
    switch (dataset->options().external_state_policy()) {
      case ExternalStatePolicy::POLICY_WARN:
        external_state_policy_ =
            SerializationContext::ExternalStatePolicy::kWarn;
        break;
      case ExternalStatePolicy::POLICY_IGNORE:
        external_state_policy_ =
            SerializationContext::ExternalStatePolicy::kIgnore;
        break;
      case ExternalStatePolicy::POLICY_FAIL:
        external_state_policy_ =
            SerializationContext::ExternalStatePolicy::kFail;
        break;
      default: {
        LOG(ERROR) << "Dataset " << dataset->type_string()
                   << " has an unknown external_state_policy enum value: "
                   << dataset->options().external_state_policy();
      }
    }
  }
  SerializationContext::Params params(ctx);
  params.external_state_policy = external_state_policy_;

  GraphDef graph_def;
  Status s = AsGraphDef(ctx, dataset, SerializationContext(params), &graph_def);
  if (!s.ok()) {
    ctx->CtxFailure(errors::FailedPrecondition(
        "Failed to serialize the input pipeline graph: ", s.error_message()));
    return;
  }
  if (strip_device_assignment_) {
    auto library = graph_def.mutable_library();
    for (auto& function : (*library->mutable_function())) {
      for (auto& node : (*function.mutable_node_def())) {
        // We do not strip the device assignment from `PyFunc` ops because they
        // need to be pinned to a host that is known to have Python interpreter.
        if (!node.device().empty() && node.op() != kPyFunc) {
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
  result->scalar<int64_t>()() = dataset->Cardinality();
}

void DatasetFromGraphOp::Compute(OpKernelContext* ctx) {
  tstring graph_def_string;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kGraphDef, &graph_def_string));
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
REGISTER_KERNEL_BUILDER(Name("DatasetToGraphV2").Device(DEVICE_CPU),
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
#endif  // !IS_MOBILE_PLATFORM
