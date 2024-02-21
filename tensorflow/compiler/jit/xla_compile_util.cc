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

#include "tensorflow/compiler/jit/xla_compile_util.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/tfrt/common/global_state.h"
#include "tensorflow/core/util/determinism.h"

namespace tensorflow {
namespace {
constexpr const char* kPjRtDeviceCompilerResourceName = "pjrt_device_compiler";
constexpr const char* kPjRtDeviceCompilationProfilerResourceName =
    "pjrt_device_compilation_profiler";
}  // namespace

absl::StatusOr<std::unique_ptr<Graph>> CreateSingleOpGraph(
    const NodeDef& node_def, absl::Span<const XlaArgument> args,
    absl::Span<const DataType> result_types) {
  // TODO(b/74182462): We implement this by creating a new dummy Graph including
  // _Arg nodes, and let CompileGraph walk it. This could be optimized.
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  // First create the actual node we care about computing.
  TF_ASSIGN_OR_RETURN(Node * main_node, graph->AddNode(node_def));

  // Create dummy _Arg nodes. Link these to `node` and also via a control
  // dependency edge to the _SOURCE node.
  for (int64_t i = 0, end = args.size(); i < end; ++i) {
    Node* node;
    string arg_name = absl::StrCat("_arg", i);
    Status status =
        NodeBuilder(arg_name, FunctionLibraryDefinition::kArgOp)
            .ControlInput(graph->source_node())
            .Attr("T", args[i].kind == XlaArgument::kResource ? DT_RESOURCE
                                                              : args[i].type)
            .Attr("index", i)
            .Finalize(graph.get(), &node);
    TF_RETURN_IF_ERROR(status);
    graph->AddEdge(node, 0, main_node, i);
  }

  // Similarly with return values, create dummy _Retval nodes fed by `node`.
  for (int64_t i = 0, end = result_types.size(); i < end; ++i) {
    Node* node;
    string retval_name = absl::StrCat("_retval", i);
    Status status = NodeBuilder(retval_name, FunctionLibraryDefinition::kRetOp)
                        .Input(main_node, i)
                        .Attr("T", result_types[i])
                        .Attr("index", i)
                        .Finalize(graph.get(), &node);
    TF_RETURN_IF_ERROR(status);
  }
  FixupSourceAndSinkEdges(graph.get());
  return graph;
}

bool UsePjRtForSingleDeviceCompilation(const DeviceType& device_type) {
  const auto& rollout_config = GetXlaOpsCommonFlags()->tf_xla_use_device_api;
  return rollout_config.IsEnabledInXlaLaunchForDevice(device_type) ||
         rollout_config.IsEnabledInXlaCompileOnDemandForDevice(device_type) ||
         rollout_config.IsEnabledInXlaCompileAndRunForDevice(device_type);
}

std::string GetPjRtDeviceCompilerResourceName(const DeviceType& device_type) {
  return absl::StrCat(kPjRtDeviceCompilerResourceName, "_",
                      device_type.type_string());
}

std::string GetPjRtDeviceCompilationProfilerResourceName(
    const DeviceType& device_type) {
  return absl::StrCat(kPjRtDeviceCompilationProfilerResourceName, "_",
                      device_type.type_string());
}

absl::StatusOr<ResourceMgr*> GetResourceMgrForDeviceCompiler(
    const OpKernelContext& ctx, const DeviceType& device_type) {
  // We store information about the JIT-compiled XLA computation in the
  // ResourceMgr. The DeviceCompiler (which contains the DeviceCompilationCache)
  // is stored in the tfrt_global ResourceMgr for TPU and the Device ResourceMgr
  // for CPU/GPU. This is to make sure the DeviceCompiler's lifecycle is
  // maintained appropriately.
  ResourceMgr* rm = nullptr;
  if (device_type == DEVICE_TPU) {
    rm = tfrt_global::GetTFGlobalResourceMgr();
  } else {
    rm = ctx.resource_manager();
  }

  if (!rm) {
    return absl::InternalError("No resource manager found.");
  }
  return rm;
}

}  // namespace tensorflow
