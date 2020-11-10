/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/jit/xla_kernel_creator.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/jit/compilability_check_util.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/kernels/xla_ops.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/mlir_bridge_pass.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {

// Returns true iff 'ndef' is a call to a function that is compilable.  A
// function is compilable iff every operator in the function body is
// compilable. If 'ndef' is not compilable and 'uncompilable_node_info' is not
// null, we will populate 'uncompilable_node_info' with uncompilable node info.
static bool IsCompilable(FunctionLibraryRuntime* flr, const NodeDef& ndef,
                         RecursiveCompilabilityChecker::UncompilableNodesMap*
                             uncompilable_node_info) {
  Device* device = flr->device();
  const XlaOpRegistry::DeviceRegistration* registration;
  CHECK(XlaOpRegistry::GetCompilationDevice(device->device_type(),
                                            &registration));

  // We can always *compile* resource operations, stateful RNGs and dummy ops,
  // even if we are sometimes unable to auto-cluster them.
  RecursiveCompilabilityChecker::OperationFilter op_filter;
  op_filter.allow_resource_ops_in_called_functions = true;
  op_filter.allow_stack_ops = true;
  op_filter.allow_tensor_array_ops = true;
  op_filter.allow_stateful_rng_ops = true;
  op_filter.allow_control_trigger = true;
  op_filter.allow_eliding_assert_and_checknumerics_ops = true;
  op_filter.allow_ops_producing_or_consuming_variant = true;
  op_filter.allow_slow_ops = true;
  op_filter.allow_inaccurate_ops = true;

  RecursiveCompilabilityChecker checker{
      op_filter, DeviceType{registration->compilation_device_name}};
  if (!uncompilable_node_info) {
    // We do not need uncompilable node info. Just return the result.
    return checker.IsCompilableCall(ndef, flr);
  }

  RecursiveCompilabilityChecker::UncompilableNodesMap uncompilable_node_result =
      checker.FindUncompilableNodes(ndef, flr);
  uncompilable_node_info->swap(uncompilable_node_result);
  return uncompilable_node_info->empty();
}

bool XlaKernelCreator::CanCreateKernel(
    const FunctionLibraryRuntime& flr,
    const std::shared_ptr<const NodeProperties>& props) const {
  return CanCreateXlaKernel(props->node_def) &&
         !XlaOpRegistry::IsCompilationDevice(flr.device()->device_type());
}

static Status CreateXlaKernel(FunctionLibraryRuntime* flr,
                              const NodeDef& node_def,
                              std::unique_ptr<OpKernel>* kernel) {
  if (!CanCreateXlaKernel(node_def)) {
    return errors::Internal("Invalid node: ", node_def.ShortDebugString());
  }

  VLOG(3) << "Attempting to create XlaLaunchOp for " << node_def.DebugString();

  // Make sure that kernels have been registered on the JIT device.
  XlaOpRegistry::RegisterCompilationKernels();

  // Get function body, constant args, and resource args.
  NameAttrList function;
  TF_RETURN_IF_ERROR(NameAndAttrsFromFunctionCall(node_def, &function));
  const FunctionBody* fbody = nullptr;
  std::vector<int> constant_arg_indices;
  std::vector<int> resource_arg_indices;
  TF_RETURN_IF_ERROR(GetBodyAndConstantsAndResources(
      flr, function, &fbody, &constant_arg_indices, &resource_arg_indices));

  // Only check for compilability if the MLIR bridge is not enabled.
  absl::optional<ConfigProto> config_proto;
  if (flr->config_proto()) {
    config_proto = *flr->config_proto();
  }
  if (!IsMlirBridgePassEnabled(*fbody->graph, config_proto)) {
    RecursiveCompilabilityChecker::UncompilableNodesMap uncompilable_nodes_map;
    if (!IsCompilable(flr, node_def, &uncompilable_nodes_map)) {
      std::vector<RecursiveCompilabilityChecker::UncompilableNodeInfo>
          uncompilable_node_info;
      for (const auto& it : uncompilable_nodes_map) {
        for (const auto& info : it.second.second) {
          uncompilable_node_info.emplace_back(info);
        }
      }
      string message = absl::StrCat(
          "Function invoked by the following node is not compilable: ",
          SummarizeNodeDef(node_def, /*max_inputs_in_summary=*/10), ".\n");
      absl::StrAppend(&message, "Uncompilable nodes:");
      for (const auto& node_info : uncompilable_node_info) {
        string node_message = absl::StrCat("\n", node_info.name, ": ",
                                           node_info.uncompilable_reason, "\n",
                                           "\tStacktrace:\n");
        for (const auto& stack_frame : node_info.stack_trace) {
          absl::StrAppendFormat(&node_message, "\t\tNode: %s, function: %s\n",
                                stack_frame.name, stack_frame.function_name);
        }
        absl::StrAppend(&message, node_message);
      }
      VLOG(1) << message;
      return errors::InvalidArgument(message);
    }
  }

  MemoryTypeVector input_memory_types =
      GetInputMemoryTypes(fbody, constant_arg_indices, resource_arg_indices);
  MemoryTypeVector output_memory_types = GetOutputMemoryTypes(fbody);

  // Create the kernel.
  Device* dev = flr->device();
  Status s;
  auto props = std::make_shared<NodeProperties>(
      &fbody->fdef.signature(), node_def, fbody->arg_types, fbody->ret_types);
  OpKernelConstruction construction(DeviceType(dev->device_type()), dev,
                                    dev->GetAllocator(AllocatorAttributes()),
                                    flr, dev->resource_manager(), props,
                                    input_memory_types, output_memory_types,
                                    flr->graph_def_version(), &s);

  *kernel = absl::make_unique<XlaLocalLaunchBase>(
      &construction, constant_arg_indices, resource_arg_indices, function,
      /*has_ref_vars=*/false);
  return s;
}

Status XlaKernelCreator::CreateKernel(
    FunctionLibraryRuntime* flr,
    const std::shared_ptr<const NodeProperties>& props,
    std::unique_ptr<OpKernel>* kernel) const {
  return CreateXlaKernel(flr, props->node_def, kernel);
}

static bool RegisterLaunchOpCreator() {
  XlaKernelCreator* xla_kernel_creator = new XlaKernelCreator();
  RegisterDefaultCustomKernelCreator(xla_kernel_creator);
  return true;
}

static bool register_me = RegisterLaunchOpCreator();

}  // namespace tensorflow
