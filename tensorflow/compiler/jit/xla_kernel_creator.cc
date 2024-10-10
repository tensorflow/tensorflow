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

#include <memory>
#include <vector>

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
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/function_utils.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/node_properties.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tsl/platform/errors.h"

namespace tensorflow {

bool XlaKernelCreator::CanCreateKernel(
    const FunctionLibraryRuntime& flr,
    const std::shared_ptr<const NodeProperties>& props) const {
  return CanCreateXlaKernel(props->node_def) &&
         !XlaOpRegistry::IsCompilationDevice(flr.device()->device_type());
}

static absl::Status CreateXlaKernel(FunctionLibraryRuntime* flr,
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

  MemoryTypeVector input_memory_types =
      GetInputMemoryTypes(fbody, constant_arg_indices, resource_arg_indices);
  MemoryTypeVector output_memory_types = GetOutputMemoryTypes(fbody);

  // Create the kernel.
  Device* dev = flr->device();
  absl::Status s;
  auto props = std::make_shared<NodeProperties>(
      &fbody->record->fdef().signature(), node_def, fbody->arg_types,
      fbody->ret_types);
  OpKernelConstruction construction(DeviceType(dev->device_type()), dev,
                                    dev->GetAllocator(AllocatorAttributes()),
                                    flr, dev->resource_manager(), props,
                                    input_memory_types, output_memory_types,
                                    flr->graph_def_version(), &s);

  *kernel = std::make_unique<XlaLocalLaunchBase>(
      &construction, constant_arg_indices, resource_arg_indices, function,
      /*has_ref_vars=*/false);
  return s;
}

absl::Status XlaKernelCreator::CreateKernel(
    FunctionLibraryRuntime* flr,
    const std::shared_ptr<const NodeProperties>& props,
    std::unique_ptr<OpKernel>* kernel) const {
  return CreateXlaKernel(flr, props->node_def, kernel);
}

bool RegisterLaunchOpCreator() {
  XlaKernelCreator* xla_kernel_creator = new XlaKernelCreator();
  RegisterDefaultCustomKernelCreator(xla_kernel_creator);
  return true;
}

static bool register_me = RegisterLaunchOpCreator();

}  // namespace tensorflow
