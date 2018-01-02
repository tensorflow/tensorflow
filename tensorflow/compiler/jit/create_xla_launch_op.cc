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

#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/kernels/xla_launch_op.h"
#include "tensorflow/compiler/jit/mark_for_compilation_pass.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace {

// Givens a NodeDef 'ndef' and the function library runtime 'flr', if
// 'ndef' is a call to a compilable function defined in 'flr', returns OK
// and fills in 'kernel' with a XlaLaunchOp kernel which computes the
// node. Otherwise, returns a non-OK.
//
// This routine is here so that FunctionLibraryRuntime can jit a
// specific function call as requested.
Status CreateXlaLaunchOp(FunctionLibraryRuntime* flr, const NodeDef& ndef,
                         std::unique_ptr<OpKernel>* kernel) {
  bool xla_compile = false;
  if (!flr->GetFunctionLibraryDefinition()
           ->GetAttr(ndef, kXlaCompileAttr, &xla_compile)
           .ok() ||
      !xla_compile) {
    // Not marked as _XlaCompile=true.
    return errors::InvalidArgument("No ", kXlaCompileAttr, " for ", ndef.op());
  }
  // Make sure that kernels have been registered on the JIT device.
  XlaOpRegistry::RegisterCompilationKernels();
  if (!IsCompilable(flr, ndef)) {
    // ndef is calling a function that XLA can't compile.
    return errors::InvalidArgument("Not compilable: ", ndef.ShortDebugString());
  }
  FunctionLibraryRuntime::Handle handle;
  // If ndef is not instantiable, e.g., the function does not exist,
  // simply bail out.
  TF_RETURN_IF_ERROR(
      flr->Instantiate(ndef.op(), AttrSlice(&ndef.attr()), &handle));
  const FunctionBody* fbody = flr->GetFunctionBody(handle);
  CHECK(fbody);  // Can't be nullptr since we just instantiated it.
  std::vector<bool> const_args(fbody->arg_types.size());
  // If we can't analyze the const args. Bail out.
  TF_RETURN_IF_ERROR(BackwardsConstAnalysis(*(fbody->graph), &const_args));

  for (int i = 0; i < const_args.size(); ++i) {
    if (const_args[i]) {
      // There is a const arg. Bail out.
      return errors::InvalidArgument("Const arg: ", i, " in ",
                                     DebugString(fbody->fdef));
    }
  }

  NodeDef launch_def;
  launch_def.set_name(ndef.name());
  launch_def.set_op("_XlaLaunch");
  launch_def.set_device(flr->device()->name());
  AddNodeAttr("Tconstants", DataTypeVector{}, &launch_def);
  AddNodeAttr("Nresources", 0, &launch_def);
  AddNodeAttr("Targs", fbody->arg_types, &launch_def);
  AddNodeAttr("Tresults", fbody->ret_types, &launch_def);
  NameAttrList func;
  func.set_name(ndef.op());
  *(func.mutable_attr()) = ndef.attr();
  AddNodeAttr("function", func, &launch_def);

  // TODO(b/32387911): Handles the host memory types across function
  // calls properly. For now, we assume all inputs and outputs are on
  // the device memory.
  MemoryTypeVector input_memory_types(fbody->arg_types.size(), DEVICE_MEMORY);
  MemoryTypeVector output_memory_types(fbody->ret_types.size(), DEVICE_MEMORY);

  Device* dev = flr->device();
  Status s;
  OpKernelConstruction construction(
      DeviceType(dev->device_type()), dev,
      dev->GetAllocator(AllocatorAttributes()), &launch_def,
      &fbody->fdef.signature(), flr, fbody->arg_types, input_memory_types,
      fbody->ret_types, output_memory_types, flr->graph_def_version(), &s);
  kernel->reset(new XlaLocalLaunchOp(&construction));
  return s;
}

bool RegisterLaunchOpCreator() {
  RegisterDefaultCustomKernelCreator(CreateXlaLaunchOp);
  return true;
}

static bool register_me = RegisterLaunchOpCreator();

}  // end namespace
}  // namespace tensorflow
