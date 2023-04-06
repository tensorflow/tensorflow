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
#include "tensorflow/core/common_runtime/next_pluggable_device/pjrt_compile_on_demand_op.h"

#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/compiler/jit/device_compiler_client.h"
#include "tensorflow/compiler/jit/variable_info.h"
#include "tensorflow/compiler/jit/variable_info_util.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/c_api_conversions.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device_api.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/utils.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/common/create_pjrt_client_util.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {

static StatusOr<xla::Shape> DeviceShapeRepresentation(
    const TensorShape& shape, DataType type, bool use_fast_memory,
    XlaLayoutPreference layout_preference) {
  xla::Shape xla_shape;
  TF_RETURN_IF_ERROR(
      tensorflow::TensorShapeToXLAShape(type, shape, &xla_shape));
  ApiConverter::StackHelper<XLA_Shape> c_xla_shape(xla_shape);
  ApiConverter::StackHelper<XLA_Shape> c_device_shape;
  TF_Status* tf_status = TF_NewStatus();
  TfnpdApi()->TFNPD_XlaShapeToDeviceShapeRepresentation(
      &c_xla_shape.value, type, use_fast_memory,
      ConvertToCXlaLayoutPreference(layout_preference), &c_device_shape.value,
      tf_status);
  const Status status = StatusFromTF_Status(tf_status);
  TF_DeleteStatus(tf_status);
  TF_RETURN_IF_ERROR(status);
  return c_device_shape.AsCpp<xla::Shape>();
}

static int GetDeviceOrdinal(const DeviceBase* device) {
  return device->parsed_name().id;
}

static DeviceType GetDeviceType(OpKernelContext* ctx) {
  auto* device =
      tensorflow::down_cast<Device*>(ctx->device()->UnderlyingDevice());
  return DeviceType(device->device_type());
}

// LINT.IfChange
static XlaCompiler::Options GenerateXlaCompilerOptions(
    const FunctionLibraryRuntime& function_library, DeviceBase* device_base) {
  XlaCompiler::Options options;
  options.device_ordinal = GetDeviceOrdinal(device_base);
  options.flib_def = function_library.GetFunctionLibraryDefinition();
  options.graph_def_version = function_library.graph_def_version();
  auto* next_pluggable_device =
      dynamic_cast<NextPluggableDevice*>(device_base->UnderlyingDevice());
  // TODO(b/267499840): support setting compilation device type and
  // shape_determination_fns for non-NextPluggableDevice case.
  // TODO(b/273348427): Set these fields in a XlaDevice::Metadata and set
  // XlaCompiler::Options using the XlaDevice::Metadata instead.
  // XlaDevice::Metadata should be moved out of XlaDevice and made more general
  // so that it could be used with other devices that support XLA compilation
  // (eg. NextPluggableDevice).
  if (next_pluggable_device != nullptr) {
    options.device_type =
        DeviceType(next_pluggable_device->GetCompilationDeviceType());
    options.shape_determination_fns =
        XlaShapeLayoutHelpers::ShapeDeterminationFns{UseNoPreferenceLayoutFn(),
                                                     DeviceShapeRepresentation};
  }
  options.allow_cpu_custom_calls = false;
  options.alias_passthrough_params = false;
  options.detailed_logging = false;
  return options;
}
// LINT.ThenChange(//tensorflow/compiler/jit/xla_compiler_options_util.cc)

Status PjRtCompileOnDemandOp::Compile(
    OpKernelContext* ctx, xla::PjRtClient* pjrt_client,
    const std::vector<XlaCompiler::Argument>& args,
    XlaCompiler::CompilationResult* compilation_result,
    std::unique_ptr<xla::PjRtLoadedExecutable>* executable) {
  // TODO(b/260798754): use caching when it is ready.
  const XlaCompiler::Options options =
      GenerateXlaCompilerOptions(*ctx->function_library(), ctx->device());
  XlaCompiler::CompileOptions compile_options;
  compile_options.is_entry_computation = true;
  compile_options.use_tuple_arg = false;
  compile_options.always_return_tuple = true;
  XlaCompiler compiler(options);
  TF_RETURN_IF_ERROR(compiler.CompileSingleOp(
      compile_options, XlaCompiler::SingleOpCompileArgument(*ctx), args,
      compilation_result));
  xla::ExecutableBuildOptions build_options =
      GetExecutableBuildOptions(options, *compilation_result, -1);
  xla::CompileOptions pjrt_compile_options;
  pjrt_compile_options.executable_build_options = build_options;
  pjrt_compile_options.compile_portable_executable = true;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtLoadedExecutable> pjrt_executable,
      pjrt_client->Compile(*compilation_result->computation,
                           pjrt_compile_options));
  VLOG(2) << "Compiled PJRT executable " << pjrt_executable->name()
          << " num_replicas " << pjrt_executable->num_replicas()
          << " num_partitions " << pjrt_executable->num_partitions();
  *executable = std::move(pjrt_executable);
  return OkStatus();
}

Status PjRtCompileOnDemandOp::Run(
    OpKernelContext* ctx, xla::PjRtClient* pjrt_client,
    const std::vector<const Tensor*>& inputs,
    const std::vector<VariableInfo>& variables,
    const XlaCompiler::CompilationResult& compilation_result,
    std::unique_ptr<xla::PjRtLoadedExecutable> executable) {
  xla::ExecuteOptions options;
  options.arguments_are_tupled = false;
  options.untuple_result = true;
  // Note: TF does not use PJRT host callbacks as of today. Setting this option
  // to true to workaround an ExecuteOptions check: [1].
  //
  // [1]:
  // tensorflow/compiler/xla/pjrt/pjrt_c_api_client.cc;l=923-927;rcl=519286815
  options.use_major_to_minor_data_layout_for_callbacks = true;
  TF_ASSIGN_OR_RETURN(
      xla::PjRtDevice * device,
      pjrt_client->LookupAddressableDevice(GetDeviceOrdinal(ctx->device())));

  const std::vector<xla::PjRtBuffer*> executable_args =
      PreparePjRtExecutableArguments(compilation_result.input_mapping, inputs,
                                     variables);
  // TODO(b/257548614): currently PJRT is compiled as portable (num_replica = 1
  // and num_partition = 1). Support multiple partitions case.
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<xla::PjRtBuffer>> execute_outputs,
      executable->ExecutePortable(executable_args, device, options));

  TF_RETURN_IF_ERROR(PopulateCtxOutputsFromPjRtExecutableOutputs(
      inputs, variables, compilation_result, execute_outputs, ctx));
  return OkStatus();
}

void PjRtCompileOnDemandOp::Compute(OpKernelContext* ctx) {
  OP_REQUIRES_VALUE(xla::PjRtClient * pjrt_client, ctx,
                    GetOrCreatePjRtClient(GetDeviceType(ctx)));
  OP_REQUIRES(ctx, ctx->function_library(),
              errors::Internal("Function library missing"));

  OP_REQUIRES_VALUE(std::vector<int> constant_input_indices, ctx,
                    GetConstantInputIndicesFromContext(ctx));
  const std::vector<int> variables_indices =
      GetResourceVariableIndicesFromContext(ctx);
  const std::vector<const Tensor*> inputs = InputsFromContext(ctx);
  std::vector<VariableInfo> variables;
  variables.reserve(variables_indices.size());
  OP_REQUIRES_OK(
      ctx, GetVariableInfosFromInputs(ctx->resource_manager(), ctx->device(),
                                      inputs, variables_indices, &variables));
  OP_REQUIRES_OK(ctx, LockVariables(absl::MakeSpan(variables)));

  // Compile
  XlaCompiler::CompilationResult result;
  std::unique_ptr<xla::PjRtLoadedExecutable> executable;
  OP_REQUIRES_VALUE(std::vector<XlaCompiler::Argument> args, ctx,
                    XlaComputationLaunchContext::BuildXlaCompilerArguments(
                        constant_input_indices, inputs, variables,
                        static_cast<Device*>(ctx->device())));
  OP_REQUIRES_OK(ctx, Compile(ctx, pjrt_client, args, &result, &executable));

  // Execute
  OP_REQUIRES_OK(ctx, Run(ctx, pjrt_client, inputs, variables, result,
                          std::move(executable)));

  ctx->SetStatus(OkStatus());
  VLOG(1) << "PjRtCompileOnDemandOp::Compute: " << ctx->op_kernel().name()
          << " on device " << ctx->device()->name() << " Done.";
}

void RegisterPjRtCompileOnDemand(const char* device, const char* jit_device) {
  // Any op assigned to the device that isn't rewritten by the graph rewriter
  // gets executed by a PjRtCompileOnDemandOp, which compiles it and executes
  // it just-in-time.
  auto factory = [](OpKernelConstruction* context) -> OpKernel* {
    return new PjRtCompileOnDemandOp(context);
  };
  XlaOpRegistry::RegisterCompilationKernels();
  static XlaDeviceOpRegistrations* registrations = RegisterXlaDeviceKernels(
      device, jit_device, factory, "PjRtCompileOnDemandOp");
  (void)registrations;
}

}  // namespace tensorflow
