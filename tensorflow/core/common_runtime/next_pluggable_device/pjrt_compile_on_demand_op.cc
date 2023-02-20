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
#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device_api.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/utils.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/common/async_value_tensor.h"
#include "tensorflow/core/tfrt/common/create_pjrt_client_util.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {

static StatusOr<xla::Shape> TpuShapeRepresentation(
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

static XlaCompiler::Options GenerateXlaCompilerOptions(
    const FunctionLibraryRuntime& function_library,
    const DeviceBase* device_base) {
  XlaCompiler::Options options;
  options.device_ordinal = GetDeviceOrdinal(device_base);
  options.flib_def = function_library.GetFunctionLibraryDefinition();
  options.graph_def_version = function_library.graph_def_version();
  // TODO(b/260799193): currently device_type and shape_determination_fns are
  // hardcoded for TPU. Support generating compiler options for different
  // devices. We may introduce a plugin c API to provide options.device_type.
  options.device_type = DEVICE_TPU_XLA_JIT;
  options.shape_determination_fns =
      XlaShapeLayoutHelpers::ShapeDeterminationFns{UseNoPreferenceLayoutFn(),
                                                   TpuShapeRepresentation};
  options.allow_cpu_custom_calls = false;
  options.alias_passthrough_params = false;
  options.detailed_logging = false;
  return options;
}

static std::vector<xla::PjRtBuffer*> PrepareExecutableArguments(
    int xla_input_sizes, const std::vector<int>& input_mapping,
    const std::vector<const Tensor*>& inputs,
    const std::vector<VariableInfo>& variables,
    const absl::flat_hash_map<int, int>& variable_lookup) {
  std::vector<xla::PjRtBuffer*> args;
  args.reserve(xla_input_sizes);
  for (auto arg_num : input_mapping) {
    const Tensor* tensor;
    if (auto it = variable_lookup.find(arg_num); it != variable_lookup.end()) {
      tensor = variables[it->second].var()->tensor();
    } else {
      tensor = inputs[arg_num];
    }
    AsyncValueTensor* av_tensor = AsyncValueTensor::FromTensor(tensor);
    if (av_tensor->GetBuffer() == nullptr) {
      // TODO(b/260799971): verify size 0 argument is supported (cl/387160525).
      CHECK_EQ(tensor->NumElements(), 0);  // Crash OK
      continue;
    }
    args.push_back(av_tensor->GetBuffer().get());
  }
  return args;
}

static Status PopulateOutputs(
    OpKernelContext* ctx, const std::vector<const Tensor*>& inputs,
    const std::vector<VariableInfo>& variables,
    const absl::flat_hash_map<int, int>& variable_lookup,
    const XlaCompiler::CompilationResult& compilation_result,
    std::vector<std::unique_ptr<xla::PjRtBuffer>>& execute_outputs) {
  // Copy XLA results to the OpOutputList.
  int output_num = 0;
  for (int i = 0, end = ctx->num_outputs(); i < end; ++i) {
    const DataType& type = compilation_result.outputs[i].type;
    VLOG(2) << "Populating output for retval " << i << " type "
            << DataTypeString(type);

    if (compilation_result.outputs[i].is_constant) {
      Device* device = dynamic_cast<Device*>(ctx->device());
      bool requires_copy_to_device = device->device_type() != DEVICE_CPU;
      TF_RETURN_IF_ERROR(SetOutputForConstant(ctx, requires_copy_to_device,
                                              &compilation_result, i));
    } else if (type == DT_RESOURCE) {
      int input_index = compilation_result.outputs[i].input_index;
      TF_RET_CHECK(input_index >= 0 && input_index < ctx->num_inputs())
          << "Invalid input for outputs " << i << ": " << input_index;
      ctx->set_output(i, *inputs[input_index]);
    } else {
      Tensor* output_tensor;
      TensorShape shape = TensorShape(
          execute_outputs[output_num]->on_device_shape().dimensions());
      TF_RETURN_IF_ERROR(ctx->allocate_output(i, shape, &output_tensor));
      auto output_avt = AsyncValueTensor::FromTensor(output_tensor);
      output_avt->SetBuffer(std::move(execute_outputs[output_num]));
      ++output_num;
    }
  }

  // Apply variable updates, if any.
  for (int i = 0, end = compilation_result.resource_updates.size(); i < end;
       ++i) {
    const XlaCompiler::ResourceUpdate& write =
        compilation_result.resource_updates[i];
    int actual_input_index = write.input_index;
    CHECK_GE(actual_input_index, 0);                  // Crash OK
    CHECK_LT(actual_input_index, ctx->num_inputs());  // Crash OK
    auto it = variable_lookup.find(actual_input_index);
    if (it == variable_lookup.end()) {
      continue;
    }
    Var* var = variables[it->second].var();
    CHECK(var);  // Crash OK

    VLOG(2) << "Updating variable #" << i
            << " at input index: " << actual_input_index << " with shape "
            << write.shape.DebugString() << "; variable tensor has shape: "
            << var->tensor()->shape().DebugString();

    if (var->is_initialized && var->tensor()->dtype() != write.type) {
      return errors::Internal("Mismatched type in variable write");
    }

    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        var->tensor()->dtype(), var->tensor()->shape(), var->tensor()));
    AsyncValueTensor::FromTensor(var->tensor())
        ->SetBuffer(std::move(execute_outputs[output_num]));
    var->is_initialized |= write.modified;
    ++output_num;
  }
  return OkStatus();
}

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
  TF_ASSIGN_OR_RETURN(
      xla::PjRtDevice * device,
      pjrt_client->LookupAddressableDevice(GetDeviceOrdinal(ctx->device())));

  absl::flat_hash_map<int, int> variable_lookup;
  for (int i = 0; i < variables.size(); i++) {
    variable_lookup[variables[i].index()] = i;
  }
  const std::vector<xla::PjRtBuffer*> executable_args =
      PrepareExecutableArguments(compilation_result.xla_input_shapes.size(),
                                 compilation_result.input_mapping, inputs,
                                 variables, variable_lookup);
  // TODO(b/257548614): currently PJRT is compiled as portable (num_replica = 1
  // and num_partition = 1). Support multiple partitions case.
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<xla::PjRtBuffer>> execute_outputs,
      executable->ExecutePortable(executable_args, device, options));

  TF_RETURN_IF_ERROR(PopulateOutputs(ctx, inputs, variables, variable_lookup,
                                     compilation_result, execute_outputs));
  return OkStatus();
}

void PjRtCompileOnDemandOp::Compute(OpKernelContext* ctx) {
  OP_REQUIRES_VALUE(xla::PjRtClient * pjrt_client, ctx,
                    GetOrCreatePjRtClient(DEVICE_TPU));
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
