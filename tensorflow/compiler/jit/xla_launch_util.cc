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

#include "tensorflow/compiler/jit/xla_launch_util.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/jit/pjrt_tensor_buffer.h"
#include "tensorflow/compiler/jit/pjrt_tensor_buffer_util.h"
#include "tensorflow/compiler/jit/variable_info.h"
#include "tensorflow/compiler/jit/variable_info_util.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/tf2xla/xla_resource.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_execute_options.h"
#include "xla/runtime/device_id.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/tsl/framework/device_id_utils.h"
#include "xla/tsl/framework/serving_device_selector_policies.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tensorflow/core/common_runtime/composite_device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/gpu/gpu_serving_device_selector.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/common/async_value_tensor.h"

namespace tensorflow {
namespace {

absl::flat_hash_map<int, int> CreateVariableLookup(
    const std::vector<VariableInfo>& variables) {
  absl::flat_hash_map<int, int> variable_lookup;
  for (int i = 0; i < variables.size(); i++) {
    variable_lookup[variables[i].index()] = i;
  }
  return variable_lookup;
}

}  // anonymous namespace

std::vector<const Tensor*> InputsFromContext(OpKernelContext* ctx) {
  std::vector<const Tensor*> inputs;
  inputs.reserve(ctx->num_inputs());
  for (int input_idx = 0; input_idx < ctx->num_inputs(); input_idx++) {
    inputs.push_back(&ctx->input(input_idx));
  }
  return inputs;
}

absl::StatusOr<std::vector<int>> GetConstantInputIndicesFromContext(
    OpKernelContext* ctx) {
  std::vector<int> constant_input_indices;
  TF_RETURN_IF_ERROR(GetCompileTimeConstInputs(
      &ctx->op_kernel(), &constant_input_indices, ctx->function_library()));
  if (!absl::c_all_of(constant_input_indices, [&](int idx) {
        return ctx->input_memory_type(idx) == HOST_MEMORY;
      })) {
    return absl::InternalError(
        "Unexpected device placement for a constant input");
  }
  return constant_input_indices;
}

// Sets output `output_num` for `ctx` provided it is known at a compile time.
absl::Status SetOutputForConstant(
    OpKernelContext* ctx, bool requires_copy_to_device,
    const XlaCompiler::CompilationResult* compilation_result, int output_num) {
  CHECK(compilation_result->outputs[output_num].is_constant);
  const Tensor& const_tensor =
      compilation_result->outputs[output_num].constant_value;
  Tensor* output_tensor;
  if (requires_copy_to_device && const_tensor.TotalBytes() > 0) {
    // Copy host -> device. (Empty tensors don't have backing buffers.)
    // Manually allocate memory so we can allocate as much memory as the device
    // requires (as given by GetByteSizeRequirement). This avoids
    // XlaTransferManager having to reallocate the device buffer later if
    // XlaTransferManager is used.
    VLOG(1) << "Constant output tensor on device";

    TF_RETURN_IF_ERROR(
        ctx->allocate_output(output_num, const_tensor.shape(), &output_tensor));
    Device* device = dynamic_cast<Device*>(ctx->device());
    if (device == nullptr) {
      return absl::InternalError("DeviceBase was not a Device.");
    }
    ctx->op_device_context()->CopyCPUTensorToDevice(
        &const_tensor, device, output_tensor,
        [&](absl::Status status) { CHECK_OK(status); });

    if (device->device_type() == DEVICE_GPU) {
      // The GPUDeviceContext enqueues the host->device transfer in a
      // separate stream from the main compute stream. We must ensure the
      // compute stream is synchronized with the host->device transfer
      // stream now otherwise we will create a race condition.
      auto* gpu_device_context =
          static_cast<GPUDeviceContext*>(ctx->op_device_context());
      TF_RETURN_IF_ERROR(gpu_device_context->stream()->WaitFor(
          gpu_device_context->host_to_device_stream()));
    }
  } else {
    // No copy required.
    ctx->set_output(output_num, const_tensor);
    output_tensor = ctx->mutable_output(output_num);
  }
  return absl::OkStatus();
}

static absl::StatusOr<Var*> GetOrCreateResourceVar(
    OpKernelContext* ctx, const ResourceHandle& handle,
    const XlaCompiler::ResourceUpdate& write) {
  Var* variable = nullptr;
  TF_RETURN_IF_ERROR(
      LookupOrCreateResource<Var>(ctx, handle, &variable, [&write](Var** ptr) {
        *ptr = new Var(write.type);
        return absl::OkStatus();
      }));
  return variable;
}

absl::StatusOr<std::vector<VariableInfo>> GatherVariableInfo(
    OpKernelContext* ctx,
    const XlaCompiler::CompilationResult& compilation_result,
    int missing_ctx_input_prefix) {
  std::vector<VariableInfo> out;
  out.reserve(compilation_result.resource_updates.size());
  for (int i = 0; i < compilation_result.resource_updates.size(); ++i) {
    const XlaCompiler::ResourceUpdate& write =
        compilation_result.resource_updates[i];
    int actual_input_index = write.input_index - missing_ctx_input_prefix;
    if (actual_input_index < 0 || actual_input_index >= ctx->num_inputs()) {
      return xla::Internal("Invalid input index for variable write.");
    }

    const ResourceHandle handle = HandleFromInput(ctx, actual_input_index);
    TF_ASSIGN_OR_RETURN(Var * variable,
                        GetOrCreateResourceVar(ctx, handle, write));
    out.emplace_back(actual_input_index, handle.name(), variable,
                     handle.definition_stack_trace());
  }
  return std::move(out);
}

absl::StatusOr<std::vector<XlaCompiler::Argument>>
XlaComputationLaunchContext::BuildXlaCompilerArguments(
    absl::Span<int const> must_be_constant_idxs,
    absl::Span<const Tensor* const> inputs,
    absl::Span<VariableInfo const> variable_args, Device* device) {
  if (!must_be_constant_idxs.empty() &&
      !absl::c_is_sorted(must_be_constant_idxs)) {
    return absl::InvalidArgumentError("must_be_constant_idxs is not sorted");
  }
  VLOG(2) << "Must be const args: {"
          << absl::StrJoin(must_be_constant_idxs, ",") << "} out of "
          << inputs.size() << " args";
  std::vector<XlaCompiler::Argument> out;
  out.reserve(inputs.size());

  // TODO(cheshire): Avoid duplication with framework/op_kernel.h
  DeviceContext* device_context = nullptr;
  if (device != nullptr) {
    TF_RETURN_IF_ERROR(device->TryGetDeviceContext(&device_context));
    bool using_default_context = false;
    auto cleanup = absl::MakeCleanup([&] {
      if (device_context != nullptr && !using_default_context) {
        device_context->Unref();
      }
    });
    if (device_context == nullptr) {
      using_default_context = true;
      auto* dev_info = device->tensorflow_accelerator_device_info();
      if (dev_info) device_context = dev_info->default_context;
    }
  }

  absl::flat_hash_map<int, const VariableInfo*> variable_info_lookup;
  CHECK_OK(CreateVariableInfoLookup(variable_args, variable_info_lookup));
  for (int64_t input_num = 0; input_num < inputs.size(); ++input_num) {
    const Tensor* input = inputs[input_num];
    XlaCompiler::Argument& arg = out.emplace_back();

    if (variable_info_lookup.count(input_num) && device != nullptr) {
      // Handles resource variables.
      TF_RET_CHECK(input->dtype() == DT_RESOURCE);
      const VariableInfo& variable = *variable_info_lookup[input_num];
      arg.name = std::string(variable.name());
      arg.kind = XlaCompiler::Argument::kResource;
      arg.resource_kind = XlaResource::kVariable;
      arg.definition_stack_trace = variable.definition_stack_trace();
      if (variable.var() && variable.var()->is_initialized) {
        const Tensor* value = variable.var()->tensor();
        arg.type = value->dtype();
        arg.shape = value->shape();
        arg.initialized = true;
      } else {
        // The values of uninitialized variables are not passed as inputs, since
        // they are meaningless. However, it is legal to assign to a resource
        // variable for the first time inside the XLA computation, so we do
        // permit uninitialized variables.
        arg.initialized = false;
        arg.type = DT_INVALID;
        arg.shape = TensorShape();
      }

      if (absl::c_binary_search(must_be_constant_idxs, input_num)) {
        TF_RET_CHECK(variable.var() && variable.var()->is_initialized);
        const Tensor* value = variable.var()->tensor();
        Tensor value_on_host(value->dtype(), value->shape());
        if (!device_context) {
          value_on_host = *value;
        } else {
          TF_RETURN_IF_ERROR(device_context->CopyDeviceTensorToCPUSync(
              value, "", device, &value_on_host));
        }
        arg.kind = XlaCompiler::Argument::kConstantResource;
        arg.constant_value = value_on_host;
      }
    } else if (absl::c_binary_search(must_be_constant_idxs, input_num)) {
      arg.kind = XlaCompiler::Argument::kConstant;
      arg.type = input->dtype();
      arg.shape = input->shape();
      arg.constant_value = *input;
    } else {
      // Normal inputs.
      TF_RET_CHECK(input->dtype() != DT_RESOURCE);
      if (input->NumElements() > 0) {
        arg.kind = XlaCompiler::Argument::kParameter;
      } else {
        arg.kind = XlaCompiler::Argument::kConstant;
        arg.constant_value = *input;
      }
      arg.type = input->dtype();
      arg.shape = input->shape();
    }
  }

  return out;
}

// TODO(b/289002708) Create a unit test to cover use_pjrt_tensor_buffer=true.
absl::Status PreparePjRtExecutableArguments(
    int num_missing_prefix_ctx_inputs, const std::vector<int>& input_mapping,
    const std::vector<const Tensor*>& inputs,
    const absl::flat_hash_map<int, const Tensor*>& variable_snapshots,
    xla::PjRtClient* pjrt_client, xla::PjRtDevice* pjrt_device,
    bool use_pjrt_tensor_buffer, std::vector<xla::PjRtBuffer*>* args,
    std::vector<std::unique_ptr<xla::PjRtBuffer>>* owned_args,
    absl::flat_hash_set<int>* non_donatable_input_indices) {
  for (auto arg_num : input_mapping) {
    const Tensor* tensor;
    if (auto it = variable_snapshots.find(arg_num);
        it != variable_snapshots.end()) {
      tensor = it->second;
    } else {
      tensor = inputs[arg_num - num_missing_prefix_ctx_inputs];
    }

    // The input tensor can have the following cases.
    // 1. Tensor with PjRtTensorBuffer, containing a PjRtBuffer. This case
    // occurs when the producer is a XLA kernel (e.g.XlaLocalLaunch), or if this
    // tensor is produced by host-to-device transfer via PjRtDeviceContext.
    //
    // 2. Old fashion Tensor with raw device memory pointer. This case occurs
    // when the producer is a non-XLA TF GPU kernel or function (e.g.
    // tf.matmul).
    //
    // 3. AsyncValueTensor, containing a PjRtBuffer. This is the legacy mode
    // and certain device type (e.g. TPU) still uses this path.
    AsyncValueTensor* av_tensor = AsyncValueTensor::FromTensor(tensor);
    if (use_pjrt_tensor_buffer) {
      if (av_tensor != nullptr) {
        return absl::InvalidArgumentError(
            "If use_pjrt_tensor_buffer is set, the input tensor should not "
            "contain an AsyncValueTensor.");
      }
      const PjRtTensorBuffer* pjrt_tensor_buffer =
          dynamic_cast<const PjRtTensorBuffer*>(DMAHelper::buffer(tensor));
      if (pjrt_tensor_buffer != nullptr) {
        args->push_back(pjrt_tensor_buffer->pjrt_buffer());
      } else {
        // Creates a PjRtBuffer from DeviceMemoryBase. The newly created
        // PjRtBuffer needs to be persisted till XLA execution is completed.
        xla::Shape device_shape;
        TF_RETURN_IF_ERROR(TensorShapeToXLAShape(
            tensor->dtype(), tensor->shape(), &device_shape));
        TF_ASSIGN_OR_RETURN(
            std::unique_ptr<xla::PjRtBuffer> pjrt_buffer,
            pjrt_client->CreateViewOfDeviceBuffer(
                const_cast<char*>(tensor->tensor_data().data()), device_shape,
                pjrt_device->default_memory_space().value_or(nullptr),
                [tensor = *tensor]() {}));
        owned_args->push_back(std::move(pjrt_buffer));
        args->push_back(owned_args->back().get());
      }
    } else {
      if (av_tensor->GetBuffer() == nullptr) {
        CHECK_EQ(tensor->NumElements(), 0);  // Crash OK
        continue;
      }
      args->push_back(av_tensor->GetBuffer().get());
    }

    if (!tensor->RefCountIsOne()) {
      non_donatable_input_indices->insert(args->size() - 1);
    }
  }
  return absl::OkStatus();
}

// TODO(b/289002708) Create a unit test to cover use_pjrt_tensor_buffer=true.
absl::Status PopulateCtxOutputsFromPjRtExecutableOutputs(
    int num_missing_prefix_ctx_inputs, const std::vector<const Tensor*>& inputs,
    const std::vector<VariableInfo>& variables,
    const XlaCompiler::CompilationResult& compilation_result,
    const bool use_pjrt_tensor_buffer,
    std::vector<std::unique_ptr<xla::PjRtBuffer>>& executable_outputs,
    OpKernelContext* ctx) {
  // Copy XLA results to the OpOutputList.
  int output_num = 0;
  for (int i = 0, end = ctx->num_outputs(); i < end; ++i) {
    const DataType& type = compilation_result.outputs[i].type;
    VLOG(2) << "Populating output for retval " << i << " type "
            << DataTypeString(type);
    if (type == DT_VARIANT) {
      return absl::UnimplementedError(
          "Support for TensorList crossing the XLA/TF boundary "
          "is not implemented");
    }

    if (compilation_result.outputs[i].is_constant) {
      bool requires_copy_to_device = GetDeviceType(ctx) != DEVICE_CPU;
      TF_RETURN_IF_ERROR(SetOutputForConstant(ctx, requires_copy_to_device,
                                              &compilation_result, i));
    } else if (type == DT_RESOURCE) {
      int input_index = compilation_result.outputs[i].input_index -
                        num_missing_prefix_ctx_inputs;
      TF_RET_CHECK(input_index >= 0 && input_index < ctx->num_inputs())
          << "Invalid input for outputs " << i << ": " << input_index;
      ctx->set_output(i, *inputs[input_index]);
    } else {
      xla::PjRtBuffer* output_buffer = executable_outputs[output_num].get();
      if (output_buffer->IsTuple()) {
        return absl::InvalidArgumentError(
            "Tuple PJRT buffer output is not supported.");
      }
      absl::Span<const int64_t> dims;
      std::optional<std::vector<int64_t>> logical_dims_storage;
      if (output_buffer->has_dynamic_dimensions()) {
        TF_ASSIGN_OR_RETURN(std::vector<int64_t> logical_dims,
                            output_buffer->logical_dimensions());
        logical_dims_storage.emplace(std::move(logical_dims));
        dims = *logical_dims_storage;
      } else {
        dims = output_buffer->dimensions();
      }
      TensorShape tensor_shape;
      for (int i = 0; i < dims.size(); ++i) {
        TF_RETURN_IF_ERROR(tensor_shape.AddDimWithStatus(dims[i]));
      }
      if (use_pjrt_tensor_buffer) {
        TF_ASSIGN_OR_RETURN(
            Tensor output_tensor,
            MakeTensorFromPjRtBuffer(
                type, tensor_shape, std::move(executable_outputs[output_num])));
        ctx->set_output(i, output_tensor);
      } else {
        // Uses AsyncValueTensor. This path currently used by TPU but is going
        // to be deprecated.
        Tensor* output_tensor;
        TF_RETURN_IF_ERROR(
            ctx->allocate_output(i, tensor_shape, &output_tensor));
        auto output_avt = AsyncValueTensor::FromTensor(output_tensor);
        output_avt->SetBuffer(std::move(executable_outputs[output_num]));
      }
      ++output_num;
    }
  }

  // Apply variable updates, if any.
  const auto& variable_lookup = CreateVariableLookup(variables);
  for (int i = 0; i < compilation_result.resource_updates.size(); ++i) {
    const XlaCompiler::ResourceUpdate& write =
        compilation_result.resource_updates[i];
    int actual_input_index = write.input_index - num_missing_prefix_ctx_inputs;
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
      return absl::InternalError("Mismatched type in variable write");
    }

    if (use_pjrt_tensor_buffer) {
      TF_RETURN_IF_ERROR(PjRtTensorBufferUtil::UpdateOrMakeTensorWithPjRtBuffer(
          write.type, write.shape, std::move(executable_outputs[output_num]),
          var->tensor()));
    } else {
      TF_RETURN_IF_ERROR(
          ctx->allocate_temp(write.type, write.shape, var->tensor()));
      AsyncValueTensor::FromTensor(var->tensor())
          ->SetBuffer(std::move(executable_outputs[output_num]));
    }

    var->is_initialized |= write.modified;
    ++output_num;
  }
  return absl::OkStatus();
}

xla::ExecuteOptions GetPjRtExecuteOptions(
    const DeviceType& device_type,
    absl::flat_hash_set<int> non_donatable_input_indices,
    xla::ExecuteContext* execute_context) {
  xla::ExecuteOptions options;
  // Hardcode run id to always be one: TF distributed strategy
  // differentiates between subsequent runs using dependency edges. This
  // is safe, as only TF dist-strat can produce distributed ops, and we
  // can rely on TF dist-strat invariants.
  options.launch_id = 1;
  // TODO(b/293186653): investigate we should turn on strict shape checking for
  // GPU.
  if (device_type == DEVICE_GPU) {
    options.strict_shape_checking = false;
  }
  // Note: TF does not use PJRT host callbacks as of today. Setting this option
  // to true to workaround an ExecuteOptions check: [1].
  //
  // [1]:
  // tensorflow/compiler/xla/pjrt/pjrt_c_api_client.cc;l=923-927;rcl=519286815
  options.use_major_to_minor_data_layout_for_callbacks = true;
  options.non_donatable_input_indices = std::move(non_donatable_input_indices);
  options.context = execute_context;
  return options;
}

DeviceType GetDeviceType(OpKernelContext* ctx) {
  auto* device = absl::down_cast<Device*>(ctx->device()->UnderlyingDevice());
  return DeviceType(device->device_type());
}

absl::Status RunPjRtExecutable(
    const std::vector<const Tensor*>& inputs,
    const std::vector<VariableInfo>& variables,
    const XlaCompiler::CompilationResult& compilation_result,
    xla::PjRtClient* pjrt_client, xla::PjRtLoadedExecutable* executable,
    OpKernelContext* ctx) {
  absl::flat_hash_map<int, const Tensor*> variable_snapshots;
  for (int i = 0; i < variables.size(); i++) {
    variable_snapshots[variables[i].index()] = variables[i].var()->tensor();
  }
  return RunPjRtExecutable(/*num_missing_prefix_ctx_inputs=*/0, inputs,
                           variable_snapshots, variables, compilation_result,
                           pjrt_client, executable, ctx);
}

// TODO(b/289421064): Add unit test for this.
absl::Status RunPjRtExecutable(
    int num_missing_prefix_ctx_inputs, const std::vector<const Tensor*>& inputs,
    const absl::flat_hash_map<int, const Tensor*>& variable_snapshots,
    const std::vector<VariableInfo>& updated_variables,
    const XlaCompiler::CompilationResult& compilation_result,
    xla::PjRtClient* pjrt_client, xla::PjRtLoadedExecutable* executable,
    OpKernelContext* ctx) {
  const CompositeDevice::AcceleratorDeviceInfo* accelerator_device_info =
      ctx->device()->tensorflow_accelerator_device_info();
  const DeviceType& device_type = GetDeviceType(ctx);
  const bool use_pjrt_tensor_buffer =
      device_type == DEVICE_GPU || accelerator_device_info == nullptr ||
      accelerator_device_info->use_pjrt_tensor_buffer;
  int pjrt_device_id = 0;
  if (device_type == DEVICE_CPU || device_type == DEVICE_XLA_CPU) {
    pjrt_device_id = 0;
  } else {
    pjrt_device_id =
        tsl::GetDeviceIdFromDeviceParsedName(ctx->device()->parsed_name());
  }
  TF_ASSIGN_OR_RETURN(
      xla::PjRtDevice * device,
      pjrt_client->LookupAddressableDevice(xla::LocalDeviceId(pjrt_device_id)));

  gpu::GpuServingDeviceSelectorResource* device_selector_resource = nullptr;
  if (device_type == DEVICE_GPU) {
    auto rm = ctx->resource_manager();
    TF_RETURN_IF_ERROR(rm->LookupOrCreate<
                       gpu::GpuServingDeviceSelectorResource>(
        rm->default_container(), gpu::kGpuServingDeviceSelectorResourceName,
        &device_selector_resource,
        [&](gpu::GpuServingDeviceSelectorResource** device_selector_resource) {
          *device_selector_resource = new gpu::GpuServingDeviceSelectorResource(
              pjrt_client->addressable_device_count(),
              std::make_unique<tsl::RoundRobinPolicy>());
          return absl::OkStatus();
        }));
    core::ScopedUnref device_selector_resource_ref(device_selector_resource);

    TF_ASSIGN_OR_RETURN(absl::string_view fingerprint,
                        executable->FingerprintExecutable());
    device_selector_resource->selector()->Enqueue(pjrt_device_id, fingerprint);
  }
  const Eigen::ThreadPoolDevice* intra_op_thread_pool = nullptr;
  if (ctx != nullptr && ctx->device() != nullptr &&
      ctx->device()->has_eigen_cpu_device()) {
    intra_op_thread_pool = ctx->device()->eigen_cpu_device();
  }
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<xla::PjRtBuffer>> execute_outputs,
      RunPjRtExecutable(num_missing_prefix_ctx_inputs, inputs,
                        variable_snapshots, updated_variables, device_type,
                        use_pjrt_tensor_buffer, compilation_result, device,
                        pjrt_client, executable, intra_op_thread_pool));
  if (device_selector_resource != nullptr) {
    device_selector_resource->selector()->Completed(pjrt_device_id,
                                                    /*had_error=*/false);
  }

  TF_RETURN_IF_ERROR(PopulateCtxOutputsFromPjRtExecutableOutputs(
      num_missing_prefix_ctx_inputs, inputs, updated_variables,
      compilation_result, use_pjrt_tensor_buffer, execute_outputs, ctx));
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>> RunPjRtExecutable(
    int num_missing_prefix_ctx_inputs, const std::vector<const Tensor*>& inputs,
    const absl::flat_hash_map<int, const Tensor*>& variable_snapshots,
    const std::vector<VariableInfo>& updated_variables,
    const DeviceType& device_type, bool use_pjrt_tensor_buffer,
    const XlaCompiler::CompilationResult& compilation_result,
    xla::PjRtDevice* device, xla::PjRtClient* pjrt_client,
    xla::PjRtLoadedExecutable* executable,
    const Eigen::ThreadPoolDevice* intra_op_thread_pool) {
  std::vector<xla::PjRtBuffer*> executable_args;
  executable_args.reserve(compilation_result.input_mapping.size());
  std::vector<std::unique_ptr<xla::PjRtBuffer>> owned_executable_args;
  absl::flat_hash_set<int> non_donatable_input_indices;

  TF_RETURN_IF_ERROR(PreparePjRtExecutableArguments(
      num_missing_prefix_ctx_inputs, compilation_result.input_mapping, inputs,
      variable_snapshots, pjrt_client, device, use_pjrt_tensor_buffer,
      &executable_args, &owned_executable_args, &non_donatable_input_indices));

  std::vector<std::unique_ptr<xla::PjRtBuffer>> execute_outputs;
  std::optional<tsl::Future<void>> future;

  std::unique_ptr<xla::CpuExecuteContext> cpu_execute_context;
  if (pjrt_client->platform_id() == xla::CpuId() &&
      intra_op_thread_pool != nullptr) {
    cpu_execute_context = std::make_unique<xla::CpuExecuteContext>();
    cpu_execute_context->set_intra_op_thread_pool(intra_op_thread_pool);
  }
  xla::ExecuteOptions execute_options =
      GetPjRtExecuteOptions(device_type, std::move(non_donatable_input_indices),
                            cpu_execute_context.get());

  if (executable->num_replicas() != 1 || executable->num_partitions() != 1) {
    TF_ASSIGN_OR_RETURN(execute_outputs,
                        executable->ExecuteSharded(executable_args, device,
                                                   execute_options, future));
  } else {
    TF_ASSIGN_OR_RETURN(execute_outputs,
                        executable->ExecutePortable(executable_args, device,
                                                    execute_options, future));
  }

  // We need to ensure the PjRtBuffers owned by `owned_executable_args` and
  // the `cpu_execute_context` live until execution is complete. We do this by
  // capturing them by move, so they are owned by the lambda that is executed
  // when the future returned by ExecutePortable/ExecuteSharded is ready
  // i.e. when the execution is complete.
  if ((!owned_executable_args.empty() || cpu_execute_context != nullptr) &&
      future.has_value()) {
    future->OnReady([owned_executable_args = std::move(owned_executable_args),
                     cpu_execute_context =
                         std::move(cpu_execute_context)](absl::Status s) {});
  }

  return execute_outputs;
}

}  // namespace tensorflow
