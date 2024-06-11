/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tpu/kernels/tpu_execute_op.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/jit/variable_info.h"
#include "tensorflow/compiler/jit/variable_info_util.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_tensor.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "xla/debug_options_flags.h"
#include "xla/literal.h"
#include "xla/service/backend.h"
#include "xla/service/computation_placer.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/tpu/tpu_node_context.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_entry.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_lookup.h"
#include "tensorflow/core/tpu/kernels/tpu_executable_info.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_op_consts.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group.h"
#include "tensorflow/core/tpu/tpu_configuration.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/tpu/tpu_execute.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/platform/macros.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace {
using ::tensorflow::tpu::CompilationCacheEntryRef;
using ::tensorflow::tpu::TpuCompilationCacheLookup;
using ::tensorflow::tpu::TpuNodeContext;

// Looks up the input `key` in the compilation cache, populating
// `*rendezvous_key_base` and `*entry`.
Status GetComputationCacheEntry(
    OpKernelContext* context, std::string* rendezvous_key_base,
    std::unique_ptr<CompilationCacheEntryRef>* entry) {
  const Tensor* key;
  TF_RETURN_IF_ERROR(context->input("key", &key));
  tsl::profiler::TraceMe trace_me("TpuExecuteOp::LookupProto", /*level=*/2);
  if (!TensorShapeUtils::IsVector(key->shape()) ||
      key->shape().dim_size(0) != 3) {
    return absl::InvalidArgumentError(
        "Key argument to TPUExecute must be a 3-element vector");
  }

  ResourceMgr* rmgr = GetTPUConfigResourceMgr();
  TpuCompilationCacheLookup* proto_lookup;
  TF_RETURN_IF_ERROR(rmgr->Lookup(rmgr->default_container(),
                                  tpu::kCompiledProtoCacheResourceName,
                                  &proto_lookup));
  core::ScopedUnref lookup_unref(proto_lookup);
  TF_RETURN_IF_ERROR(proto_lookup->Lookup(key->vec<tstring>()(0), entry));
  *rendezvous_key_base = key->vec<tstring>()(1);
  return absl::OkStatus();
}

struct VariableUpdateMap {
  // Maps input index to the updated output index. If the variable doesn't have
  // an updated output, the corresponding output is set to -1.
  absl::flat_hash_map<int, int> input_to_output;
  // Maps output index to (the input index, whether the update is generated from
  // compilation).
  absl::flat_hash_map<int, std::pair<int, bool>> output_to_input;
  // Part of the input indices that are from the compilation, in the compiled
  // order.
  std::vector<int> input_in_compiled_update_order;
};

// Creates a VariableUpdateMap from both the compilation and the fused variable
// reads/updates.
absl::StatusOr<VariableUpdateMap> BuildVariableUpdateMap(
    absl::Span<const TPUExecutableInfoProto::UpdateIndexPair* const>
        compiled_variable_updates,
    absl::Span<int const> fused_device_var_reads_in_computation_inputs,
    const std::vector<int>& fused_device_var_updates_in_computation_outputs,
    int64_t computation_output_count) {
  VariableUpdateMap map;
  auto add_pair = [&](int input, int output, bool from_compilation) -> Status {
    TF_RET_CHECK(map.input_to_output.emplace(input, output).second)
        << "Duplicate variable input index: " << input;
    if (output >= 0) {
      TF_RET_CHECK(map.output_to_input
                       .emplace(output, std::make_pair(input, from_compilation))
                       .second)
          << "Duplicate variable output index: " << output;
    }
    return absl::OkStatus();
  };

  // First add the updates produced by the compilation. Not all variables are
  // updated, and if not, they do not have an output in the XLA computation. The
  // update output indices in the XLA computation start after the non-variable
  // outputs.
  int num_updated_variables = 0;
  for (int i = 0; i < compiled_variable_updates.size(); ++i) {
    const bool updated = compiled_variable_updates[i]->updated();
    if (updated) ++num_updated_variables;
  }
  TF_RET_CHECK(num_updated_variables <= computation_output_count)
      << num_updated_variables << " <= " << computation_output_count;
  int64_t compiled_variable_output_index =
      computation_output_count - num_updated_variables;
  for (auto update : compiled_variable_updates) {
    map.input_in_compiled_update_order.push_back(update->index());
    if (!update->updated()) {
      TF_RETURN_IF_ERROR(add_pair(update->index(), -1, true));
      continue;
    }
    TF_RETURN_IF_ERROR(
        add_pair(update->index(), compiled_variable_output_index, true));
    ++compiled_variable_output_index;
  }

  // Now add the updates from the attributes.
  TF_RET_CHECK(fused_device_var_reads_in_computation_inputs.size() ==
               fused_device_var_updates_in_computation_outputs.size());
  for (int64_t i = 0; i < fused_device_var_reads_in_computation_inputs.size();
       ++i) {
    TF_RETURN_IF_ERROR(
        add_pair(fused_device_var_reads_in_computation_inputs[i],
                 fused_device_var_updates_in_computation_outputs[i], false));
  }
  return map;
}

// Buffers representing the inputs to a computation.
struct InputBuffers {
  explicit InputBuffers(xla::Shape device_shape)
      : buffers(std::move(device_shape)) {}

  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;

  ~InputBuffers() = default;

  xla::ShapedBuffer ToShapedBuffer(xla::Shape host_shape,
                                   se::DeviceMemoryAllocator* allocator,
                                   int device_ordinal) {
    CHECK_NE(allocator, nullptr);
    xla::ShapedBuffer shaped_buffer(std::move(host_shape), buffers.shape(),
                                    device_ordinal);
    shaped_buffer.set_buffers(buffers.Map<se::DeviceMemoryBase>(
        [](const xla::MaybeOwningDeviceMemory& buffer) {
          return buffer.AsDeviceMemoryBase();
        }));
    return shaped_buffer;
  }

  // Describes the buffer tree.
  xla::ShapeTree<xla::MaybeOwningDeviceMemory> buffers;

  // Information about resource variables passed directly to TPUExecute.
  std::vector<VariableInfo> variables;

  // Mapping from input index to offsets in 'variables'. < 0 if the input does
  // not correspond to a variable in 'variables'.
  std::vector<int> variable_index;
};

// Builds an InputBuffers object that describes the inputs to the computation.
absl::StatusOr<std::unique_ptr<InputBuffers>> BuildComputationInputs(
    OpKernelContext* context, const xla::Shape& input_host_shape,
    const VariableUpdateMap& variable_updates, xla::Backend* backend,
    int device_ordinal, se::Stream* stream) {
  tsl::profiler::TraceMe trace_me("BuildComputationInputs", /*level=*/2);
  OpInputList arg_list;
  TF_RETURN_IF_ERROR(context->input_list("args", &arg_list));

  if (arg_list.size() != xla::ShapeUtil::TupleElementCount(input_host_shape)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Number of parameters (", arg_list.size(),
                     ") does not match input shape: ",
                     xla::ShapeUtil::TupleElementCount(input_host_shape)));
  }

  auto validate_shape = [&](int i, const Tensor& tensor) {
    const xla::Shape& expected =
        xla::ShapeUtil::GetTupleElementShape(input_host_shape, i);
    VLOG(4) << "Input " << i << " TF shape " << tensor.shape().DebugString();
    XlaTensor* xla_tensor = XlaTensor::FromTensor(&tensor);

    if (xla_tensor == nullptr) {
      // FromTensor failed; tensor must be empty.
      if (!xla::ShapeUtil::IsZeroElementArray(expected)) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Run-time shape mismatch for TPUExecute argument[", i, "] (",
            context->op_kernel().requested_input(i), "). Expected ",
            expected.DebugString(),
            "; got empty tensor. If you are running "
            "with TF2 TPU, make sure you set `drop_remainder=False` when "
            "calling `dataset.batch` on the `tf.data.Dataset` so dynamic batch "
            "size can be handled"));
      }
    } else {
      // Compare host shapes, easier than getting the expected device shape.
      const xla::Shape& xla_shape = xla_tensor->shaped_buffer().on_host_shape();
      if (!xla::ShapeUtil::Compatible(expected, xla_shape)) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Run-time shape mismatch for TPUExecute argument[", i, "] (",
            context->op_kernel().requested_input(i), "). Expected ",
            expected.DebugString(), "; got ", xla_shape.DebugString()));
      }
    }

    return absl::OkStatus();
  };

  // Iterate over the inputs, validating the shapes of non-variable inputs,
  // and creating a VariableInfo object for each variable. We consider variable
  // inputs in a separate phase because we must acquire variable locks in order.
  std::vector<VariableInfo> variables;
  std::vector<int> variable_index(arg_list.size(), -1);
  variables.reserve(arg_list.size());
  ResourceHandle handle;
  for (int i = 0; i < arg_list.size(); ++i) {
    // Arguments are assumed to be variables if they have a resource type.
    // (Non-variable resources are not supported.)
    if (context->input_dtype(i) == DT_RESOURCE) {
      variable_index[i] = variables.size();
      // TODO(phawkins): we may be looking up many variables here; it would be
      // better if we did not repeatedly acquire the resource manager's lock.
      TF_RETURN_IF_ERROR(HandleFromInput(context, i, &handle));
      Var* variable;
      TF_RETURN_IF_ERROR(LookupResource(context, handle, &variable));
      variables.push_back(VariableInfo(i, handle.name(), variable));
    } else {
      TF_RETURN_IF_ERROR(validate_shape(i, arg_list[i]));
    }
  }

  // Lock the variables, and validate their shapes. We hold the variable locks
  // for the duration of the TPU execution so we can donate the variable buffers
  // to the computation. If we copied the variable's Tensor instead, its
  // reference count would be greater than one due to the reference the Var
  // object holds, and we would never be able to reuse variable buffers.
  // TODO(phawkins): add a 'reuse_buffers' attribute to TPUExecute that allows
  // the user to elect to copy the buffers and permit concurrent access instead.
  TF_RETURN_IF_ERROR(LockVariables(absl::MakeSpan(variables)));
  for (int i = 0; i < variables.size(); ++i) {
    TF_RETURN_IF_ERROR(
        validate_shape(variables[i].index(), *variables[i].var()->tensor()));
  }

  se::DeviceMemoryAllocator* const allocator = backend->memory_allocator();
  xla::TransferManager* const transfer_manager = backend->transfer_manager();

  auto input_buffers = std::make_unique<InputBuffers>(
      transfer_manager->HostShapeToDeviceShape(input_host_shape));

  // Allocates a buffer for the root tuple.
  const int64_t root_size =
      transfer_manager->GetByteSizeRequirement(input_buffers->buffers.shape());
  TF_ASSIGN_OR_RETURN(*input_buffers->buffers.mutable_element({}),
                      allocator->Allocate(device_ordinal, root_size));

  // Helper function that sets the input buffers for 'arg_index' to 'buffers'.
  // If 'donate_buffers' is true, donates ownership of the buffers in 'buffers'
  // to the computation and overwrites the entries in 'buffers' with nulls.
  auto set_input_buffers_helper = [&](int arg_index, bool donate_buffers,
                                      xla::ShapedBuffer* buffers) {
    buffers->buffers().ForEachMutableElement([&](const xla::ShapeIndex& index,
                                                 se::DeviceMemoryBase* buffer) {
      xla::ShapeIndex in_index = {arg_index};
      for (int64_t j : index) {
        in_index.push_back(j);
      }
      auto* in_buffer = input_buffers->buffers.mutable_element(in_index);
      if (donate_buffers) {
        *in_buffer = se::OwningDeviceMemory(*buffer, device_ordinal, allocator);
        *buffer = se::DeviceMemoryBase();
      } else {
        *in_buffer = *buffer;
      }
    });
  };

  // Assigns the buffers of 'tensor' as computation input 'i'. Allocates fresh
  // buffers for zero-element tensors where required.
  auto assign_input = [&](int i, const Tensor& tensor,
                          bool may_reuse) -> absl::Status {
    XlaTensor* xla_tensor = XlaTensor::FromTensor(&tensor);

    // Size 0 tensors have no backing XlaTensor, but may still need to have
    // tuple buffers allocated.
    if (xla_tensor == nullptr) {
      CHECK_EQ(tensor.NumElements(), 0);
      const xla::Shape& host_shape =
          xla::ShapeUtil::GetSubshape(input_host_shape, {i});
      TF_ASSIGN_OR_RETURN(xla::ScopedShapedBuffer buffers,
                          transfer_manager->AllocateScopedShapedBuffer(
                              host_shape, allocator, device_ordinal));
      set_input_buffers_helper(/*arg_index=*/i, /*donate_buffers=*/true,
                               &buffers);
    } else {
      bool can_reuse_buffers = tensor.RefCountIsOne() && may_reuse;
      set_input_buffers_helper(/*arg_index=*/i,
                               /*donate_buffers=*/can_reuse_buffers,
                               &xla_tensor->shaped_buffer());
      xla_tensor->WaitForDefinitionEventOnStream(stream);
    }
    return absl::OkStatus();
  };

  for (int i = 0; i < arg_list.size(); ++i) {
    auto it = variable_updates.input_to_output.find(i);
    if (it == variable_updates.input_to_output.end()) {
      TF_RETURN_IF_ERROR(assign_input(i, arg_list[i], /*may_reuse=*/true));
      continue;
    }
    // input i is a variable
    bool updated = it->second >= 0;
    if (arg_list[i].dtype() != DT_RESOURCE) {
      TF_RETURN_IF_ERROR(assign_input(i, arg_list[i], updated));
    } else {
      int vi = variable_index[i];
      TF_RETURN_IF_ERROR(
          assign_input(i, *variables[vi].var()->tensor(), updated));
    }
  }

  input_buffers->variables = std::move(variables);
  input_buffers->variable_index = std::move(variable_index);

  return std::move(input_buffers);
}

struct OutputBuffers {
  OutputBuffers(xla::ScopedShapedBuffer b, se::DeviceMemoryAllocator* allocator)
      : owned_buffers(b.on_device_shape(), true),
        buffers(b.release()),
        memory_allocator(allocator) {}

  ~OutputBuffers() {
    buffers.buffers().ForEachElement(
        [&](const xla::ShapeIndex& index, const se::DeviceMemoryBase& buffer) {
          if (owned_buffers.element(index) && !buffer.is_null()) {
            Status status =
                memory_allocator->Deallocate(buffers.device_ordinal(), buffer);
            if (!status.ok()) {
              LOG(ERROR) << "Error deallocating buffer " << status;
            }
          }
        });
  }

  // Which of the buffers do we own?
  xla::ShapeTree<bool> owned_buffers;

  xla::ShapedBuffer buffers;

  se::DeviceMemoryAllocator* const memory_allocator;
};

// Allocates Tensors for the outputs of the computation. Ownership of most
// output buffers is passed to the output Tensors. Returns an OutputBuffer that
// owns the root buffer that should be passed to the XLA computation, as well as
// any output buffers that do not have corresponding output tensors. The latter
// may happen for zero-element tensors of type int64 or complex64 which still
// require a tuple buffer but do not have a corresponding XlaTensor.
absl::StatusOr<std::unique_ptr<OutputBuffers>> AllocateOutputTensors(
    OpKernelContext* context, xla::ScopedShapedBuffer scoped_buffers,
    absl::Span<const TensorShapeProto* const> output_tensor_shape_protos,
    const VariableUpdateMap& variable_updates, TpuNodeContext* node_context,
    se::Stream* stream, int device_ordinal, InputBuffers* input_buffers,
    const std::shared_ptr<se::Event>& definition_event) {
  VLOG(4) << "Output buffers: " << scoped_buffers.ToString();

  tsl::profiler::TraceMe trace_me("AllocateOutputTensors", /*level=*/2);
  // Shapes of the outputs, in TensorShape form.
  const int64_t sub_elements =
      xla::ShapeUtil::TupleElementCount(scoped_buffers.on_host_shape());
  if (sub_elements != output_tensor_shape_protos.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Mismatched numbers of output shapes: ", sub_elements,
                     " vs. ", output_tensor_shape_protos.size()));
  }

  xla::TransferManager* const transfer_manager =
      node_context->backend()->transfer_manager();

  std::vector<TensorShape> output_tensor_shapes;
  output_tensor_shapes.reserve(sub_elements);
  for (int64_t i = 0; i < sub_elements; ++i) {
    TF_RETURN_IF_ERROR(
        TensorShape::IsValidShape(*output_tensor_shape_protos[i]));
    TensorShape shape(*output_tensor_shape_protos[i]);
    const xla::Shape& xla_shape =
        xla::ShapeUtil::GetSubshape(scoped_buffers.on_host_shape(), {i});
    if (!xla_shape.IsArray() ||
        xla::ShapeUtil::ElementsIn(xla_shape) != shape.num_elements()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Mismatched number of elements in output shape: ",
          xla::ShapeUtil::HumanString(xla_shape), " vs ", shape.DebugString()));
    }
    output_tensor_shapes.push_back(shape);
  }

  // Builds a shaped buffer for the outputs.
  TF_RET_CHECK(scoped_buffers.on_host_shape().IsTuple());
  TF_RET_CHECK(!xla::ShapeUtil::IsNestedTuple(scoped_buffers.on_host_shape()));

  se::DeviceMemoryAllocator* const allocator =
      node_context->backend()->memory_allocator();

  auto output_buffers =
      std::make_unique<OutputBuffers>(std::move(scoped_buffers), allocator);

  xla::Shape output_device_shape = output_buffers->buffers.on_device_shape();

  if (!output_device_shape.is_static()) {
    TF_RETURN_IF_ERROR(transfer_manager->ReadDynamicShapes(
        stream, &output_buffers->buffers, &output_device_shape));
    for (int64_t i = 0; i < sub_elements; ++i) {
      const xla::Shape& subshape =
          xla::ShapeUtil::GetSubshape(output_device_shape, {i});
      TensorShape shape;
      TF_RETURN_IF_ERROR(XLAShapeToTensorShape(subshape, &shape));
      output_tensor_shapes[i] = shape;
    }
  }

  // Transfers ownership of the buffers that back XLA computation output 'i'
  // to 'output_tensor'.
  auto transfer_buffers = [&](int i, Tensor* output_tensor) {
    const xla::Shape& device_shape =
        xla::ShapeUtil::GetTupleElementShape(output_device_shape, i);

    // Transfers ownership of the output buffers to the output Tensor, if
    // there the tensor is backed by an XlaTensor. Tensors of size 0 have no
    // backing XlaTensor, so we let retain 'output_buffers' ownership of any
    // buffers in that case.
    if (output_tensor->NumElements() > 0) {
      xla::ScopedShapedBuffer shaped_buffer(device_shape, allocator,
                                            device_ordinal);
      shaped_buffer.buffers().ForEachMutableElement(
          [&](const xla::ShapeIndex& index, se::DeviceMemoryBase* buffer) {
            xla::ShapeIndex out_index = {i};
            for (int64_t j : index) {
              out_index.push_back(j);
            }
            *buffer = output_buffers->buffers.buffers().element(out_index);
            *output_buffers->owned_buffers.mutable_element(out_index) = false;
          });

      XlaTensor* xla_tensor = XlaTensor::FromTensor(output_tensor);
      xla_tensor->set_shaped_buffer(std::move(shaped_buffer));
      xla_tensor->ResetDefinitionEvent(definition_event, stream);
    }
  };

  const int num_updated_variables = variable_updates.output_to_input.size();
  TF_RET_CHECK(num_updated_variables <= output_tensor_shapes.size())
      << num_updated_variables << " <= " << output_tensor_shapes.size();

  OpInputList arg_list;
  TF_RETURN_IF_ERROR(context->input_list("args", &arg_list));

  // The TPU program outputs the updated variables including DT_RESOURCE and
  // non-DT_RESOURCE. The TPUExecuteOp needs to output all non-DT_RESOURCE
  // variables (updated or not).
  //
  //                       updated          not_updated
  //                 |------------------|------------------|
  // DT_RESOURCE     | allocate persist |    do nothing    |
  //                 |------------------|------------------|
  //                 |     allocate     | forward Op input |
  // not DT_RESOURCE |      output      |   to Op output   | Op output
  //                 |------------------|------------------|
  //                    program output

  // Allocates a fresh tensor for each updated variable. While the variable
  // inputs need come in no particular order, the variable values are
  // always added last by XlaCompiler class, in the same order as the
  // corresponding input variables.
  int op_output_index = 0;
  int compiled_update_index = 0;
  auto process_non_updated_variable = [&](int input_index) {
    const int variable_index = input_buffers->variable_index.at(input_index);
    // If a DT_RESOURCE input is not updated, nothing needs to be done
    // because there is no corresponding output. If a non-resource input
    // is not updated, forward the input to the output.
    if (variable_index < 0) {
      context->set_output(op_output_index, arg_list[input_index]);
      ++op_output_index;
    }
  };
  for (int i = 0; i < output_tensor_shapes.size(); ++i) {
    auto it = variable_updates.output_to_input.find(i);
    if (it == variable_updates.output_to_input.end()) {
      // Not a variable update.
      // Allocates a fresh tensor for each output of the operator. We always
      // allocate a new host-side tensor, but the on-device buffers that back
      // that tensor may be aliases of input buffers.
      Tensor* output_tensor;
      TF_RETURN_IF_ERROR(context->allocate_output(
          op_output_index, output_tensor_shapes[i], &output_tensor));
      transfer_buffers(i, output_tensor);
      ++op_output_index;
      continue;
    }
    const int input_index = it->second.first;
    // We must process the compiled updates in order, which includes the
    // non-updated variables, i.e., those without an XLA output.
    const bool from_compilation = it->second.second;
    while (from_compilation &&
           variable_updates
                   .input_in_compiled_update_order[compiled_update_index] !=
               input_index) {
      process_non_updated_variable(
          variable_updates
              .input_in_compiled_update_order[compiled_update_index]);
      ++compiled_update_index;
    }
    ++compiled_update_index;
    const int variable_index = input_buffers->variable_index.at(input_index);
    if (variable_index >= 0) {
      // This output corresponds to a DT_RESOURCE input to the TPUExecute
      // operator. Update the corresponding variable.
      VariableInfo& var = input_buffers->variables[variable_index];
      TF_RETURN_IF_ERROR(context->allocate_temp(var.var()->tensor()->dtype(),
                                                output_tensor_shapes[i],
                                                var.var()->tensor()));
      transfer_buffers(i, var.var()->tensor());
    } else {
      // This output corresponds to a non-resource input to the TPUExecute
      // operator. This case occurs for the distributed TPU rewrite which
      // adds variable values as inputs and outputs rather than passing the
      // variables themselves; reading and writing the variable is handled
      // outside the op.
      // TODO(phawkins): remove this case when placement of variables on TPU
      // devices is well supported and we no longer need to place "remote"
      // variables on CPU devices.
      Tensor* output_tensor;
      TF_RETURN_IF_ERROR(context->allocate_output(
          op_output_index, output_tensor_shapes[i], &output_tensor));
      ++op_output_index;
      transfer_buffers(i, output_tensor);
    }
  }

  // Process any remaining non-updated variables.
  for (; compiled_update_index <
         variable_updates.input_in_compiled_update_order.size();
       ++compiled_update_index) {
    process_non_updated_variable(
        variable_updates.input_in_compiled_update_order[compiled_update_index]);
  }
  return std::move(output_buffers);
}

}  // namespace

// TPUExecuteOp

TPUExecuteOp::TPUExecuteOp(OpKernelConstruction* context)
    : AsyncOpKernel(context, /* is_deferred = */ true) {}

AsyncOpKernel* TPUExecuteOp::AsAsync() {
  // If TPU launches are asynchronous, we can perform the launch without
  // blocking the calling thread, and so the executor may treat this kernel as
  // a regular (synchronous) OpKernel.
  return nullptr;
}

void TPUExecuteOp::Compute(OpKernelContext* context) {
  Status s = DoWork(context);
  // NOTE: We can't use `OP_REQUIRES_OK()` here because that macro includes
  // a dynamic check that we are not in an AsyncOpKernel.
  if (TF_PREDICT_FALSE(!s.ok())) {
    context->SetStatus(s);
  }
}

void TPUExecuteOp::ComputeAsync(OpKernelContext* context, DoneCallback done) {
  // If TPU launches are asynchronous, then perform the launch on this
  // thread to avoid a thread hop, which has an observable latency cost.
  OP_REQUIRES_OK_ASYNC(context, DoWork(context), done);
  done();
}

Status TPUExecuteOp::DoWork(OpKernelContext* context) {
  VLOG(1) << "Cloud TPU: TPUExecuteOp::Compute";

  const XlaDevice::Metadata* metadata;
  TF_RETURN_IF_ERROR(XlaDevice::GetMetadata(context, &metadata));
  const int device_ordinal = metadata->device_ordinal();

  // We are guaranteed that the object underlying TpuNodeContext won't be
  // deleted out from under us, while node_context is alive.
  TF_ASSIGN_OR_RETURN(std::unique_ptr<TpuNodeContext> node_context,
                      TpuNodeContext::Create(device_ordinal));

  tsl::profiler::TraceMe trace_me(
      [device_ordinal, context] {
        return tsl::profiler::TraceMeEncode(
            "TpuExecuteOp", {{"device_ordinal", device_ordinal},
                             {"id", context->step_id()},
                             {"iter_num", context->frame_iter().iter_id}});
      },
      /*level=*/2);
  tsl::profiler::TraceMe trace_me_init("TPUExecuteOp::Init", /*level=*/2);

  std::string rendezvous_key_base;
  std::unique_ptr<CompilationCacheEntryRef> entry_ref;
  TF_RETURN_IF_ERROR(
      GetComputationCacheEntry(context, &rendezvous_key_base, &entry_ref));

  // Shapes of the inputs and outputs, in xla::Shape form.
  tpu::TpuCompilationCacheEntry entry = entry_ref->get();
  const tpu::TpuProgramGroup* tpu_program_group =
      tensorflow::down_cast<const tpu::TpuProgramGroup*>(
          entry.tpu_program_group());
  CHECK_NE(tpu_program_group, nullptr);
  const int core_index = entry.core_index();
  const TPUExecutableInfoProto& executable =
      tpu_program_group->executable_info(core_index);

  xla::Backend* const backend = node_context->backend();
  xla::TransferManager* const transfer_manager = backend->transfer_manager();
  TF_RET_CHECK(context->op_device_context());
  se::Stream* stream = context->op_device_context()->stream();

  TF_RET_CHECK(executable.input_shapes_size() == 1);

  xla::Shape host_shape(executable.input_shapes(0));

  TF_ASSIGN_OR_RETURN(
      auto variable_update_map,
      BuildVariableUpdateMap(executable.variable_indices(),
                             fused_device_var_reads_in_computation_inputs_,
                             fused_device_var_updates_in_computation_outputs_,
                             executable.output_tensor_shapes().size()));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<InputBuffers> input_buffers,
      BuildComputationInputs(context, host_shape, variable_update_map, backend,
                             device_ordinal, stream));

  // Ideally this should be the host-to-device stream from XlaDeviceContext.
  // The particular anti-dependency this is avoiding (why we need a separate
  // transfer stream) is between the executable writing tuple tables and
  // TPUExecute()'s deregister_stream; if they come from the same stream pool
  // antidependencies will occur. XlaBackend has a different pool of streams
  // to the stream->GetOrCreateSubStream() that TPUExecute() uses, so these
  // will never refer to the same stream.
  //
  // TODO(jmolloy): Add the necessary plumbing to obtain the proper
  // host-to-device stream here.
  TF_ASSIGN_OR_RETURN(auto transfer_stream_ptr,
                      backend->BorrowStream(device_ordinal));

  se::DeviceMemoryAllocator* const allocator = backend->memory_allocator();
  auto shaped_buffer = input_buffers->ToShapedBuffer(std::move(host_shape),
                                                     allocator, device_ordinal);
  if (transfer_manager->CanShapedBufferBeAccessedNow(stream->parent(),
                                                     shaped_buffer)) {
    TF_RETURN_IF_ERROR(transfer_manager->WriteRootTupleIndexTable(
        transfer_stream_ptr.get(), shaped_buffer));
    TF_RETURN_IF_ERROR(stream->WaitFor(transfer_stream_ptr.get()));
  } else {
    TF_RETURN_IF_ERROR(
        transfer_manager->WriteRootTupleIndexTable(stream, shaped_buffer));
  }
  VLOG(4) << "Input buffers: " << shaped_buffer.ToString();

  // Snapshot the inputs, if a snapshot was requested.
  std::shared_ptr<xla::HloSnapshot> hlo_snapshot;
  if (executable.has_session_module()) {
    hlo_snapshot =
        std::make_shared<xla::HloSnapshot>(executable.session_module());
    auto literal =
        std::make_shared<xla::Literal>(shaped_buffer.on_host_shape());
    transfer_manager->TransferLiteralFromDevice(
        stream, shaped_buffer, literal.get(),
        [hlo_snapshot, literal](Status status) {
          if (!status.ok()) {
            LOG(ERROR) << "TransferLiteralFromDevice for HLO snapshot inputs "
                          "failed: "
                       << status;
            return;
          }
          *hlo_snapshot->add_arguments() = literal->ToProto();
        });
  }

  TF_ASSIGN_OR_RETURN(std::shared_ptr<se::Event> definition_event,
                      stream->parent()->CreateEvent());

  trace_me_init.Stop();

  const uint32 rng_seed = GetXLARandomSeed();

  std::unique_ptr<xla::DeviceAssignment> device_assignment;
  if (executable.has_device_assignment()) {
    TF_ASSIGN_OR_RETURN(device_assignment, xla::DeviceAssignment::Deserialize(
                                               executable.device_assignment()));
  }

  VLOG(4) << "Input buffers after alias resolution: "
          << shaped_buffer.ToString();

  std::vector<xla::ExecutionInput> input;
  input.emplace_back(xla::ExecutionInput(std::move(input_buffers->buffers),
                                         shaped_buffer.on_host_shape()));

  // The buffers to be freed are in the `output` and will be automatically
  // freed when it goes out of the scope. In async mode, this means the buffers
  // will be freed before anyone calls "BlockHostUntilDone", which indicates
  // that some of the (input) buffers will be freed while the program is running
  // and looks scary. However, this turns out to be not a problem since although
  // we free a memory and reassign it to other users while a program is running,
  // all subsequent writes to the program that could possibly clobber the memory
  // will depend on the program to finish.
  const TPUHostTransferInfoProto& host_transfer_info =
      tpu_program_group->host_transfer_info(core_index);
  TF_ASSIGN_OR_RETURN(
      xla::ExecutionOutput output,
      TPUExecute(executable, host_transfer_info,
                 *tpu_program_group->hlo_metadata(core_index), std::move(input),
                 rendezvous_key_base, rng_seed, node_context.get(),
                 device_assignment.get(), context->cancellation_manager(),
                 context, stream, transfer_stream_ptr.get(),
                 tpu_program_group->tpu_program(core_index)));
  TF_RETURN_IF_ERROR(stream->RecordEvent(definition_event.get()));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<OutputBuffers> output_buffers,
      AllocateOutputTensors(
          context, output.ConsumeResult(), executable.output_tensor_shapes(),
          variable_update_map, node_context.get(), stream, device_ordinal,
          input_buffers.get(), definition_event));

  // Transfer the outputs and save the snapshot to disk.
  if (hlo_snapshot) {
    auto literal =
        std::make_shared<xla::Literal>(output_buffers->buffers.on_host_shape());
    transfer_manager->TransferLiteralFromDevice(
        stream, output_buffers->buffers, literal.get(),
        [hlo_snapshot, literal](Status status) {
          if (status.ok()) {
            *hlo_snapshot->mutable_result() = literal->ToProto();
          } else {
            LOG(ERROR) << "TransferLiteralFromDevice for HLO snapshot "
                          "outputs failed: "
                       << status;
          }
          DumpHloSnapshotIfEnabled(*hlo_snapshot,
                                   xla::GetDebugOptionsFromFlags());
        });
  }
  return absl::OkStatus();
}

TPUExecuteOp::~TPUExecuteOp() = default;

TPUExecuteAndUpdateVariablesOp::TPUExecuteAndUpdateVariablesOp(
    OpKernelConstruction* context)
    : TPUExecuteOp(context) {
  OP_REQUIRES_OK(context, context->GetAttr(
                              "device_var_reads_indices",
                              &fused_device_var_reads_in_computation_inputs_));
  OP_REQUIRES_OK(
      context,
      context->GetAttr("device_var_updates_indices",
                       &fused_device_var_updates_in_computation_outputs_));
}

REGISTER_KERNEL_BUILDER(
    Name("TPUExecute").Device(DEVICE_TPU_NODE).HostMemory("key"), TPUExecuteOp);

REGISTER_KERNEL_BUILDER(Name("TPUExecuteAndUpdateVariables")
                            .Device(DEVICE_TPU_NODE)
                            .HostMemory("key"),
                        TPUExecuteAndUpdateVariablesOp);

}  // namespace tensorflow
