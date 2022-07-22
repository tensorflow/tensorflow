/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/kernels/tpu_reshard_variables_op_util.h"

#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/jit/xla_tensor.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_lookup.h"
#include "tensorflow/core/tpu/kernels/tpu_op_consts.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group.h"
#include "tensorflow/core/tpu/tpu_configuration.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/tpu/tpu_execute.h"
#include "tensorflow/core/util/stream_executor_util.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/tpu/tpu_executor.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_interface.h"
#include "tensorflow/stream_executor/tpu/tpu_node_context.h"

namespace tensorflow {
namespace tpu {
namespace reshard_variables {

Status FlushProgramMemory(se::Platform* platform, int device_ordinal) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<tpu::TpuNodeContext> node_interfaces,
                      tpu::TpuNodeContext::Create(device_ordinal));

  auto* executor = tensorflow::down_cast<tpu::TpuExecutorInterface*>(
      node_interfaces->stream_executor()->implementation());
  return executor->UnloadAllPrograms();
}

Status CheckIsValidKey(const Tensor& key) {
  if (!TensorShapeUtils::IsVector(key.shape()) ||
      key.shape().dim_size(0) != 3) {
    return errors::InvalidArgument(
        "new_format_key argument to TPUReshardVariables  must be a 3-element "
        "vector");
  }
  if (key.dtype() != DT_STRING) {
    return errors::InvalidArgument(
        "new_format_key argument to TPUReshardVariables must be DT_STRING "
        "type");
  }
  return OkStatus();
}

bool IsDefaultKey(const Tensor& key) { return key.vec<tstring>()(0).empty(); }

// Looks up the input `key` in the compilation cache, populating
// `*rendezvous_key_base` and `*entry`.
Status GetComputationCacheEntry(
    const Tensor& key, string* rendezvous_key_base,
    std::unique_ptr<tpu::CompilationCacheEntryRef>* entry,
    tpu::CompilationCacheFetchTarget fetch_target) {
  profiler::TraceMe trace_me("TPUReshardVariablesOpKernel::LookupProto",
                             /*level=*/2);
  TF_RETURN_IF_ERROR(CheckIsValidKey(key));
  auto* rmgr = GetTPUConfigResourceMgr();
  tpu::TpuCompilationCacheLookup* proto_lookup;
  TF_RETURN_IF_ERROR(rmgr->Lookup(rmgr->default_container(),
                                  tpu::kCompiledProtoCacheResourceName,
                                  &proto_lookup));
  core::ScopedUnref lookup_unref(proto_lookup);
  TF_RETURN_IF_ERROR(
      proto_lookup->Lookup(key.vec<tstring>()(0), entry, fetch_target));
  *rendezvous_key_base = key.vec<tstring>()(1);
  return OkStatus();
}

// Builds an InputBuffers object that describes the inputs to the computation.
xla::StatusOr<xla::ShapeTree<xla::MaybeOwningDeviceMemory>> BuildInputBuffers(
    OpKernelContext* context, const std::vector<VariableInfo>& variables,
    const xla::Shape& input_host_shape, xla::Backend* backend,
    int device_ordinal, se::Stream* stream) {
  profiler::TraceMe trace_me("BuildComputationInputs", /*level=*/2);
  OpInputList var_list;
  TF_RETURN_IF_ERROR(context->input_list("vars", &var_list));

  if (var_list.size() != xla::ShapeUtil::TupleElementCount(input_host_shape)) {
    return errors::InvalidArgument(
        "Number of variables (", var_list.size(),
        ") does not match input shape: ",
        xla::ShapeUtil::TupleElementCount(input_host_shape));
  }

  auto validate_shape = [&](int i, const Tensor& tensor) {
    const xla::Shape& expected =
        xla::ShapeUtil::GetTupleElementShape(input_host_shape, i);
    VLOG(4) << "Input " << i << " TF shape " << tensor.shape().DebugString();
    XlaTensor* xla_tensor = XlaTensor::FromTensor(&tensor);

    if (xla_tensor == nullptr) {
      // FromTensor failed; tensor must be empty.
      if (!xla::ShapeUtil::IsZeroElementArray(expected)) {
        return errors::InvalidArgument(
            "Run-time shape mismatch for TPUExecute argument[", i, "] (",
            context->op_kernel().requested_input(i), "). Expected ",
            expected.DebugString(),
            "; got empty tensor. If you are running "
            "with TF2 TPU, make sure you set `drop_remainder=False` when "
            "calling `dataset.batch` on the `tf.data.Dataset` so dynamic batch "
            "size can be handled");
      }
    } else {
      const xla::Shape& xla_shape = xla_tensor->shaped_buffer().on_host_shape();
      if (!xla::ShapeUtil::Compatible(expected, xla_shape)) {
        return errors::InvalidArgument(
            "Run-time shape mismatch for TPUReshardVariables argument[", i,
            "] (", context->op_kernel().requested_input(i), "). Expected ",
            expected.DebugString(), "; got ", xla_shape.DebugString());
      }
    }

    return OkStatus();
  };

  for (int i = 0; i < variables.size(); ++i) {
    TF_RETURN_IF_ERROR(
        validate_shape(variables[i].index(), *variables[i].var()->tensor()));
  }

  se::DeviceMemoryAllocator* const allocator = backend->memory_allocator();
  xla::TransferManager* const transfer_manager = backend->transfer_manager();

  xla::ShapeTree<xla::MaybeOwningDeviceMemory> input_buffers(
      transfer_manager->HostShapeToDeviceShape(input_host_shape));

  // Allocates a buffer for the root tuple.
  const int64_t root_size =
      transfer_manager->GetByteSizeRequirement(input_buffers.shape());
  TF_ASSIGN_OR_RETURN(*input_buffers.mutable_element({}),
                      allocator->Allocate(device_ordinal, root_size));

  auto set_input_buffers_helper = [&](int arg_index, xla::ShapedBuffer* buffers,
                                      bool owning = false) {
    buffers->buffers().ForEachMutableElement(
        [&](const xla::ShapeIndex& index, se::DeviceMemoryBase* buffer) {
          xla::ShapeIndex in_index = {arg_index};
          for (int64_t j : index) {
            in_index.push_back(j);
          }
          if (owning) {
            *input_buffers.mutable_element(in_index) =
                se::OwningDeviceMemory(*buffer, device_ordinal, allocator);
            *buffer = se::DeviceMemoryBase();
          } else {
            *input_buffers.mutable_element(in_index) = *buffer;
          }
        });
  };

  // Assigns the buffers of 'tensor' as computation input 'i'. Allocates fresh
  // buffers for zero-element tensors where required.
  auto assign_input = [&](int i, const Tensor& tensor) -> xla::Status {
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
      set_input_buffers_helper(/*arg_index=*/i, &buffers);
    } else {
      set_input_buffers_helper(/*arg_index=*/i, &xla_tensor->shaped_buffer(),
                               tensor.RefCountIsOne());
      xla_tensor->WaitForDefinitionEventOnStream(stream);
    }
    return OkStatus();
  };

  for (int i = 0; i < var_list.size(); ++i) {
    TF_RET_CHECK(var_list[i].dtype() == DT_RESOURCE);
    TF_RETURN_IF_ERROR(assign_input(i, *variables[i].var()->tensor()));
  }

  return std::move(input_buffers);
}

// Perform a compaction to reduce fragmentation.
Status PerformCompaction(stream_executor::Stream* stream) {
  profiler::TraceMe trace_me("PerformCompaction", /*level=*/2);
  auto* ds_executor =
      down_cast<tpu::TpuExecutorInterface*>(stream->parent()->implementation());
  TF_RETURN_IF_ERROR(ds_executor->EnqueueCompactionOnStreamForHbm(stream));
  // LoadProgram and GetOrCreateConstantHandle are not managed by stream
  // dependencies but they write to shared memory, so we need to block here to
  // prevent those operations from racing.
  return stream->BlockHostUntilDone();
}

// Updates the variables to the execution result's buffers, and deallocates the
// root tuple buffer.
Status UpdateOutputVariables(
    OpKernelContext* context, xla::ScopedShapedBuffer result_buffers,
    absl::Span<const TensorShapeProto* const> output_tensor_shape_protos,
    xla::Backend* backend, se::Stream* stream, int device_ordinal,
    const std::vector<VariableInfo>& variables,
    const std::shared_ptr<se::Event>& definition_event) {
  profiler::TraceMe trace_me("UpdateOutputVariables", /*level=*/2);
  // Shapes of the outputs, in TensorShape form.
  const int64_t sub_elements =
      xla::ShapeUtil::TupleElementCount(result_buffers.on_host_shape());
  if (sub_elements != output_tensor_shape_protos.size()) {
    return errors::InvalidArgument(
        "Mismatched numbers of output shapes: ", sub_elements, " vs. ",
        output_tensor_shape_protos.size());
  }

  if (sub_elements != variables.size()) {
    return errors::InvalidArgument(
        "Output count does not equal varaible count: ", sub_elements, " vs. ",
        variables.size());
  }

  std::vector<TensorShape> output_tensor_shapes;
  output_tensor_shapes.reserve(sub_elements);
  for (int64_t i = 0; i < sub_elements; ++i) {
    TF_RETURN_IF_ERROR(
        TensorShape::IsValidShape(*output_tensor_shape_protos[i]));
    TensorShape shape(*output_tensor_shape_protos[i]);
    const xla::Shape& xla_shape =
        xla::ShapeUtil::GetSubshape(result_buffers.on_host_shape(), {i});
    if (!xla_shape.IsArray() ||
        xla::ShapeUtil::ElementsIn(xla_shape) != shape.num_elements()) {
      return errors::InvalidArgument(
          "Mismatched number of elements in output shape: ",
          xla::ShapeUtil::HumanString(xla_shape), " vs ", shape.DebugString());
    }
    output_tensor_shapes.push_back(shape);
    VLOG(2) << "Output " << i << " shape " << shape.DebugString();
  }

  // Build a shaped buffer for the outputs.
  TF_RET_CHECK(result_buffers.on_host_shape().IsTuple());
  TF_RET_CHECK(!xla::ShapeUtil::IsNestedTuple(result_buffers.on_host_shape()));

  se::DeviceMemoryAllocator* const allocator = backend->memory_allocator();

  auto output_buffers = result_buffers.release();
  const xla::Shape& output_host_shape = output_buffers.on_host_shape();
  const xla::Shape& output_device_shape = output_buffers.on_device_shape();

  // Transfers ownership of the buffers that back XLA computation output 'i'
  // to 'output_tensor'.
  auto transfer_buffers = [&](int i, Tensor* output_tensor) {
    const xla::Shape& host_shape =
        xla::ShapeUtil::GetTupleElementShape(output_host_shape, i);
    const xla::Shape& device_shape =
        xla::ShapeUtil::GetTupleElementShape(output_device_shape, i);
    if (output_tensor->NumElements() > 0) {
      xla::ScopedShapedBuffer shaped_buffer(host_shape, device_shape, allocator,
                                            device_ordinal);
      shaped_buffer.buffers().ForEachMutableElement(
          [&](const xla::ShapeIndex& index, se::DeviceMemoryBase* buffer) {
            xla::ShapeIndex out_index = {i};
            for (int64_t j : index) {
              out_index.push_back(j);
            }
            *buffer = output_buffers.buffers().element(out_index);
          });

      XlaTensor* xla_tensor = XlaTensor::FromTensor(output_tensor);
      xla_tensor->set_shaped_buffer(std::move(shaped_buffer));
      xla_tensor->ResetDefinitionEvent(definition_event, stream);
    }
  };

  for (int i = 0; i < variables.size(); ++i) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        variables[i].var()->tensor()->dtype(), output_tensor_shapes[i],
        variables[i].var()->tensor()));
    transfer_buffers(i, variables[i].var()->tensor());
  }
  return allocator->Deallocate(output_buffers.device_ordinal(),
                               output_buffers.buffer({}));
}

}  // namespace reshard_variables
}  // namespace tpu
}  // namespace tensorflow
