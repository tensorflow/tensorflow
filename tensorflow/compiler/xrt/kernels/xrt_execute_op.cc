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

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/service/hlo_input_output_alias_config.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/compiler/xrt/xrt_compilation_cache.h"
#include "tensorflow/compiler/xrt/xrt_device.h"
#include "tensorflow/compiler/xrt/xrt_memory_manager.h"
#include "tensorflow/compiler/xrt/xrt_metrics.h"
#include "tensorflow/compiler/xrt/xrt_state.h"
#include "tensorflow/compiler/xrt/xrt_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/monitoring/timed.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace tensorflow {

namespace {

struct InputBuffers {
  std::vector<RefPtr<XRTTupleAllocation>> input_tuples;
  std::vector<xla::ShapedBuffer> input_allocations;
  std::vector<xla::ShapedBuffer*> input_pointers;
};

uint32 InitialRandomSeed() {
  // Support plumbing the TF seed through to XLA is being worked on.
  // If a user wants deterministic behavior, their best option
  // is to start with a known checkpoint. This also handles issues when
  // multiple random calls can be invoked in any order by TF executor.
  // Another option is to use stateless random ops. They have much cleaner
  // semantics.
  // If a user really wants to set a deterministic seed for XLA-based
  // devices, this is the place to do it.
  std::random_device rd;
  // Make the starting value odd.
  return rd() | 1;
}

uint32 GetXLARandomSeed() {
  // We initialize counter with an odd number and increment it by two
  // everytime. This ensures that it will never be zero, even
  // after an overflow. When seeded with zero, some XLA backends
  // can return all zeros instead of random numbers.
  static std::atomic<uint32> counter(InitialRandomSeed());
  return counter.fetch_add(2);
}

xla::StatusOr<InputBuffers> GetInputBuffers(
    XRTMemoryManager::WorkingSet* working_set, xla::Backend* backend,
    const std::vector<InputCoords>& input_coords, bool release_inputs) {
  InputBuffers input_buffers;
  input_buffers.input_tuples.reserve(input_coords.size());
  input_buffers.input_allocations.reserve(input_coords.size());
  input_buffers.input_pointers.reserve(input_coords.size());
  for (size_t i = 0; i < input_coords.size(); ++i) {
    TF_RETURN_IF_ERROR(
        working_set->LookupAndPin(backend, input_coords[i].handle));
    auto tuple = working_set->PinnedTuples().back();
    input_buffers.input_tuples.emplace_back(tuple);
    if (release_inputs) {
      // We are holding a reference to the tuple, so we can safely delete it
      // from the resource manager here.
      TF_RETURN_IF_ERROR(
          working_set->MemoryManager()->Release(input_coords[i].handle));
      VLOG(2) << "Released allocation handle " << input_coords[i].handle;
    }
    if (input_coords[i].index.empty()) {
      TF_ASSIGN_OR_RETURN(xla::ShapedBuffer shaped_buffer,
                          tuple->ToShapedBuffer());
      input_buffers.input_allocations.emplace_back(std::move(shaped_buffer));
    } else {
      TF_ASSIGN_OR_RETURN(xla::ShapedBuffer shaped_buffer,
                          tuple->ToShapedBuffer());
      TF_ASSIGN_OR_RETURN(xla::ShapedBuffer sub_shaped_buffer,
                          shaped_buffer.SubShapedBuffer(input_coords[i].index));
      input_buffers.input_allocations.emplace_back(
          std::move(sub_shaped_buffer));
    }
  }
  for (size_t i = 0; i < input_buffers.input_allocations.size(); ++i) {
    input_buffers.input_pointers.push_back(&input_buffers.input_allocations[i]);
  }
  return std::move(input_buffers);
}

xla::StatusOr<InputBuffers> GetChainedOpInputs(
    const xrt::XRTChainedExecuteOp& op,
    absl::Span<const RefPtr<XRTTupleAllocation>> op_inputs) {
  InputBuffers input_buffers;
  input_buffers.input_tuples.reserve(op.inputs_size());
  input_buffers.input_allocations.reserve(op.inputs_size());
  input_buffers.input_pointers.reserve(op.inputs_size());
  for (int i = 0; i < op.inputs_size(); ++i) {
    auto& input = op.inputs(i);
    input_buffers.input_tuples.emplace_back(op_inputs[i]);
    // Thanks to the greatness of proto3, there is no way to query for
    // explicitly set fields, so the default for output_index (zero) means no
    // sub-index. As consequence, the real index is output_index - 1.
    if (input.output_index() == 0) {
      TF_ASSIGN_OR_RETURN(xla::ShapedBuffer shaped_buffer,
                          input_buffers.input_tuples.back()->ToShapedBuffer());
      input_buffers.input_allocations.emplace_back(std::move(shaped_buffer));
    } else {
      TF_ASSIGN_OR_RETURN(xla::ShapedBuffer shaped_buffer,
                          input_buffers.input_tuples.back()->ToShapedBuffer());
      TF_ASSIGN_OR_RETURN(
          xla::ShapedBuffer sub_shaped_buffer,
          shaped_buffer.SubShapedBuffer({input.output_index() - 1}));
      input_buffers.input_allocations.emplace_back(
          std::move(sub_shaped_buffer));
    }
  }
  for (size_t i = 0; i < input_buffers.input_allocations.size(); ++i) {
    input_buffers.input_pointers.push_back(&input_buffers.input_allocations[i]);
  }
  return std::move(input_buffers);
}

// Given a shape, returns a byte array representing the shape metadata of the
// shape. The shape metadata contains dimensions sizes stored as contiguous S32.
std::vector<int32> PrepareMetadata(const xla::Shape& shape) {
  DCHECK(shape.is_static());
  DCHECK(shape.IsArray());
  // Each dimension size is stored as a S32.
  std::vector<int32> result(shape.dimensions_size());
  for (int64 i = 0; i < shape.dimensions_size(); ++i) {
    result[i] = shape.dimensions(i);
  }
  return result;
}

// Given a buffer with dynamic shape, update buffer metadata at the correct
// offset starting from that buffer.
//
// +-----------+
// |Payload    |
// +-----------+
// | Padding   |
// +-----------+
// |dim_size_0 |  (each dim_size is a S32):
// +-----------+
// |dim_size_1 |
// +-----------+
//  ..........
// +-----------+
//
// Size of payload = ByteSizeOf(runtime_shape)
// Size of payload + padding = ByteSizeOf(compile_time_shape_static)
// Size of payload + padding + metadata = ByteSizeOf(compile_time_shape)
Status UpdateMetadata(se::Stream* stream, se::DeviceMemory<uint8>* buffer,
                      const xla::Shape& compile_time_shape,
                      const xla::Shape& runtime_shape) {
  TF_ASSIGN_OR_RETURN(auto compiler, xla::Compiler::GetForPlatform(
                                         stream->parent()->platform()));
  TF_ASSIGN_OR_RETURN(
      auto transfer_manager,
      xla::TransferManager::GetForPlatform(stream->parent()->platform()));
  auto shape_size_fn = compiler->ShapeSizeBytesFunction();
  xla::Shape compile_time_shape_static =
      xla::ShapeUtil::MakeStaticShape(compile_time_shape);
  uint64 offset = shape_size_fn(compile_time_shape_static);
  uint64 metadata_size = shape_size_fn(compile_time_shape) - offset;
  auto metadata_buffer =
      stream->parent()->GetSubBuffer(buffer, offset, metadata_size);

  auto metadata_literal = std::make_shared<xla::Literal>(
      xla::LiteralUtil::CreateR1<int32>(PrepareMetadata(runtime_shape)));
  TF_RETURN_IF_ERROR(transfer_manager->TransferArrayToDeviceAsync(
      stream, *metadata_literal, metadata_buffer));
  // Retain the literal until the end of the transfer.
  stream->ThenDoHostCallback([metadata_literal]() { return Status::OK(); });
  return Status::OK();
}

// Given a static input buffer, convert it to dynamic form by expanding it to
// the bounded size and attaching a metadata filled with dimension sizes.
//
// From:
// +--------+
// |Payload |
// +--------+
//
// To:
//
// +--------+
// |Payload |
// +--------+
// | Padding|
// +--------+
// |Metadata|
// +--------+
//
// As we can't expand the size of an existing memory allocation, a reallocation
// is required. A list of new allocations are returned after this function. The
// caller is reponsible for maintaining those allocations.
xla::StatusOr<std::vector<se::OwningDeviceMemory>> UpdateDynamicInputs(
    se::Stream* stream, se::DeviceMemoryAllocator* allocator,
    std::vector<xla::ShapedBuffer*> runtime_inputs,
    const std::vector<xla::ShapeLayout>& compile_time_shapes) {
  std::vector<se::OwningDeviceMemory> new_allocations;
  TF_RET_CHECK(runtime_inputs.size() == compile_time_shapes.size());
  TF_ASSIGN_OR_RETURN(auto compiler, xla::Compiler::GetForPlatform(
                                         stream->parent()->platform()));
  auto shape_size_fn = compiler->ShapeSizeBytesFunction();
  for (int64 i = 0; i < compile_time_shapes.size(); i++) {
    const xla::Shape& compile_time_shape = compile_time_shapes[i].shape();
    if (compile_time_shape.is_static()) {
      continue;
    }
    auto* runtime_input = runtime_inputs[i];

    bool element_modified = false;
    TF_RETURN_IF_ERROR(xla::ShapeUtil::ForEachSubshapeWithStatus(
        compile_time_shape,
        [&](const xla::Shape& compile_time_shape,
            const xla::ShapeIndex& index) -> Status {
          if (compile_time_shape.IsTuple() || compile_time_shape.is_static()) {
            return Status::OK();
          }
          const xla::Shape& runtime_shape = xla::ShapeUtil::GetSubshape(
              runtime_input->on_device_shape(), index);
          TF_RET_CHECK(!runtime_shape.IsTuple());
          TF_RET_CHECK(xla::ShapeUtil::DynamicShapeIsCompatible(
              runtime_shape, compile_time_shape));
          se::DeviceMemoryBase* static_input =
              runtime_input->buffers().mutable_element(index);
          TF_ASSIGN_OR_RETURN(
              auto dynamic_input,
              allocator->Allocate(stream->parent()->device_ordinal(),
                                  shape_size_fn(compile_time_shape)));
          new_allocations.emplace_back(std::move(dynamic_input));
          se::DeviceMemory<uint8>* dynamic_input_base =
              new_allocations.back().ptr();
          // Send the original data to the new location.
          stream->ThenMemcpyD2D(dynamic_input_base, *static_input,
                                static_input->size());
          TF_RETURN_IF_ERROR(UpdateMetadata(stream, dynamic_input_base,
                                            compile_time_shape, runtime_shape));
          // Modify the memory location in the input shape tree to point to the
          // new input.
          runtime_input->set_buffer(*dynamic_input_base, index);
          element_modified = true;
          return Status::OK();
        }));
    if (element_modified) {
      runtime_input->set_shapes(compile_time_shape, compile_time_shape);
      // The input location has been modified, need to fix tuple table to
      // point to the correct address.
      TF_ASSIGN_OR_RETURN(
          auto transfer_manager,
          xla::TransferManager::GetForPlatform(stream->parent()->platform()));
      TF_RETURN_IF_ERROR(
          transfer_manager->WriteTupleIndexTablesAsync(stream, *runtime_input));
    }
  }
  return std::move(new_allocations);
}

xla::StatusOr<xla::Literal> ReadMetadataLiteral(
    se::Stream* stream, se::DeviceMemoryBase* buffer,
    const xla::Shape& buffer_shape, xla::TransferManager* transfer_manager) {
  TF_ASSIGN_OR_RETURN(auto compiler, xla::Compiler::GetForPlatform(
                                         stream->parent()->platform()));
  auto shape_size_fn = compiler->ShapeSizeBytesFunction();
  xla::Shape buffer_shape_static =
      xla::ShapeUtil::MakeStaticShape(buffer_shape);
  const int64 offset = shape_size_fn(buffer_shape_static);
  int64 metadata_size = shape_size_fn(buffer_shape) - offset;
  TF_RET_CHECK(metadata_size != 0);
  auto buffer_8 = se::DeviceMemory<uint8>(*buffer);
  auto metadata_buffer =
      stream->parent()->GetSubBuffer(&buffer_8, offset, metadata_size);
  return transfer_manager->TransferArrayFromDevice(
      stream,
      xla::ShapeUtil::MakeShape(xla::S32, {buffer_shape.dimensions_size()}),
      metadata_buffer);
}

// For each subshape in the result buffer that's dynamic, read the dynamic
// dimension sizes from the metadata, and update output shapes. The result shape
// is a static and concrete shape.
xla::Status UpdateDynamicOutputs(se::Stream* stream,
                                 xla::ShapedBuffer* shaped_buffer,
                                 xla::Shape* output_host_shape,
                                 xla::Shape* output_device_shape) {
  DCHECK(output_device_shape->is_dynamic());
  TF_ASSIGN_OR_RETURN(
      auto transfer_manager,
      xla::TransferManager::GetForPlatform(stream->parent()->platform()));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  TF_RETURN_IF_ERROR(shaped_buffer->buffers().ForEachMutableElementWithStatus(
      [&](const xla::ShapeIndex& index, se::DeviceMemoryBase* buffer) {
        const xla::Shape& buffer_shape =
            xla::ShapeUtil::GetSubshape(*output_device_shape, index);
        if (buffer_shape.IsTuple()) {
          return Status::OK();
        }
        xla::Shape& host_shape =
            *xla::ShapeUtil::GetMutableSubshape(output_host_shape, index);
        xla::Shape& device_shape =
            *xla::ShapeUtil::GetMutableSubshape(output_device_shape, index);
        if (device_shape.is_static()) {
          return Status::OK();
        }
        TF_ASSIGN_OR_RETURN(auto metadata,
                            ReadMetadataLiteral(stream, buffer, buffer_shape,
                                                transfer_manager));
        // Update shape size from metadata.
        for (int64 i = 0; i < metadata.element_count(); ++i) {
          host_shape.mutable_dimensions()[i] = metadata.Get<int32>({i});
          device_shape.mutable_dimensions()[i] = metadata.Get<int32>({i});
        }
        return Status::OK();
      }));
  output_host_shape->clear_dynamic_dimensions();
  output_device_shape->clear_dynamic_dimensions();
  return Status::OK();
}

// Create output tuple from run_result.
xla::StatusOr<RefPtr<XRTTupleAllocation>> CreateOutputTuple(
    se::Stream* stream, xla::ScopedShapedBuffer run_result,
    xla::Backend* backend, int device_ordinal) {
  XRTTupleAllocation* output_tuple;
  xla::ShapedBuffer shaped_buffer = run_result.release();
  if (shaped_buffer.on_device_shape().is_dynamic()) {
    // Update dynamic shapes from output buffer, and create a XRT tensor with
    // dimension sizes read from metadata.
    xla::Shape output_host_shape = shaped_buffer.on_host_shape();
    xla::Shape output_device_shape = shaped_buffer.on_device_shape();
    TF_RETURN_IF_ERROR(UpdateDynamicOutputs(
        stream, &shaped_buffer, &output_host_shape, &output_device_shape));
    TF_RETURN_IF_ERROR(XRTTupleAllocation::CreateFromBuffer(
        shaped_buffer, output_host_shape, output_device_shape, backend,
        device_ordinal, &output_tuple));
  } else {
    // Fast-path: Don't copy shapes of output buffer.
    TF_RETURN_IF_ERROR(XRTTupleAllocation::CreateFromBuffer(
        shaped_buffer, backend, device_ordinal, &output_tuple));
  }
  return RefPtr<XRTTupleAllocation>(output_tuple);
}

xla::StatusOr<RefPtr<XRTTupleAllocation>> RunExecutable(
    OpKernelContext* context, XRTGenericDeviceAccessor::ScopedRef* device_ref,
    xla::LocalExecutable* executable, const InputBuffers& input_buffers,
    se::Stream* stream, int rng_seed,
    const xrt::CommonExecutionConfig& config) {
  VLOG(2) << "Executing computation.";
  xla::ExecutableRunOptions run_options;
  run_options.set_stream(stream);
  run_options.set_allocator(device_ref->backend()->memory_allocator());
  run_options.set_intra_op_thread_pool(&context->eigen_cpu_device());
  run_options.set_rng_seed(rng_seed);
  if (config.run_id() != 0) {
    run_options.set_run_id(xla::RunId(config.run_id()));
  }
  if (executable->executable()
          ->module_config()
          .has_static_device_assignment()) {
    run_options.set_device_assignment(
        &executable->executable()->module_config().static_device_assignment());
  }
  xla::GpuExecutableRunOptions gpu_options;
  std::vector<xla::GlobalDeviceId> gpu_global_ids;
  if (config.local_replica_mapping_size() > 0) {
    gpu_global_ids.reserve(config.local_replica_mapping_size());
    for (auto& gid : config.local_replica_mapping()) {
      gpu_global_ids.emplace_back(xla::GlobalDeviceId(gid));
    }
    gpu_options.set_gpu_global_device_ids(gpu_global_ids);
  }
  std::shared_ptr<NcclUniqueIdFactory> nccl_factory = GetNcclUniqueIdFactory();
  if (nccl_factory != nullptr) {
    auto uid_callback =
        [&](const xla::NcclCliqueKey& key) -> xla::StatusOr<std::string> {
      std::vector<xla::int64> replicas;
      for (auto& device : key.devices()) {
        replicas.push_back(device.value());
      }
      return nccl_factory->GetUniqueId(replicas);
    };
    gpu_options.set_nccl_unique_id_callback(uid_callback);
  }
  run_options.set_gpu_executable_run_options(&gpu_options);

  Env* env = Env::Default();
  auto start_time = env->NowMicros();
  const std::vector<xla::ShapeLayout>& shape_layouts =
      executable->executable()
          ->module_config()
          .entry_computation_layout()
          .parameter_layouts();
  TF_ASSIGN_OR_RETURN(auto new_allocations,
                      UpdateDynamicInputs(stream, run_options.allocator(),
                                          input_buffers.input_pointers,
                                          shape_layouts));
  auto new_allocations_ptr =
      std::make_shared<std::vector<se::OwningDeviceMemory>>(
          std::move(new_allocations));
  TF_ASSIGN_OR_RETURN(
      xla::ScopedShapedBuffer run_result,
      executable->Run(input_buffers.input_pointers, run_options));
  // Retain the new allocation for input memory until the end of execution.
  stream->ThenDoHostCallback([new_allocations_ptr]() { return Status::OK(); });

  auto elapsed = env->NowMicros() - start_time;
  VLOG(2) << "Elapsed time: " << elapsed << "us";

  TF_ASSIGN_OR_RETURN(
      RefPtr<XRTTupleAllocation> output_tuple_ptr,
      CreateOutputTuple(stream, std::move(run_result), device_ref->backend(),
                        device_ref->device_ordinal()));

  // The ScopedShapedBuffer returned by the executable Run() API, in case of
  // input/output buffer aliasing, might have holes in it, which need to be
  // filled using the proper input tuples buffers which are the source of
  // aliasing.
  const xla::HloInputOutputAliasConfig& input_output_alias =
      executable->executable()->module().input_output_alias_config();
  auto alias_function =
      [&](const xla::ShapeIndex& output_index,
          const xla::HloInputOutputAliasConfig::Alias& alias) -> Status {
    TF_RET_CHECK(alias.parameter_number < input_buffers.input_tuples.size());
    return alias.kind == xla::HloInputOutputAliasConfig::AliasKind::kUserAlias
               ? output_tuple_ptr->AliasBufferFrom(
                     *input_buffers.input_tuples[alias.parameter_number],
                     alias.parameter_index, output_index)
               : Status::OK();
  };
  TF_RETURN_IF_ERROR(input_output_alias.ForEachAliasWithStatus(alias_function));

  return std::move(output_tuple_ptr);
}

xla::StatusOr<RefPtr<XRTTupleAllocation>> ExecuteComputation(
    OpKernelContext* context, XRTMemoryManager* memory_manager,
    XRTGenericDeviceAccessor::ScopedRef* device_ref,
    xla::LocalExecutable* executable, const InputBuffers& input_buffers,
    se::Stream* stream, int rng_seed,
    const xrt::CommonExecutionConfig& config) {
  auto runfn = [&]() {
    return RunExecutable(context, device_ref, executable, input_buffers, stream,
                         rng_seed, config);
  };

  // We pass zero as requested_free_size as there is no simple way to get the
  // peak heap size. Upon zero, the Run() API will try to free chunks of device
  // memory, until either the runfn can run, or we run out of freeable memory.
  return memory_manager->Run<RefPtr<XRTTupleAllocation>>(
      runfn, device_ref->backend(), device_ref->device_ordinal(),
      /*requested_free_size=*/0);
}

xla::StatusOr<RefPtr<XRTTupleAllocation>> ExecuteComputation(
    OpKernelContext* context, const RefPtr<XRTMemoryManager>& memory_manager,
    XRTGenericDeviceAccessor::ScopedRef* device_ref,
    xla::LocalExecutable* executable,
    const std::vector<InputCoords>& input_coords, bool release_inputs,
    se::Stream* stream, int rng_seed,
    const xrt::CommonExecutionConfig& config) {
  XRTMemoryManager::WorkingSet working_set(memory_manager);
  TF_ASSIGN_OR_RETURN(InputBuffers input_buffers,
                      GetInputBuffers(&working_set, device_ref->backend(),
                                      input_coords, release_inputs));
  return ExecuteComputation(context, memory_manager.get(), device_ref,
                            executable, input_buffers, stream, rng_seed,
                            config);
}

// XRTExecuteOp

class XRTExecuteOp : public AsyncOpKernel {
 public:
  explicit XRTExecuteOp(OpKernelConstruction* context);
  ~XRTExecuteOp() override;

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override;

 private:
  Status DoWork(OpKernelContext* context);
};

XRTExecuteOp::XRTExecuteOp(OpKernelConstruction* context)
    : AsyncOpKernel(context) {}

void XRTExecuteOp::ComputeAsync(OpKernelContext* context, DoneCallback done) {
  // Schedule onto the default queue, for unbounded concurrency. See b/73520706
  Env::Default()->SchedClosure([this, context, done]() {
    OP_REQUIRES_OK_ASYNC(context, DoWork(context), done);
    done();
  });
}

Status XRTExecuteOp::DoWork(OpKernelContext* context) {
  VLOG(1) << "XRTExecuteOp::Compute";
  auto timed = monitoring::MakeTimed(xrt_metrics::GetExecuteCell());
  ResourceMgr* rm;
  TF_RETURN_IF_ERROR(
      XRTGenericDeviceAccessor::GetResourceManager(context, &rm));

  const Tensor& execution_input = context->input(0);
  TF_RET_CHECK(TensorShapeUtils::IsScalar(execution_input.shape()));
  int64 compilation_handle = execution_input.scalar<int64>()();

  const Tensor& execution_config = context->input(1);
  TF_RET_CHECK(TensorShapeUtils::IsScalar(execution_config.shape()));
  xrt::XRTExecutionConfig config_proto;
  TF_RET_CHECK(
      ParseFromTString(execution_config.scalar<tstring>()(), &config_proto));

  int core_index_in_replica = config_proto.core_index_in_replica();
  TF_RET_CHECK(core_index_in_replica == 0);
  bool release_inputs = config_proto.release_input_handles();
  bool release_compilation = config_proto.release_compilation_handle();

  TF_ASSIGN_OR_RETURN(auto cache,
                      XRTGenericDeviceAccessor::GetOrCreateCompilationCache(
                          context, /*max_number_of_entries=*/0));
  // We are guaranteed that the underlying device object won't be deleted out
  // from under us, while the ScopedRef is live.
  class XRTGenericDeviceAccessor::ScopedRef device_ref;
  TF_RETURN_IF_ERROR(
      XRTGenericDeviceAccessor::InitScopedRef(context, &device_ref));

  int rng_seed = config_proto.rng_seed();
  if (rng_seed == 0) {
    rng_seed = GetXLARandomSeed();
  }

  se::Stream* stream = context->op_device_context()
                           ? context->op_device_context()->stream()
                           : nullptr;
  RefPtr<XRTMemoryManager> memory_manager = XRTMemoryManager::Get(rm);
  TF_ASSIGN_OR_RETURN(std::vector<InputCoords> input_coords,
                      GetComputationInputs(context, "input_handles"));

  std::unique_ptr<XRTCompilationCacheEntryRef> entry;
  TF_RETURN_IF_ERROR(cache->Lookup(compilation_handle, &entry));
  xla::LocalExecutable* executable = entry->get().get_executable();
  if (release_compilation) {
    // Process-wide cache of XLA executables.
    TF_RETURN_IF_ERROR(cache->Release(compilation_handle));
    VLOG(2) << "Released compilation handle " << compilation_handle;
  }

  TF_ASSIGN_OR_RETURN(
      RefPtr<XRTTupleAllocation> output_tuple,
      ExecuteComputation(context, memory_manager, &device_ref, executable,
                         input_coords, release_inputs, stream, rng_seed,
                         config_proto.common_config()));

  return CreateExecuteOutput(context, memory_manager.get(),
                             std::move(output_tuple),
                             config_proto.return_exploded_tuple());
}

XRTExecuteOp::~XRTExecuteOp() = default;

class XRTExecuteChainedOp : public AsyncOpKernel {
 public:
  explicit XRTExecuteChainedOp(OpKernelConstruction* context);
  ~XRTExecuteChainedOp() override;

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override;

 private:
  Status DoWork(OpKernelContext* context);
};

XRTExecuteChainedOp::XRTExecuteChainedOp(OpKernelConstruction* context)
    : AsyncOpKernel(context) {}

void XRTExecuteChainedOp::ComputeAsync(OpKernelContext* context,
                                       DoneCallback done) {
  // Schedule onto the default queue, for unbounded concurrency. See b/73520706
  Env::Default()->SchedClosure([this, context, done]() {
    OP_REQUIRES_OK_ASYNC(context, DoWork(context), done);
    done();
  });
}

Status XRTExecuteChainedOp::DoWork(OpKernelContext* context) {
  VLOG(1) << "XRTExecuteChainedOp::Compute";
  auto timed = monitoring::MakeTimed(xrt_metrics::GetExecuteChainedCell());
  ResourceMgr* rm;
  TF_RETURN_IF_ERROR(
      XRTGenericDeviceAccessor::GetResourceManager(context, &rm));

  const Tensor& execution_plan = context->input(0);
  TF_RET_CHECK(TensorShapeUtils::IsScalar(execution_plan.shape()));
  xrt::XRTChainedExecutePlan plan;
  TF_RET_CHECK(ParseFromTString(execution_plan.scalar<tstring>()(), &plan));

  const Tensor& execution_config = context->input(1);
  TF_RET_CHECK(TensorShapeUtils::IsScalar(execution_config.shape()));
  xrt::XRTChainedExecuteConfig config;
  TF_RET_CHECK(ParseFromTString(execution_config.scalar<tstring>()(), &config));

  TF_ASSIGN_OR_RETURN(auto cache,
                      XRTGenericDeviceAccessor::GetOrCreateCompilationCache(
                          context, /*max_number_of_entries=*/0));
  // We are guaranteed that the underlying device object won't be deleted out
  // from under us, while the ScopedRef is live.
  class XRTGenericDeviceAccessor::ScopedRef device_ref;
  TF_RETURN_IF_ERROR(
      XRTGenericDeviceAccessor::InitScopedRef(context, &device_ref));

  int rng_seed = config.rng_seed();
  if (rng_seed == 0) {
    rng_seed = GetXLARandomSeed();
  }

  se::Stream* stream = context->op_device_context()
                           ? context->op_device_context()->stream()
                           : nullptr;
  RefPtr<XRTMemoryManager> memory_manager = XRTMemoryManager::Get(rm);
  auto execute_op = [&](const xrt::XRTChainedExecuteOp& op,
                        absl::Span<const RefPtr<XRTTupleAllocation>> op_inputs)
      -> xla::StatusOr<RefPtr<XRTTupleAllocation>> {
    TF_ASSIGN_OR_RETURN(InputBuffers input_buffers,
                        GetChainedOpInputs(op, op_inputs));

    std::unique_ptr<XRTCompilationCacheEntryRef> entry;
    TF_RETURN_IF_ERROR(cache->Lookup(op.computation_handle(), &entry));
    xla::LocalExecutable* executable = entry->get().get_executable();

    return ExecuteComputation(context, memory_manager.get(), &device_ref,
                              executable, input_buffers, stream, rng_seed,
                              config.common_config());
  };

  return ExecuteChained(context, memory_manager, device_ref.backend(),
                        device_ref.device_ordinal(), plan, config, execute_op);
}

XRTExecuteChainedOp::~XRTExecuteChainedOp() = default;

}  // namespace

REGISTER_KERNEL_BUILDER(Name("XRTExecute")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("computation_handle")
                            .HostMemory("execution_config")
                            .HostMemory("input_handles")
                            .HostMemory("output_handle"),
                        XRTExecuteOp);

REGISTER_KERNEL_BUILDER(Name("XRTExecute")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("computation_handle")
                            .HostMemory("execution_config")
                            .HostMemory("input_handles")
                            .HostMemory("output_handle"),
                        XRTExecuteOp);

REGISTER_KERNEL_BUILDER(Name("XRTExecuteChained")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("execution_plan")
                            .HostMemory("execution_config")
                            .HostMemory("output_handle"),
                        XRTExecuteChainedOp);

REGISTER_KERNEL_BUILDER(Name("XRTExecuteChained")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("execution_plan")
                            .HostMemory("execution_config")
                            .HostMemory("output_handle"),
                        XRTExecuteChainedOp);

}  // namespace tensorflow
