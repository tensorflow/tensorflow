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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_input_output_alias_config.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory_allocator.h"
#include "tensorflow/compiler/xla/stream_executor/platform.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor_internal.h"
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

namespace tensorflow {

namespace {

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

std::vector<bool> GetDynamicInputInfo(
    const xla::ComputationLayout& computation_layout) {
  std::vector<bool> input_is_dynamic;
  input_is_dynamic.reserve(computation_layout.parameter_count());
  for (int64_t i = 0; i < computation_layout.parameter_count(); ++i) {
    input_is_dynamic.push_back(
        !computation_layout.parameter_shape(i).is_static());
  }
  return input_is_dynamic;
}

xla::StatusOr<std::vector<RefPtr<XRTTupleAllocation>>> GetInputTuples(
    xla::LocalExecutable* executable, XRTMemoryManager::WorkingSet* working_set,
    xla::Backend* backend, const std::vector<InputCoords>& input_coords,
    bool release_inputs, se::DeviceMemoryAllocator* allocator) {
  const xla::ComputationLayout& computation_layout =
      executable->executable()->module_config().entry_computation_layout();

  return GetInputTupleAllocations(
      input_coords, working_set, backend, computation_layout.parameter_count(),
      [&](int64_t i) { return computation_layout.parameter_shape(i); },
      release_inputs, allocator);
}

xla::StatusOr<std::vector<RefPtr<XRTTupleAllocation>>> GetChainedOpInputTuples(
    const xrt::XRTChainedExecuteOp& op,
    absl::Span<const RefPtr<XRTTupleAllocation>> op_inputs) {
  std::vector<RefPtr<XRTTupleAllocation>> input_tuples;
  input_tuples.reserve(op.inputs_size());
  for (int i = 0; i < op.inputs_size(); ++i) {
    auto& input = op.inputs(i);
    // Thanks to the greatness of proto3, there is no way to query for
    // explicitly set fields, so the default for output_index (zero) means no
    // sub-index. As consequence, the real index is output_index - 1.
    if (input.output_index() == 0) {
      input_tuples.emplace_back(op_inputs[i]);
    } else {
      XRTTupleAllocation* sub_tuple;
      TF_RETURN_IF_ERROR(XRTTupleAllocation::MakeSubBuffer(
          op_inputs[i].get(), {input.output_index() - 1}, &sub_tuple,
          /*alias_parent_allocation=*/true));
      input_tuples.emplace_back(sub_tuple);
    }
  }
  return input_tuples;
}

// Given a shape, returns a byte array representing the shape metadata of the
// shape. The shape metadata contains dimensions sizes stored as contiguous S32.
std::vector<int32> PrepareMetadata(const xla::Shape& shape) {
  DCHECK(shape.is_static());
  DCHECK(shape.IsArray());
  // Each dimension size is stored as a S32.
  std::vector<int32> result(shape.dimensions_size());
  for (int64_t i = 0; i < shape.dimensions_size(); ++i) {
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
  stream->ThenDoHostCallback([metadata_literal]() { return OkStatus(); });
  return OkStatus();
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
Status UpdateDynamicInputs(
    se::Stream* stream, se::DeviceMemoryAllocator* allocator,
    std::vector<xla::ExecutionInput>* execution_inputs,
    const std::vector<xla::ShapeLayout>& compile_time_shapes) {
  TF_RET_CHECK(execution_inputs->size() == compile_time_shapes.size());
  TF_ASSIGN_OR_RETURN(auto compiler, xla::Compiler::GetForPlatform(
                                         stream->parent()->platform()));
  auto shape_size_fn = compiler->ShapeSizeBytesFunction();
  for (int64_t i = 0; i < compile_time_shapes.size(); i++) {
    const xla::Shape& compile_time_shape = compile_time_shapes[i].shape();
    if (compile_time_shape.is_static()) {
      continue;
    }
    xla::ExecutionInput* execution_input = &(*execution_inputs)[i];
    bool element_modified = false;
    TF_RETURN_IF_ERROR(xla::ShapeUtil::ForEachSubshapeWithStatus(
        compile_time_shape,
        [&](const xla::Shape& sub_shape,
            const xla::ShapeIndex& index) -> Status {
          if (sub_shape.IsTuple() || sub_shape.is_static()) {
            return OkStatus();
          }
          TF_ASSIGN_OR_RETURN(
              const xla::Shape* runtime_shape,
              xla::ShapeUtil::TryGetSubshape(execution_input->shape(), index));
          TF_RET_CHECK(!runtime_shape->IsTuple());
          TF_RET_CHECK(xla::ShapeUtil::DynamicArrayShapeIsCompatible(
              *runtime_shape, sub_shape));
          TF_ASSIGN_OR_RETURN(
              se::OwningDeviceMemory dynamic_input,
              allocator->Allocate(stream->parent()->device_ordinal(),
                                  shape_size_fn(sub_shape)));

          se::DeviceMemoryBase static_input =
              execution_input->Buffer(index).AsDeviceMemoryBase();
          se::DeviceMemory<uint8>* dynamic_input_base = dynamic_input.ptr();
          // Send the original data to the new location.
          stream->ThenMemcpyD2D(dynamic_input_base, static_input,
                                static_input.size());
          TF_RETURN_IF_ERROR(UpdateMetadata(stream, dynamic_input_base,
                                            sub_shape, *runtime_shape));
          // Modify the memory location in the input shape tree to point to the
          // new input.
          execution_input->SetBuffer(
              index, xla::MaybeOwningDeviceMemory(std::move(dynamic_input)));
          execution_input->ClearUnownedIndex(index);
          element_modified = true;
          return OkStatus();
        }));
    if (element_modified) {
      TF_RETURN_IF_ERROR(execution_input->SetDynamicShape(compile_time_shape));
      TF_ASSIGN_OR_RETURN(xla::ShapedBuffer shaped_buffer,
                          execution_input->ToShapedBuffer(
                              allocator, stream->parent()->device_ordinal()));
      // The input location has been modified, need to fix tuple table to
      // point to the correct address.
      TF_ASSIGN_OR_RETURN(
          auto transfer_manager,
          xla::TransferManager::GetForPlatform(stream->parent()->platform()));
      TF_RETURN_IF_ERROR(
          transfer_manager->WriteTupleIndexTablesAsync(stream, shaped_buffer));
    }
  }
  return OkStatus();
}

xla::StatusOr<RefPtr<XRTTupleAllocation>> CreateOutputTuple(
    se::Stream* stream, xla::ExecutionOutput run_result, xla::Backend* backend,
    int device_ordinal, se::DeviceMemoryAllocator* allocator) {
  XRTTupleAllocation* output_tuple;
  xla::ScopedShapedBuffer* shaped_buffer = run_result.MutableResult();
  if (shaped_buffer->on_device_shape().is_dynamic()) {
    // Update dynamic shapes from output buffer, and create a XRT tensor with
    // dimension sizes read from metadata.
    xla::Shape output_device_shape = shaped_buffer->on_device_shape();
    TF_ASSIGN_OR_RETURN(
        auto transfer_manager,
        xla::TransferManager::GetForPlatform(stream->parent()->platform()));
    TF_RETURN_IF_ERROR(transfer_manager->ReadDynamicShapes(
        stream, shaped_buffer, &output_device_shape));
    TF_RETURN_IF_ERROR(XRTTupleAllocation::CreateFromBuffer(
        *shaped_buffer,
        xla::ShapeUtil::DeviceShapeToHostShape(output_device_shape),
        output_device_shape, backend, device_ordinal, &output_tuple,
        allocator));
  } else {
    // Fast-path: Don't copy shapes of output buffer.
    TF_RETURN_IF_ERROR(XRTTupleAllocation::CreateFromBuffer(
        *shaped_buffer, backend, device_ordinal, &output_tuple, allocator));
  }
  // After the output tuple is created, we can release the output result
  // buffers, to make sure they won't be cleared by its destructor.
  (void)run_result.ConsumeResult().release();
  return RefPtr<XRTTupleAllocation>(output_tuple);
}

xla::StatusOr<RefPtr<XRTTupleAllocation>> RunExecutable(
    OpKernelContext* context, XRTGenericDeviceAccessor::ScopedRef* device_ref,
    xla::LocalExecutable* executable,
    absl::Span<const RefPtr<XRTTupleAllocation>> input_tuples,
    bool release_inputs, se::Stream* stream, int rng_seed,
    const xrt::CommonExecutionConfig& config) {
  const xla::ComputationLayout& computation_layout =
      executable->executable()->module_config().entry_computation_layout();
  std::vector<bool> input_is_dynamic = GetDynamicInputInfo(computation_layout);
  TF_ASSIGN_OR_RETURN(
      std::vector<xla::ExecutionInput> execution_inputs,
      GetArgumentsBuffers(
          executable->executable()->module().input_output_alias_config(),
          input_tuples, input_is_dynamic, release_inputs));

  se::DeviceMemoryAllocator* allocator = device_ref->allocator();
  xla::ExecutableRunOptions run_options;
  run_options.set_stream(stream);
  run_options.set_allocator(allocator);
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
  xla::gpu::GpuExecutableRunOptions gpu_options;
  std::map<int, xla::GlobalDeviceId> gpu_global_ids;
  if (config.local_replica_mapping_size() > 0) {
    int i = 0;
    for (auto& gid : config.local_replica_mapping()) {
      gpu_global_ids[i++] = xla::GlobalDeviceId(gid);
    }
    gpu_options.set_gpu_global_device_ids(gpu_global_ids);
  }
  std::shared_ptr<NcclUniqueIdFactory> nccl_factory = GetNcclUniqueIdFactory();
  if (nccl_factory != nullptr) {
    auto uid_callback =
        [&](const xla::gpu::NcclCliqueKey& key) -> xla::StatusOr<std::string> {
      std::vector<int64_t> replicas;
      const auto key_devices = key.devices();
      replicas.reserve(key_devices.size());
      for (auto& device : key_devices) {
        replicas.push_back(device.value());
      }
      return nccl_factory->GetUniqueId(replicas);
    };
    gpu_options.set_nccl_unique_id_callback(uid_callback);
  }
  run_options.set_gpu_executable_run_options(&gpu_options);

  const std::vector<xla::ShapeLayout>& shape_layouts =
      executable->executable()
          ->module_config()
          .entry_computation_layout()
          .parameter_layouts();
  TF_RETURN_IF_ERROR(UpdateDynamicInputs(stream, run_options.allocator(),
                                         &execution_inputs, shape_layouts));
  TF_ASSIGN_OR_RETURN(
      xla::ExecutionOutput run_result,
      executable->Run(std::move(execution_inputs), run_options));

  TF_ASSIGN_OR_RETURN(
      RefPtr<XRTTupleAllocation> output_tuple_ptr,
      CreateOutputTuple(stream, std::move(run_result), device_ref->backend(),
                        device_ref->device_ordinal(), allocator));
  // The ScopedShapedBuffer returned by the executable Run() API, in case of
  // input/output buffer aliasing, might have holes in it, which need to be
  // filled using the proper input tuples buffers which are the source of
  // aliasing.
  TF_RETURN_IF_ERROR(RebuildOutputAliases(
      output_tuple_ptr, input_tuples,
      executable->executable()->module().input_output_alias_config()));

  return std::move(output_tuple_ptr);
}

xla::StatusOr<RefPtr<XRTTupleAllocation>> ExecuteComputation(
    OpKernelContext* context, XRTMemoryManager* memory_manager,
    XRTGenericDeviceAccessor::ScopedRef* device_ref,
    xla::LocalExecutable* executable,
    absl::Span<const RefPtr<XRTTupleAllocation>> input_tuples,
    bool release_inputs, se::Stream* stream, int rng_seed,
    const xrt::CommonExecutionConfig& config) {
  auto runfn = [&]() {
    return RunExecutable(context, device_ref, executable, input_tuples,
                         release_inputs, stream, rng_seed, config);
  };

  // We pass zero as requested_free_size as there is no simple way to get the
  // peak heap size. Upon zero, the Run() API will try to free chunks of device
  // memory, until either the runfn can run, or we run out of freeable memory.
  return memory_manager->Run<RefPtr<XRTTupleAllocation>>(
      runfn, device_ref->backend(), device_ref->device_ordinal(),
      /*requested_free_size=*/0, device_ref->allocator());
}

xla::StatusOr<RefPtr<XRTTupleAllocation>> ExecuteComputation(
    OpKernelContext* context, const RefPtr<XRTMemoryManager>& memory_manager,
    XRTGenericDeviceAccessor::ScopedRef* device_ref,
    xla::LocalExecutable* executable,
    const std::vector<InputCoords>& input_coords, bool release_inputs,
    se::Stream* stream, int rng_seed,
    const xrt::CommonExecutionConfig& config) {
  XRTMemoryManager::WorkingSet working_set(memory_manager);
  TF_ASSIGN_OR_RETURN(
      std::vector<RefPtr<XRTTupleAllocation>> input_tuples,
      GetInputTuples(executable, &working_set, device_ref->backend(),
                     input_coords, release_inputs, device_ref->allocator()));
  return ExecuteComputation(context, memory_manager.get(), device_ref,
                            executable, input_tuples, release_inputs, stream,
                            rng_seed, config);
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
  int64_t compilation_handle = execution_input.scalar<int64_t>()();

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
    std::unique_ptr<XRTCompilationCacheEntryRef> entry;
    TF_RETURN_IF_ERROR(cache->Lookup(op.computation_handle(), &entry));
    xla::LocalExecutable* executable = entry->get().get_executable();

    TF_ASSIGN_OR_RETURN(std::vector<RefPtr<XRTTupleAllocation>> input_tuples,
                        GetChainedOpInputTuples(op, op_inputs));

    return ExecuteComputation(
        context, memory_manager.get(), &device_ref, executable, input_tuples,
        /*release_inputs=*/false, stream, rng_seed, config.common_config());
  };

  return ExecuteChained(context, memory_manager, device_ref.backend(),
                        device_ref.device_ordinal(), plan, config, execute_op,
                        device_ref.allocator());
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
