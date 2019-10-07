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
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/hlo_input_output_alias_config.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/compiler/xrt/xrt_compilation_cache.h"
#include "tensorflow/compiler/xrt/xrt_device.h"
#include "tensorflow/compiler/xrt/xrt_memory_manager.h"
#include "tensorflow/compiler/xrt/xrt_state.h"
#include "tensorflow/compiler/xrt/xrt_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/types.h"
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

xla::StatusOr<RefPtr<XRTTupleAllocation>> RunExecutable(
    OpKernelContext* context, XRTGenericDeviceAccessor::ScopedRef* device_ref,
    xla::LocalExecutable* executable, const InputBuffers& input_buffers,
    se::Stream* stream, int rng_seed) {
  VLOG(2) << "Executing computation.";
  xla::ExecutableRunOptions run_options;
  run_options.set_stream(stream);
  run_options.set_allocator(device_ref->backend()->memory_allocator());
  run_options.set_intra_op_thread_pool(&context->eigen_cpu_device());
  run_options.set_rng_seed(rng_seed);

  Env* env = Env::Default();
  auto start_time = env->NowMicros();
  TF_ASSIGN_OR_RETURN(
      xla::ScopedShapedBuffer run_result,
      executable->Run(input_buffers.input_pointers, run_options));
  auto elapsed = env->NowMicros() - start_time;
  VLOG(2) << "Elapsed time: " << elapsed << "us";

  auto shaped_buffer = run_result.release();
  XRTTupleAllocation* output_tuple;
  TF_RETURN_IF_ERROR(XRTTupleAllocation::CreateFromBuffer(
      shaped_buffer, device_ref->backend(), device_ref->device_ordinal(),
      &output_tuple));
  RefPtr<XRTTupleAllocation> output_tuple_ptr(output_tuple);

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
               ? output_tuple->AliasBufferFrom(
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
    se::Stream* stream, int rng_seed) {
  auto runfn = [&]() {
    return RunExecutable(context, device_ref, executable, input_buffers, stream,
                         rng_seed);
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
    se::Stream* stream, int rng_seed) {
  XRTMemoryManager::WorkingSet working_set(memory_manager);
  TF_ASSIGN_OR_RETURN(InputBuffers input_buffers,
                      GetInputBuffers(&working_set, device_ref->backend(),
                                      input_coords, release_inputs));
  return ExecuteComputation(context, memory_manager.get(), device_ref,
                            executable, input_buffers, stream, rng_seed);
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
      config_proto.ParseFromString(execution_config.scalar<tstring>()()));

  int core_index_in_replica = config_proto.core_index_in_replica();
  TF_RET_CHECK(core_index_in_replica == 0);
  bool release_inputs = config_proto.release_input_handles();
  bool release_compilation = config_proto.release_compilation_handle();

  TF_ASSIGN_OR_RETURN(
      auto cache, GetOrCreateCompilationCache(rm, /*max_number_of_entries=*/0));
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
                         input_coords, release_inputs, stream, rng_seed));

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
  ResourceMgr* rm;
  TF_RETURN_IF_ERROR(
      XRTGenericDeviceAccessor::GetResourceManager(context, &rm));

  const Tensor& execution_plan = context->input(0);
  TF_RET_CHECK(TensorShapeUtils::IsScalar(execution_plan.shape()));
  xrt::XRTChainedExecutePlan plan;
  TF_RET_CHECK(plan.ParseFromString(execution_plan.scalar<tstring>()()));

  const Tensor& execution_config = context->input(1);
  TF_RET_CHECK(TensorShapeUtils::IsScalar(execution_config.shape()));
  xrt::XRTChainedExecuteConfig config;
  TF_RET_CHECK(config.ParseFromString(execution_config.scalar<tstring>()()));

  TF_ASSIGN_OR_RETURN(
      auto cache, GetOrCreateCompilationCache(rm, /*max_number_of_entries=*/0));
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
                              executable, input_buffers, stream, rng_seed);
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
