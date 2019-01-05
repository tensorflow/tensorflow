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
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/compiler/xrt/xrt_compilation_cache.h"
#include "tensorflow/compiler/xrt/xrt_device.h"
#include "tensorflow/compiler/xrt/xrt_state.h"
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

// Populates `inputs` with the input tensors to the computation.
Status GetComputationInputs(OpKernelContext* context, ResourceMgr* rm,
                            bool release_inputs,
                            std::vector<XRTTupleAllocation*>* input_tuples,
                            std::vector<xla::ShapedBuffer>* input_allocations,
                            std::vector<xla::ShapedBuffer*>* input_pointers) {
  std::vector<int64> input_uids;
  OpInputList arg_list;
  TF_RETURN_IF_ERROR(context->input_list("input_handles", &arg_list));

  // Concatenate all input uids from list of scalars-or-vectors carrying them.
  for (int i = 0; i < arg_list.size(); ++i) {
    const Tensor& arg = arg_list[i];
    if (TensorShapeUtils::IsScalar(arg.shape())) {
      input_uids.push_back(arg.scalar<int64>()());
    } else {
      TF_RET_CHECK(TensorShapeUtils::IsVector(arg.shape()));
      auto arg_vec = arg.vec<int64>();
      const int64 num_elts = arg.shape().dim_size(0);
      for (int i = 0; i < num_elts; ++i) {
        input_uids.push_back(arg_vec(i));
      }
    }
  }

  // Retrieve allocations for the uids.
  input_tuples->resize(input_uids.size());
  input_pointers->resize(input_uids.size());
  for (int i = 0; i < input_uids.size(); ++i) {
    const int64 input_uid = input_uids[i];
    TF_RETURN_IF_ERROR(
        XRTTupleAllocation::Lookup(rm, input_uid, &(*input_tuples)[i]));
    if (release_inputs) {
      // We are holding a reference to the tuple, so we can safely delete it
      // from the resource manager here.
      TF_RETURN_IF_ERROR(
          XRTTupleAllocation::DeleteFromResourceManager(rm, input_uid));
      VLOG(2) << "Released allocation handle " << input_uid;
    }
    XRTTupleAllocation* tuple = (*input_tuples)[i];
    input_allocations->emplace_back(tuple->ToShapedBuffer());
  }
  for (int i = 0; i < input_uids.size(); ++i) {
    (*input_pointers)[i] = &(*input_allocations)[i];
  }
  return Status::OK();
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
      config_proto.ParseFromString(execution_config.scalar<string>()()));

  int core_index_in_replica = config_proto.core_index_in_replica();
  TF_RET_CHECK(core_index_in_replica == 0);
  bool release_inputs = config_proto.release_input_handles();
  bool release_compilation = config_proto.release_compilation_handle();

  XRTCompilationCache* cache;
  TF_RETURN_IF_ERROR(rm->Lookup<XRTCompilationCache>(
      rm->default_container(), kXRTCompilationCacheResourceName, &cache));
  core::ScopedUnref cache_unref(cache);

  std::unique_ptr<XRTCompilationCacheEntryRef> entry;
  TF_RETURN_IF_ERROR(cache->Lookup(compilation_handle, &entry));

  if (release_compilation) {
    // Process-wide cache of XLA executables.
    TF_RETURN_IF_ERROR(cache->Release(compilation_handle));
    VLOG(2) << "Released compilation handle " << compilation_handle;
  }

  std::vector<XRTTupleAllocation*> input_tuples;
  // Make a cleanup method so that we can safely return in error conditions
  // without leaking references to allocations.
  auto buffer_releaser = gtl::MakeCleanup([&input_tuples]() {
    for (auto tuple : input_tuples) {
      if (tuple != nullptr) {
        tuple->Unref();
      }
    }
  });
  std::vector<xla::ShapedBuffer> input_allocations;
  std::vector<xla::ShapedBuffer*> input_pointers;
  TF_RETURN_IF_ERROR(GetComputationInputs(context, rm, release_inputs,
                                          &input_tuples, &input_allocations,
                                          &input_pointers));

  // We are guaranteed that the underlying device object won't be deleted out
  // from under us, while the ScopedRef is live.
  class XRTGenericDeviceAccessor::ScopedRef device_ref;
  TF_RETURN_IF_ERROR(
      XRTGenericDeviceAccessor::InitScopedRef(context, 0, &device_ref));

  int rng_seed = config_proto.rng_seed();
  if (rng_seed == 0) {
    rng_seed = GetXLARandomSeed();
  }

  se::Stream* stream = context->op_device_context()
                           ? context->op_device_context()->stream()
                           : nullptr;

  // Execute the computation.
  VLOG(2) << "Executing computation.";
  xla::ExecutableRunOptions run_options;
  run_options.set_stream(stream);
  run_options.set_allocator(device_ref.backend()->memory_allocator());
  run_options.set_intra_op_thread_pool(&context->eigen_cpu_device());
  run_options.set_rng_seed(rng_seed);

  Env* env = Env::Default();
  auto start_time = env->NowMicros();

  xla::LocalExecutable* executable = entry->get().get_executable();
  auto run_result = executable->Run(input_pointers, run_options);
  if (!run_result.ok()) {
    return run_result.status();
  }

  auto elapsed = env->NowMicros() - start_time;
  VLOG(2) << "Elapsed time: " << elapsed << "us";

  auto scoped_buffer = run_result.ConsumeValueOrDie();
  auto shaped_buffer = scoped_buffer.release();
  XRTTupleAllocation* output_tuple;
  TF_RETURN_IF_ERROR(XRTTupleAllocation::CreateFromBuffer(
      shaped_buffer, device_ref.backend(), device_ref.device_ordinal(),
      &output_tuple));
  if (config_proto.return_exploded_tuple() &&
      output_tuple->on_device_shape().IsTuple()) {
    int64 tuple_element_count =
        xla::ShapeUtil::TupleElementCount(output_tuple->on_device_shape());
    Tensor* output_tensor;
    TF_RETURN_IF_ERROR(context->allocate_output(
        0, TensorShape({tuple_element_count}), &output_tensor));

    for (int64 i = 0; i < tuple_element_count; ++i) {
      xla::ShapeIndex shape_index;
      shape_index.push_back(i);

      XRTTupleAllocation* suballocation;
      TF_RETURN_IF_ERROR(XRTTupleAllocation::MakeSubBuffer(
          output_tuple, shape_index, &suballocation,
          /*alias_parent_allocation=*/false));
      int64 key;
      TF_RETURN_IF_ERROR(suballocation->Intern(rm, &key));
      output_tensor->vec<int64>()(i) = key;
    }
    output_tuple->Unref();
  } else {
    Tensor* output_tensor;
    TF_RETURN_IF_ERROR(
        context->allocate_output(0, TensorShape({}), &output_tensor));
    int64 key;
    TF_RETURN_IF_ERROR(output_tuple->Intern(rm, &key));
    output_tensor->scalar<int64>()() = key;
  }
  return Status::OK();
}

XRTExecuteOp::~XRTExecuteOp() = default;

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

}  // namespace tensorflow
