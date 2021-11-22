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

#include "tensorflow/core/tpu/tpu_execute.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "absl/base/casts.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/hlo_input_output_alias_config.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/tpu/kernels/tpu_execute_op_options.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/tpu/c_api_conversions.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"
#include "tensorflow/stream_executor/tpu/tpu_op_executable.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_interface.h"

namespace tensorflow {

namespace {

using ::tensorflow::tpu::TpuNodeContext;

// These are placeholders for absl flags.
static bool tpu_cancellation_terminates_process = false;
static bool tpu_cancellation_closes_chips = true;

// Host-side runtime for transfers between TPU and host.
// TODO(b/161940519): Implement this class.
class HostTransferManager {
 public:
  explicit HostTransferManager(TpuNodeContext*, xla::Backend*) {}

  using HostCommmandHandler = TpuOpExecutable::HostCommandHandler;

  // Returns a function to be called when the TPU triggers a host command
  // interrupt while executing the current program.
  xla::StatusOr<HostCommmandHandler> Initialize(
      const TPUHostTransferInfoProto& program,
      const std::string& rendezvous_key_base, OpKernelContext* ctx);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(HostTransferManager);
};

xla::StatusOr<HostTransferManager::HostCommmandHandler>
HostTransferManager::Initialize(const TPUHostTransferInfoProto& program,
                                const string& rendezvous_key_base,
                                OpKernelContext* ctx) {
  return HostCommmandHandler([](uint32, int64_t) {
    LOG(WARNING) << "HostTransferManager is unimplemented.";
  });
}

// Sleep for 5 seconds, then call std::quick_exit(42) to quickly restart.
void ExitCountdown(Env* env) {
  const int kSleepSeconds = 5;
  LOG(INFO) << "TpuExecute was cancelled. Sleeping for " << kSleepSeconds
            << " seconds before terminating the process to give time "
               "for other errors to propagate";
  env->SleepForMicroseconds(kSleepSeconds * 1000000);
  LOG(ERROR) << "Aborting process due to cancelled TPUExecute. Consult "
                "the anomalies reported above (if any), run state of job "
                "(including failed RPCs) and worker logs. This "
                "termination is to ensure a consistent state, if your job "
                "does not restart, modify the retries allowed. See "
                "b/62262381 and b/65223927.";
  std::quick_exit(42);
}

xla::Shape HostShapeToDeviceShape(const xla::Shape& host_shape) {
  XLA_Shape c_host_shape;
  XLA_Shape c_device_shape;
  ApiConverter::ToC(host_shape, &c_host_shape);
  tensorflow::tpu::OpsApiFn()->HardwareLayout_HostShapeToDeviceShapeFn(
      &c_host_shape, &c_device_shape);
  xla::Shape device_shape = ApiConverter::FromC(&c_device_shape);
  ApiConverter::Free(&c_host_shape);
  ApiConverter::Free(&c_device_shape);
  return device_shape;
}

int64_t ShapeSizeCompact(const xla::Shape& shape) {
  XLA_Shape c_shape;
  ApiConverter::ToC(shape, &c_shape);
  int64_t size =
      tensorflow::tpu::OpsApiFn()->HardwareLayout_ShapeSizeCompactFn(&c_shape);
  ApiConverter::Free(&c_shape);
  return size;
}

int64_t ShapeSizeCompactRaw(const xla::Shape& shape) {
  XLA_Shape c_shape;
  ApiConverter::ToC(shape, &c_shape);
  int64_t size =
      tensorflow::tpu::OpsApiFn()->HardwareLayout_ShapeSizeCompactRawFn(
          &c_shape);
  ApiConverter::Free(&c_shape);
  return size;
}

// Given a tuple, fix all non-leaf nodes (tuples) such that the tuple tables
// point to the correct leaf nodes.
xla::Status FixTupleTableAsync(se::Stream* stream,
                               const xla::Shape& tuple_shape,
                               xla::ExecutionInput* mem,
                               xla::TransferManager* transfer_manager) {
  return xla::ShapeUtil::ForEachSubshapeWithStatus(
      tuple_shape,
      [&](const xla::Shape& element_shape,
          const xla::ShapeIndex& index) -> Status {
        if (!element_shape.IsTuple()) {
          return Status::OK();
        }
        std::vector<se::DeviceMemoryBase> elements;
        xla::ShapeIndex element_index = index;
        element_index.push_back(0);
        for (int i = 0; i < element_shape.tuple_shapes_size(); ++i) {
          // Gather all children of the tuple element.
          element_index.back() = i;
          elements.push_back(mem->Buffer(element_index).AsDeviceMemoryBase());
        }
        se::DeviceMemoryBase tuple_table_addr =
            mem->Buffer(index).AsDeviceMemoryBase();
        return transfer_manager->WriteSingleTupleIndexTable(
            stream, elements, element_shape, &tuple_table_addr);
      });
}

// Returns true if `dynamic_shape` has dimensions that are less-equal to the
// "bounded_shape".
bool DynamicShapeIsCompatible(const xla::Shape& dynamic_shape,
                              const xla::Shape& bounded_shape) {
  if (dynamic_shape.rank() != bounded_shape.rank()) {
    return false;
  }
  for (int64_t i = 0; i < dynamic_shape.rank(); ++i) {
    if (dynamic_shape.dimensions(i) > bounded_shape.dimensions(i)) {
      return false;
    }
  }
  return true;
}

// For dynamic inputs, copy them and attach metadata of shape sizes to the
// end of the tensor.
//
// The buffer for dynamic shapes contains three parts:
// +--------+
// |Payload |
// +--------+
// | Padding|
// +--------+
// |Metadata|
// +--------+
//
// Metadata contains the sizes of shape without padding, eventually
// representing the size of valid data.
xla::Status UpdateDynamicInputs(
    se::Stream* stream, se::DeviceMemoryAllocator* allocator,
    std::vector<xla::ExecutionInput>* runtime_inputs,
    const std::vector<xla::Shape>& compile_time_shapes) {
  TF_RET_CHECK(runtime_inputs->size() == compile_time_shapes.size());
  for (int64_t i = 0; i < compile_time_shapes.size(); i++) {
    // TODO(yunxing): Iterating over thousands of elements can be slow. One way
    // to optimize for fast path without dynamic shapes is add a field in
    // compilation result indicating if dynamic input is presented.
    if (compile_time_shapes[i].is_static()) {
      continue;
    }
    auto& runtime_input = (*runtime_inputs)[i];
    xla::Shape compile_time_shapes_on_device =
        HostShapeToDeviceShape(compile_time_shapes[i]);
    bool element_modified = false;
    TF_RETURN_IF_ERROR(xla::ShapeUtil::ForEachSubshapeWithStatus(
        compile_time_shapes_on_device,
        [&](const xla::Shape& compile_time_shape,
            const xla::ShapeIndex& index) -> Status {
          if (compile_time_shape.IsTuple() || compile_time_shape.is_static()) {
            return Status::OK();
          }

          const xla::Shape& runtime_shape =
              xla::ShapeUtil::GetSubshape(runtime_input.shape(), index);

          TF_RET_CHECK(!runtime_shape.IsTuple());
          TF_RET_CHECK(
              DynamicShapeIsCompatible(runtime_shape, compile_time_shape));

          xla::MaybeOwningDeviceMemory* mutable_input_mem =
              runtime_input.MutableBuffer(index);
          auto padded_data = std::make_shared<std::vector<int8>>(
              ShapeSizeCompact(compile_time_shape), -1);
          auto raw_input_runtime = std::make_shared<std::vector<uint32>>(
              ShapeSizeCompact(runtime_shape) / sizeof(uint32));
          stream->ThenMemcpyD2H(
              se::DeviceMemory<int8>(mutable_input_mem->AsDeviceMemoryBase()),
              absl::MakeSpan(absl::bit_cast<int8*>(raw_input_runtime->data()),
                             ShapeSizeCompactRaw(runtime_shape)));
          stream->ThenDoHostCallback([raw_input_runtime, padded_data,
                                      runtime_shape, compile_time_shape]() {
            // After getting the data onto the host, transpose the data to
            // the correct layout by delinearizing it and linearizing it again.
            XLA_Shape c_runtime_shape, c_compile_time_shape;
            ApiConverter::ToC(runtime_shape, &c_runtime_shape);
            ApiConverter::ToC(compile_time_shape, &c_compile_time_shape);
            StatusHelper status;

            TpuExecute_RuntimeInputToPaddedData_Params params;
            params.struct_size =
                TpuExecute_RuntimeInputToPaddedData_Params_SIZE;
            params.priv = nullptr;
            params.runtime_input_ptr = raw_input_runtime->data();
            params.runtime_input_size = raw_input_runtime->size();
            params.padded_data_ptr = padded_data->data();
            params.padded_data_size = padded_data->size();
            params.runtime_shape = &c_runtime_shape;
            params.compile_time_shape = &c_compile_time_shape;
            params.status = status.c_status;

            tensorflow::tpu::OpsApiFn()->TpuExecute_RuntimeInputToPaddedDataFn(
                &params);
            ApiConverter::Free(&c_runtime_shape);
            ApiConverter::Free(&c_compile_time_shape);
            return status.status();
          });
          // Allocate new input and transfer the padded and transposed data to
          // the new input location.
          TF_ASSIGN_OR_RETURN(
              auto new_input,
              allocator->Allocate(stream->parent()->device_ordinal(),
                                  ShapeSizeCompact(compile_time_shape)));
          auto typed_new_input_memory =
              se::DeviceMemory<int8>(new_input.cref());
          stream->ThenMemcpyH2D<int8>(*padded_data, &typed_new_input_memory);

          // Retain the memory until the end of the transfer.
          stream->ThenDoHostCallback([padded_data]() { return Status::OK(); });

          // Modify the memory location in the input shape tree to point to the
          // new input.
          *mutable_input_mem =
              xla::MaybeOwningDeviceMemory(std::move(new_input));
          element_modified = true;
          return Status::OK();
        }));
    if (element_modified) {
      // The input location has been modified, need to fix tuple table to
      // point to the correct address.
      TF_ASSIGN_OR_RETURN(
          auto transfer_manager,
          xla::TransferManager::GetForPlatform(stream->parent()->platform()));
      TF_RETURN_IF_ERROR(FixTupleTableAsync(stream,
                                            compile_time_shapes_on_device,
                                            &runtime_input, transfer_manager));
    }
  }
  return Status::OK();
}

void TPUCancelExecution(Env* env, int device_ordinal) {
  if (tpu_cancellation_terminates_process) {
    LOG(INFO) << "TPUCancelExecution StopChipHeartbeats on device "
              << device_ordinal;
    Status status = TpuNodeContext::StopChipHeartbeats();
    LOG(INFO) << "TPUCancelExecution StopChipHeartbeats done: " << status
              << " on device " << device_ordinal;
    // Sleep and exit in another thread so the cancellation manager can
    // continue running callbacks. The new thread will call quick_exit,
    // so we discard the returned Thread pointer because we won't have
    // an opportunity to delete it.
    (void)env->StartThread(ThreadOptions(), "tpu_execute_exit_countdown",
                           [env]() { ExitCountdown(env); });
  } else if (tpu_cancellation_closes_chips) {
    LOG(INFO) << "TPUCancelExecution CloseTPUHost on device " << device_ordinal;
    Status status = TpuNodeContext::CloseTpuHost();
    LOG(INFO) << "TPUCancelExecution CloseTPUHost done: " << status
              << " on device " << device_ordinal;
  } else {
    LOG(INFO) << "TPUCancelExecution CloseTPUHost on device " << device_ordinal
              << " is suppressed";
  }
}

std::pair<CancellationToken, bool> RegisterCancellation(
    OpKernelContext* ctx, CancellationManager* cancellation_manager,
    int device_ordinal) {
  // Set up a cancellation callback, to ensure the TPU program we run will
  // halt if the RPC is cancelled. Without this the TPU program might block
  // forever. The mechanism itself is a big hammer; we close all devices
  // attached to this host on each cancellation callback. This is necessary to
  // ensure the system will eventually halt, since the TensorNodes on each
  // chip may be stuck waiting for mutual communication.
  //
  // By closing all devices, we ensure all subsequent attempts to use the
  // device will fail, until the devices are re-initialized via a new call to
  // tpu.initialize_system.
  //
  // In a multi-TensorNode setup, CloseTPUHost may be called once for each
  // TensorNode, and each call will close all TensorNodes. This quadratic
  // behavior ensures the mechanism is robust to various orderings
  // (i.e. races) between the TPU programs, which are run on separate threads.
  // In practice the quadratic behavior isn't that bad; the first call will
  // actually halt any running TPU programs (which may be expensive), while
  // subsequent calls will attempt to close an already-closed device (which is
  // cheap).
  //
  // TODO(b/62262381): The cancellation manager is shared between multiple TPU
  // execute ops and the cancellation will not be invoked only when RPC is
  // cancelled (it may also be induced by OOM errors from a different TPU
  // execute), this results in a pretty coarse cancellation domain. This
  // cancellation callback should only execute in a narrower scope to not be
  // triggered in such cases.
  CancellationToken token = cancellation_manager->get_cancellation_token();
  // Don't rely on OpKernelContext being available when the callback runs.
  Env* env = ctx->env();
  bool already_cancelled =
      !cancellation_manager->RegisterCallbackWithErrorLogging(
          token,
          [device_ordinal, env]() { TPUCancelExecution(env, device_ordinal); },
          absl::StrCat("TPUCancellation on device ", device_ordinal));
  return std::pair<CancellationToken, bool>(token, already_cancelled);
}

void UnregisterCancellation(
    OpKernelContext* ctx, CancellationManager* cancellation_manager,
    se::Stream* stream, int device_ordinal, CancellationToken token,
    std::shared_ptr<HostTransferManager> host_transfer_manager) {
  // If execution reaches this point, the host callback enqueued below will get
  // called regardless of stream status. Call inc_num_deferred_ops_function here
  // and dec_num_deferred_ops_function in the host callback.
  ctx->inc_num_deferred_ops_function()();
  auto dec_num_deferred_ops_function = ctx->dec_num_deferred_ops_function();

  // Try to avoid running callbacks on the compute stream, because this reduces
  // the frequency of back-to-back programs (which are most efficient because
  // they don't require host synchronization). Instead, borrow a substream and
  // have the substream wait on the compute stream.
  se::Stream* deregister_stream = stream->GetOrCreateSubStream();
  deregister_stream->ThenWaitFor(stream);
  deregister_stream->ThenDoHostCallback([=]() {
    // Ensure the host_transfer_manager is copied into the callback scope.
    (void)host_transfer_manager;

    // We must deregister the callback in the success case, to avoid closing all
    // devices. In the failure case we must NOT call DeregisterCallback as that
    // waits for all previous cancellation callbacks to complete and any call
    // to XlaDevice::Sync() will cause deadlock. Consider:
    //   1) CancellationManager::StartCancel() is in progress (state is
    //      cancelling_).
    //   2) The call below to DeregisterCallback will block until state is
    //   cancelled_ (all callbacks are completed).
    //   3) A different cancellation callback has called XlaDevice::Sync(),
    //   which will block until (2) is done.
    //   4) StartCancel() in (1) cannot complete until (3) is done.
    //
    // Instead, call TryDeregisterCallback. The functional difference is
    // TryDeregisterCallback will not block if cancellation is in proress
    // so makes no guarantees as to the state of any callbacks.
    // This is not a problem, as our cancellation handler does not rely on
    // any external state.
    VLOG(1) << "cancellation_manager->TryDeregisterCallback on device "
            << device_ordinal;
    cancellation_manager->TryDeregisterCallback(token);
    VLOG(1) << "cancellation_manager->TryDeregisterCallback done on device "
            << device_ordinal;

    // ExecutorState is held alive until at least this point to ensure
    // cancellation_manager is valid. After all outstanding
    // dec_num_deferred_ops_function are called, ExecutorState::Finish will be
    // allowed to proceed.
    dec_num_deferred_ops_function();
  });
  stream->ReturnSubStream(deregister_stream);
}

}  // namespace

xla::StatusOr<xla::ExecutionOutput> TPUExecute(
    const TPUExecutableInfoProto& executable,
    const TPUHostTransferInfoProto& host_transfers,
    const xla::HloProto& hlo_metadata,
    std::vector<xla::ExecutionInput> arguments,
    const string& rendezvous_key_base, uint32 rng_seed,
    TpuNodeContext* node_context, xla::DeviceAssignment* device_assignment,
    CancellationManager* cancellation_manager, OpKernelContext* ctx,
    stream_executor::Stream* stream,
    stream_executor::Stream* host_to_device_stream,
    const XLA_TpuProgram* tpu_program) {
  profiler::TraceMe traceme("TPUExecute", 2);
  TF_RET_CHECK(tpu::TpuPlatformInterface::GetRegisteredPlatform() != nullptr);
  TF_RET_CHECK(tpu_program != nullptr);
  VLOG(1) << "TPUExecute on device " << node_context->device_ordinal();

  xla::Backend* backend = node_context->backend();

  // Create a HostTransferManager to handle Send/Recv operations from the TPU.
  std::shared_ptr<HostTransferManager> host_transfer_manager =
      std::make_shared<HostTransferManager>(node_context, backend);
  TF_ASSIGN_OR_RETURN(HostTransferManager::HostCommmandHandler handler,
                      host_transfer_manager->Initialize(
                          host_transfers, rendezvous_key_base, ctx));

  VLOG(2) << "Cloud TPU: Executing computation on device "
          << node_context->device_ordinal();

  xla::ExecutableRunOptions run_options;
  run_options.set_stream(stream);
  run_options.set_device_assignment(device_assignment);
  run_options.set_rng_seed(rng_seed);
  run_options.set_allocator(backend->memory_allocator());
  run_options.set_host_to_device_stream(host_to_device_stream);

  const xla::ServiceExecutableRunOptions service_run_options(run_options);

  std::unique_ptr<xla::HloModule> module;
  std::vector<xla::Shape> input_shapes;
  {
    xla::ComputationLayout computation_layout(
        xla::ShapeLayout(xla::Shape(executable.output_shape())));
    for (const xla::ShapeProto& shape_proto : executable.input_shapes()) {
      xla::Shape shape(shape_proto);
      computation_layout.add_parameter_layout(xla::ShapeLayout(shape));
      input_shapes.push_back(std::move(shape));
    }
    module = absl::make_unique<xla::HloModule>(
        "TpuExecutableModule",
        xla::HloModuleConfig(std::move(computation_layout)));
  }

  TF_ASSIGN_OR_RETURN(
      module->input_output_alias_config(),
      xla::HloInputOutputAliasConfig::CreateFromProto(
          backend->transfer_manager()->HostShapeToDeviceShape(
              module->config().entry_computation_layout().result_shape()),
          hlo_metadata.hlo_module().input_output_alias()));
  TF_RET_CHECK(executable.input_shapes().size() == arguments.size());

  for (auto& prefetch : hlo_metadata.hlo_module().cross_program_prefetches()) {
    module->AddCrossProgramPrefetch(
        prefetch.parameter(),
        xla::ShapeIndex(prefetch.index().begin(), prefetch.index().end()));
  }

  TF_RETURN_IF_ERROR(UpdateDynamicInputs(stream, backend->memory_allocator(),
                                         &arguments, input_shapes));

  auto tpu_executable = absl::make_unique<TpuOpExecutable>(
      tpu_program, std::move(module), /*host_command_handler=*/handler);

  const int32_t device_ordinal = node_context->device_ordinal();
  CancellationToken token;
  bool already_cancelled;
  std::tie(token, already_cancelled) =
      RegisterCancellation(ctx, cancellation_manager, device_ordinal);

  // If the RPC was already cancelled before we managed to register the
  // cancellation callback, we shouldn't attempt to run the TPU program, since
  // it might block forever.
  if (already_cancelled) {
    return errors::Cancelled(
        "RPC cancelled, not running TPU program on device ", device_ordinal);
  }

  xla::StatusOr<xla::ExecutionOutput> output =
      tpu_executable->ExecuteAsyncOnStream(&service_run_options,
                                           std::move(arguments),
                                           /*hlo_execution_profile=*/nullptr);

  // If !output.ok(), it means we failed to enqueue the program the TPU. This is
  // possibly caused by a failed cancellation callback closing the chips.
  if (!output.ok()) {
    // If cancellation manager is already cancelled or cancelling, it means
    // another failure has occurred earlier and this TpuExecuteOp is cancelled
    // regardless of whether itself is an error.
    already_cancelled = cancellation_manager->IsCancelling() ||
                        cancellation_manager->IsCancelled();
    if (already_cancelled) {
      return errors::Cancelled(
          "RPC cancelled, not running TPU program on device ", device_ordinal);
    }
  }
  UnregisterCancellation(ctx, cancellation_manager, stream, device_ordinal,
                         token, host_transfer_manager);
  VLOG(1) << "Cloud TPU: TPUExecute done";
  return output;
}

}  // namespace tensorflow
