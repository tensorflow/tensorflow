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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/backend.h"
#include "xla/service/computation_layout.h"
#include "xla/service/executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/tpu/c_api_conversions.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/c_api_defn.h"
#include "xla/stream_executor/tpu/proto_helper.h"
#include "xla/stream_executor/tpu/status_helper.h"
#include "xla/stream_executor/tpu/tpu_api.h"
#include "xla/stream_executor/tpu/tpu_node_context.h"
#include "xla/stream_executor/tpu/tpu_op_executable.h"
#include "xla/stream_executor/tpu/tpu_ops_c_api.h"
#include "xla/stream_executor/tpu/tpu_platform_interface.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/outside_compilation_params.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_rendezvous_c_api_internal.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/tpu/kernels/tpu_executable_info.pb.h"

namespace tensorflow {

namespace {

using ::tensorflow::tpu::TpuNodeContext;

// This is a placeholder for an absl::Flag.
static bool tpu_cancellation_closes_chips = true;

int64_t ShapeSizeCompact(const xla::Shape& shape) {
  XLA_Shape c_shape;
  ApiConverter::ToC(shape, &c_shape);
  int64_t size =
      stream_executor::tpu::OpsApiFn()->HardwareLayout_ShapeSizeCompactFn(
          &c_shape);
  ApiConverter::Destroy(&c_shape);
  return size;
}

int64_t ShapeSizeCompactRaw(const xla::Shape& shape) {
  XLA_Shape c_shape;
  ApiConverter::ToC(shape, &c_shape);
  int64_t size =
      stream_executor::tpu::OpsApiFn()->HardwareLayout_ShapeSizeCompactRawFn(
          &c_shape);
  ApiConverter::Destroy(&c_shape);
  return size;
}

// Given a tuple, fix all non-leaf nodes (tuples) such that the tuple tables
// point to the correct leaf nodes.
absl::Status FixTupleTableAsync(se::Stream* stream,
                                const xla::Shape& tuple_shape,
                                xla::ExecutionInput* mem,
                                xla::TransferManager* transfer_manager) {
  return xla::ShapeUtil::ForEachSubshapeWithStatus(
      tuple_shape,
      [&](const xla::Shape& element_shape,
          const xla::ShapeIndex& index) -> absl::Status {
        if (!element_shape.IsTuple()) {
          return absl::OkStatus();
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
  if (dynamic_shape.dimensions_size() != bounded_shape.dimensions_size()) {
    return false;
  }
  for (int64_t i = 0; i < dynamic_shape.dimensions_size(); ++i) {
    if (dynamic_shape.dimensions(i) > bounded_shape.dimensions(i)) {
      return false;
    }
  }
  return true;
}

// For dynamic inputs, copy them and attach metadata of shape sizes to the
// beginning of the tensor.
//
// The buffer for dynamic shapes contains three parts:
// +--------+
// |Metadata|
// +--------+
// |Payload |
// +--------+
// |Padding |
// +--------+
//
// Metadata contains the sizes of shape without padding, eventually
// representing the size of valid data.
absl::Status UpdateDynamicInputs(
    se::Stream* stream, se::DeviceMemoryAllocator* allocator,
    std::vector<xla::ExecutionInput>* runtime_inputs,
    const std::vector<xla::Shape>& compile_time_shapes) {
  TF_RET_CHECK(runtime_inputs->size() == compile_time_shapes.size());
  TF_ASSIGN_OR_RETURN(
      auto transfer_manager,
      xla::TransferManager::GetForPlatform(stream->parent()->GetPlatform()));
  for (int64_t i = 0; i < compile_time_shapes.size(); i++) {
    // TODO(yunxing): Iterating over thousands of elements can be slow. One way
    // to optimize for fast path without dynamic shapes is add a field in
    // compilation result indicating if dynamic input is presented.
    if (compile_time_shapes[i].is_static()) {
      continue;
    }
    auto& runtime_input = (*runtime_inputs)[i];
    xla::Shape compile_time_shapes_on_device =
        transfer_manager->HostShapeToDeviceShape(compile_time_shapes[i]);
    bool element_modified = false;
    TF_RETURN_IF_ERROR(xla::ShapeUtil::ForEachSubshapeWithStatus(
        compile_time_shapes_on_device,
        [&](const xla::Shape& compile_time_shape,
            const xla::ShapeIndex& index) -> absl::Status {
          if (compile_time_shape.IsTuple() || compile_time_shape.is_static()) {
            return absl::OkStatus();
          }

          const xla::Shape& runtime_shape =
              xla::ShapeUtil::GetSubshape(runtime_input.shape(), index);

          TF_RET_CHECK(!runtime_shape.IsTuple());
          TF_RET_CHECK(
              DynamicShapeIsCompatible(runtime_shape, compile_time_shape));

          xla::MaybeOwningDeviceMemory* mutable_input_mem =
              runtime_input.MutableBuffer(index);
          auto padded_data = std::make_shared<std::vector<int8_t>>(
              ShapeSizeCompact(compile_time_shape), -1);
          auto raw_input_runtime = std::make_shared<std::vector<uint32_t>>(
              ShapeSizeCompact(runtime_shape) / sizeof(uint32_t));
          TF_RETURN_IF_ERROR(stream->MemcpyD2H(
              se::DeviceMemory<int8_t>(mutable_input_mem->AsDeviceMemoryBase()),
              absl::MakeSpan(absl::bit_cast<int8_t*>(raw_input_runtime->data()),
                             ShapeSizeCompactRaw(runtime_shape))));
          TF_RETURN_IF_ERROR(stream->DoHostCallbackWithStatus(
              [raw_input_runtime, padded_data, runtime_shape,
               compile_time_shape]() {
                // After getting the data onto the host, transpose the data to
                // the correct layout by delinearizing it and linearizing it
                // again.
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

                stream_executor::tpu::OpsApiFn()
                    ->TpuExecute_RuntimeInputToPaddedDataFn(&params);
                ApiConverter::Destroy(&c_runtime_shape);
                ApiConverter::Destroy(&c_compile_time_shape);
                return status.status();
              }));
          // Allocate new input and transfer the padded and transposed data to
          // the new input location.
          TF_ASSIGN_OR_RETURN(
              auto new_input,
              allocator->Allocate(stream->parent()->device_ordinal(),
                                  ShapeSizeCompact(compile_time_shape)));
          auto typed_new_input_memory =
              se::DeviceMemory<int8_t>(new_input.cref());
          TF_RETURN_IF_ERROR(
              stream->MemcpyH2D<int8_t>(*padded_data, &typed_new_input_memory));

          // Retain the memory until the end of the transfer.
          TF_RETURN_IF_ERROR(stream->DoHostCallback([padded_data] {}));

          // Modify the memory location in the input shape tree to point to the
          // new input.
          *mutable_input_mem =
              xla::MaybeOwningDeviceMemory(std::move(new_input));
          element_modified = true;
          return absl::OkStatus();
        }));
    if (element_modified) {
      // The input location has been modified, need to fix tuple table to
      // point to the correct address.
      TF_RETURN_IF_ERROR(FixTupleTableAsync(stream,
                                            compile_time_shapes_on_device,
                                            &runtime_input, transfer_manager));
    }
  }
  return absl::OkStatus();
}

void TPUCancelExecution(int device_ordinal) {
  if (tpu_cancellation_closes_chips) {
    LOG(INFO) << "TPUCancelExecution CloseTPUHost on device " << device_ordinal;
    absl::Status status = TpuNodeContext::CloseTpuHost();
    LOG(INFO) << "TPUCancelExecution CloseTPUHost done: " << status
              << " on device " << device_ordinal;
  } else {
    LOG(INFO) << "TPUCancelExecution CloseTPUHost on device " << device_ordinal
              << " is suppressed";
  }
}

std::pair<CancellationToken, bool> RegisterCancellation(
    CancellationManager* cancellation_manager, int device_ordinal) {
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
  bool already_cancelled =
      !cancellation_manager->RegisterCallbackWithErrorLogging(
          token, [device_ordinal]() { TPUCancelExecution(device_ordinal); },
          absl::StrCat("TPUCancellation on device ", device_ordinal));
  return std::pair<CancellationToken, bool>(token, already_cancelled);
}

struct DestroyOCParams {
  void operator()(SE_OutsideCompilationParams* params) {
    if (params == nullptr) {
      return;
    }
    delete[] params->device_name;
    delete[] params->rendezvous_key;
    Destroy(params->rendezvous);
    delete params->rendezvous;
    if (params->host_transfers.size > 0) {
      StreamExecutor_Tpu_FreeSerializedProto(&params->host_transfers);
    }
    delete params;
  }
};

typedef std::unique_ptr<SE_OutsideCompilationParams, DestroyOCParams>
    OcParamsPtr;

void UnregisterCancellation(OpKernelContext* ctx,
                            CancellationManager* cancellation_manager,
                            se::Stream* stream, int device_ordinal,
                            CancellationToken token) {
  // If execution reaches this point, the host callback enqueued below will get
  // called regardless of stream status. Call inc_num_deferred_ops_function here
  // and dec_num_deferred_ops_function in the host callback.
  ctx->inc_num_deferred_ops_function()();
  auto dec_num_deferred_ops_function = ctx->dec_num_deferred_ops_function();

  // Try to avoid running callbacks on the compute stream, because this reduces
  // the frequency of back-to-back programs (which are most efficient because
  // they don't require host synchronization). Instead, borrow a substream and
  // have the substream wait on the compute stream.
  se::Stream* deregister_stream =
      stream->GetOrCreateSubStream().value_or(nullptr);
  if (deregister_stream == nullptr) {
    return;
  }
  deregister_stream->WaitFor(stream).IgnoreError();
  deregister_stream
      ->DoHostCallback([=]() {
        // We must deregister the callback in the success case, to avoid closing
        // all devices. In the failure case we must NOT call DeregisterCallback
        // as that waits for all previous cancellation callbacks to complete and
        // any call to XlaDevice::Sync() will cause deadlock. Consider:
        //   1) CancellationManager::StartCancel() is in progress (state is
        //      cancelling_).
        //   2) The call below to DeregisterCallback will block until state is
        //   cancelled_ (all callbacks are completed).
        //   3) A different cancellation callback has called XlaDevice::Sync(),
        //   which will block until (2) is done.
        //   4) StartCancel() in (1) cannot complete until (3) is done.
        //
        // Instead, call TryDeregisterCallback. The functional difference is
        // TryDeregisterCallback will not block if cancellation is in progress
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
        // dec_num_deferred_ops_function are called, ExecutorState::Finish will
        // be allowed to proceed.
        dec_num_deferred_ops_function();
      })
      .IgnoreError();
  stream->ReturnSubStream(deregister_stream);
}

OcParamsPtr CreateOcParams(const std::string& rendezvous_key_base,
                           OpKernelContext* op_kernel_context,
                           const TPUHostTransferInfoProto& host_transfers) {
  OcParamsPtr oc_params(new SE_OutsideCompilationParams());
  const std::string& device_name = op_kernel_context->device()->name();
  oc_params->device_name = new char[device_name.size() + 1];
  std::strncpy(oc_params->device_name, device_name.c_str(),
               device_name.size() + 1);
  oc_params->rendezvous_key = new char[rendezvous_key_base.size() + 1];
  std::strncpy(oc_params->rendezvous_key, rendezvous_key_base.c_str(),
               rendezvous_key_base.size() + 1);
  oc_params->rendezvous = ToC(op_kernel_context->rendezvous());
  oc_params->host_transfers =
      stream_executor::tpu::SerializeProto(host_transfers);
  return oc_params;
}

}  // namespace

absl::StatusOr<xla::ExecutionOutput> TPUExecute(
    const TPUExecutableInfoProto& executable,
    const TPUHostTransferInfoProto& host_transfers,
    const xla::HloProto& hlo_metadata,
    std::vector<xla::ExecutionInput> arguments,
    const std::string& rendezvous_key_base, uint32_t rng_seed,
    TpuNodeContext* node_context, xla::DeviceAssignment* device_assignment,
    CancellationManager* cancellation_manager, OpKernelContext* ctx,
    stream_executor::Stream* stream,
    stream_executor::Stream* host_to_device_stream,
    const XLA_TpuProgram* tpu_program) {
  tsl::profiler::TraceMe traceme("TPUExecute", 2);
  TF_RET_CHECK(tpu::TpuPlatformInterface::GetRegisteredPlatform() != nullptr);
  TF_RET_CHECK(tpu_program != nullptr);
  const int device_ordinal = node_context->device_ordinal();
  VLOG(1) << "TPUExecute on device " << device_ordinal;
  xla::Backend* backend = node_context->backend();

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
    module = std::make_unique<xla::HloModule>(
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
        xla::ShapeIndex(prefetch.index().begin(), prefetch.index().end()),
        prefetch.offset());
  }

  VLOG(1) << "TPUExecute: Updating dynamic HLO inputs on " << device_ordinal;

  TF_RETURN_IF_ERROR(UpdateDynamicInputs(stream, backend->memory_allocator(),
                                         &arguments, input_shapes));

  // Retrieve the TPU embedding memory addresses to be fed to the TPU. The
  // memory addresses are communicated with a dynamically allocated C array
  // (which needs to be free'd once the function terminates).
  VLOG(1) << "TPUExecute: Updating TPUEmbedding memory addresses on "
          << device_ordinal;

  SE_DeviceMemoryBase* device_memory_addrs = nullptr;
  size_t device_memory_addrs_count;
  auto device_memory_cleanup =
      absl::MakeCleanup([device_memory_addrs, device_ordinal]() {
        if (device_memory_addrs != nullptr) {
          stream_executor::tpu::OpsApiFn()
              ->TpuExecute_FreeTpuEmbeddingMemoryAllocationsFn(
                  device_ordinal, device_memory_addrs);
        }
      });

  StatusHelper status;
  stream_executor::tpu::OpsApiFn()
      ->TpuExecute_GetTpuEmbeddingMemoryAllocationsFn(
          device_ordinal, &device_memory_addrs, &device_memory_addrs_count,
          status.c_status);
  if (!status.ok()) {
    return status.status();
  }

  // Add the TPU embedding memory addresses as additional arguments for the TPU
  // executable.
  VLOG(1) << "TPUExecute: Adding " << device_memory_addrs_count
          << " TPUEmbedding memory addresses to HLO parameters.";
  for (int i = 0; i < device_memory_addrs_count; ++i) {
    xla::ShapeTree<xla::MaybeOwningDeviceMemory> tree(
        xla::ShapeUtil::MakeOpaqueShape());
    const SE_DeviceMemoryBase& addr = device_memory_addrs[i];
    VLOG(2) << absl::StrFormat("Device memory addr[%i] = {%p, %llu, %llu}", i,
                               addr.opaque, addr.size, addr.payload);
    *tree.mutable_element({}) = ApiConverter::FromC(addr);
    xla::ExecutionInput input(std::move(tree));
    arguments.push_back(std::move(input));
  }

  OcParamsPtr oc_params =
      CreateOcParams(rendezvous_key_base, ctx, host_transfers);

  auto tpu_executable = std::make_unique<TpuOpExecutable>(
      tpu_program, std::move(module), oc_params.get());

  CancellationToken token;
  bool already_cancelled;
  std::tie(token, already_cancelled) =
      RegisterCancellation(cancellation_manager, device_ordinal);

  // If the RPC was already cancelled before we managed to register the
  // cancellation callback, we shouldn't attempt to run the TPU program, since
  // it might block forever.
  if (already_cancelled) {
    return absl::CancelledError(absl::StrCat(
        "RPC cancelled, not running TPU program on device ", device_ordinal));
  }

  absl::StatusOr<xla::ExecutionOutput> output =
      tpu_executable->ExecuteAsyncOnStream(&service_run_options,
                                           std::move(arguments));

  // If !output.ok(), it means we failed to enqueue the program the TPU. This is
  // possibly caused by a failed cancellation callback closing the chips.
  if (!output.ok()) {
    // If cancellation manager is already cancelled or cancelling, it means
    // another failure has occurred earlier and this TpuExecuteOp is cancelled
    // regardless of whether itself is an error.
    already_cancelled = cancellation_manager->IsCancelling() ||
                        cancellation_manager->IsCancelled();
    if (already_cancelled) {
      return absl::CancelledError(absl::StrCat(
          "RPC cancelled, not running TPU program on device ", device_ordinal));
    }
  }
  UnregisterCancellation(ctx, cancellation_manager, stream, device_ordinal,
                         token);

  VLOG(1) << "Cloud TPU: TPUExecute done";
  return output;
}

}  // namespace tensorflow
