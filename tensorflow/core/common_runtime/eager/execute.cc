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

#include "tensorflow/core/common_runtime/eager/execute.h"

#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/copy_to_device_node.h"
#include "tensorflow/core/common_runtime/eager/execute_node.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/lib/core/errors.h"
#ifndef __ANDROID__
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/distributed_runtime/eager/remote_execute_node.h"
#endif
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {

namespace {

// Copy of the definition in third_party/tensorflow/compiler/jit/defs.h
// Copied here because we don't currently compile XLA on windows. So, can't
// depend on it directly.
const char* const kXlaCompileAttr = "_XlaCompile";

// Initializes the step stats if needed.
void MaybeInitializeStepStats(StepStats* step_stats, EagerContext* ctx) {
  // Lazily initialize the RunMetadata with information about all devices if
  // this is the first call.
  while (step_stats->dev_stats_size() < ctx->devices()->size()) {
    int device_idx = step_stats->dev_stats_size();
    auto* dev_stats = step_stats->add_dev_stats();
    dev_stats->set_device(ctx->devices()->at(device_idx)->name());
  }
}

int StepStatsDeviceIndex(StepStats* step_stats, EagerContext* ctx,
                         Device* device) {
  // Find the current device's index.
  if (device == nullptr) {
    device = ctx->HostCPU();
  }
  for (int i = 0; i < ctx->devices()->size(); ++i) {
    if (ctx->devices()->at(i) == device ||
        ctx->devices()->at(i)->name() == device->name()) {
      return i;
    }
  }
  // TODO(apassos) do not fall back to host CPU if device is unknown.
  return 0;
}

// This function expects *handle to point to an existing tensor handle. The
// function will (maybe) update the *handle to be pointed to the newly copied
// tensor handle.
//
// The passed in *handle will be Unreffed if it is replaced.
Status MaybeCopyInputToExpectedDevice(EagerOperation* op, int i,
                                      const Device* expected_device,
                                      RunMetadata* run_metadata,
                                      TensorHandle** handle) {
  EagerContext* ctx = op->EagerContext();
  Device* handle_device = (*handle)->device();
  const Device* actual_device =
      handle_device == nullptr ? ctx->HostCPU() : handle_device;
  const Device* op_device =
      op->Device() == nullptr ? ctx->HostCPU() : op->Device();

  if (expected_device != actual_device) {
    switch (ctx->GetDevicePlacementPolicy()) {
      case DEVICE_PLACEMENT_SILENT_FOR_INT32:
        // TODO(xpan): See if we could bubble python related error up
        // to python level.
        if ((*handle)->dtype == DT_INT32) {
          // Note: enabling silent copies of int32 tensors to match behavior
          // of graph mode.
          break;
        }
        TF_FALLTHROUGH_INTENDED;
      case DEVICE_PLACEMENT_EXPLICIT:
        return errors::InvalidArgument(
            "Tensors on conflicting devices:"
            " cannot compute ",
            op->Name(), " as input #", i, " was expected to be on ",
            expected_device->name(), " but is actually on ",
            actual_device->name(), " (operation running on ", op_device->name(),
            ")",
            " Tensors can be copied explicitly using .gpu() or .cpu() "
            "methods,"
            " or transparently copied by using tf.enable_eager_execution("
            "device_policy=tfe.DEVICE_PLACEMENT_SILENT). Copying tensors "
            "between devices"
            " may slow down your model");
      case DEVICE_PLACEMENT_WARN:
        LOG(WARNING) << "before computing " << op->Name() << " input #" << i
                     << " was expected to be on " << expected_device->name()
                     << " but is actually on " << actual_device->name()
                     << " (operation running on " << op_device->name()
                     << "). This triggers a copy which can be a performance "
                        "bottleneck.";
        break;
      case DEVICE_PLACEMENT_SILENT:  // Do nothing.
        break;
    }
    // We are only here if the policy is warn or silent copies, so we should
    // trigger a copy.
    auto pre_time_nanos = Env::Default()->NowNanos();
    TensorHandle* result_handle = nullptr;
    Status status = EagerCopyToDevice(
        *handle, ctx, expected_device->name().c_str(), &result_handle);
    if (run_metadata != nullptr) {
      auto* step_stats = run_metadata->mutable_step_stats();
      MaybeInitializeStepStats(step_stats, ctx);
      // Record the sending on the source device for now.
      int device_idx = StepStatsDeviceIndex(step_stats, ctx, handle_device);
      auto* dev_stats = step_stats->mutable_dev_stats(device_idx);
      auto* node_stats = dev_stats->add_node_stats();
      node_stats->set_node_name("_Send");
      node_stats->set_all_start_micros(pre_time_nanos /
                                       EnvTime::kMicrosToNanos);
      node_stats->set_all_start_nanos(pre_time_nanos);
      int64 now_nanos = Env::Default()->NowNanos();
      node_stats->set_op_end_rel_micros((now_nanos - pre_time_nanos) /
                                        EnvTime::kMicrosToNanos);
      node_stats->set_op_end_rel_nanos(now_nanos - pre_time_nanos);
      node_stats->set_all_end_rel_micros((now_nanos - pre_time_nanos) /
                                         EnvTime::kMicrosToNanos);
      node_stats->set_all_end_rel_nanos(now_nanos - pre_time_nanos);
    }
    if (!status.ok()) {
      if (result_handle != nullptr) result_handle->Unref();
      return errors::Internal("Failed copying input tensor from ",
                              actual_device->name(), " to ",
                              expected_device->name(), " in order to run ",
                              op->Name(), ": ", status.error_message());
    }

    (*handle)->Unref();
    *handle = result_handle;
  }
  return Status::OK();
}

Status ValidateInputTypeAndPlacement(EagerContext* ctx, Device* op_device,
                                     EagerOperation* op, const OpKernel* kernel,
                                     RunMetadata* run_metadata) {
  Device* host_device = ctx->HostCPU();
  const MemoryTypeVector& memtypes = kernel->input_memory_types();
  if (memtypes.size() != op->Inputs().size()) {
    return errors::InvalidArgument("expected ", memtypes.size(),
                                   " inputs, got ", op->Inputs().size());
  }
  for (int i = 0; i < op->Inputs().size(); ++i) {
    const Device* expected_device =
        memtypes[i] == HOST_MEMORY ? host_device : op_device;
    TF_RETURN_IF_ERROR(MaybeCopyInputToExpectedDevice(
        op, i, expected_device, run_metadata, &((*op->MutableInputs())[i])));
    tensorflow::TensorHandle* handle = op->Inputs()[i];
    if (handle->dtype != kernel->input_type(i)) {
      return errors::InvalidArgument(
          "cannot compute ", op->Name(), " as input #", i, "(zero-based)",
          " was expected to be a ", DataTypeString(kernel->input_type(i)),
          " tensor but is a ", DataTypeString(handle->dtype), " tensor");
    }
  }
  return Status::OK();
}

Status SelectDevice(const NodeDef& ndef, EagerContext* ctx, Device** device) {
  PrioritizedDeviceTypeVector final_devices;
  TF_RETURN_IF_ERROR(SupportedDeviceTypesForNode(
      ctx->prioritized_device_type_list(), ndef, &final_devices));
  if (final_devices.empty()) {
    return errors::Internal("Could not find valid device for node.\nNode: ",
                            FormatNodeDefForError(ndef),
                            "\nAll kernels registered for op ", ndef.op(),
                            " :\n", KernelsRegisteredForOp(ndef.op()));
  }
  for (Device* d : *ctx->devices()) {
    if (d->device_type() == final_devices[0].first.type_string()) {
      *device = d;
      return Status::OK();
    }
  }
  return errors::Unknown("Could not find a device for node ",
                         FormatNodeDefForError(ndef));
}

Status GetOutputDTypes(EagerOperation* op, DataTypeVector* output_dtypes) {
  const auto& node_def = op->MutableAttrs()->BuildNodeDef();
  const OpDef* op_def = nullptr;

  const FunctionDef* function_def =
      op->EagerContext()->FuncLibDef()->Find(op->Name());
  if (function_def != nullptr) {
    op_def = &(function_def->signature());
  } else {
    TF_RETURN_IF_ERROR(OpDefForOp(op->Name().c_str(), &op_def));
  }

  TF_RETURN_IF_ERROR(OutputTypesForNode(node_def, *op_def, output_dtypes));

  return Status::OK();
}

}  // namespace

namespace {
bool IsLocal(EagerContext* ctx, tensorflow::Device* d) {
  if (d == nullptr || ctx->remote_device_mgr() == nullptr) return true;
  tensorflow::Device* tmp;
  return ctx->local_device_mgr()->LookupDevice(d->name(), &tmp).ok();
}

bool OnSameTask(EagerContext* ctx, Device* first, Device* second) {
  if (first == nullptr) first = ctx->HostCPU();
  if (second == nullptr) second = ctx->HostCPU();
  return first->parsed_name().job == second->parsed_name().job &&
         first->parsed_name().replica == second->parsed_name().replica &&
         first->parsed_name().task == second->parsed_name().task;
}

Status EagerLocalExecute(EagerOperation* op,
                         gtl::InlinedVector<TensorHandle*, 2>* retvals,
                         int* num_retvals) {
  EagerContext* ctx = op->EagerContext();
  auto status = ctx->GetStatus();
  if (!status.ok()) return status;
  Device* device = op->Device();

  Fprint128 cache_key = op->MutableAttrs()->CacheKey(
      device == nullptr ? "unspecified" : device->name());
  KernelAndDevice* kernel = ctx->GetCachedKernel(cache_key);
  if (kernel == nullptr) {
    // If we are running a function on explicitly requested TPU,
    // compile it with XLA.
    // Note that it is not ideal, but currently ok, to set this
    // attribute after computing the kernel cache key above.
    if (op->is_function() && device != nullptr &&
        (device->device_type() == "TPU" || device->device_type() == "XLA_GPU" ||
         device->device_type() == "XLA_CPU")) {
      op->MutableAttrs()->Set(kXlaCompileAttr, true);
    }

    const NodeDef& ndef = op->MutableAttrs()->BuildNodeDef();
    if (device == nullptr) {
      status = SelectDevice(ndef, ctx, &device);
      if (!status.ok()) return status;
    }
    CHECK(device != nullptr);
    if (ctx->LogDevicePlacement()) {
      LOG(INFO) << "Executing op " << ndef.op() << " in device "
                << device->name();
    }

    auto* flr = ctx->func_lib(device);
    if (flr == nullptr) {
      return errors::Unavailable(
          "Unable to find a FunctionLibraryRuntime corresponding to device ",
          device->name());
    }
    kernel = new KernelAndDevice(ctx->GetRendezvous(), ctx->LogMemory(),
                                 ctx->GetCollectiveExecutorHandle());
    status = KernelAndDevice::Init(ndef, flr, ctx->runner(), kernel);
    if (!status.ok()) {
      delete kernel;
      return status;
    }

    ctx->AddKernelToCache(cache_key, kernel);
  }
  const DataTypeVector& output_dtypes = kernel->output_dtypes();
  const int output_dtypes_size = static_cast<int>(output_dtypes.size());
  if (output_dtypes_size > *num_retvals) {
    return errors::InvalidArgument("Expecting ", output_dtypes.size(),
                                   " outputs, but *num_retvals is ",
                                   *num_retvals);
  }
  *num_retvals = output_dtypes_size;
  if (device == nullptr) {
    // TODO(apassos) debug how the assignment below might return a different
    // device from the one requested above.
    device = kernel->device();
  }
  status = ValidateInputTypeAndPlacement(
      ctx, device, op, kernel->kernel(),
      ctx->ShouldStoreMetadata() ? ctx->RunMetadataProto() : nullptr);
  if (!status.ok()) return status;
  std::unique_ptr<NodeExecStats> maybe_stats;
  StepStats* maybe_step_stats = nullptr;
  GraphCollector* graph_collector = nullptr;
  if (ctx->ShouldStoreMetadata()) {
    graph_collector = ctx->GetGraphCollector();
    maybe_step_stats = ctx->RunMetadataProto()->mutable_step_stats();
    int64 now_nanos = Env::Default()->NowNanos();
    maybe_stats.reset(new NodeExecStats);
    maybe_stats->set_node_name(op->Name());
    maybe_stats->set_all_start_micros(now_nanos / EnvTime::kMicrosToNanos);
    maybe_stats->set_all_start_nanos(now_nanos);
    maybe_stats->set_op_start_rel_micros(0);
    maybe_stats->set_op_start_rel_nanos(0);
    maybe_stats->set_scheduled_micros(now_nanos / EnvTime::kMicrosToNanos);
    maybe_stats->set_scheduled_nanos(now_nanos);
    // TODO(apassos) track referenced tensors
  }
  retvals->resize(*num_retvals);
  if (ctx->Async()) {
    // Note that for async mode, execution order will make sure that all
    // input handles are ready before executing them.
    // TODO(agarwal): Consider executing "cheap" kernels inline for performance.
    tensorflow::uint64 id = ctx->NextId();
    for (int i = 0; i < *num_retvals; ++i) {
      (*retvals)[i] = new TensorHandle(id, /* d= */ kernel->OutputDevice(i),
                                       /* op_device= */ kernel->device(),
                                       output_dtypes[i], ctx);
    }
    EagerNode* node = new ExecuteNode(
        id, ctx, op->Device(), op->Inputs(), kernel, maybe_stats.release(),
        maybe_step_stats, graph_collector, output_dtypes, *retvals);
    ctx->ExecutorAdd(node);
  } else {
    // Execute checks if retvals[i] is nullptr or not to figure if it needs to
    // allocate it.
    status = EagerExecute(ctx, op->Device(), op->Inputs(), kernel,
                          maybe_stats.get(), maybe_step_stats, graph_collector,
                          retvals->data(), *num_retvals);
  }

  return status;
}

#ifndef __ANDROID__
std::function<void()> GetRemoteTensorDestructor(
    EagerContext* ctx, eager::EagerClient* eager_client, uint64 context_id,
    uint64 op_id, int output_num) {
  return [ctx, eager_client, context_id, op_id, output_num]() {
    if (!ctx->HasActiveRemoteContext(context_id)) {
      // This means that this tensor was pointing to a remote device, which has
      // been changed out from under us. Simply return since there is nothing we
      // can do.
      return tensorflow::Status::OK();
    }

    std::unique_ptr<eager::EnqueueRequest> request(new eager::EnqueueRequest);
    request->set_context_id(context_id);

    auto* handle_to_decref = request->add_queue()->mutable_handle_to_decref();
    handle_to_decref->set_op_id(op_id);
    handle_to_decref->set_output_num(output_num);

    if (ctx->Async()) {
      tensorflow::uint64 id = ctx->NextId();
      auto* node =
          new eager::RemoteExecuteNode(id, std::move(request), eager_client);
      ctx->ExecutorAdd(node);
    } else {
      eager::EnqueueRequest* actual_request = request.release();
      eager::EnqueueResponse* response = new eager::EnqueueResponse;
      eager_client->EnqueueAsync(
          actual_request, response,
          [actual_request, response](const tensorflow::Status& s) {
            delete actual_request;
            delete response;
          });
    }

    return tensorflow::Status::OK();
  };
}
#endif

// When !ctx->UseSendTensorRPC(), then tensors are shipped between remote
// devices by the receiver invoking the WorkerService.RecvTensor RPC *on the
// sender* (Rendezvous::RecvAsync() invoked by the _Recv kernel).
//
// However, in some configurations the node that has the tensor to be copied
// isn't running a server (WorkerService RPC interface). For such cases,
// this function enables sending tensors using the EagerService.SendTensor RPC
// *on the receiver*.
Status EagerRemoteSendTensor(EagerContext* ctx, TensorHandle* h,
                             Device* recv_device, TensorHandle** result) {
#ifdef __ANDROID__
  return errors::Unimplemented(
      "Eager's remote execution is not available on Android devices.");
#else
  eager::EagerClient* eager_client;
  uint64 context_id;
  TF_RETURN_IF_ERROR(
      ctx->GetClientAndContextID(recv_device, &eager_client, &context_id));

  eager::SendTensorRequest request;
  eager::SendTensorResponse response;

  request.set_context_id(context_id);
  request.set_op_id(ctx->NextId());
  request.set_device_name(recv_device->name());

  Device* tensor_handle_device = h->device();

  // AsProtoTensorContent doesn't work when the tensor is on the GPU, hence copy
  // it to the CPU before copying it out.
  // TODO(nareshmodi): this is currently slow, but can be fixed by making tensor
  // handles aware of more than one device.
  TensorHandle* actual_handle;
  if (tensor_handle_device != nullptr &&
      tensor_handle_device->device_type() != "CPU") {
    TF_RETURN_IF_ERROR(h->CopyToDevice(ctx, ctx->HostCPU(), &actual_handle));
  } else {
    actual_handle = h;
    actual_handle->Ref();
  }

  const Tensor* tensor;
  TF_RETURN_IF_ERROR(actual_handle->Tensor(&tensor));
  tensor->AsProtoTensorContent(request.add_tensors());

  const tensorflow::uint64 id = request.op_id();

  // TODO(nareshmodi): support making this call async.
  Notification n;
  Status status;
  eager_client->SendTensorAsync(&request, &response,
                                [&n, &status](const Status& s) {
                                  status = s;
                                  n.Notify();
                                });
  n.WaitForNotification();
  if (!status.ok()) return status;

  std::function<void()> destructor =
      GetRemoteTensorDestructor(ctx, eager_client, context_id, id, 0);

  *result = new TensorHandle(id, /*output_num=*/0, /*remote_shape_node_id=*/0,
                             tensor->dtype(), std::move(destructor),
                             recv_device, recv_device, ctx);
  (*result)->SetRemoteShape(MakeUnique<TensorShape>(tensor->shape()));

  actual_handle->Unref();

  return Status::OK();
#endif
}

Status EagerRemoteExecute(EagerOperation* op, TensorHandle** retvals,
                          int* num_retvals) {
#ifdef __ANDROID__
  return errors::Unimplemented(
      "Eager's remote execution is not available on Android devices.");
#else
  EagerContext* ctx = op->EagerContext();

  eager::EagerClient* eager_client;
  uint64 context_id;
  TF_RETURN_IF_ERROR(
      ctx->GetClientAndContextID(op->Device(), &eager_client, &context_id));

  std::unique_ptr<eager::EnqueueRequest> request(new eager::EnqueueRequest);
  eager::EnqueueResponse response;

  request->set_context_id(context_id);

  auto* remote_op = request->add_queue()->mutable_operation();

  for (int i = 0; i < op->Inputs().size(); i++) {
    tensorflow::Device* input_device = op->Inputs()[i]->device();
    if (op->Device() != input_device &&
        // If the expected and actual devices are on the same task, don't
        // explicitly copy, and instead depend on the copy to happen locally
        // when the op is executed on the device.
        !OnSameTask(ctx, op->Device(), input_device)) {
      // TODO(b/110044833): It's possible the same tensor gets copied to the
      // remote device repeatedly.
      TF_RETURN_IF_ERROR(MaybeCopyInputToExpectedDevice(
          op, i, op->Device(), /* run_metadata= */ nullptr,
          &(*op->MutableInputs())[i]));
    }

    tensorflow::TensorHandle* input = op->Inputs()[i];

    tensorflow::int64 op_id;
    int32 output_num;
    TF_RETURN_IF_ERROR(input->RemoteAddress(&op_id, &output_num));

    auto* remote_op_input = remote_op->add_inputs();
    remote_op_input->set_op_id(op_id);
    remote_op_input->set_output_num(output_num);
  }

  remote_op->set_id(op->EagerContext()->NextId());
  remote_op->set_name(op->Name());
  // Inputs set above.
  op->Attrs().FillAttrValueMap(remote_op->mutable_attrs());
  remote_op->set_device(op->Device()->name());

  DataTypeVector output_dtypes;
  TF_RETURN_IF_ERROR(GetOutputDTypes(op, &output_dtypes));

  if (*num_retvals != output_dtypes.size()) {
    return errors::InvalidArgument(
        "num_retvals does not match expected output dtypes");
  }

  tensorflow::Device* op_device = op->Device();

  bool is_async = op->EagerContext()->Async();
  uint64 remote_node_id = 0;

  if (is_async) {
    remote_node_id = op->EagerContext()->NextId();
  }

  const tensorflow::uint64 id = remote_op->id();
  for (int i = 0; i < *num_retvals; i++) {
    // TODO(nareshmodi): Change the callback to instead add the decref to a list
    // of pending decrefs that we can send as a batch with the next execute.
    std::function<void()> destructor =
        GetRemoteTensorDestructor(ctx, eager_client, context_id, id, i);

    retvals[i] = new TensorHandle(remote_op->id(), i, remote_node_id,
                                  output_dtypes[i], std::move(destructor),
                                  op_device, op_device, op->EagerContext());
  }

  if (is_async) {
    // Copy the output handles, since the container for them might get
    // destroyed.
    gtl::InlinedVector<TensorHandle*, 2> retvals_copy;
    for (int i = 0; i < *num_retvals; i++) {
      retvals_copy.push_back(retvals[i]);
      retvals_copy[i]->Ref();
    }
    // Unable to capture via std::move, so bind instead.
    auto* node = new eager::RemoteExecuteNode(
        remote_node_id, std::move(request), eager_client, op->Inputs(),
        std::bind(
            [](const gtl::InlinedVector<TensorHandle*, 2>& retvals,
               const Status& status, const eager::EnqueueResponse& response) {
              if (!status.ok()) return;
              for (int i = 0; i < retvals.size(); i++) {
                retvals[i]->SetRemoteShape(MakeUnique<TensorShape>(
                    response.queue_response(0).shape(i)));
                retvals[i]->Unref();
              }
            },
            std::move(retvals_copy), std::placeholders::_1,
            std::placeholders::_2));
    op->EagerContext()->ExecutorAdd(node);
  } else {
    Notification n;
    Status status;
    eager_client->EnqueueAsync(request.get(), &response,
                               [&n, &status](const Status& s) {
                                 status = s;
                                 n.Notify();
                               });
    n.WaitForNotification();

    if (!status.ok()) return status;

    for (int i = 0; i < *num_retvals; i++) {
      retvals[i]->SetRemoteShape(
          MakeUnique<TensorShape>(response.queue_response(0).shape(i)));
    }
  }

  return Status::OK();
#endif
}

// These ops are not pinnable since they generate data. It can be slower to
// generate and then copy the data instead of just generating the data on the
// device directly.
bool IsPinnableOp(const string& op_type) {
  static const gtl::FlatSet<string>* unpinnable_ops = new gtl::FlatSet<string>({
      "RandomUniform",
      "RandomUniformInt",
      "RandomNormal",
      "StatelessRandomUniform",
      "StatelessRandomUniformInt",
      "StatelessRandomNormal",
  });

  return unpinnable_ops->find(op_type) == unpinnable_ops->end();
}

// The Op device may be updated if:
// - A resource touching input is specified: all resource-touching ops run in
// the device the resource is, regardless of anything else that has been
// specified. This is identical to the graph mode behavior.
//
// - All op inputs are on the CPU, small (<64 elements) and integers
// (int32/int64). This can be disabled by setting the environment variable
// "TF_EAGER_ENABLE_SMALL_TENSOR_CPU_PINNING" to "0" or "false".
Status MaybeUpdateOpDevice(EagerOperation* op) {
  EagerContext* ctx = op->EagerContext();
  bool device_set_for_resource_variable = false;
  bool all_inputs_eligible_for_cpu_pinning =
      ctx->PinSmallOpsToCPU() && IsPinnableOp(op->Name());

  for (int i = 0; i < op->Inputs().size(); ++i) {
    Device* input_op_device = op->Inputs()[i]->op_device();
    VLOG(2) << "for op " << op->Name() << " input " << i << " "
            << DataTypeString(op->Inputs()[i]->dtype) << " "
            << (input_op_device == nullptr ? "cpu" : input_op_device->name())
            << " " << (op->Device() == nullptr ? "cpu" : op->Device()->name());
    if (op->Inputs()[i]->dtype == DT_RESOURCE &&
        (input_op_device != op->Device() || input_op_device == nullptr)) {
      Device* d = input_op_device == nullptr ? ctx->HostCPU() : input_op_device;
      VLOG(1) << "Changing device of operation " << op->Name() << " to "
              << d->name() << " because input #" << i
              << " is a resource in this device.";
      op->SetDevice(d);

      device_set_for_resource_variable = true;
      all_inputs_eligible_for_cpu_pinning = false;
    } else if (all_inputs_eligible_for_cpu_pinning) {
      TensorHandle* handle = op->Inputs()[i];

      // Input is on CPU.
      if (input_op_device != nullptr && input_op_device != ctx->HostCPU()) {
        all_inputs_eligible_for_cpu_pinning = false;
        continue;
      }

      if (handle->dtype != DataType::DT_INT32 &&
          handle->dtype != DataType::DT_INT64) {
        all_inputs_eligible_for_cpu_pinning = false;
        continue;
      }

      int64 num_elements;
      TF_RETURN_IF_ERROR(handle->NumElements(&num_elements));
      if (num_elements > 64) {
        all_inputs_eligible_for_cpu_pinning = false;
      }
    }
  }

  // Ops without inputs are usually ops that generate a tensor in some way and
  // usually require being present on whatever device they are scheduled on
  // - for e.g. VarHandleOp or _Recv).
  // TODO(nareshmodi): Is it possible there is no int32/int64 CPU kernel for
  // an op, but there is a GPU kernel?
  if (!op->Inputs().empty() && all_inputs_eligible_for_cpu_pinning) {
    VLOG(1) << "Forcing op " << op->Name()
            << " to be on the CPU since all input tensors have an "
               "int32/int64 dtype, and are small (less than 64 elements).";
    op->SetDevice(ctx->HostCPU());
  }

  return Status::OK();
}
}  // namespace

Status EagerExecute(EagerOperation* op,
                    gtl::InlinedVector<TensorHandle*, 2>* retvals,
                    int* num_retvals) {
  TF_RETURN_IF_ERROR(MaybeUpdateOpDevice(op));

  bool op_is_local = IsLocal(op->EagerContext(), op->Device());

  if (op_is_local) {
    return EagerLocalExecute(op, retvals, num_retvals);
  }

  if (op->EagerContext()->LogDevicePlacement()) {
    LOG(INFO) << "Executing op " << op->Name() << " in device "
              << op->Device()->name();
  }

  return EagerRemoteExecute(op, retvals->data(), num_retvals);
}

Status EagerExecute(EagerContext* ctx, Device* device,
                    const gtl::InlinedVector<TensorHandle*, 4>& op_inputs,
                    KernelAndDevice* kernel, NodeExecStats* maybe_stats,
                    StepStats* maybe_step_stats,
                    GraphCollector* graph_collector, TensorHandle** retvals,
                    int num_retvals) {
  if (device == nullptr) {
    // TODO(apassos) debug how the assignment below might return a different
    // device from the one requested above.
    device = kernel->device();
  }

  std::vector<Tensor> outputs(1);
  const MemoryTypeVector* output_memory_types = nullptr;
  output_memory_types = &kernel->kernel()->output_memory_types();

  // If there are multiple references to a TensorHandle in 'op_inputs' we must
  // increment the reference count of the corresponding Tensor or risk it being
  // overwritten during kernel execution. The reference count is incremented
  // below when we insert a copy of the Tensor into protected_tensors, and will
  // be decremented once execution is complete.
  std::vector<tensorflow::Tensor> protected_tensors;
  for (int i = 0; i < op_inputs.size(); ++i) {
    if (!op_inputs[i]->RefCountIsOne()) {
      const Tensor* input_tensor = nullptr;
      TF_RETURN_IF_ERROR(op_inputs[i]->Tensor(&input_tensor));
      protected_tensors.push_back(*input_tensor);
    }
  }

  gtl::InlinedVector<TensorValue, 4> input_vector(op_inputs.size());
  for (int i = 0; i < op_inputs.size(); ++i) {
    TF_RETURN_IF_ERROR(op_inputs[i]->TensorValue(&input_vector[i]));
  }

  //  TODO(apassos) figure out how to record stats for ops which are a part of
  //  functions.
  // TODO(agarwal): change Run to take vector of handles ?
  ScopedStepContainer* container = ctx->StepContainer();
  if (container == nullptr) {
    TF_RETURN_IF_ERROR(kernel->Run(input_vector, &outputs, maybe_stats,
                                   maybe_step_stats, graph_collector));
  } else {
    TF_RETURN_IF_ERROR(kernel->Run(container, input_vector, &outputs,
                                   maybe_stats, maybe_step_stats,
                                   graph_collector));
  }
  if (maybe_stats != nullptr) {
    int64 nanos = Env::Default()->NowNanos();
    maybe_stats->set_op_end_rel_micros(nanos / EnvTime::kMicrosToNanos -
                                       maybe_stats->all_start_micros());
    maybe_stats->set_op_end_rel_nanos(nanos - maybe_stats->all_start_nanos());
    maybe_stats->set_all_end_rel_micros(nanos / EnvTime::kMicrosToNanos -
                                        maybe_stats->all_start_micros());
    maybe_stats->set_all_end_rel_nanos(nanos - maybe_stats->all_start_nanos());
    mutex_lock ml(*ctx->MetadataMu());
    if (ctx->ShouldStoreMetadata()) {
      {
        GraphCollector* collector = ctx->GetGraphCollector();
        mutex_lock mll(collector->mu);
        for (const auto& graph : collector->graphs) {
          *ctx->RunMetadataProto()->add_partition_graphs() = graph;
        }
        collector->graphs.clear();
      }
      auto* step_stats = ctx->RunMetadataProto()->mutable_step_stats();
      // Lazily initialize the RunMetadata with information about all devices if
      // this is the first call.
      while (step_stats->dev_stats_size() < ctx->devices()->size()) {
        step_stats->add_dev_stats();
      }
      // Find the current device's index.
      int device_idx = 0;
      for (int i = 0; i < ctx->devices()->size(); ++i) {
        if (ctx->devices()->at(i) == device) {
          device_idx = i;
          break;
        }
      }
      // Populate the device stats for this device.
      auto* dev_stats = step_stats->mutable_dev_stats(device_idx);
      dev_stats->set_device(device->name());
      *dev_stats->add_node_stats() = *maybe_stats;
    }
  }
  DCHECK_EQ(num_retvals, outputs.size());
  for (int i = 0; i < num_retvals; ++i) {
    if (retvals[i] == nullptr) {
      retvals[i] =
          new TensorHandle(outputs[i], /* d= */ kernel->OutputDevice(i),
                           /* op_device= */ device, ctx);
    } else {
      // In the async case, the retval is not a nullptr, and its device is
      // already set since all TensorHandles always have their device set during
      // construction.
      DCHECK_EQ(device, retvals[i]->op_device());
      DCHECK_EQ(kernel->OutputDevice(i), retvals[i]->device());

      retvals[i]->SetTensor(outputs[i]);
    }
  }
  return Status::OK();
}

namespace {

Status LocalEagerCopyToDevice(TensorHandle* h, EagerContext* ctx, Device* dstd,
                              TensorHandle** result) {
  TF_RETURN_IF_ERROR(ctx->GetStatus());
  if (ctx->Async()) {
    // Note that `h` may not be currently ready. However execution order will
    // make sure that `h` is ready before the copy is actually done.
    CopyToDeviceNode* node = new CopyToDeviceNode(h, dstd, ctx);
    TensorHandle* output = node->dst();
    // Note that calling Add makes `node` accessible by the EagerExecutor
    // thread. So further accesses need to be thread-safe.
    ctx->ExecutorAdd(node);
    *result = output;
    return Status::OK();
  } else {
    TF_RETURN_IF_ERROR(h->CopyToDevice(ctx, dstd, result));
    return Status::OK();
  }
}

Status FindDeviceFromName(EagerContext* ctx, const char* device_name,
                          Device** device) {
  *device = ctx->HostCPU();
  if (device_name == nullptr || strlen(device_name) == 0) {
    return Status::OK();
  }

  auto status = ctx->local_device_mgr()->LookupDevice(device_name, device);
  if (status.ok()) {
    return status;
  }

  if (ctx->remote_device_mgr() != nullptr) {
    return ctx->remote_device_mgr()->LookupDevice(device_name, device);
  }

  return status;
}

Status ExecuteSend(EagerContext* ctx, tensorflow::Device* device,
                   TensorHandle* h, StringPiece wire_id,
                   const string& recv_device) {
  const tensorflow::AttrTypeMap* types;
  bool is_function = false;
  TF_RETURN_IF_ERROR(
      tensorflow::AttrTypeMapForOp("_Send", &types, &is_function));
  DCHECK(!is_function);
  tensorflow::EagerOperation op(ctx, "_Send", /*is_function=*/false, types);

  op.AddInput(h);

  op.SetDevice(device);

  op.MutableAttrs()->Set("tensor_name", wire_id);
  op.MutableAttrs()->Set("send_device", device->name());
  op.MutableAttrs()->Set(
      "send_device_incarnation",
      static_cast<int64>(device->attributes().incarnation()));
  op.MutableAttrs()->Set("recv_device", recv_device);
  op.MutableAttrs()->Set("client_terminated", false);

  op.MutableAttrs()->Set("T", h->dtype);

  int num_outputs = 0;
  gtl::InlinedVector<TensorHandle*, 2> retvals;

  return EagerExecute(&op, &retvals, &num_outputs);
}

Status ExecuteRecv(EagerContext* ctx, tensorflow::Device* device,
                   DataType dtype, StringPiece wire_id,
                   const string& send_device, int64 send_device_incarnation,
                   TensorHandle** result) {
  const tensorflow::AttrTypeMap* types;
  bool is_function = false;
  TF_RETURN_IF_ERROR(
      tensorflow::AttrTypeMapForOp("_Recv", &types, &is_function));
  DCHECK(!is_function);
  tensorflow::EagerOperation op(ctx, "_Recv", /*is_function=*/false, types);

  op.SetDevice(device);

  op.MutableAttrs()->Set("tensor_name", wire_id);
  op.MutableAttrs()->Set("send_device", send_device);
  op.MutableAttrs()->Set("send_device_incarnation", send_device_incarnation);
  op.MutableAttrs()->Set("recv_device", device->name());
  op.MutableAttrs()->Set("client_terminated", false);

  op.MutableAttrs()->Set("tensor_type", dtype);

  int num_outputs = 1;
  gtl::InlinedVector<TensorHandle*, 2> retvals(num_outputs);

  TF_RETURN_IF_ERROR(EagerExecute(&op, &retvals, &num_outputs));

  *result = retvals.at(0);

  return Status::OK();
}

// This gets a unique wire ID. We add a random identifier so that if the worker
// has other clients that it is servicing, we don't have any collision.
string GetUniqueWireID() {
  static tensorflow::uint64 random_seed = random::New64();
  static tensorflow::mutex wireid_mutex(tensorflow::LINKER_INITIALIZED);
  static tensorflow::int64 wireid GUARDED_BY(wireid_mutex) = 0;
  tensorflow::mutex_lock l(wireid_mutex);
  return strings::StrCat(random_seed, "_", wireid++);
}

}  // namespace

Status EagerCopyToDevice(TensorHandle* h, EagerContext* ctx,
                         const char* device_name, TensorHandle** result) {
  tensorflow::Device* send_device = h->device();

  if (send_device == nullptr) {
    send_device = ctx->HostCPU();
  }

  bool sender_is_local = IsLocal(ctx, send_device);

  tensorflow::Device* recv_device;
  TF_RETURN_IF_ERROR(FindDeviceFromName(ctx, device_name, &recv_device));

  bool recver_is_local = IsLocal(ctx, recv_device);

  if (sender_is_local && recver_is_local) {
    return LocalEagerCopyToDevice(h, ctx, recv_device, result);
  } else if (ctx->UseSendTensorRPC() && sender_is_local && !recver_is_local) {
    return EagerRemoteSendTensor(ctx, h, recv_device, result);
  } else {
    string wire_id = GetUniqueWireID();

    TF_RETURN_IF_ERROR(
        ExecuteSend(ctx, send_device, h, wire_id, recv_device->name()));

    return ExecuteRecv(ctx, recv_device, h->dtype, wire_id, send_device->name(),
                       send_device->attributes().incarnation(), result);
  }
}
}  // namespace tensorflow
