/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xrt/client/xrt_tf_client.h"

#include <stack>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/distributed_runtime/request_id.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/protobuf/eager_service.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace tensorflow {

XrtTfClient::XrtTfClient(ClusterDef cluster_def,
                         std::shared_ptr<GrpcChannelCache> channel_cache)
    : cluster_def_(cluster_def), channel_cache_(std::move(channel_cache)) {
  eager_client_cache_ =
      absl::make_unique<XrtGrpcEagerClientCache>(channel_cache_);
}

xla::StatusOr<std::shared_ptr<XrtTfContext>> XrtTfContext::Create(
    const XrtTfContext::Options& options,
    std::shared_ptr<XrtTfClient> tf_client, const std::string& job, int task) {
  int64 rendezvous_id = random::New64();

  eager::CreateContextRequest request;
  ServerDef* server_def = request.mutable_server_def();
  *server_def->mutable_cluster() = tf_client->cluster_def();
  server_def->set_job_name(job);
  server_def->set_protocol("grpc");
  request.set_keep_alive_secs(60);
  request.set_rendezvous_id(rendezvous_id);
  request.set_async(options.async);

  eager::CreateContextResponse response;

  std::string target = absl::StrFormat("/job:%s/task:%d/replica:0", job, task);
  TF_ASSIGN_OR_RETURN(XrtGrpcEagerClient * eager_client,
                      tf_client->eager_client_cache()->GetClient(target));

  TF_RETURN_IF_ERROR(eager_client->SyncCall(
      &XrtGrpcEagerClient::CreateContextAsync, &request, &response));

  // Due to a TFE server-side bug, devices returned by the eager CreateContext
  // method have the wrong device incarnation numbers, which we need to call
  // RecvTensor. Use the device attributes from WorkerService.GetStatus instead.
  // TODO(phawkins): revert to using device information from CreateContext once
  // the bug is fixed.
  GetStatusRequest status_request;
  GetStatusResponse status_response;
  TF_RETURN_IF_ERROR(eager_client->SyncCall(&XrtGrpcEagerClient::GetStatusAsync,
                                            &status_request, &status_response));

  std::vector<DeviceAttributes> devices(
      status_response.device_attributes().begin(),
      status_response.device_attributes().end());
  VLOG(1) << "Remote devices: " << devices.size();
  int cpu_device_id = -1;
  for (int i = 0; i < devices.size(); ++i) {
    const auto& device = devices[i];
    VLOG(2) << "Remote device: " << device.DebugString();
    if (cpu_device_id < 0 && device.device_type() == "CPU") {
      cpu_device_id = i;
      VLOG(1) << "Remote CPU device: " << i << " name: " << device.name();
    }
  }
  if (cpu_device_id < 0) {
    return errors::FailedPrecondition(
        "Remote TensorFlow worker does not have a CPU device.");
  }

  std::sort(devices.begin(), devices.end(),
            [](const DeviceAttributes& a, const DeviceAttributes& b) {
              return a.name() < b.name();
            });
  return std::make_shared<XrtTfContext>(options, tf_client, eager_client,
                                        rendezvous_id, response.context_id(),
                                        std::move(devices), cpu_device_id);
}

XrtTfContext::XrtTfContext(const XrtTfContext::Options& options,
                           std::shared_ptr<XrtTfClient> tf_client,
                           XrtGrpcEagerClient* eager_client,
                           int64 rendezvous_id, int64 context_id,
                           std::vector<DeviceAttributes> devices,
                           int cpu_device_id)
    : options_(options),
      tf_client_(tf_client),
      eager_client_(eager_client),
      rendezvous_id_(rendezvous_id),
      context_id_(context_id),
      devices_(std::move(devices)),
      cpu_device_id_(cpu_device_id) {
  CHECK_GE(cpu_device_id_, 0);
  enqueue_request_ = absl::make_unique<eager::EnqueueRequest>();
  queue_thread_.reset(Env::Default()->StartThread(ThreadOptions(),
                                                  "xrt_tf_client_queue_thread",
                                                  [this]() { QueueThread(); }));
}

XrtTfContext::~XrtTfContext() {
  Status status = Close();
  if (!status.ok()) {
    LOG(ERROR) << "XrtTfContext::Close failed with error: " << status;
  }
}

Status XrtTfContext::Close() {
  {
    absl::MutexLock lock(&mu_);
    shutting_down_ = true;
  }

  eager::CloseContextRequest request;
  request.set_context_id(context_id_);

  Status status;
  absl::Notification done;
  eager::CloseContextResponse response;
  eager_client_->CloseContextAsync(&request, &response, [&](Status s) {
    status = s;
    done.Notify();
  });
  done.WaitForNotification();
  return status;
}

void XrtTfContext::QueueThread() {
  auto should_flush_queue = [this]() {
    mu_.AssertHeld();  // For annotalysis.
    return enqueue_request_->queue_size() > options_.max_queue_size ||
           flush_requested_ || shutting_down_;
  };
  while (true) {
    auto request = absl::make_unique<eager::EnqueueRequest>();
    {
      absl::MutexLock lock(&mu_);
      // To keep the connection alive, make sure we send an EnqueueRequest
      // regularly, currently every 5 seconds.
      mu_.AwaitWithTimeout(absl::Condition(&should_flush_queue),
                           absl::Seconds(5));
      if (shutting_down_) break;
      std::swap(request, enqueue_request_);
      flush_requested_ = false;
    }

    std::vector<OperationId> op_ids;
    for (const auto& item : request->queue()) {
      if (item.has_operation()) {
        op_ids.push_back(item.operation().id());
      }
    }
    request->set_context_id(context_id_);

    VLOG(10) << "Enqueue:\n" << request->DebugString();
    eager::EnqueueResponse response;
    Status status;
    absl::Notification done;
    eager_client_->EnqueueAsync(request.get(), &response, [&](Status s) {
      status = s;
      done.Notify();
    });

    done.WaitForNotification();

    VLOG(10) << "EnqueueResponse: " << status << "\n" << response.DebugString();
    {
      absl::MutexLock lock(&mu_);
      if (status.ok()) {
        for (OperationId op_id : op_ids) {
          DeleteOperation(op_id);
        }
      } else {
        ReportError(op_ids, status);
      }
    }
  }
}

void XrtTfContext::ReportError(absl::Span<const OperationId> op_ids,
                               Status status) {
  auto shared_error = std::make_shared<Status>(status);
  absl::flat_hash_set<OperationId> visited(op_ids.begin(), op_ids.end());
  std::stack<Operation*> stack;
  for (OperationId op_id : op_ids) {
    stack.push(LookupOperation(op_id));
  }
  while (!stack.empty()) {
    Operation* op = stack.top();
    stack.pop();
    VLOG(10) << "Reporting error for " << op->id;
    for (const std::shared_ptr<XrtRecvTensorFuture>& future :
         op->tensor_futures) {
      VLOG(10) << "Reporting error for " << op->id << " future";
      future->call_options_.StartCancel();
      future->Notify(status);
    }
    for (OperationId consumer_id : op->consumers) {
      Operation* consumer = LookupOperation(consumer_id);
      stack.push(consumer);
    }
    DeleteOperation(op->id);
  }
}

XrtTfContext::Operation* XrtTfContext::AddOperation() {
  OperationId id = ++next_op_id_;
  auto result = operations_.emplace(id, Operation(id));
  return &result.first->second;
}

void XrtTfContext::DeleteOperation(OperationId id) {
  CHECK_GT(operations_.erase(id), 0);
}

XrtTfContext::Operation* XrtTfContext::LookupOperation(OperationId id) {
  auto it = operations_.find(id);
  CHECK(it != operations_.end()) << id;
  return &it->second;
}

std::vector<XrtTensorHandle> XrtTfContext::EnqueueOp(
    absl::string_view name, absl::Span<const XrtTensorHandle* const> inputs,
    int output_arity, protobuf::Map<std::string, AttrValue> attrs,
    int device_id, std::shared_ptr<XrtRecvTensorFuture> future) {
  std::vector<XrtTensorHandle> outputs;
  absl::MutexLock lock(&mu_);
  Operation* op = AddOperation();

  eager::Operation* proto = enqueue_request_->add_queue()->mutable_operation();
  proto->set_id(op->id);
  proto->set_name(static_cast<std::string>(name));
  for (const XrtTensorHandle* input : inputs) {
    input->Serialize(proto->add_inputs());
  }
  proto->mutable_attrs()->swap(attrs);
  proto->set_device(devices_.at(device_id).name());

  outputs.reserve(output_arity);
  for (int i = 0; i < output_arity; ++i) {
    outputs.push_back(
        XrtTensorHandle(shared_from_this(), device_id, TensorId{op->id, i}));
  }
  if (future) {
    op->tensor_futures.push_back(future);
  }

  return outputs;
}

XrtTensorHandle XrtTfContext::SendTensor(
    std::unique_ptr<TensorProto> tensor_proto, int device_id,
    bool host_memory) {
  DataType dtype = tensor_proto->dtype();
  bool transfer_via_cpu_device = host_memory && device_id != cpu_device_id_;
  int rpc_device_id = transfer_via_cpu_device ? cpu_device_id_ : device_id;
  OperationId op_id;
  {
    absl::MutexLock lock(&mu_);
    Operation* op = AddOperation();
    op_id = op->id;
  }

  eager::SendTensorRequest request;
  request.set_context_id(context_id_);
  request.set_op_id(op_id);
  request.mutable_tensors()->AddAllocated(tensor_proto.release());
  request.set_device_name(devices_.at(rpc_device_id).name());
  auto response = std::make_shared<eager::SendTensorResponse>();
  auto context_ptr = shared_from_this();
  absl::Notification done;
  eager_client_->SendTensorAsync(
      &request, response.get(),
      [context_ptr, op_id, response, &done](Status status) {
        absl::MutexLock lock(&context_ptr->mu_);
        if (!status.ok()) {
          context_ptr->ReportError({op_id}, status);
        } else {
          context_ptr->DeleteOperation(op_id);
        }
        done.Notify();
      });
  XrtTensorHandle handle(context_ptr, rpc_device_id, TensorId{op_id, 0});

  // TODO(phawkins): we block here to avoid a race. We must not
  // enqueue any dependent operations until the SendTensor has been
  // acknowledged.
  done.WaitForNotification();

  // TODO(phawkins): EagerService.SendTensor could use a host_memory option.
  if (!transfer_via_cpu_device) {
    return handle;
  }
  std::string wire_id = XrtGetUniqueWireID();
  EnqueueSend(this, handle, dtype, device_id, wire_id, /*host_memory=*/false);
  return EnqueueRecv(this, dtype, rpc_device_id, device_id, wire_id,
                     /*host_memory=*/true);
}

// This gets a unique wire ID. We add a random identifier so that if the
// worker has other clients that it is servicing, we don't have any collision.
std::string XrtGetUniqueWireID() {
  static uint64 random_seed = random::New64();
  static std::atomic<int64> wireid(0);
  return absl::StrCat(random_seed, "_", ++wireid);
}

static std::string GetReceiverDevice(XrtTfContext* context,
                                     int recv_device_id) {
  if (recv_device_id < 0) {
    return "/job:xrt_client/task:0/replica:0/device:CPU:0";
  } else {
    return context->devices().at(recv_device_id).name();
  }
}

static std::string GetRendezvousKey(absl::string_view send_device,
                                    absl::string_view recv_device,
                                    const uint64 send_device_incarnation,
                                    absl::string_view tensor_name) {
  return absl::StrCat(send_device, ";",
                      strings::FpToString(send_device_incarnation), ";",
                      recv_device, ";", tensor_name, ";0:0");
}

std::shared_ptr<XrtRecvTensorFuture> XrtTfContext::RecvTensor(
    const XrtTensorHandle& tensor, DataType dtype, bool host_memory) {
  auto response = std::make_shared<XrtRecvTensorFuture>();

  int device_id = tensor.device_id();

  std::string wire_id = XrtGetUniqueWireID();
  EnqueueSend(this, tensor, dtype, /*recv_device_id=*/-1, wire_id,
              /*host_memory=*/host_memory, /*future=*/response);

  const DeviceAttributes& device = devices().at(device_id);
  RecvTensorRequest request;
  request.set_step_id(rendezvous_id_);
  request.set_rendezvous_key(GetRendezvousKey(device.name(),
                                              GetReceiverDevice(this, -1),
                                              device.incarnation(), wire_id));
  request.set_request_id(GetUniqueRequestId());
  // TODO(phawkins): verify uniqueness of request ID. Random IDs won't collide
  // with high probability, but we should probably add code to guard against
  // collisions nonetheless.

  eager_client_->RecvTensorAsync(
      &request, &response->value_,
      [response, wire_id](Status status) {
        VLOG(10) << "RecvTensor complete for " << wire_id;
        response->Notify(status);
      },
      &response->call_options_);
  return response;
}

Status XrtTfContext::RegisterFunction(const FunctionDef& def) {
  eager::RegisterFunctionRequest request;
  request.set_context_id(context_id_);
  *request.mutable_function_def() = def;

  eager::RegisterFunctionResponse response;
  Status status;
  absl::Notification done;
  eager_client_->RegisterFunctionAsync(&request, &response, [&](Status s) {
    status = s;
    done.Notify();
  });
  done.WaitForNotification();
  return status;
}
void XrtTfContext::EnqueueDecrefTensorHandle(eager::RemoteTensorHandle handle) {
  absl::MutexLock lock(&mu_);
  eager::QueueItem* item = enqueue_request_->add_queue();
  *item->mutable_handle_to_decref() = handle;
}

void XrtTfContext::FlushQueue() {
  absl::MutexLock lock(&mu_);
  FlushQueueLocked();
}

void XrtTfContext::FlushQueueLocked() { flush_requested_ = true; }

XrtTensorHandle::XrtTensorHandle() = default;
XrtTensorHandle::~XrtTensorHandle() {
  if (context_) {
    eager::RemoteTensorHandle proto;
    Serialize(&proto);
    context_->EnqueueDecrefTensorHandle(proto);
  }
}

XrtTensorHandle::XrtTensorHandle(XrtTensorHandle&& other) {
  context_ = other.context_;
  device_id_ = other.device_id_;
  tensor_id_ = other.tensor_id_;

  other.context_ = nullptr;
  other.device_id_ = -1;
  other.tensor_id_ = XrtTfContext::TensorId{-1, -1};
}

XrtTensorHandle& XrtTensorHandle::operator=(XrtTensorHandle&& other) {
  context_ = other.context_;
  device_id_ = other.device_id_;
  tensor_id_ = other.tensor_id_;

  other.context_ = nullptr;
  other.device_id_ = -1;
  other.tensor_id_ = XrtTfContext::TensorId{-1, -1};
  return *this;
}

void XrtTensorHandle::Serialize(eager::RemoteTensorHandle* proto) const {
  proto->set_op_id(tensor_id_.first);
  proto->set_output_num(tensor_id_.second);
}

AttrValue MakeAttrValue(std::string s) {
  AttrValue a;
  a.set_s(std::move(s));
  return a;
}

AttrValue MakeAttrValue(int64 i) {
  AttrValue a;
  a.set_i(i);
  return a;
}

AttrValue MakeBoolAttrValue(bool b) {
  AttrValue a;
  a.set_b(b);
  return a;
}

AttrValue MakeAttrValue(DataType dtype) {
  AttrValue a;
  a.set_type(dtype);
  return a;
}

AttrValue MakeAttrValue(TensorProto tensor) {
  AttrValue a;
  *a.mutable_tensor() = tensor;
  return a;
}

AttrValue MakeAttrValue(absl::Span<const DataType> dtypes) {
  AttrValue a;
  auto* list = a.mutable_list();
  for (DataType dtype : dtypes) {
    list->add_type(dtype);
  }
  return a;
}

void EnqueueSend(XrtTfContext* context, const XrtTensorHandle& tensor,
                 DataType dtype, int recv_device_id, std::string wire_id,
                 bool host_memory,
                 std::shared_ptr<XrtRecvTensorFuture> future) {
  protobuf::Map<std::string, AttrValue> attrs;
  const DeviceAttributes& device = context->devices().at(tensor.device_id());
  attrs["tensor_name"] = MakeAttrValue(wire_id);
  attrs["send_device"] = MakeAttrValue(device.name());
  attrs["send_device_incarnation"] = MakeAttrValue(device.incarnation());
  attrs["recv_device"] =
      MakeAttrValue(GetReceiverDevice(context, recv_device_id));
  attrs["client_terminated"] = MakeBoolAttrValue(false);
  attrs["T"] = MakeAttrValue(dtype);

  context->EnqueueOp(host_memory ? "_HostSend" : "_Send", {&tensor},
                     /*output_arity=*/0, std::move(attrs), tensor.device_id(),
                     future);
}

XrtTensorHandle EnqueueRecv(XrtTfContext* context, DataType dtype,
                            int send_device_id, int recv_device_id,
                            std::string wire_id, bool host_memory) {
  protobuf::Map<std::string, AttrValue> attrs;
  const DeviceAttributes& send_device = context->devices().at(send_device_id);
  const DeviceAttributes& recv_device = context->devices().at(recv_device_id);
  attrs["tensor_name"] = MakeAttrValue(wire_id);
  attrs["send_device"] = MakeAttrValue(send_device.name());
  attrs["send_device_incarnation"] = MakeAttrValue(send_device.incarnation());
  attrs["recv_device"] = MakeAttrValue(recv_device.name());
  attrs["client_terminated"] = MakeBoolAttrValue(false);
  attrs["tensor_type"] = MakeAttrValue(dtype);

  return std::move(context->EnqueueOp(host_memory ? "_HostRecv" : "_Recv",
                                      /*inputs=*/{},
                                      /*output_arity=*/1, std::move(attrs),
                                      recv_device_id)[0]);
}

XrtTensorHandle EnqueueConst(XrtTfContext* context, int device_id,
                             TensorProto value, bool host_memory) {
  protobuf::Map<std::string, AttrValue> attrs;
  attrs["value"] = MakeAttrValue(value);
  attrs["dtype"] = MakeAttrValue(value.dtype());

  return std::move(context->EnqueueOp(host_memory ? "HostConst" : "Const",
                                      /*inputs=*/{},
                                      /*output_arity=*/1, std::move(attrs),
                                      device_id)[0]);
}

}  // namespace tensorflow
