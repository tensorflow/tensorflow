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

// This file contains a simplified TF client that talks to a remote TF eager
// service over gRPC. Unlike the standard TF client libraries, this is a small
// self-contained client with few TF core dependencies that allows clients to:
// * send tensors to and from remote devices
// * run ops
// which is all the functionality we need for running XLA programs using XRT.
// The API is intended to be minimal and does not take dependencies on classes
// such as Tensor or Device.
//
// The main feature this client adds over the remote eager TF client is
// batching. Rather than synchronously executing each operator, the client
// accumulates batches of operators and enqueues them as a unit. This is
// important to hide latency; clients of XRT make large numbers of cheap
// operator calls to perform operations like allocation and deallocation.
//
// The auto-batching client is also more ergonomic that using graph mode or
// functions to batch computations. The graphs an XRT client runs may often be
// ephemeral and may rarely be the same. By allowing the XRT to enqueue
// operators eagerly and performing batching in the RPC client we can hide
// latency without requiring users to manage functions/graphs and their
// lifetimes.
//
// An important future direction for the client and something that cannot be
// supported by the TF graph mode API is asynchronous execution. However, we
// do not yet use asynchronous execution, mostly because of some problematic
// error handling semantics in the remote eager service API that make it
// difficult to attribute errors to asynchronously-launched operations.

// TODO(phawkins): handle client shutdown more gracefully; abandon all pending
// operations on shutdown.

#ifndef TENSORFLOW_COMPILER_XRT_CLIENT_XRT_TF_CLIENT_H_
#define TENSORFLOW_COMPILER_XRT_CLIENT_XRT_TF_CLIENT_H_

#include <memory>

#include "absl/container/inlined_vector.h"
#include "absl/container/node_hash_map.h"
#include "absl/synchronization/notification.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xrt/client/xrt_grpc_eager_client.h"
#include "tensorflow/compiler/xrt/client/xrt_tf_client.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/cluster.pb.h"

namespace tensorflow {

// Class that manages a connection to a TensorFlow cluster.
class XrtTfClient {
 public:
  XrtTfClient(ClusterDef cluster_def,
              std::shared_ptr<GrpcChannelCache> channel_cache);

  const ClusterDef& cluster_def() const { return cluster_def_; }
  XrtGrpcEagerClientCache* eager_client_cache() const {
    return eager_client_cache_.get();
  }

 private:
  const ClusterDef cluster_def_;
  const std::shared_ptr<GrpcChannelCache> channel_cache_;

  std::unique_ptr<XrtGrpcEagerClientCache> eager_client_cache_;
};

class XrtTensorHandle;
class XrtRecvTensorFuture;

// Class that manages a TensorFlow Eager context.
// TODO(phawkins): Intended to be thread-safe
class XrtTfContext : public std::enable_shared_from_this<XrtTfContext> {
 public:
  struct Options {
    // Enable async mode.
    // TODO(phawkins): this is not tested.
    bool async = false;

    // Maximum number of ops to keep queued.
    int max_queue_size = 100;
  };
  static xla::StatusOr<std::shared_ptr<XrtTfContext>> Create(
      const Options& options, std::shared_ptr<XrtTfClient> client,
      const std::string& job, int task);

  XrtTfContext(const Options& options, std::shared_ptr<XrtTfClient> client,
               XrtGrpcEagerClient* eager_client, int64 rendezvous_id,
               int64 context_id, std::vector<DeviceAttributes> devices,
               int cpu_device_id);

  ~XrtTfContext();

  const Options& options() const { return options_; }

  // The set of devices that were known to the remote worker when the context
  // was created.
  const std::vector<DeviceAttributes>& devices() const { return devices_; }

  // The CPU device on the remote worker.
  int cpu_device_id() const { return cpu_device_id_; }

  // Sends `tensor_proto` to `devices_[device_id]`. If `host_memory` is true,
  // sends to the tensor to host memory on `device_id`.
  XrtTensorHandle SendTensor(std::unique_ptr<TensorProto> tensor_proto,
                             int device_id, bool host_memory = false);

  // Receives `tensor` from the remote host. Does not flush the queue.
  std::shared_ptr<XrtRecvTensorFuture> RecvTensor(const XrtTensorHandle& tensor,
                                                  DataType dtype,
                                                  bool host_memory);

  // Enqueues an operator onto the remote host.
  // 'future' is an optional future that depends on the op.
  std::vector<XrtTensorHandle> EnqueueOp(
      absl::string_view name, absl::Span<const XrtTensorHandle* const> inputs,
      int output_arity, protobuf::Map<string, AttrValue> attrs, int device_id,
      std::shared_ptr<XrtRecvTensorFuture> future = {});

  // Registers a function `def` on the remote host.
  Status RegisterFunction(const FunctionDef& def);

  // Flushes any enqueued work to the remote host.
  void FlushQueue();

 private:
  friend class XrtTensorHandle;

  // An operation ID on the remote worker.
  typedef int64 OperationId;

  // Names a tensor on the remote worker.
  typedef std::pair<int64, int32> TensorId;

  // An Operation describes an operation to be enqueued to a remote worker,
  // together with its consumers. We need to know the set of consumers so we
  // can propagate errors to dependent operations in the event of failure.
  struct Operation {
    explicit Operation(OperationId id) : id(id) {}

    OperationId id;

    // Operations that depend on the output of this operation.
    absl::InlinedVector<OperationId, 2> consumers;

    // Tensor futures that consume the output of this operator.
    std::vector<std::shared_ptr<XrtRecvTensorFuture>> tensor_futures;
  };

  // Allocates and returns new operation. Does not return ownership.
  Operation* AddOperation() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Flushes the queue of pending work.
  void FlushQueueLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Shuts down the context, abandoning any pending operations.
  Status Close();

  // Enqueues an operation that releases the client's handle to a remote tensor.
  void EnqueueDecrefTensorHandle(eager::RemoteTensorHandle handle);

  // Reports the failure of a set of operations. Propagates the failure to
  // any dependent operations.
  void ReportError(absl::Span<const OperationId> op_ids, Status status)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Looks up operation 'id'. Dies if 'id' does not exist.
  Operation* LookupOperation(OperationId id) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Deletes an operation 'id'. Dies if 'id' does not exist.
  void DeleteOperation(OperationId id) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const Options options_;

  std::shared_ptr<XrtTfClient> tf_client_;
  XrtGrpcEagerClient* eager_client_;

  // The rendezvous ID to use when performing Send/Recv operations.
  const int64 rendezvous_id_;

  // A unique ID for this context on the remote worker.
  const int64 context_id_;

  // The set of devices present on the remote worker.
  std::vector<DeviceAttributes> devices_;

  // The CPU device on the remote worker. A CPU device must exist or Create()
  // fails with an error.
  int cpu_device_id_;

  absl::Mutex mu_;

  // The next available operation ID.
  int64 next_op_id_ GUARDED_BY(mu_) = 0;
  // The set of pending operations.
  absl::node_hash_map<OperationId, Operation> operations_ GUARDED_BY(mu_);

  // The queue of operations to run next.
  std::unique_ptr<eager::EnqueueRequest> enqueue_request_ GUARDED_BY(mu_);

  // Requests the queue thread to flush the queue.
  bool flush_requested_ GUARDED_BY(mu_) = false;

  // Notifies the queue thread that we are shutting down.
  bool shutting_down_ GUARDED_BY(mu_) = false;

  // Thread responsible for enqueueing queued ops to the remote worker.
  // Also responsible for sending regular RPCs to keep the connection alive.
  void QueueThread();
  std::unique_ptr<Thread> queue_thread_;
};

// RAII class that owns a reference to a remote TF tensor.
class XrtTensorHandle {
 public:
  XrtTensorHandle();
  XrtTensorHandle(std::shared_ptr<XrtTfContext> context, int device_id,
                  XrtTfContext::TensorId tensor_id)
      : context_(context), device_id_(device_id), tensor_id_(tensor_id) {}
  ~XrtTensorHandle();

  // Moveable but not copyable; the handle cannot be duplicated.
  XrtTensorHandle(const XrtTensorHandle&) = delete;
  XrtTensorHandle& operator=(const XrtTensorHandle&) = delete;
  XrtTensorHandle(XrtTensorHandle&& other);
  XrtTensorHandle& operator=(XrtTensorHandle&& other);

  // Serializes the handle's ID to a protocol buffer.
  void Serialize(eager::RemoteTensorHandle* proto) const;

  // The context to which the handle belongs.
  const std::shared_ptr<XrtTfContext>& context() const { return context_; }

  int device_id() const { return device_id_; }
  void set_device_id(int device_id) { device_id_ = device_id; }

  // Returns true if the handle refers to valid context.
  bool valid() const { return context_ != nullptr; }

 private:
  friend class XrtTfContext;
  std::shared_ptr<XrtTfContext> context_;
  int device_id_ = -1;
  XrtTfContext::TensorId tensor_id_ = {-1, -1};
};

// Future that holds the result of a RecvTensor call.
class XrtRecvTensorFuture {
 public:
  XrtRecvTensorFuture() = default;

  // Returns either an error or a pointer to the RecvTensorResponse.
  // Blocks waiting for the future if it is not yet available.
  xla::StatusOr<RecvTensorResponse*> Get() {
    done_.WaitForNotification();
    absl::MutexLock lock(&mu_);
    if (!status_.ok()) return status_;
    return &value_;
  }

 private:
  friend class XrtTfContext;

  // Marks the future as completed, with `status`.
  void Notify(Status status) {
    absl::MutexLock lock(&mu_);
    if (done_.HasBeenNotified()) {
      LOG(ERROR) << "Duplicate notification for XrtRecvTensorFuture. "
                    "Previous status: "
                 << status_ << " new status: " << status;
      return;
    }
    status_ = status;
    done_.Notify();
  }

  absl::Mutex mu_;
  absl::Notification done_;
  Status status_ GUARDED_BY(mu_);
  RecvTensorResponse value_ GUARDED_BY(mu_);

  CallOptions call_options_;
};

// This gets a unique wire ID. We add a random identifier so that if the
// worker has other clients that it is servicing, we don't have collisions.
std::string XrtGetUniqueWireID();

// Helpers for enqueuing common TF ops.

// Enqueues a _Send operator that sends a tensor located on a device on the
// remote worker. If recv_device_id < 0 the target of the send is the client,
// and a fake device name is used (since the client has no real name in the
// TF cluster).
// 'future' may be null. If non-null it gives a future that depends on the
// output of the send and that must be aborted if the send fails.
void EnqueueSend(XrtTfContext* context, const XrtTensorHandle& tensor,
                 DataType dtype, int recv_device_id, std::string wire_id,
                 bool host_memory,
                 std::shared_ptr<XrtRecvTensorFuture> future = {});

// Enqueues a _Recv operator that receives a tensor onto a remote device.
XrtTensorHandle EnqueueRecv(XrtTfContext* context, DataType dtype,
                            int send_device_id, int recv_device_id,
                            std::string wire_id, bool host_memory);

// Enqueues a Const operator operator on a remote device.
XrtTensorHandle EnqueueConst(XrtTfContext* context, int device_id,
                             TensorProto value, bool host_memory);

// Helpers for building AttrValue protos. We have our own versions of these
// to avoid depending on TF framework code.
AttrValue MakeAttrValue(std::string s);
AttrValue MakeAttrValue(int64 i);
AttrValue MakeBoolAttrValue(bool b);
AttrValue MakeAttrValue(DataType dtype);
AttrValue MakeAttrValue(TensorProto tensor);
AttrValue MakeAttrValue(absl::Span<const DataType> dtypes);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XRT_CLIENT_XRT_TF_CLIENT_H_
