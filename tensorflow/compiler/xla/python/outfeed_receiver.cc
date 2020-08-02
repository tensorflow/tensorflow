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

#include "tensorflow/compiler/xla/python/outfeed_receiver.h"

#include <sys/types.h>

#include <memory>
#include <sstream>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/profiler/lib/traceme.h"

// Implementation notes:
//
// Startup:
// -------
//
// The startup is initiated by a call from Python to StartOutfeedReceiver,
// which starts N threads for listening to the N devices and for enqueueing
// the received data into a callback queue. There is one additional callback
// thread for dequeing the data and invoking the Python callback.
//
// Framing protocol
// ----------------
//
// The outfeed mechanism has a single channel and the receiver must know
// exactly the shape and number of outfeed operations issued by the compiled
// code. This makes it hard to use outfeed in conditionals and loops and
// especially when outfeeding different-shaped data.
//
// To address this, when we compile the code we capture the shape of the
// data being outfed, and we generate a consumer ID (uint32_t) that is unique
// across the lifetime of the program to: the Python callable to callback to,
// the shape of the arguments, the keyword arguments to pass to the callable.
// Each outfeed payload is preceeded by a header (of shape u32[2]) with a
// special first value and the consumer ID. We maintain a registry of shapes
// by consumer ID. When receiving we lookup the shape by consumer ID, and then
// we read the payload.
//
// Back pressure:
// --------------
//
// We maintain a sum of the bytes from all the data waiting in the callback
// queue. The listening threads will wait for the sum to drop below a
// configurable threshold, default 256Mb. While the listening thread is waiting,
// on CPU and GPU the next outfeed operation from the device will block. On
// TPU there is a buffer, but eventually the TPU will also block.
//
// Shutdown:
// ---------
//
// The shutdown is initiated automatically when the last reference to the
// outfeed receiver object is dropped, and the Python garbage collector invokes
// the destructor.
//
// The shutdown sequence is implemented as follows:
// * we enqueue on all devices a computation that outfeeds a special header
//   with customer ID kOutfeedCidShutdown.
// * when each listening threads gets the shutdown header, it decrements
//   a counter of listening threads, and if the counter reaches 0, it
//   enqueues a special shutdown callback.
// * when the callback thread gets the shutdown callback marker, it terminates.
// * the shutdown code waits until all threads terminate.
//
// Since we currently keep the shape registry in the OutfeedReceiver, it is
// not safe to replace the OutfeedReceiver instance during the lifetime of
// the JAX program, or else previously cached jitted computations may refer
// to previously cached shapes. This can be solved, but for now we disallow
// replacing the OutfeedReceiver, and do not provide a Shutdown API to the
// Python program.

namespace xla {

// The header contains:
// 0. kOutfeedHeaderStart
// 1. consumer id
int constexpr kOutfeedHeaderWords = 2;
uint32_t constexpr kOutfeedHeaderStart = 271828;
// Special consumer IDs, without outfeed payload.
uint32_t constexpr kOutfeedCidShutdown = 0;

// Encapsulates data received from a device outfeed.
class OutfeedData {
 public:
  OutfeedData(Device* device, uint32_t consumer_id, Shape shape)
      : device_(device),
        consumer_id_(consumer_id),
        shape_(shape),
        literal_(nullptr),
        literal_size_bytes_(0) {}

  Device* device() { return device_; }
  uint32_t consumer_id() const { return consumer_id_; }
  Shape shape() const { return shape_; }
  std::unique_ptr<Literal> literal() {
    CHECK(literal_);
    return std::move(literal_);
  }

  void SetLiteral(std::unique_ptr<Literal> literal);

  ssize_t literal_size_bytes() const { return literal_size_bytes_; }

  std::string DebugString() const;

 private:
  Device* device_;
  uint32_t consumer_id_;
  Shape shape_;
  std::unique_ptr<Literal> literal_;
  ssize_t literal_size_bytes_;
};

void OutfeedData::SetLiteral(std::unique_ptr<Literal> literal) {
  literal_ = std::move(literal);
  shape_ = literal_->shape();
  int total_size_bytes = 0;
  ShapeUtil::ForEachSubshape(
      shape_, [&](const Shape& literal_subshape, const ShapeIndex& index) {
        if (!literal_subshape.IsTuple()) {
          total_size_bytes += ShapeUtil::ByteSizeOf(literal_subshape, 8);
        }
      });
  literal_size_bytes_ = total_size_bytes;
}

std::string OutfeedData::DebugString() const {
  return absl::StrFormat("dev=%s; cons=%d; shape=%s", device_->DebugString(),
                         consumer_id_, shape_.ToString());
}

class OutfeedReceiverImpl {
 public:
  OutfeedReceiverImpl(OutfeedReceiver::Callback callback,
                      absl::Span<PjRtClient* const> clients,
                      ssize_t max_callback_queue_size_bytes);

  OutfeedReceiverImpl(const OutfeedReceiverImpl&) = delete;
  OutfeedReceiverImpl& operator=(const OutfeedReceiverImpl&) = delete;

  // Blocks until all data has been received from devices and all data
  // in the queue has been passed to Python.
  ~OutfeedReceiverImpl();

  void Start();

  StatusOr<XlaOp> AddOutfeedToBuilder(XlaBuilder* builder, XlaOp token,
                                      uint32_t consumer_id,
                                      std::vector<XlaOp> arrays);

 private:
  bool CallbackQueueNotEmpty() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return !callback_queue_.empty();
  }

  bool CallbackQueueHasSpace() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return callback_queue_size_bytes_ < max_callback_queue_size_bytes_;
  }

  bool ShutdownDone() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return (num_working_callback_threads_ == 0 && num_listening_threads_ == 0);
  }

  void CallbackThreadLoop();
  void DeviceListenerThreadLoop(int device_idx);

  // Enqueues to a device an outfeed operation with a shutdown consumer ID.
  Status SendShutdownOutfeedHeader(int device_idx);

  // Receives a raw Literal from a device outfeed.
  StatusOr<std::unique_ptr<Literal>> ReceiveRawFromOutfeed(const Device* device,
                                                           const Shape& shape);

  // Enqueues received data in the callbaback queue.
  void EnqueueReceivedData(std::unique_ptr<OutfeedData> received)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Shuts down the threads. See implementation notes at top of file.
  // It is not safe to restart an OutfeedReceiver after shutting down one.
  void Shutdown();

  OutfeedReceiver::Callback callback_;
  // The devices on which we are listening.
  std::vector<Device*> devices_;
  // Maximum bytes capacity of the callback queue.
  uint64_t max_callback_queue_size_bytes_;

  absl::Mutex mu_;
  // Registered shapes by consumer id.
  // The shape registry must be alive as long as the program exists.
  // Right now we tell the user to never restart after Shutdown.
  absl::flat_hash_map<uint32_t, Shape> shape_registry_ TF_GUARDED_BY(mu_);
  // How many bytes of Literal are in the callback queue.
  uint64_t callback_queue_size_bytes_ TF_GUARDED_BY(mu_);
  // Threads listening.
  int num_listening_threads_ TF_GUARDED_BY(mu_);
  bool shutdown_started_ TF_GUARDED_BY(mu_);

  // How many callback threads are still working. Used for shutdown.
  int num_working_callback_threads_ TF_GUARDED_BY(mu_);

  std::queue<std::unique_ptr<OutfeedData>> callback_queue_ TF_GUARDED_BY(mu_);
  // The threadpool must come last to ensure the queue exists
  // when the pool destructor is called.
  std::unique_ptr<tensorflow::thread::ThreadPool> threads_;
};

OutfeedReceiverImpl::OutfeedReceiverImpl(
    OutfeedReceiver::Callback callback, absl::Span<PjRtClient* const> clients,
    ssize_t max_callback_queue_size_bytes) {
  callback_ = callback;
  max_callback_queue_size_bytes_ = max_callback_queue_size_bytes;
  for (const auto& client : clients) {
    for (const auto& device : client->devices()) {
      devices_.push_back(device.get());
    }
  }
  CHECK_GT(devices_.size(), 0);

  callback_queue_size_bytes_ = 0;
  num_listening_threads_ = 0;
  num_working_callback_threads_ = 0;
  shutdown_started_ = false;
}

void OutfeedReceiverImpl::Start() {
  {
    absl::MutexLock lock(&mu_);
    CHECK(!shutdown_started_);
  }
  int num_threads = 1 + devices_.size();
  threads_ = absl::make_unique<tensorflow::thread::ThreadPool>(
      tensorflow::Env::Default(), "outfeed_receiver", num_threads);
  threads_->Schedule([this]() { CallbackThreadLoop(); });
  for (int device_idx = 0; device_idx < devices_.size(); ++device_idx) {
    threads_->Schedule(
        [this, device_idx]() { DeviceListenerThreadLoop(device_idx); });
  }
}

void OutfeedReceiverImpl::Shutdown() {
  VLOG(2) << "Shutdown start";
  {
    absl::MutexLock lock(&mu_);
    CHECK(!shutdown_started_);
    shutdown_started_ = true;
  }
  for (int device_idx = 0; device_idx < devices_.size(); ++device_idx) {
    CHECK(SendShutdownOutfeedHeader(device_idx).ok());
  }
  VLOG(2) << "Shutdown waiting for listening and callback threads to stop";
  absl::MutexLock lock(&mu_);
  mu_.Await(absl::Condition(this, &OutfeedReceiverImpl::ShutdownDone));
  VLOG(2) << "Shutdown done";
}

OutfeedReceiverImpl::~OutfeedReceiverImpl() {
  VLOG(2) << "~OutfeedReceiverImpl";
  Shutdown();
}

void OutfeedReceiverImpl::DeviceListenerThreadLoop(int device_idx) {
  {
    absl::MutexLock lock(&mu_);
    ++num_listening_threads_;
  }
  Device* device = devices_[device_idx];
  while (true) {
    Shape header_shape = ShapeUtil::MakeShape(U32, {kOutfeedHeaderWords});
    std::unique_ptr<Literal> header =
        ReceiveRawFromOutfeed(device, header_shape).ValueOrDie();
    absl::Span<uint32_t> header_data = header->data<uint32>();
    CHECK_EQ(header_data.size(), kOutfeedHeaderWords);
    CHECK_EQ(header_data[0], kOutfeedHeaderStart);
    uint32_t consumer_id = header_data[1];
    Shape shape;
    {
      absl::MutexLock lock(&mu_);
      auto registered_shape = shape_registry_.find(consumer_id);
      if (registered_shape == shape_registry_.end()) {
        LOG(FATAL)
            << "[" << device->DebugString()
            << "] Cannot find registered shape for consumer ID " << consumer_id
            << ". Perhaps the code was compiled with a different instance "
            << "of OutfeedReceiver.";
      }
      shape = registered_shape->second;
    }
    auto received = absl::make_unique<OutfeedData>(device, consumer_id, shape);
    VLOG(2) << "Listener received header " << received->DebugString();
    if (consumer_id == kOutfeedCidShutdown) {
      VLOG(2) << "[" << device->DebugString()
              << "] Listener received shutdown header";
      absl::MutexLock lock(&mu_);
      --num_listening_threads_;
      if (num_listening_threads_ == 0) {
        VLOG(2) << "Last listener shutdown; enqueue shutdown callback";
        EnqueueReceivedData(std::move(received));
      }
      return;
    }
    std::unique_ptr<Literal> data =
        ReceiveRawFromOutfeed(device, shape).ValueOrDie();
    received->SetLiteral(std::move(data));
    absl::MutexLock lock(&mu_);
    EnqueueReceivedData(std::move(received));
  }
}

void OutfeedReceiverImpl::EnqueueReceivedData(
    std::unique_ptr<OutfeedData> received) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  mu_.Await(absl::Condition(this, &OutfeedReceiverImpl::CallbackQueueHasSpace));
  ssize_t literal_size_bytes = received->literal_size_bytes();
  callback_queue_size_bytes_ += literal_size_bytes;
  VLOG(2) << "Listener enqueues data " << received->DebugString() << " of size "
          << literal_size_bytes << " bytes; " << (1 + callback_queue_.size())
          << " callbacks in queue of total size " << callback_queue_size_bytes_
          << " bytes.\n";
  callback_queue_.push(std::move(received));
}

StatusOr<std::unique_ptr<Literal>> OutfeedReceiverImpl::ReceiveRawFromOutfeed(
    const Device* device, const Shape& shape) {
  std::shared_ptr<Literal> literal_shared;

  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device,
                      device->GetLocalDeviceState());
  TF_ASSIGN_OR_RETURN(Literal literal,
                      local_device->client()->TransferFromOutfeedLocal(
                          shape, local_device->device_ordinal()));

  return absl::make_unique<Literal>(std::move(literal));
}

void OutfeedReceiverImpl::CallbackThreadLoop() {
  {
    absl::MutexLock lock(&mu_);
    num_working_callback_threads_++;
    CHECK_EQ(num_working_callback_threads_, 1);
  }
  while (true) {
    std::unique_ptr<OutfeedData> received;
    {
      absl::MutexLock lock(&mu_);
      mu_.Await(
          absl::Condition(this, &OutfeedReceiverImpl::CallbackQueueNotEmpty));
      received = std::move(callback_queue_.front());
      callback_queue_.pop();
      callback_queue_size_bytes_ -= received->literal_size_bytes();
      VLOG(2) << "Dequeued callback for " << received->DebugString() << "; "
              << callback_queue_.size() << " callbacks in queue of total size "
              << callback_queue_size_bytes_ << " bytes.\n";
    }
    if (received->consumer_id() == kOutfeedCidShutdown) {
      VLOG(2) << "Callback loop received shutdown signal";
      {
        absl::MutexLock lock(&mu_);
        CHECK(callback_queue_.empty());
        CHECK_EQ(callback_queue_size_bytes_, 0);
        --num_working_callback_threads_;
      }
      VLOG(2) << "Callback loop done";
      return;
    }
    {
      tensorflow::profiler::TraceMe traceme("OutfeedReceiver::Callback");
      callback_(received->device(), received->consumer_id(),
                received->literal());
    }
  }
}

Status OutfeedReceiverImpl::SendShutdownOutfeedHeader(int device_idx) {
  const Device* device = devices_[device_idx];
  constexpr int consumer_id = kOutfeedCidShutdown;
  VLOG(2) << "[" << device->DebugString()
          << "] SendSpecialHeader cons=" << consumer_id;
  XlaBuilder builder(
      absl::StrFormat("special_outfeed_header_%d_%d", consumer_id, device_idx));
  XlaOp send =
      AddOutfeedToBuilder(&builder, CreateToken(&builder), consumer_id, {})
          .ValueOrDie();
  XlaComputation computation = builder.Build(send).ValueOrDie();

  CompileOptions compile_options;
  compile_options.executable_build_options.set_num_replicas(1);
  compile_options.executable_build_options.set_num_partitions(1);
  DeviceAssignment device_assignment(1, 1);
  device_assignment(0, 0) = device->id();
  compile_options.executable_build_options.set_device_assignment(
      device_assignment);

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtExecutable> executable,
      PjRtExecutable::Compile(computation, devices_[device_idx]->client(),
                              std::move(compile_options)));
  ExecuteOptions execute_options;
  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<PjRtBuffer>> output_buffers,
                      executable->Execute({}, execute_options));
  return Status::OK();
}

StatusOr<XlaOp> OutfeedReceiverImpl::AddOutfeedToBuilder(
    XlaBuilder* builder, XlaOp token, uint32_t consumer_id,
    std::vector<XlaOp> arrays) {
  XlaOp data = Tuple(builder, std::move(arrays));
  Shape shape_with_layout = builder->GetShape(data).ValueOrDie();
  ShapeUtil::ForEachMutableSubshape(
      &shape_with_layout, [](Shape* subshape, const ShapeIndex&) {
        if (!subshape->has_layout()) {
          LayoutUtil::SetToDefaultLayout(subshape);
        }
      });
  VLOG(2) << "RegisterShape cons=" << consumer_id
          << "; shape=" << shape_with_layout.ToString();
  {
    absl::MutexLock lock(&mu_);
    auto found = shape_registry_.find(consumer_id);
    if (found != shape_registry_.end()) {
      if (!ShapeUtil::Equal(shape_with_layout, found->second)) {
        return InvalidArgument(
            "Shape %s does not match previous shape %s used "
            "for consumer id %d",
            shape_with_layout.DebugString(), found->second.DebugString(),
            consumer_id);
      }
    } else {
      shape_registry_.insert({consumer_id, shape_with_layout});
    }
  }

  std::vector<uint32_t> header{kOutfeedHeaderStart, consumer_id};
  XlaOp header_op = ConstantR1<uint32_t>(builder, header);
  token = OutfeedWithToken(
      header_op, token, ShapeUtil::MakeShape(U32, {kOutfeedHeaderWords}), "");
  if (consumer_id != kOutfeedCidShutdown) {
    token = OutfeedWithToken(data, token, shape_with_layout, "");
  }
  return token;
}

OutfeedReceiver::OutfeedReceiver(Callback callback,
                                 absl::Span<PjRtClient* const> clients,
                                 ssize_t max_callback_queue_size_bytes) {
  p_impl_ = absl::make_unique<OutfeedReceiverImpl>(
      callback, clients, max_callback_queue_size_bytes);
}

OutfeedReceiver::~OutfeedReceiver() {}

void OutfeedReceiver::Start() { p_impl_->Start(); }

StatusOr<XlaOp> OutfeedReceiver::AddOutfeedToBuilder(
    XlaBuilder* builder, XlaOp token, uint32_t consumer_id,
    std::vector<XlaOp> arrays) {
  if (consumer_id == kOutfeedCidShutdown) {
    return InvalidArgument("Consumer ID cannot be a reserved value: %d",
                           consumer_id);
  }
  return p_impl_->AddOutfeedToBuilder(builder, token, consumer_id, arrays);
}

}  // namespace xla
