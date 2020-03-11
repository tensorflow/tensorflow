// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include <complex>
#include <cstddef>
#include <functional>
#include <memory>

#include "grpcpp/grpcpp.h"
#include "absl/base/thread_annotations.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/python/tpu_driver/event_id.h"
#include "tensorflow/compiler/xla/python/tpu_driver/platform/external/compat.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.pb.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_service.grpc.pb.h"
#include "tensorflow/compiler/xla/util.h"

namespace tpu_driver {
namespace {

using xla::Status;

const int64_t kMaxStreamWriteSize = 10 * 1000 * 1000;
const absl::Duration kWriteEpochDuration = absl::Microseconds(10);

constexpr char kGrpcProtocol[] = "grpc://";

class GrpcTpuStream;
class GrpcTpuDriver;

class GrpcEvent : public Event {
 public:
  explicit GrpcEvent(EventId id, GrpcTpuStream* stream)
      : id_(id), stream_(stream) {}
  ~GrpcEvent() override;

  xla::Status Await() override;
  absl::optional<xla::Status> AwaitWithTimeout(
      absl::Duration duration) override;
  void AddCallback(std::function<void(Status)> callback) override;

  EventId id() const { return id_; }
  GrpcTpuStream* stream() const { return stream_; }

 private:
  const EventId id_;
  GrpcTpuStream* stream_;
};

class ErrorEvent : public GrpcEvent {
 public:
  explicit ErrorEvent(Status status) : GrpcEvent(EventId{0, 0}, nullptr) {
    status_ = status;
  }

  xla::Status Await() override { return status_; }
  absl::optional<xla::Status> AwaitWithTimeout(
      absl::Duration duration) override {
    return status_;
  }
  void AddCallback(std::function<void(Status)> callback) override {
    callback(status_);
  }

 private:
  Status status_;
};

class GrpcBufferHandle : public BufferHandle {
 public:
  explicit GrpcBufferHandle(
      EventId id, std::shared_ptr<GrpcEvent> event, int64_t bytes,
      absl::optional<xla::ShapeProto> shape = absl::nullopt)
      : id_(id),
        stream_(event->stream()),
        event_(std::move(event)),
        bytes_(bytes),
        shape_(shape) {}

  std::shared_ptr<Event> OnReady() override { return event_; }
  int64_t size_in_bytes() override { return bytes_; }

  EventId id() const { return id_; }
  GrpcTpuStream* stream() const { return stream_; }

  absl::optional<xla::ShapeProto> shape() override { return shape_; }

 private:
  const EventId id_;
  GrpcTpuStream* stream_;
  std::shared_ptr<GrpcEvent> event_;
  int64_t bytes_;
  absl::optional<xla::ShapeProto> shape_;
};

class GrpcCompiledProgramHandle : public CompiledProgramHandle {
 public:
  explicit GrpcCompiledProgramHandle(EventId id,
                                     std::shared_ptr<GrpcEvent> event)
      : id_(id),
        stream_(event->stream()),
        event_(std::move(event)),
        metadata_(std::make_shared<CompiledProgramMetadata>()) {}

  std::shared_ptr<Event> OnReady() override { return event_; }

  EventId id() const { return id_; }
  GrpcTpuStream* stream() const { return stream_; }

  Status program_shape(xla::ProgramShapeProto* program_shape) override {
    auto opt_status = OnReady()->AwaitWithTimeout(absl::Hours(1));
    if (!opt_status.has_value()) {
      return xla::InternalError("Compile failed to finish within 1 hour.");
    }

    Status status = opt_status.value();
    if (!status.ok()) {
      return status;
    }
    *program_shape = metadata_->program_shape();
    return Status::OK();
  }

  std::shared_ptr<CompiledProgramMetadata> metadata() { return metadata_; }

 private:
  const EventId id_;
  GrpcTpuStream* stream_;
  std::shared_ptr<GrpcEvent> event_;

  // Using a shared pointer here because the program handle can go out of scope
  // before we get a response back, but we want a valid location to write things
  // into regardless.
  std::shared_ptr<CompiledProgramMetadata> metadata_;
};

class GrpcLoadedProgramHandle : public LoadedProgramHandle {
 public:
  explicit GrpcLoadedProgramHandle(EventId id, std::shared_ptr<GrpcEvent> event)
      : id_(id), stream_(event->stream()), event_(std::move(event)) {}

  std::shared_ptr<Event> OnReady() override { return event_; }

  EventId id() const { return id_; }
  GrpcTpuStream* stream() const { return stream_; }

 private:
  const EventId id_;
  GrpcTpuStream* stream_;
  std::shared_ptr<GrpcEvent> event_;
};

class GrpcTpuStream {
 public:
  explicit GrpcTpuStream(int32_t id, GrpcTpuDriver* driver,
                         std::unique_ptr<grpc::CloudTpuDriver::Stub> stub);
  virtual ~GrpcTpuStream();

  std::unique_ptr<BufferHandle> Allocate(int32_t core_id, MemoryRegion region,
                                         int64_t num_bytes,
                                         absl::Span<Event* const> wait_for);
  std::unique_ptr<BufferHandle> Allocate(int32_t core_id, MemoryRegion region,
                                         const xla::ShapeProto& shape,
                                         absl::Span<Event* const> wait_for);
  std::unique_ptr<BufferHandle> AllocateTuple(
      int32_t core_id, MemoryRegion region,
      absl::Span<BufferHandle* const> children,
      absl::Span<Event* const> wait_for);
  std::shared_ptr<Event> Deallocate(std::unique_ptr<BufferHandle> handle,
                                    absl::Span<Event* const> wait_for);

  std::shared_ptr<Event> TransferToDevice(const void* src, BufferHandle* dst,
                                          absl::Span<Event* const> wait_for);
  std::shared_ptr<Event> TransferFromDevice(const BufferHandle* src, void* dst,
                                            absl::Span<Event* const> wait_for);

  std::shared_ptr<Event> TransferFromDeviceToDevice(
      const BufferHandle* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for);

  std::unique_ptr<CompiledProgramHandle> CompileProgram(
      const xla::HloProto& source, int32_t num_replicas,
      absl::Span<Event* const> wait_for);
  std::unique_ptr<LoadedProgramHandle> LoadProgram(
      int32_t core_id, const CompiledProgramHandle* handle,
      absl::Span<Event* const> wait_for);
  std::shared_ptr<Event> UnloadProgram(
      std::unique_ptr<LoadedProgramHandle> handle,
      absl::Span<Event* const> wait_for);
  std::shared_ptr<Event> ExecuteProgram(
      LoadedProgramHandle* program, absl::Span<BufferHandle* const> inputs,
      absl::Span<BufferHandle* const> outputs,
      const xla::DeviceAssignmentProto& device_assignment,
      absl::Span<Event* const> wait_for);

 private:
  friend class GrpcEvent;
  friend class GrpcTpuDriver;

  struct EventInfo {
    bool all_deps_done = false;
    bool done = false;     // response received
    bool deleted = false;  // deleted by the user
    Status status;
    absl::InlinedVector<std::function<void(Status)>, 1> callbacks;
    // Most events should have <= 2 requirement events.
    absl::InlinedVector<EventId, 2> deps;
  };

  struct TransferInfo {
    explicit TransferInfo(void* dst, int64_t num_bytes)
        : dst(dst), num_bytes(num_bytes) {}

    void* const dst;
    const uint64_t num_bytes;
  };

  struct CompileMetadataInfo {
    explicit CompileMetadataInfo(
        std::shared_ptr<CompiledProgramMetadata> metadata) {
      compiled_metadata = metadata;
    }
    std::shared_ptr<CompiledProgramMetadata> compiled_metadata;
  };

  // Every public method above should call this first.
  void InitializeRequest(StreamRequest::Entry* req,
                         absl::Span<Event* const> wait_for)
      ABSL_LOCKS_EXCLUDED(events_mutex_);

  // The first update to an event marks it done and calls registered callbacks.
  // All subsequent updates must have the same OK-ness as the first update.
  // Among non-OK updates, only the first error status is remembered.
  void UpdateEventStatus(EventId id, Status status)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(events_mutex_);

  // To ensure callbacks are still triggered, after this is called, we do not
  // remove the event from the event mapping until a response is received from
  // the server.
  void DeleteEvent(EventId id) ABSL_LOCKS_EXCLUDED(events_mutex_);

  // Wait at most `duration` for event `id` to complete. Returns the event
  // status or an empty optional if the event does not complete in time.
  absl::optional<Status> WaitForEvent(EventId id, absl::Duration duration)
      ABSL_LOCKS_EXCLUDED(events_mutex_);

  void AddEventCallback(EventId id, std::function<void(Status)> callback)
      ABSL_LOCKS_EXCLUDED(events_mutex_);

  void AddWriteRequest(std::unique_ptr<StreamRequest::Entry> req) {
    absl::MutexLock m(&request_lock_);
    VLOG(2) << "Adding request: " << req->DebugString();
    requests_.push_back(std::move(req));
  }

  // Unique identifier for this stream.
  int32_t id_;
  // The parent driver that created this stream.
  GrpcTpuDriver* driver_;

  std::unique_ptr<grpc::CloudTpuDriver::Stub> stub_;
  ::grpc::ClientContext ctx_;
  std::unique_ptr<
      ::grpc::ClientReaderWriterInterface<StreamRequest, StreamResponse>>
      stream_;

  absl::Mutex request_lock_;
  std::deque<std::unique_ptr<StreamRequest::Entry>> requests_
      ABSL_GUARDED_BY(request_lock_);
  int64_t num_pending_requests_ ABSL_GUARDED_BY(request_lock_) = 0;

  bool shutting_down_ ABSL_GUARDED_BY(request_lock_) = false;

  void StreamWriterFn();
  Thread writer_thread_;

  void StreamReaderFn();
  Thread reader_thread_;

  // Map from operation ID to event information.
  absl::Mutex events_mutex_;
  absl::flat_hash_map<EventId, EventInfo> events_
      ABSL_GUARDED_BY(events_mutex_);

  // Map from operation ID to transfer information.
  // When a D2H transfer completes, received data is copied into the `dst`
  // pointer in `TransferInfo`.
  absl::Mutex transfers_mutex_;
  absl::flat_hash_map<EventId, TransferInfo> transfers_
      ABSL_GUARDED_BY(transfers_mutex_);

  absl::Mutex compiles_mutex_;
  absl::flat_hash_map<EventId, CompileMetadataInfo> compiles_
      ABSL_GUARDED_BY(compiles_mutex_);
};

class GrpcTpuDriver : public TpuDriver {
 public:
  explicit GrpcTpuDriver(const TpuDriverConfig& config,
                         std::shared_ptr<::grpc::ChannelCredentials> creds,
                         int32_t client_id)
      : config_(config), creds_(creds), client_id_(client_id) {
    SystemInfo system_info;
    QuerySystemInfo(&system_info);
    for (auto& chip_info : system_info.tpu_chip()) {
      for (auto& core_info : chip_info.core()) {
        int32_t core_id = core_info.id();
        // We have one stream per core, so use core ID as stream ID.
        streams_[core_id] = AllocateStream(core_id);
      }
    }
    CHECK_GT(streams_.size(), 0) << "Can't find any TPU chip in the system.";

    host_stream_ = AllocateStream(-1);
  }

  ~GrpcTpuDriver() override {
    if (closed_) {
      return;
    }
    auto status = Close();
    if (!status.ok()) {
      LOG(ERROR) << status;
    }
  }

  void QuerySystemInfo(SystemInfo* system_info) override;
  Status Reset() override;

  std::unique_ptr<BufferHandle> Allocate(
      int32_t core_id, MemoryRegion region, int64_t num_bytes,
      absl::Span<Event* const> wait_for) override {
    return streams_[core_id]->Allocate(core_id, region, num_bytes, wait_for);
  }
  std::unique_ptr<BufferHandle> Allocate(
      int32_t core_id, MemoryRegion region, const xla::ShapeProto& shape,
      absl::Span<Event* const> wait_for) override {
    return streams_[core_id]->Allocate(core_id, region, shape, wait_for);
  }
  std::unique_ptr<BufferHandle> AllocateTuple(
      int32_t core_id, MemoryRegion region,
      absl::Span<BufferHandle* const> children,
      absl::Span<Event* const> wait_for) override {
    return streams_[core_id]->AllocateTuple(core_id, region, children,
                                            wait_for);
  }
  std::shared_ptr<Event> Deallocate(
      std::unique_ptr<BufferHandle> handle,
      absl::Span<Event* const> wait_for) override {
    auto* stream = static_cast<GrpcBufferHandle*>(handle.get())->stream();
    return stream->Deallocate(std::move(handle), wait_for);
  }

  std::shared_ptr<Event> TransferToDevice(
      const void* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for) override {
    auto* stream = static_cast<GrpcBufferHandle*>(dst)->stream();
    return stream->TransferToDevice(src, dst, wait_for);
  }
  std::shared_ptr<Event> TransferFromDevice(
      const BufferHandle* src, void* dst,
      absl::Span<Event* const> wait_for) override {
    auto* stream = static_cast<const GrpcBufferHandle*>(src)->stream();
    return stream->TransferFromDevice(src, dst, wait_for);
  }

  std::shared_ptr<Event> TransferFromDeviceToDevice(
      const BufferHandle* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for) override {
    auto* stream = static_cast<const GrpcBufferHandle*>(src)->stream();
    return stream->TransferFromDeviceToDevice(src, dst, wait_for);
  }

  std::unique_ptr<CompiledProgramHandle> CompileProgram(
      const xla::HloProto& source, int32_t num_replicas,
      absl::Span<Event* const> wait_for) override {
    // Always compile using the first/default core's stream.
    return streams_[0]->CompileProgram(source, num_replicas, wait_for);
  }
  std::unique_ptr<LoadedProgramHandle> LoadProgram(
      int32_t core_id, const CompiledProgramHandle* handle,
      absl::Span<Event* const> wait_for) override {
    return streams_[core_id]->LoadProgram(core_id, handle, wait_for);
  }
  std::shared_ptr<Event> UnloadProgram(
      std::unique_ptr<LoadedProgramHandle> handle,
      absl::Span<Event* const> wait_for) override {
    auto* stream =
        static_cast<const GrpcLoadedProgramHandle*>(handle.get())->stream();
    return stream->UnloadProgram(std::move(handle), wait_for);
  }
  std::shared_ptr<Event> ExecuteProgram(
      LoadedProgramHandle* program, absl::Span<BufferHandle* const> inputs,
      absl::Span<BufferHandle* const> outputs,
      const xla::DeviceAssignmentProto& device_assignment,
      absl::Span<Event* const> wait_for) override {
    auto* stream =
        static_cast<const GrpcLoadedProgramHandle*>(program)->stream();
    return stream->ExecuteProgram(program, inputs, outputs, device_assignment,
                                  wait_for);
  }

  EventId NewOperationId() { return EventId{client_id_, ++operation_id_}; }

  static std::unique_ptr<grpc::CloudTpuDriver::Stub> CreateTpuDriverStub(
      const TpuDriverConfig& config,
      std::shared_ptr<::grpc::ChannelCredentials> creds);

  uint32_t client_id() const { return client_id_; }

 private:
  Status Close();
  std::unique_ptr<GrpcTpuStream> AllocateStream(int32_t core_id);

  const TpuDriverConfig config_;
  std::shared_ptr<::grpc::ChannelCredentials> creds_;
  const uint32_t client_id_;
  // Map from stream IDs to streams.
  absl::flat_hash_map<int32_t, std::unique_ptr<GrpcTpuStream>> streams_;
  std::unique_ptr<GrpcTpuStream> host_stream_;
  // Shared by all streams.
  std::atomic<uint64_t> operation_id_{0};
  std::atomic<bool> closed_{false};
};  // namespace

GrpcEvent::~GrpcEvent() { stream_->DeleteEvent(id_); }

Status GrpcEvent::Await() {
  auto opt_status = stream_->WaitForEvent(id_, absl::InfiniteDuration());
  return opt_status.value();
}

absl::optional<Status> GrpcEvent::AwaitWithTimeout(absl::Duration duration) {
  return stream_->WaitForEvent(id_, duration);
}

void GrpcEvent::AddCallback(std::function<void(Status)> callback) {
  stream_->AddEventCallback(id_, std::move(callback));
}

GrpcTpuStream::GrpcTpuStream(int32_t id, GrpcTpuDriver* driver,
                             std::unique_ptr<grpc::CloudTpuDriver::Stub> stub)
    : id_(id),
      driver_(driver),
      stub_(std::move(stub)),
      stream_(stub_->StreamExecute(&ctx_)),
      writer_thread_(&GrpcTpuStream::StreamWriterFn, this),
      reader_thread_(&GrpcTpuStream::StreamReaderFn, this) {}

GrpcTpuStream::~GrpcTpuStream() {
  {
    absl::MutexLock lock(&request_lock_);
    shutting_down_ = true;
  }

  VLOG(1) << "Shutting down stream.";
  {
    // Mark all remaining events invalid.
    absl::MutexLock lock(&events_mutex_);
    for (auto e : events_) {
      if (!e.second.done) {
        LOG(ERROR) << "Resetting: " << e.first;
        UpdateEventStatus(e.first, xla::Status(tensorflow::error::Code::ABORTED,
                                               "Driver was closed."));
      }
    }
  }
  VLOG(1) << "Closing stream.";
  stream_->WritesDone();
  stream_->Finish().IgnoreError();
  VLOG(1) << "Waiting for writer.";
  writer_thread_.join();
  VLOG(1) << "Waiting for reader.";
  reader_thread_.join();
}

void GrpcTpuStream::InitializeRequest(StreamRequest::Entry* req,
                                      absl::Span<Event* const> wait_for) {
  auto operation_id = driver_->NewOperationId();
  EventInfo event_info;

  req->set_operation_id(operation_id.AsInt());
  if (wait_for.empty()) {
    event_info.all_deps_done = true;
  } else {
    event_info.deps.reserve(wait_for.size());
    for (auto* event : wait_for) {
      auto grpc_event = static_cast<const GrpcEvent*>(event);
      req->add_wait_for_id(grpc_event->id().AsInt());
      event_info.deps.push_back(grpc_event->id());
    }
  }

  absl::MutexLock lock(&events_mutex_);
  events_[operation_id] = event_info;
}

void GrpcTpuStream::UpdateEventStatus(EventId id, Status status) {
  auto it = events_.find(id);

  // These should only happen when the server shuts down, and our local event
  // cancellation interleaves with server responses. It should be safe to ignore
  // the second updates in these situations.
  if (it == events_.end()) {
    VLOG(1) << "Received a status update: " << status
            << ", but cannot find GrpcEvent " << id;
    return;
  }
  if (it->second.done) {
    // Done and deleted events must have already been removed.
    CHECK(!it->second.deleted);
    VLOG(1) << "Received a second status update: " << status.error_message()
            << ", for GrpcEvent " << id << " already done with status: "
            << it->second.status.error_message();
    return;
  }

  // This is the first time this event finishes. Remember the results and call
  // the callbacks.
  VLOG(1) << "Response received for GrpcEvent " << id << ". "
          << status.ToString() << ". Firing " << it->second.callbacks.size()
          << " callbacks.";
  it->second.done = true;
  it->second.status = status;
  for (const auto& callback : it->second.callbacks) {
    callback(status);
  }

  // Truly remove the event if it's both done and deleted.
  if (it->second.deleted) {
    events_.erase(it);
  }
}

void GrpcTpuStream::DeleteEvent(EventId id) {
  absl::MutexLock lock(&events_mutex_);
  auto it = events_.find(id);
  CHECK(it != events_.end());
  CHECK(!it->second.deleted);
  it->second.deleted = true;
  // Truly remove the event if it's both done and deleted.
  if (it->second.done) {
    events_.erase(it);
  }
}

absl::optional<Status> GrpcTpuStream::WaitForEvent(EventId id,
                                                   absl::Duration duration) {
  events_mutex_.Lock();
  auto it = events_.find(id);

  if (it == events_.end()) {
    // This event has already been marked as done and deleted. Assume success.
    events_mutex_.Unlock();
    return Status::OK();
  }

  if (!it->second.all_deps_done) {
    absl::InlinedVector<EventId, 2> deps = it->second.deps;
    events_mutex_.Unlock();
    for (auto dep : deps) {
      // If a requirement event timed out, no point in any further waiting.
      if (!WaitForEvent(dep, duration)) {
        return absl::nullopt;
      }
    }
    events_mutex_.Lock();
  }

  // Set the flag here, as we're guaranteed they have all completed at this
  // point. This helps terminate recursion on a chain of completed events as
  // soon as possible, at this event.
  it = events_.find(id);
  if (it != events_.end()) {
    it->second.all_deps_done = true;
  }

  auto done = [this, id]() {
    events_mutex_.AssertHeld();
    return !events_.contains(id) || events_[id].done;
  };
  if (events_mutex_.AwaitWithTimeout(absl::Condition(&done), duration)) {
    auto status = events_.contains(id) ? events_[id].status : Status::OK();
    events_mutex_.Unlock();
    return status;
  }
  events_mutex_.Unlock();
  return absl::nullopt;
}

void GrpcTpuStream::AddEventCallback(EventId id,
                                     std::function<void(Status)> callback) {
  absl::MutexLock lock(&events_mutex_);
  auto it = events_.find(id);
  if (it == events_.end()) {
    callback(Status());
    return;
  }
  if (it->second.done) {
    callback(it->second.status);
    return;
  }
  it->second.callbacks.push_back(std::move(callback));
}

static bool ShouldBeginWriting(int64_t* pending_requests) {
  return *pending_requests > 32;
}

void GrpcTpuStream::StreamWriterFn() {
  while (true) {
    request_lock_.LockWhenWithTimeout(
        absl::Condition(&ShouldBeginWriting, &num_pending_requests_),
        kWriteEpochDuration);
    if (shutting_down_) {
      request_lock_.Unlock();
      return;
    }

    if (requests_.empty()) {
      request_lock_.Unlock();
      continue;
    }

    std::vector<StreamRequest> reqs;
    int64_t request_bytes = 0;
    while (!requests_.empty()) {
      StreamRequest::Entry* e = requests_.front().release();
      requests_.pop_front();
      const int64_t entry_bytes = e->ByteSizeLong();
      if (reqs.empty() || request_bytes + entry_bytes > kMaxStreamWriteSize) {
        reqs.push_back(StreamRequest());
        request_bytes = 0;
      }
      VLOG(1) << "Sending request: " << EventId::FromInt(e->operation_id());
      VLOG(2) << "Sending request: " << e->DebugString();
      reqs.back().mutable_entry()->AddAllocated(e);
    }
    num_pending_requests_ = 0;
    request_lock_.Unlock();

    for (const auto& r : reqs) {
      TraceMe activity(absl::StrCat("GrpcTpuStream::Send "));
      ::grpc::WriteOptions opts;
      opts.set_no_compression().clear_buffer_hint();
      stream_->Write(r, opts);
    }
  }
}

void GrpcTpuStream::StreamReaderFn() {
  StreamResponse resp;
  while (stream_->Read(&resp)) {
    VLOG(2) << "Received response: " << resp.DebugString();
    for (const StreamResponse::Entry entry : resp.entry()) {
      EventId event_id = EventId::FromInt(entry.operation_id());
      VLOG(1) << "Received response for: " << event_id;

      TraceMe activity("GrpcTpuStream::RequestComplete");
      if (entry.has_transfer_from()) {
        TraceMe activity("GrpcTpuStream::TransferFromComplete");
        absl::MutexLock lock(&transfers_mutex_);
        auto it = transfers_.find(event_id);
        CHECK(it != transfers_.end());
        VLOG(1) << "Copying: " << it->second.num_bytes << " to position "
                << it->second.dst;
        if (entry.transfer_from().data().size() != it->second.num_bytes) {
          absl::MutexLock lock(&events_mutex_);
          UpdateEventStatus(
              event_id,
              Status(
                  tensorflow::error::Code::DATA_LOSS,
                  absl::StrCat("Expected ", it->second.num_bytes, " received ",
                               entry.transfer_from().data().size())));
          continue;
        }
        memcpy(it->second.dst, entry.transfer_from().data().data(),
               it->second.num_bytes);
      }

      if (entry.has_compile()) {
        TraceMe activity("GrpcTpuStream::CompileComplete");
        absl::MutexLock lock(&compiles_mutex_);
        auto it = compiles_.find(event_id);
        CHECK(it != compiles_.end());
        *it->second.compiled_metadata = entry.compile().metadata();
      }

      absl::MutexLock lock(&events_mutex_);
      if (entry.status().code() != tensorflow::error::Code::OK) {
        UpdateEventStatus(
            event_id,
            Status(static_cast<tensorflow::error::Code>(entry.status().code()),
                   entry.status().message()));
      } else {
        UpdateEventStatus(event_id, Status::OK());
      }
    }
  }
}

std::unique_ptr<BufferHandle> GrpcTpuStream::Allocate(
    int32_t core_id, MemoryRegion region, int64_t num_bytes,
    absl::Span<Event* const> wait_for) {
  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  TraceMe activity(absl::StrCat("GrpcTpuStream::Allocate(num_bytes)"));
  req->mutable_alloc()->set_core_id(core_id);
  req->mutable_alloc()->set_region(region);
  req->mutable_alloc()->set_num_bytes(num_bytes);
  auto event =
      std::make_shared<GrpcEvent>(EventId::FromInt(req->operation_id()), this);
  AddWriteRequest(std::move(req));
  return absl::make_unique<GrpcBufferHandle>(event->id(), std::move(event),
                                             num_bytes);
}

std::unique_ptr<BufferHandle> GrpcTpuStream::Allocate(
    int32_t core_id, MemoryRegion region, const xla::ShapeProto& shape,
    absl::Span<Event* const> wait_for) {
  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  TraceMe activity(absl::StrCat("GrpcTpuStream::Allocate(shape)"));
  req->mutable_alloc()->set_core_id(core_id);
  req->mutable_alloc()->set_region(region);
  *req->mutable_alloc()->mutable_shape() = shape;
  auto event =
      std::make_shared<GrpcEvent>(EventId::FromInt(req->operation_id()), this);
  AddWriteRequest(std::move(req));
  return absl::make_unique<GrpcBufferHandle>(
      event->id(), std::move(event), ComputeBytesFromShape(shape), shape);
}

std::unique_ptr<BufferHandle> GrpcTpuStream::AllocateTuple(
    int32_t core_id, MemoryRegion region,
    absl::Span<BufferHandle* const> children,
    absl::Span<Event* const> wait_for) {
  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  TraceMe activity(absl::StrCat("GrpcTpuStream::AllocateTuple"));
  req->mutable_alloc_tuple()->set_core_id(core_id);
  req->mutable_alloc_tuple()->set_region(region);
  for (auto child : children) {
    auto grpc_child = static_cast<GrpcBufferHandle*>(child);
    req->mutable_alloc_tuple()->add_children(grpc_child->id().AsInt());
  }
  auto event =
      std::make_shared<GrpcEvent>(EventId::FromInt(req->operation_id()), this);
  AddWriteRequest(std::move(req));
  return absl::make_unique<GrpcBufferHandle>(event->id(), std::move(event), 0);
}

std::shared_ptr<Event> GrpcTpuStream::Deallocate(
    std::unique_ptr<BufferHandle> handle, absl::Span<Event* const> wait_for) {
  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  TraceMe activity(absl::StrCat("GrpcTpuStream::Deallocate"));
  auto grpc_handle = static_cast<GrpcBufferHandle*>(handle.get());
  req->mutable_dealloc()->set_handle(grpc_handle->id().AsInt());
  auto event =
      std::make_shared<GrpcEvent>(EventId::FromInt(req->operation_id()), this);
  AddWriteRequest(std::move(req));
  return event;
}

std::shared_ptr<Event> GrpcTpuStream::TransferToDevice(
    const void* src, BufferHandle* dst, absl::Span<Event* const> wait_for) {
  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  TraceMe activity(absl::StrCat("GrpcTpuStream::TransferToDevice"));
  req->mutable_transfer_to()->mutable_data()->assign(
      static_cast<const char*>(src), dst->size_in_bytes());
  req->mutable_transfer_to()->set_target_handle(
      static_cast<GrpcBufferHandle*>(dst)->id().AsInt());
  auto event =
      std::make_shared<GrpcEvent>(EventId::FromInt(req->operation_id()), this);
  AddWriteRequest(std::move(req));
  return event;
}

std::shared_ptr<Event> GrpcTpuStream::TransferFromDevice(
    const BufferHandle* src, void* dst, absl::Span<Event* const> wait_for) {
  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  TraceMe activity(absl::StrCat("GrpcTpuStream::TransferFromDevice"));
  req->mutable_transfer_from()->set_source_handle(
      static_cast<const GrpcBufferHandle*>(src)->id().AsInt());
  EventId event_id = EventId::FromInt(req->operation_id());
  {
    absl::MutexLock lock(&transfers_mutex_);
    TransferInfo info(dst, const_cast<BufferHandle*>(src)->size_in_bytes());
    transfers_.insert(std::make_pair(event_id, info));
  }
  auto event = std::make_shared<GrpcEvent>(event_id, this);
  AddWriteRequest(std::move(req));
  return event;
}

std::shared_ptr<Event> GrpcTpuStream::TransferFromDeviceToDevice(
    const BufferHandle* src, BufferHandle* dst,
    absl::Span<Event* const> wait_for) {
  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  TraceMe activity(absl::StrCat("GrpcTpuStream::TransferFromDeviceToDevice",
                                req->operation_id()));

  req->mutable_transfer_from_to()->set_source_handle(
      static_cast<const GrpcBufferHandle*>(src)->id().AsInt());
  req->mutable_transfer_from_to()->set_target_handle(
      static_cast<const GrpcBufferHandle*>(dst)->id().AsInt());
  EventId event_id = EventId::FromInt(req->operation_id());
  auto event = std::make_shared<GrpcEvent>(event_id, this);
  AddWriteRequest(std::move(req));
  return event;
}

std::unique_ptr<CompiledProgramHandle> GrpcTpuStream::CompileProgram(
    const xla::HloProto& source, int32_t num_replicas,
    absl::Span<Event* const> wait_for) {
  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  TraceMe activity(absl::StrCat("GrpcTpuStream::CompileProgram"));
  *req->mutable_compile()->mutable_hlo_program() = source;
  req->mutable_compile()->set_num_replicas(num_replicas);
  EventId event_id = EventId::FromInt(req->operation_id());

  auto event =
      std::make_shared<GrpcEvent>(EventId::FromInt(req->operation_id()), this);

  auto handle = absl::make_unique<GrpcCompiledProgramHandle>(event->id(),
                                                             std::move(event));
  {
    absl::MutexLock lock(&compiles_mutex_);
    CompileMetadataInfo info(handle->metadata());
    compiles_.insert(std::make_pair(event_id, info));
  }

  AddWriteRequest(std::move(req));
  return std::move(handle);
}

std::unique_ptr<LoadedProgramHandle> GrpcTpuStream::LoadProgram(
    int32_t core_id, const CompiledProgramHandle* handle,
    absl::Span<Event* const> wait_for) {
  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  TraceMe activity(absl::StrCat("GrpcTpuStream::LoadProgram"));
  req->mutable_load()->set_core_id(core_id);
  auto grpc_handle = static_cast<const GrpcCompiledProgramHandle*>(handle);
  if (grpc_handle->id().client_id != driver_->client_id()) {
    auto event = std::make_shared<ErrorEvent>(
        xla::InvalidArgument("Invalid program handle (wrong client id). Did "
                             "you restart the server or use a stale handle?"));
    return absl::make_unique<GrpcLoadedProgramHandle>(event->id(),
                                                      std::move(event));
  }
  req->mutable_load()->set_compiled_program_handle(grpc_handle->id().AsInt());
  auto event =
      std::make_shared<GrpcEvent>(EventId::FromInt(req->operation_id()), this);
  AddWriteRequest(std::move(req));
  return absl::make_unique<GrpcLoadedProgramHandle>(event->id(),
                                                    std::move(event));
}

std::shared_ptr<Event> GrpcTpuStream::UnloadProgram(
    std::unique_ptr<LoadedProgramHandle> handle,
    absl::Span<Event* const> wait_for) {
  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  TraceMe activity(absl::StrCat("GrpcTpuStream::UnloadProgram"));
  req->mutable_unload()->set_loaded_program_handle(
      static_cast<GrpcLoadedProgramHandle*>(handle.get())->id().AsInt());
  auto event =
      std::make_shared<GrpcEvent>(EventId::FromInt(req->operation_id()), this);
  AddWriteRequest(std::move(req));
  return event;
}

std::shared_ptr<Event> GrpcTpuStream::ExecuteProgram(
    LoadedProgramHandle* program, absl::Span<BufferHandle* const> inputs,
    absl::Span<BufferHandle* const> outputs,
    const xla::DeviceAssignmentProto& device_assignment,
    absl::Span<Event* const> wait_for) {
  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  auto program_handle = static_cast<GrpcLoadedProgramHandle*>(program);
  if (program_handle->id().client_id != driver_->client_id()) {
    return std::make_shared<ErrorEvent>(
        xla::InvalidArgument("Invalid program handle (wrong client id). Did "
                             "you restart the server or use a stale handle?"));
  }

  req->mutable_execute()->set_loaded_program_handle(
      program_handle->id().AsInt());

  for (BufferHandle* input : inputs) {
    auto* grpc_handle = static_cast<GrpcBufferHandle*>(input);
    if (grpc_handle->id().client_id != driver_->client_id()) {
      return std::make_shared<ErrorEvent>(xla::InvalidArgument(
          "Invalid input buffer (wrong client id). Did you restart the server "
          "or use a stale handle?"));
    }
    req->mutable_execute()->add_input_handle(grpc_handle->id().AsInt());
  }

  for (BufferHandle* output : outputs) {
    auto* grpc_handle = static_cast<GrpcBufferHandle*>(output);
    if (grpc_handle->id().client_id != driver_->client_id()) {
      return std::make_shared<ErrorEvent>(xla::InvalidArgument(
          "Invalid output buffer (wrong client id). Did you restart the server "
          "or use a stale handle?"));
    }
    req->mutable_execute()->add_output_handle(
        static_cast<GrpcBufferHandle*>(output)->id().AsInt());
  }
  // Only pass along device_assignment if it's not default constructed.
  if (!(device_assignment.replica_count() == 0 &&
        device_assignment.computation_count() == 0)) {
    *req->mutable_execute()->mutable_device_assignment() = device_assignment;
  }
  auto event =
      std::make_shared<GrpcEvent>(EventId::FromInt(req->operation_id()), this);
  AddWriteRequest(std::move(req));
  return event;
}

/*static*/ std::unique_ptr<grpc::CloudTpuDriver::Stub>
GrpcTpuDriver::CreateTpuDriverStub(
    const TpuDriverConfig& config,
    std::shared_ptr<::grpc::ChannelCredentials> creds) {
  ::grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());
  args.SetMaxSendMessageSize(std::numeric_limits<int>::max());

  // Send at least 20 keep-alives before giving up.
  int keepalive_timeout_ms = config.grpc().keepalive_timeout_secs() * 1000;
  int keepalive_interval_ms = keepalive_timeout_ms / 20;

  grpc_arg client_arg_vals[] = {
      {.type = GRPC_ARG_INTEGER,
       .key = const_cast<char*>(
           GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS),
       .value = {.integer = keepalive_interval_ms}},
      {.type = GRPC_ARG_INTEGER,
       .key = const_cast<char*>(GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA),
       .value = {.integer = 0}},  // unlimited
      {.type = GRPC_ARG_INTEGER,
       .key = const_cast<char*>(GRPC_ARG_KEEPALIVE_TIME_MS),
       .value = {.integer = keepalive_interval_ms}},
      {.type = GRPC_ARG_INTEGER,
       .key = const_cast<char*>(GRPC_ARG_KEEPALIVE_TIMEOUT_MS),
       .value = {.integer = keepalive_timeout_ms}},
      {.type = GRPC_ARG_INTEGER,
       .key = const_cast<char*>(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS),
       .value = {.integer = 1}},
      {.type = GRPC_ARG_INTEGER,
       .key = const_cast<char*>(GRPC_ARG_HTTP2_WRITE_BUFFER_SIZE),
       .value = {.integer = 64 * 1000 * 1000}}};

  grpc_channel_args client_args = {.num_args = 6, .args = client_arg_vals};
  args.SetChannelArgs(&client_args);

  // strips out 'grpc://'
  auto worker_addr = absl::StripPrefix(config.worker(), kGrpcProtocol);
  std::shared_ptr<::grpc::Channel> channel =
      ::grpc::CreateCustomChannel(std::string(worker_addr), creds, args);
  return grpc::CloudTpuDriver::NewStub(channel);
}

std::unique_ptr<GrpcTpuStream> GrpcTpuDriver::AllocateStream(int32_t id) {
  auto stub = CreateTpuDriverStub(config_, creds_);
  ::grpc::ClientContext ctx;
  ctx.set_fail_fast(false);
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
  return absl::make_unique<GrpcTpuStream>(id, this, std::move(stub));
}

void GrpcTpuDriver::QuerySystemInfo(SystemInfo* system_info) {
  auto stub = CreateTpuDriverStub(config_, creds_);
  ::grpc::ClientContext ctx;
  ctx.set_fail_fast(false);
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));

  QuerySystemInfoRequest req;
  QuerySystemInfoResponse resp;
  ::grpc::Status status = stub->QuerySystemInfo(&ctx, req, &resp);
  if (!status.ok()) {
    LOG(ERROR) << "QuerySystemInfo request failed: " << status.error_code()
               << ": " << status.error_message() << ": "
               << status.error_details();
    return;
  }
  *system_info = resp.system_info();
}

Status GrpcTpuDriver::Reset() {
  auto stub = CreateTpuDriverStub(config_, creds_);
  ::grpc::ClientContext ctx;
  ctx.set_fail_fast(false);
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
  ResetRequest req;
  ResetResponse resp;
  ::grpc::Status status = stub->Reset(&ctx, req, &resp);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to reset the gRPC driver: " << status.error_code()
               << ": " << status.error_message() << ": "
               << status.error_details();
    return xla::Status(tensorflow::error::Code(status.error_code()),
                       absl::StrCat("Failed to reset TPU driver. Error was: ",
                                    status.error_message(),
                                    ". Details: ", status.error_details()));
  }
  streams_.clear();
  host_stream_.reset();
  return Close();
}

Status GrpcTpuDriver::Close() {
  auto stub = CreateTpuDriverStub(config_, creds_);
  ::grpc::ClientContext ctx;
  ctx.set_fail_fast(false);
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
  CloseRequest req;
  req.set_client_id(client_id_);
  CloseResponse resp;
  ::grpc::Status status = stub->Close(&ctx, req, &resp);
  if (!status.ok()) {
    return xla::Status(tensorflow::error::Code(status.error_code()),
                       absl::StrCat("Failed to close TPU driver. Error was: ",
                                    status.error_message(),
                                    ". Details: ", status.error_details()));
  }
  closed_ = true;
  return Status::OK();
}
}  // namespace

xla::StatusOr<std::unique_ptr<TpuDriver>> CreateGrpcTpuDriver(
    const TpuDriverConfig& config,
    std::shared_ptr<::grpc::ChannelCredentials> creds) {
  auto stub = GrpcTpuDriver::CreateTpuDriverStub(config, creds);
  ::grpc::ClientContext ctx;
  ctx.set_fail_fast(false);
  ctx.set_deadline(
      std::chrono::system_clock::now() +
      std::chrono::seconds(config.grpc().connection_timeout_secs()));
  OpenRequest req;
  OpenResponse resp;
  ::grpc::Status status = stub->Open(&ctx, req, &resp);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to open the gRPC driver: " << status.error_code()
               << ": " << status.error_message() << ": "
               << status.error_details();
    return xla::Status(
        tensorflow::error::Code(status.error_code()),
        absl::StrCat(
            "Failed to connect to remote server at address: ", config.worker(),
            ". Error from gRPC: ", status.error_message(),
            ". Details: ", status.error_details()));
  }
  return std::unique_ptr<TpuDriver>(
      new GrpcTpuDriver(config, creds, resp.client_id()));
}

REGISTER_TPU_DRIVER(
    "grpc://",
    [](const TpuDriverConfig& config)
        -> xla::StatusOr<std::unique_ptr<TpuDriver>> {
      if (absl::StartsWith(config.worker(), "grpc://localhost")) {
        LOG(INFO) << "Using local credentials for localhost: connection.";
        return CreateGrpcTpuDriver(
            config, ::grpc::experimental::LocalCredentials(LOCAL_TCP));
      } else {
        return CreateGrpcTpuDriver(config,
                                   ::grpc::InsecureChannelCredentials());
      }
    });

}  // namespace tpu_driver
