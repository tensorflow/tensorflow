// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
// =============================================================================

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/pjrt/semaphore.h"
#include "tensorflow/compiler/xla/pjrt/worker_thread.h"
#include "tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tpu_driver {
namespace {

using xla::Status;
using xla::WorkerThread;

const char kPodTpuDriverPrefix[] = "grpc+pod://";

class PodTpuDriver;

class PodEvent : public Event {
 public:
  explicit PodEvent(PodTpuDriver* driver, int64 operation_id)
      : driver_(driver), operation_id_(operation_id) {}
  int64_t operation_id() const { return operation_id_; }

  xla::Status Await() override;

  absl::optional<xla::Status> AwaitWithTimeout(
      absl::Duration duration) override;

  void AddCallback(std::function<void(Status)> callback) override;

 private:
  PodTpuDriver* driver_;
  const int64_t operation_id_;
};

class CombinedEvent : public PodEvent {
 public:
  explicit CombinedEvent(PodTpuDriver* driver, int64 operation_id,
                         std::vector<std::shared_ptr<Event>> events)
      : PodEvent(driver, operation_id), events_(events) {}

  xla::Status Await() override {
    for (auto& event : events_) {
      TF_RETURN_IF_ERROR(event->Await());
    }
    return Status::OK();
  }

  absl::optional<xla::Status> AwaitWithTimeout(
      absl::Duration duration) override {
    // TODO(frankchn): This might extend the timeout.
    for (auto& event : events_) {
      auto status = event->AwaitWithTimeout(duration);
      if (status == absl::nullopt) {
        return absl::nullopt;
      } else {
        TF_RETURN_IF_ERROR(status.value());
      }
    }
    return Status::OK();
  }

  void AddCallback(std::function<void(Status)> callback) override {
    // TODO(frankchn): This may return before every event is done.
    events_[0]->AddCallback(std::move(callback));
  }

 private:
  std::vector<std::shared_ptr<Event>> events_;
};

class PodBufferHandle : public BufferHandle {
 public:
  explicit PodBufferHandle(PodTpuDriver* driver, int64_t operation_id,
                           int64_t size_in_bytes,
                           absl::optional<xla::ShapeProto> shape,
                           int64_t core_id)
      : driver_(driver),
        operation_id_(operation_id),
        size_in_bytes_(size_in_bytes),
        shape_(shape),
        event_(std::make_shared<PodEvent>(driver_, operation_id_)),
        core_id_(core_id) {}

  std::shared_ptr<Event> OnReady() override { return event_; }
  int64_t size_in_bytes() override { return size_in_bytes_; }
  absl::optional<xla::ShapeProto> shape() override { return shape_; }

  int64_t operation_id() const { return operation_id_; }
  int64_t core_id() const { return core_id_; }

 private:
  PodTpuDriver* driver_;
  const int64_t operation_id_;
  const int64_t size_in_bytes_;
  const absl::optional<xla::ShapeProto> shape_;
  std::shared_ptr<PodEvent> event_;
  const int64_t core_id_;
};

class PodCompiledProgramHandle : public CompiledProgramHandle {
 public:
  explicit PodCompiledProgramHandle(PodTpuDriver* driver, int64_t operation_id)
      : driver_(driver),
        operation_id_(operation_id),
        event_(std::make_shared<PodEvent>(driver_, operation_id_)) {}

  std::shared_ptr<Event> OnReady() override { return event_; }

  xla::Status program_shape(xla::ProgramShapeProto* program_shape) override;

  int64_t operation_id() const { return operation_id_; }

 private:
  PodTpuDriver* driver_;
  const int64_t operation_id_;
  std::shared_ptr<PodEvent> event_;
};

class PodLoadedProgramHandle : public LoadedProgramHandle {
 public:
  explicit PodLoadedProgramHandle(PodTpuDriver* driver, int64_t operation_id,
                                  int64_t core_id)
      : driver_(driver),
        operation_id_(operation_id),
        core_id_(core_id),
        event_(std::make_shared<PodEvent>(driver_, operation_id_)) {}

  std::shared_ptr<Event> OnReady() override { return event_; }

  int64_t operation_id() const { return operation_id_; }
  int64_t core_id() const { return core_id_; }

 private:
  PodTpuDriver* driver_;
  const int64_t operation_id_;
  const int64_t core_id_;
  std::shared_ptr<PodEvent> event_;
};

struct EventInFlight {
  std::shared_ptr<Event> underlying_event;
  std::function<std::shared_ptr<Event>(void)> create_fn;

  absl::flat_hash_set<int64_t> incomplete_deps;
  std::vector<std::function<void(Status)>> callbacks;
};

class PodTpuDriver : public TpuDriver {
 public:
  explicit PodTpuDriver(const TpuDriverConfig& config,
                        std::shared_ptr<::grpc::ChannelCredentials> creds)
      : config_(config),
        creds_(creds),
        event_thread_(tensorflow::Env::Default(), "grpc_pod_event_thread") {
    std::vector<std::string> workers = absl::StrSplit(
        absl::StripPrefix(config.worker(), kPodTpuDriverPrefix), ',');
    for (const auto& worker : workers) {
      TpuDriverConfig worker_config(config_);
      *(worker_config.mutable_worker()) = absl::StrCat("grpc://", worker);
      drivers_.push_back(
          CreateGrpcTpuDriver(worker_config, creds_).ConsumeValueOrDie());
    }

    for (int driver_num = 0; driver_num < workers.size(); ++driver_num) {
      SystemInfo driver_info;
      drivers_[driver_num]->QuerySystemInfo(&driver_info);

      for (const auto& tpu_chip : driver_info.tpu_chip()) {
        *(pod_info_.add_tpu_chip()) = tpu_chip;
      }

      int core_num = 0;
      for (const auto& tpu_core : driver_info.local_core()) {
        *(pod_info_.add_local_core()) = tpu_core;
        core_to_driver_.push_back(drivers_[driver_num].get());
        core_to_driver_id_.push_back(driver_num);
        core_to_driver_core_.push_back(core_num++);
      }
      *(pod_info_.mutable_cpu()) = driver_info.cpu();
      pod_info_.set_host_count(pod_info_.host_count() + 1);
      pod_info_.set_chip_count(pod_info_.chip_count() +
                               driver_info.chip_count());
      pod_info_.set_core_count(pod_info_.core_count() +
                               driver_info.core_count());
    }
    pod_info_.set_host_id(0);
  }

  ~PodTpuDriver() override {
    // TODO(frankchn): Unload all handles, and wait for all events to finish.
  }

  void QuerySystemInfo(SystemInfo* system_info) override {
    *system_info = pod_info_;
  }

  xla::Status Reset() override {
    for (auto& driver : drivers_) {
      TF_RETURN_IF_ERROR(driver->Reset());
    }
    return xla::Status::OK();
  }

  std::unique_ptr<BufferHandle> Allocate(
      int32_t core_id, MemoryRegion region, int64_t num_bytes,
      absl::Span<Event* const> wait_for) override {
    int64_t operation_id = GetOperationId();
    auto deps = GetDependencyOperationIds(wait_for);

    ScheduleRequest(
        operation_id,
        [this, core_id, region, num_bytes, operation_id]() {
          absl::MutexLock l(&mu_);
          underlying_buffers_.insert(
              {operation_id,
               core_to_driver_[core_id]->Allocate(core_to_driver_core_[core_id],
                                                  region, num_bytes, {})});
          return underlying_buffers_[operation_id]->OnReady();
        },
        deps);

    return absl::make_unique<PodBufferHandle>(this, operation_id, num_bytes,
                                              absl::nullopt, core_id);
  }

  std::unique_ptr<BufferHandle> Allocate(
      int32_t core_id, MemoryRegion region, const xla::ShapeProto& shape,
      absl::Span<Event* const> wait_for) override {
    int64_t operation_id = GetOperationId();
    auto deps = GetDependencyOperationIds(wait_for);

    ScheduleRequest(
        operation_id,
        [this, core_id, region, shape, operation_id]() {
          absl::MutexLock l(&mu_);
          underlying_buffers_.insert(
              {operation_id,
               core_to_driver_[core_id]->Allocate(core_to_driver_core_[core_id],
                                                  region, shape, {})});
          return underlying_buffers_[operation_id]->OnReady();
        },
        deps);

    return absl::make_unique<PodBufferHandle>(
        this, operation_id, ComputeBytesFromShape(shape), shape, core_id);
  }

  std::unique_ptr<BufferHandle> AllocateTuple(
      int32_t core_id, MemoryRegion region,
      absl::Span<BufferHandle* const> children,
      absl::Span<Event* const> wait_for) override {
    int64_t operation_id = GetOperationId();
    auto deps = GetDependencyOperationIds(wait_for);

    std::vector<int64_t> children_ids;
    for (int i = 0; i < children.size(); ++i) {
      auto child_op_id =
          static_cast<PodBufferHandle* const>(children[i])->operation_id();
      deps.insert(child_op_id);
      children_ids.push_back(child_op_id);
    }

    ScheduleRequest(
        operation_id,
        [this, core_id, region, children_ids, operation_id]() {
          absl::MutexLock l(&mu_);

          std::vector<BufferHandle*> child_buffers;
          child_buffers.reserve(children_ids.size());
          for (int i = 0; i < children_ids.size(); ++i) {
            child_buffers.push_back(underlying_buffers_[children_ids[i]].get());
          }

          underlying_buffers_.insert(
              {operation_id,
               core_to_driver_[core_id]->AllocateTuple(
                   core_to_driver_core_[core_id], region, child_buffers, {})});
          return underlying_buffers_[operation_id]->OnReady();
        },
        deps);

    return absl::make_unique<PodBufferHandle>(this, operation_id, 0,
                                              absl::nullopt, core_id);
  }

  std::shared_ptr<Event> Deallocate(
      std::unique_ptr<BufferHandle> handle,
      absl::Span<Event* const> wait_for) override {
    int64_t operation_id = GetOperationId();
    auto deps = GetDependencyOperationIds(wait_for);
    deps.insert(static_cast<PodBufferHandle*>(handle.get())->operation_id());

    auto op_id = static_cast<PodBufferHandle*>(handle.get())->operation_id();
    auto core_id = static_cast<PodBufferHandle*>(handle.get())->core_id();

    ScheduleRequest(
        operation_id,
        [this, op_id, core_id]() {
          absl::MutexLock l(&mu_);
          auto buf_iter = underlying_buffers_.find(op_id);
          auto underlying_hn = std::move(buf_iter->second);
          underlying_buffers_.erase(buf_iter);

          return core_to_driver_[core_id]->Deallocate(std::move(underlying_hn),
                                                      {});
        },
        deps);

    return std::make_shared<PodEvent>(this, operation_id);
  }

  std::shared_ptr<Event> TransferToDevice(
      const void* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for) override {
    int64_t operation_id = GetOperationId();
    auto deps = GetDependencyOperationIds(wait_for);
    deps.insert(static_cast<PodBufferHandle*>(dst)->operation_id());

    auto op_id = static_cast<PodBufferHandle*>(dst)->operation_id();
    auto core_id = static_cast<PodBufferHandle*>(dst)->core_id();

    ScheduleRequest(
        operation_id,
        [this, src, op_id, core_id]() {
          absl::MutexLock l(&mu_);
          auto buf_iter = underlying_buffers_.find(op_id);
          return core_to_driver_[core_id]->TransferToDevice(
              src, buf_iter->second.get(), {});
        },
        deps);

    return std::make_shared<PodEvent>(this, operation_id);
  }

  std::shared_ptr<Event> TransferFromDevice(
      const BufferHandle* src, void* dst,
      absl::Span<Event* const> wait_for) override {
    int64_t operation_id = GetOperationId();
    auto deps = GetDependencyOperationIds(wait_for);
    deps.insert(static_cast<const PodBufferHandle*>(src)->operation_id());

    auto op_id = static_cast<const PodBufferHandle*>(src)->operation_id();
    auto core_id = static_cast<const PodBufferHandle*>(src)->core_id();

    ScheduleRequest(
        operation_id,
        [this, dst, op_id, core_id]() {
          absl::MutexLock l(&mu_);
          auto buf_iter = underlying_buffers_.find(op_id);
          return core_to_driver_[core_id]->TransferFromDevice(
              buf_iter->second.get(), dst, {});
        },
        deps);

    return std::make_shared<PodEvent>(this, operation_id);
  }

  std::shared_ptr<Event> TransferFromDeviceToDevice(
      const BufferHandle* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for) override {
    int64_t operation_id = GetOperationId();
    auto deps = GetDependencyOperationIds(wait_for);
    deps.insert(static_cast<const PodBufferHandle*>(src)->operation_id());
    deps.insert(static_cast<PodBufferHandle*>(dst)->operation_id());

    auto src_op_id = static_cast<const PodBufferHandle*>(src)->operation_id();
    auto dst_op_id = static_cast<PodBufferHandle*>(dst)->operation_id();
    auto core_id = static_cast<PodBufferHandle*>(dst)->core_id();

    ScheduleRequest(
        operation_id,
        [this, src_op_id, dst_op_id, core_id]() {
          absl::MutexLock l(&mu_);
          auto src_iter = underlying_buffers_.find(src_op_id);
          auto dst_iter = underlying_buffers_.find(dst_op_id);
          return core_to_driver_[core_id]->TransferFromDeviceToDevice(
              src_iter->second.get(), dst_iter->second.get(), {});
        },
        deps);

    return std::make_shared<PodEvent>(this, operation_id);
  }

  std::unique_ptr<CompiledProgramHandle> CompileProgram(
      const xla::HloProto& source, int32_t num_replicas,
      absl::Span<Event* const> wait_for) override {
    int64_t operation_id = GetOperationId();
    auto deps = GetDependencyOperationIds(wait_for);

    ScheduleRequest(
        operation_id,
        [this, operation_id, source, num_replicas]() {
          absl::MutexLock l(&mu_);
          auto cph_iterator =
              underlying_cph_
                  .insert(
                      {operation_id,
                       std::vector<std::unique_ptr<CompiledProgramHandle>>()})
                  .first;

          std::vector<std::shared_ptr<Event>> collected_events;
          for (int i = 0; i < drivers_.size(); ++i) {
            auto current_cph =
                drivers_[i]->CompileProgram(source, num_replicas, {});
            cph_iterator->second.push_back(std::move(current_cph));
            collected_events.push_back(cph_iterator->second[i]->OnReady());
          }
          return std::make_shared<CombinedEvent>(this, operation_id,
                                                 collected_events);
        },
        deps);

    return absl::make_unique<PodCompiledProgramHandle>(this, operation_id);
  }

  std::unique_ptr<LoadedProgramHandle> LoadProgram(
      int32_t core_id, const CompiledProgramHandle* handle,
      absl::Span<Event* const> wait_for) override {
    int64_t operation_id = GetOperationId();
    auto deps = GetDependencyOperationIds(wait_for);
    deps.insert(
        static_cast<const PodCompiledProgramHandle*>(handle)->operation_id());
    auto cph_op_id =
        static_cast<const PodCompiledProgramHandle*>(handle)->operation_id();

    ScheduleRequest(
        operation_id,
        [this, operation_id, cph_op_id, core_id]() {
          absl::MutexLock l(&mu_);
          auto cph_iter = underlying_cph_.find(cph_op_id);

          underlying_lph_.insert(
              {operation_id,
               core_to_driver_[core_id]->LoadProgram(
                   core_to_driver_core_[core_id],
                   cph_iter->second[core_to_driver_id_[core_id]].get(), {})});

          return underlying_lph_[operation_id]->OnReady();
        },
        deps);

    return absl::make_unique<PodLoadedProgramHandle>(this, operation_id,
                                                     core_id);
  }

  std::shared_ptr<Event> UnloadProgram(
      std::unique_ptr<LoadedProgramHandle> handle,
      absl::Span<Event* const> wait_for) override {
    int64_t operation_id = GetOperationId();
    auto deps = GetDependencyOperationIds(wait_for);
    deps.insert(
        static_cast<PodLoadedProgramHandle*>(handle.get())->operation_id());
    auto op_id =
        static_cast<PodLoadedProgramHandle*>(handle.get())->operation_id();
    auto core_id =
        static_cast<PodLoadedProgramHandle*>(handle.get())->core_id();

    ScheduleRequest(
        operation_id,
        [this, op_id, core_id]() {
          absl::MutexLock l(&mu_);

          auto lph_iter = underlying_lph_.find(op_id);
          auto event = core_to_driver_[core_id]->UnloadProgram(
              std::move(lph_iter->second), {});
          underlying_lph_.erase(lph_iter);

          return event;
        },
        deps);

    return std::make_shared<PodEvent>(this, operation_id);
  }

  std::shared_ptr<Event> ExecuteProgram(
      LoadedProgramHandle* program, absl::Span<BufferHandle* const> inputs,
      absl::Span<BufferHandle* const> outputs,
      const xla::DeviceAssignmentProto& device_assignment,
      absl::Span<Event* const> wait_for) override {
    int64_t operation_id = GetOperationId();
    auto deps = GetDependencyOperationIds(wait_for);
    deps.insert(static_cast<PodLoadedProgramHandle*>(program)->operation_id());

    auto op_id = static_cast<PodLoadedProgramHandle*>(program)->operation_id();
    auto core_id = static_cast<PodLoadedProgramHandle*>(program)->core_id();

    std::vector<int64_t> input_op_ids;
    std::vector<int64_t> output_op_ids;

    for (auto* input : inputs) {
      auto input_dep =
          static_cast<PodBufferHandle* const>(input)->operation_id();
      input_op_ids.push_back(input_dep);
      deps.insert(input_dep);
    }
    for (auto* output : outputs) {
      auto output_dep =
          static_cast<PodBufferHandle* const>(output)->operation_id();
      output_op_ids.push_back(output_dep);
      deps.insert(output_dep);
    }

    ScheduleRequest(
        operation_id,
        [this, core_id, op_id, input_op_ids, output_op_ids,
         device_assignment]() {
          absl::MutexLock l(&mu_);

          std::vector<BufferHandle*> underlying_inputs;
          std::vector<BufferHandle*> underlying_outputs;

          underlying_inputs.reserve(input_op_ids.size());
          for (auto input_op_id : input_op_ids) {
            underlying_inputs.push_back(underlying_buffers_[input_op_id].get());
          }
          underlying_outputs.reserve(output_op_ids.size());
          for (auto output_op_id : output_op_ids) {
            underlying_outputs.push_back(
                underlying_buffers_[output_op_id].get());
          }

          LoadedProgramHandle* handle = underlying_lph_[op_id].get();
          return core_to_driver_[core_id]->ExecuteProgram(
              handle, underlying_inputs, underlying_outputs, device_assignment,
              {});
        },
        deps);

    return std::make_shared<PodEvent>(this, operation_id);
  }

  std::unique_ptr<TpuLinearizer> GetLinearizer() override {
    return drivers_[0]->GetLinearizer();
  }

  // Helper methods for Event scheduling

  absl::optional<Status> WaitForEvent(int64_t event_id,
                                      absl::Duration duration) {
    std::shared_ptr<Event> underlying_event;

    {
      absl::MutexLock l(&event_mu_);
      auto event = events_.find(event_id);

      if (event == events_.end()) {
        auto event_status = abnormal_event_status_.find(event_id);
        if (event_status == abnormal_event_status_.end()) {
          return Status::OK();
        } else {
          return event_status->second;
        }
      }

      auto done = [this, event_id]() {
        event_mu_.AssertHeld();
        return events_[event_id].underlying_event != nullptr;
      };

      auto status =
          event_mu_.AwaitWithTimeout(absl::Condition(&done), duration);
      if (!status) {
        return absl::nullopt;
      }
      underlying_event = events_[event_id].underlying_event;
    }

    // Wait for the underlying event without holding on to the event_lock_, or
    // else incoming events will not be processed.
    return underlying_event->AwaitWithTimeout(duration);
  }

  void AddCallbackForEvent(int64_t event_id, std::function<void(Status)> fn) {
    absl::MutexLock l(&event_mu_);
    auto event = events_.find(event_id);

    if (event == events_.end()) {
      auto event_status = abnormal_event_status_.find(event_id);
      if (event_status == abnormal_event_status_.end()) {
        fn(Status::OK());
      } else {
        fn(event_status->second);
      }
    }

    if (event->second.underlying_event != nullptr) {
      event->second.underlying_event->AddCallback(fn);
    } else {
      event->second.callbacks.push_back(std::move(fn));
    }
  }

  xla::Status GetCompiledProgramShape(int64_t op_id,
                                      xla::ProgramShapeProto* program_shape) {
    absl::MutexLock l(&mu_);

    auto done = [this, op_id]() {
      mu_.AssertHeld();
      return underlying_cph_.contains(op_id);
    };
    mu_.Await(absl::Condition(&done));

    return underlying_cph_[op_id][0]->program_shape(program_shape);
  }

 private:
  const TpuDriverConfig& config_;
  std::shared_ptr<::grpc::ChannelCredentials> creds_;

  std::vector<std::unique_ptr<TpuDriver>> drivers_;
  std::vector<int32> core_to_driver_id_;
  std::vector<TpuDriver*> core_to_driver_;
  std::vector<int32> core_to_driver_core_;
  SystemInfo pod_info_;

  absl::Mutex mu_;
  absl::Mutex event_mu_;

  absl::flat_hash_map<int64_t, std::unique_ptr<BufferHandle>>
      underlying_buffers_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<int64_t,
                      std::vector<std::unique_ptr<CompiledProgramHandle>>>
      underlying_cph_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<int64_t, std::unique_ptr<LoadedProgramHandle>>
      underlying_lph_ ABSL_GUARDED_BY(mu_);

  absl::btree_map<int64_t, EventInFlight> events_ ABSL_GUARDED_BY(event_mu_);
  absl::flat_hash_map<int64_t, Status> abnormal_event_status_
      ABSL_GUARDED_BY(event_mu_);

  std::atomic<int64_t> operation_id_counter_{0};

  WorkerThread event_thread_;

  int64_t GetOperationId() { return operation_id_counter_++; }

  absl::flat_hash_set<int64_t> GetDependencyOperationIds(
      absl::Span<Event* const> wait_for) {
    absl::flat_hash_set<int64_t> deps;
    for (auto* event : wait_for) {
      deps.insert(static_cast<PodEvent* const>(event)->operation_id());
    }
    return deps;
  }

  // EventCompleted is executed on the event_thread_ worker thread. We want
  // to propagate the fact that the event is completed to any subsequent events
  // that might depend on this event.
  void EventCompleted(int64_t event_id, Status status) {
    absl::MutexLock l(&event_mu_);

    absl::btree_map<int64_t, EventInFlight>::iterator curr_event;
    if (!status.ok()) abnormal_event_status_.insert({event_id, status});
    curr_event = events_.find(event_id);

    DCHECK(curr_event->second.callbacks.empty());
    DCHECK(curr_event->second.incomplete_deps.empty());

    for (auto& event : events_) {
      event.second.incomplete_deps.erase(event_id);
      // The if statement conditions on both
      //  - all previous events have completed (incomplete_deps.empty())
      //  - the op creating this event has not been called yet
      //    (event.second.create_fn != nullptr)
      // We call the create_fn that creates the event and adds any relevant
      // callbacks to the actual event, before setting create_fn to nullptr
      // to indicate that it has already been called
      if (event.second.incomplete_deps.empty() &&
          event.second.create_fn != nullptr) {
        // We were the last unfilled dependency, all other dependencies are
        // filled. We can now fire the create function.
        event.second.underlying_event = event.second.create_fn();
        for (auto& fn : event.second.callbacks) {
          event.second.underlying_event->AddCallback(std::move(fn));
        }
        event.second.callbacks.clear();
        event.second.create_fn = nullptr;
      }
    }

    // We erase the current event to signal that it has finished.
    events_.erase(curr_event);
  }

  void ScheduleRequest(int64_t operation_id,
                       std::function<std::shared_ptr<Event>(void)> fn,
                       const absl::flat_hash_set<int64_t>& deps) {
    absl::MutexLock l(&event_mu_);
    absl::btree_map<int64_t, EventInFlight>::iterator event;
    absl::flat_hash_set<int64_t> incomplete_deps;

    event = events_.insert({operation_id, {}}).first;
    for (const auto& dep : deps) {
      if (events_.count(dep) > 0) incomplete_deps.insert(dep);
    }

    if (incomplete_deps.empty()) {
      // All dependencies have been fulfilled, we execute the request
      // immediately and add a callback to inform our event fulfilled thread
      // when it is done.
      event->second.create_fn = nullptr;
      event->second.underlying_event = fn();
      event->second.underlying_event->AddCallback(
          [this, operation_id](Status status) {
            event_thread_.Schedule([this, operation_id, status]() {
              EventCompleted(operation_id, status);
            });
          });
    } else {
      // There are some dependencies that are not yet fulfilled. We attach
      // the request to the event, and will execute it in the EventFulfilled
      // worker thread when all its dependencies are fulfilled.
      event->second.create_fn = std::move(fn);
      event->second.incomplete_deps = std::move(incomplete_deps);
      event->second.callbacks.push_back([this, operation_id](Status status) {
        event_thread_.Schedule([this, operation_id, status]() {
          EventCompleted(operation_id, status);
        });
      });
    }
  }
};

xla::Status PodEvent::Await() {
  return driver_->WaitForEvent(operation_id_, absl::InfiniteDuration()).value();
}

absl::optional<xla::Status> PodEvent::AwaitWithTimeout(
    absl::Duration duration) {
  return driver_->WaitForEvent(operation_id_, duration);
}

void PodEvent::AddCallback(std::function<void(Status)> callback) {
  driver_->AddCallbackForEvent(operation_id_, std::move(callback));
}

xla::StatusOr<std::unique_ptr<TpuDriver>> CreatePodTpuDriver(
    const TpuDriverConfig& config,
    std::shared_ptr<::grpc::ChannelCredentials> creds) {
  return std::unique_ptr<TpuDriver>(new PodTpuDriver(config, creds));
}

xla::Status PodCompiledProgramHandle::program_shape(
    xla::ProgramShapeProto* program_shape) {
  return driver_->GetCompiledProgramShape(operation_id(), program_shape);
}

}  // namespace

REGISTER_TPU_DRIVER(kPodTpuDriverPrefix,
                    [](const TpuDriverConfig& config)
                        -> xla::StatusOr<std::unique_ptr<TpuDriver>> {
                      return CreatePodTpuDriver(
                          config,
                          ::grpc::InsecureChannelCredentials());  // NOLINT
                    });

}  // namespace tpu_driver
