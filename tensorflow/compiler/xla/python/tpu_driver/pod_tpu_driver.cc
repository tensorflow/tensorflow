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
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
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

#define CHECK_EXISTS_OR_RETURN(container, target_op_id, operation_id)  \
  {                                                                    \
    auto p = CheckHandleExists(container, target_op_id, operation_id); \
    if (p != nullptr) return p;                                        \
  }

using xla::Status;
using xla::WorkerThread;

const char kPodTpuDriverPrefix[] = "grpc+pod://";

class PodTpuDriver;

class PodEvent : public Event {
 public:
  explicit PodEvent(PodTpuDriver* driver, int64_t operation_id)
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

class ErrorEvent : public PodEvent {
 public:
  explicit ErrorEvent(PodTpuDriver* driver, int64_t operation_id, Status status)
      : PodEvent(driver, operation_id) {
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

class CombinedEvent : public PodEvent {
 public:
  explicit CombinedEvent(PodTpuDriver* driver, int64_t operation_id,
                         std::vector<std::shared_ptr<Event>> events)
      : PodEvent(driver, operation_id), events_(events) {
    for (auto& event : events_) {
      event->AddCallback([this](Status s) { IncrementAndCheckComplete(s); });
    }
  }

  xla::Status Await() override {
    for (auto& event : events_) {
      TF_RETURN_IF_ERROR(event->Await());
    }
    return Status::OK();
  }

  absl::optional<xla::Status> AwaitWithTimeout(
      absl::Duration duration) override {
    for (auto& event : events_) {
      auto start_time = absl::Now();
      auto status = event->AwaitWithTimeout(duration);
      duration -= absl::Now() - start_time;
      if (status == absl::nullopt) {
        return absl::nullopt;
      } else {
        TF_RETURN_IF_ERROR(status.value());
      }
    }
    return Status::OK();
  }

  void AddCallback(std::function<void(Status)> callback)
      TF_LOCKS_EXCLUDED(mu_) override {
    bool all_events_completed = false;
    {
      absl::MutexLock l(&mu_);
      all_events_completed = events_completed_ == events_.size();
    }
    if (all_events_completed) {
      callback(event_status_);
    } else {
      absl::MutexLock l(&mu_);
      callbacks_.push_back(std::move(callback));
    }
  }

 private:
  void IncrementAndCheckComplete(Status s) TF_LOCKS_EXCLUDED(mu_) {
    std::vector<std::function<void(Status)>> callbacks;
    {
      absl::MutexLock l(&mu_);

      event_status_ = s;
      events_completed_++;
      if (events_completed_ == events_.size()) {
        // Copy callbacks to a temporary to be invoked outside the mutex.
        callbacks.assign(callbacks_.begin(), callbacks_.end());
        callbacks_.clear();
      } else {
        return;
      }
    }

    for (const auto& callback : callbacks) {
      callback(event_status_);
    }
  }

  absl::Mutex mu_;
  std::vector<std::shared_ptr<Event>> events_;
  std::vector<std::function<void(Status)>> callbacks_ ABSL_GUARDED_BY(mu_);
  int64_t events_completed_ ABSL_GUARDED_BY(mu_) = 0;
  Status event_status_;
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
  EventInFlight()
      : underlying_event(nullptr),
        create_fn(nullptr),
        incomplete_deps(),
        callbacks() {}

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

    int worker_count = 0;

    // Flag for environments where local core # == all cores in TPU system #,
    // which means that we are connecting to separate TPU systems or we are in
    // a test environment.
    bool in_local_core_environment = false;

    for (const auto& worker : workers) {
      TpuDriverConfig worker_config(config_);
      *(worker_config.mutable_worker()) = absl::StrCat("grpc://", worker);
      auto tpu_driver =
          CreateGrpcTpuDriver(worker_config, creds_).ConsumeValueOrDie();

      SystemInfo driver_info;
      tpu_driver->QuerySystemInfo(&driver_info);

      if (driver_info.core_count() == driver_info.local_core_size()) {
        drivers_.insert({worker_count, std::move(tpu_driver)});
        in_local_core_environment = true;
      } else {
        drivers_.insert({driver_info.host_id(), std::move(tpu_driver)});
      }

      worker_count++;
    }

    absl::flat_hash_set<std::tuple<int, int, int>> processed_chips;

    for (int driver_num = 0; driver_num < workers.size(); ++driver_num) {
      SystemInfo driver_info;
      drivers_[driver_num]->QuerySystemInfo(&driver_info);

      for (const auto& tpu_chip : driver_info.tpu_chip()) {
        std::tuple<int, int, int> coord{tpu_chip.chip_coord().x(),
                                        tpu_chip.chip_coord().y(),
                                        tpu_chip.chip_coord().z()};
        // We only want to add chips that we have not seen before if we are in a
        // TPU pod slice, or we are only seeing local cores (e.g. we are
        // connected to individual TPUs or we are in a test environment).
        if (!processed_chips.contains(coord) ||
            driver_info.core_count() == driver_info.local_core_size()) {
          *(pod_info_.add_tpu_chip()) = tpu_chip;
          processed_chips.insert(coord);
        }
      }

      *(pod_info_.mutable_cpu()) = driver_info.cpu();
    }

    // Process all the unique chips that we have seen.
    int core_count = 0;
    for (auto& tpu_chip : *pod_info_.mutable_tpu_chip()) {
      for (auto& tpu_core : *tpu_chip.mutable_core()) {
        int current_core = tpu_core.id();
        if (in_local_core_environment) {
          current_core = core_count;
        }

        core_to_driver_.insert(
            {current_core, drivers_[tpu_chip.host_id()].get()});
        core_to_driver_id_.insert({current_core, tpu_chip.host_id()});
        core_to_driver_core_.insert({current_core, tpu_core.id()});

        tpu_core.set_id(current_core);
        tpu_core.set_core_on_host_index(current_core);
        *(pod_info_.add_local_core()) = tpu_core;

        core_count++;
      }

      // We are setting host_id to zero because we want this to look like one
      // host with many cores from the perspective of tpu_client.cc.
      tpu_chip.set_host_id(0);
    }

    pod_info_.set_chip_count(pod_info_.tpu_chip_size());
    pod_info_.set_core_count(pod_info_.local_core_size());

    // We want this to look like one host with many TPU chips/cores connected.
    pod_info_.set_host_count(1);
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
      TF_RETURN_IF_ERROR(driver.second->Reset());
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
        [this, core_id, region, num_bytes,
         operation_id]() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
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
        [this, core_id, region, shape,
         operation_id]() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
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
    const size_t children_ids_size = children.size();
    children_ids.reserve(children_ids_size);
    for (size_t i = 0; i < children_ids_size; ++i) {
      auto child_op_id =
          static_cast<PodBufferHandle* const>(children[i])->operation_id();
      deps.insert(child_op_id);
      children_ids.push_back(child_op_id);
    }

    ScheduleRequest(
        operation_id,
        [this, core_id, region, children_ids,
         operation_id]() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_)
            -> std::shared_ptr<Event> {
          std::vector<BufferHandle*> child_buffers;
          child_buffers.reserve(children_ids.size());
          for (size_t i = 0; i < children_ids.size(); ++i) {
            CHECK_EXISTS_OR_RETURN(underlying_buffers_, children_ids[i],
                                   operation_id);
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
        [this, operation_id, op_id,
         core_id]() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) -> std::shared_ptr<Event> {
          CHECK_EXISTS_OR_RETURN(underlying_buffers_, op_id, operation_id);

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
        [this, src, operation_id, op_id,
         core_id]() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) -> std::shared_ptr<Event> {
          CHECK_EXISTS_OR_RETURN(underlying_buffers_, op_id, operation_id);

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
        [this, dst, operation_id, op_id,
         core_id]() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) -> std::shared_ptr<Event> {
          CHECK_EXISTS_OR_RETURN(underlying_buffers_, op_id, operation_id);
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
    auto src_core_id = static_cast<const PodBufferHandle*>(src)->core_id();
    auto dst_core_id = static_cast<PodBufferHandle*>(dst)->core_id();

    auto src_driver_id = core_to_driver_id_[src_core_id];
    auto dst_driver_id = core_to_driver_id_[dst_core_id];

    if (src_driver_id == dst_driver_id) {
      // They are in the same host, we can schedule it normally
      int64_t operation_id = GetOperationId();
      auto deps = GetDependencyOperationIds(wait_for);
      deps.insert(static_cast<const PodBufferHandle*>(src)->operation_id());
      deps.insert(static_cast<PodBufferHandle*>(dst)->operation_id());

      auto src_op_id = static_cast<const PodBufferHandle*>(src)->operation_id();
      auto dst_op_id = static_cast<PodBufferHandle*>(dst)->operation_id();

      ScheduleRequest(
          operation_id,
          [this, operation_id, src_op_id, dst_op_id, dst_core_id]()
              TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) -> std::shared_ptr<Event> {
                CHECK_EXISTS_OR_RETURN(underlying_buffers_, src_op_id,
                                       operation_id);
                CHECK_EXISTS_OR_RETURN(underlying_buffers_, dst_op_id,
                                       operation_id);

                auto src_iter = underlying_buffers_.find(src_op_id);
                auto dst_iter = underlying_buffers_.find(dst_op_id);
                return core_to_driver_[dst_core_id]->TransferFromDeviceToDevice(
                    src_iter->second.get(), dst_iter->second.get(), {});
              },
          deps);
      return std::make_shared<PodEvent>(this, operation_id);
    } else {
      // src and dst are on different hosts, we have to bounce through us.
      auto dst_size = dst->size_in_bytes();
      char* host_buf = new char[dst_size];

      auto src_event = TransferFromDevice(src, host_buf, wait_for);
      auto dst_event = TransferToDevice(host_buf, dst, {src_event.get()});
      dst_event->AddCallback(
          [src_event, host_buf](xla::Status status) { delete[] host_buf; });
      return dst_event;
    }
  }

  std::unique_ptr<CompiledProgramHandle> CompileProgram(
      const xla::HloProto& source, int32_t num_replicas,
      absl::Span<Event* const> wait_for) override {
    int64_t operation_id = GetOperationId();
    auto deps = GetDependencyOperationIds(wait_for);

    ScheduleRequest(
        operation_id,
        [this, operation_id, source,
         num_replicas]() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
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
        [this, operation_id, cph_op_id,
         core_id]() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) -> std::shared_ptr<Event> {
          CHECK_EXISTS_OR_RETURN(underlying_cph_, cph_op_id, operation_id);
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
        [this, operation_id, op_id,
         core_id]() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) -> std::shared_ptr<Event> {
          CHECK_EXISTS_OR_RETURN(underlying_lph_, op_id, operation_id);
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
    input_op_ids.reserve(inputs.size());
    output_op_ids.reserve(outputs.size());

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
        [this, operation_id, core_id, op_id, input_op_ids, output_op_ids,
         device_assignment]() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_)
            -> std::shared_ptr<Event> {
          std::vector<BufferHandle*> underlying_inputs;
          std::vector<BufferHandle*> underlying_outputs;

          underlying_inputs.reserve(input_op_ids.size());
          for (auto input_op_id : input_op_ids) {
            CHECK_EXISTS_OR_RETURN(underlying_buffers_, input_op_id,
                                   operation_id);
            underlying_inputs.push_back(underlying_buffers_[input_op_id].get());
          }
          underlying_outputs.reserve(output_op_ids.size());
          for (auto output_op_id : output_op_ids) {
            CHECK_EXISTS_OR_RETURN(underlying_buffers_, output_op_id,
                                   operation_id);
            underlying_outputs.push_back(
                underlying_buffers_[output_op_id].get());
          }

          CHECK_EXISTS_OR_RETURN(underlying_lph_, op_id, operation_id);
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

  absl::optional<Status> WaitForEvent(int64_t event_id, absl::Duration duration)
      TF_LOCKS_EXCLUDED(mu_) {
    std::shared_ptr<Event> underlying_event;

    {
      absl::MutexLock l(&mu_);
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
        mu_.AssertHeld();
        // The event was either completed and erased from the map or we have
        // an underlying event available to us.
        return events_.count(event_id) == 0 ||
               (events_[event_id]->underlying_event != nullptr &&
                events_[event_id]->underlying_event.use_count() != 0);
      };

      auto status = mu_.AwaitWithTimeout(absl::Condition(&done), duration);
      if (!status) {
        return absl::nullopt;
      }

      if (events_.count(event_id) > 0) {
        underlying_event = events_[event_id]->underlying_event;
      } else {
        underlying_event = nullptr;
      }
    }

    // Wait for the underlying event without holding on to the event_lock_, or
    // else incoming events will not be processed.
    if (underlying_event != nullptr) {
      return underlying_event->AwaitWithTimeout(duration);
    } else {
      absl::MutexLock l(&mu_);
      auto event_status = abnormal_event_status_.find(event_id);
      if (event_status == abnormal_event_status_.end()) {
        return Status::OK();
      } else {
        return event_status->second;
      }
    }
  }

  void AddCallbackForEvent(int64_t event_id, std::function<void(Status)> fn)
      TF_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock l(&mu_);
    auto event = events_.find(event_id);

    if (event == events_.end()) {
      auto event_status = abnormal_event_status_.find(event_id);
      if (event_status == abnormal_event_status_.end()) {
        fn(Status::OK());
      } else {
        fn(event_status->second);
      }
    } else {
      if (event->second->underlying_event != nullptr &&
          event->second->underlying_event.use_count() != 0) {
        event->second->underlying_event->AddCallback(fn);
      } else {
        event->second->callbacks.push_back(std::move(fn));
      }
    }
  }

  xla::Status GetCompiledProgramShape(int64_t op_id,
                                      xla::ProgramShapeProto* program_shape)
      TF_LOCKS_EXCLUDED(mu_) {
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

  absl::flat_hash_map<int32_t, std::unique_ptr<TpuDriver>> drivers_;
  absl::flat_hash_map<int32_t, int32_t> core_to_driver_id_;
  absl::flat_hash_map<int32_t, TpuDriver*> core_to_driver_;
  absl::flat_hash_map<int32_t, int32_t> core_to_driver_core_;
  SystemInfo pod_info_;

  absl::Mutex mu_;

  absl::flat_hash_map<int64_t, std::unique_ptr<BufferHandle>>
      underlying_buffers_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<int64_t,
                      std::vector<std::unique_ptr<CompiledProgramHandle>>>
      underlying_cph_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<int64_t, std::unique_ptr<LoadedProgramHandle>>
      underlying_lph_ ABSL_GUARDED_BY(mu_);

  absl::btree_map<int64_t, std::unique_ptr<EventInFlight>> events_
      ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<int64_t, Status> abnormal_event_status_
      ABSL_GUARDED_BY(mu_);

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
  void EventCompleted(int64_t event_id, Status status) TF_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock l(&mu_);

    absl::btree_map<int64_t, std::unique_ptr<EventInFlight>>::iterator
        curr_event;
    if (!status.ok()) abnormal_event_status_.insert({event_id, status});
    curr_event = events_.find(event_id);

    DCHECK(curr_event->second->callbacks.empty());
    DCHECK(curr_event->second->incomplete_deps.empty());

    for (auto& event : events_) {
      event.second->incomplete_deps.erase(event_id);
      // The if statement conditions on both
      //  - all previous events have completed (incomplete_deps.empty())
      //  - the op creating this event has not been called yet
      //    (event.second.create_fn != nullptr)
      // We call the create_fn that creates the event and adds any relevant
      // callbacks to the actual event, before setting create_fn to nullptr
      // to indicate that it has already been called
      if (event.second->incomplete_deps.empty() &&
          event.second->create_fn != nullptr) {
        // We were the last unfilled dependency, all other dependencies are
        // filled. We can now fire the create function.
        event.second->underlying_event = event.second->create_fn();
        for (auto& fn : event.second->callbacks) {
          event.second->underlying_event->AddCallback(std::move(fn));
        }
        event.second->callbacks.clear();
        event.second->create_fn = nullptr;
      }
    }

    // We erase the current event to signal that it has finished.
    events_.erase(curr_event);
  }

  void ScheduleRequest(int64_t operation_id,
                       std::function<std::shared_ptr<Event>(void)> fn,
                       const absl::flat_hash_set<int64_t>& deps)
      TF_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock l(&mu_);
    absl::btree_map<int64_t, std::unique_ptr<EventInFlight>>::iterator event;
    absl::flat_hash_set<int64_t> incomplete_deps;

    event = events_.insert({operation_id, absl::make_unique<EventInFlight>()})
                .first;
    for (const auto& dep : deps) {
      if (events_.count(dep) > 0) incomplete_deps.insert(dep);
    }

    if (incomplete_deps.empty()) {
      // All dependencies have been fulfilled, we execute the request
      // immediately and add a callback to inform our event fulfilled thread
      // when it is done.
      event->second->create_fn = nullptr;
      event->second->underlying_event = fn();
      event->second->underlying_event->AddCallback(
          [this, operation_id](Status status) {
            event_thread_.Schedule([this, operation_id, status]() {
              EventCompleted(operation_id, status);
            });
          });
    } else {
      // There are some dependencies that are not yet fulfilled. We attach
      // the request to the event, and will execute it in the EventFulfilled
      // worker thread when all its dependencies are fulfilled.
      event->second->create_fn = std::move(fn);
      event->second->incomplete_deps = std::move(incomplete_deps);
      event->second->callbacks.push_back([this, operation_id](Status status) {
        event_thread_.Schedule([this, operation_id, status]() {
          EventCompleted(operation_id, status);
        });
      });
    }
  }

  template <typename T>
  std::shared_ptr<Event> CheckHandleExists(
      absl::flat_hash_map<int64_t, T>& container, int64_t target_op_id,
      int64_t operation_id) {
    if (container.count(target_op_id) == 0) {
      return std::make_shared<ErrorEvent>(
          this, operation_id,
          tensorflow::errors::InvalidArgument("Handle ", target_op_id,
                                              " does not exist."));
    }
    return nullptr;
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
