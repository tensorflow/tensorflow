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
// =============================================================================
#include <atomic>
#include <functional>
#include <optional>

#include "absl/base/internal/sysinfo.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/python/tpu_driver/platform/external/compat.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.pb.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_service.grpc.pb.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/threadpool.h"

/*
 * The ReplayDriver wraps a concrete TpuDriver implementation and records the
 * stream of operations to a log file. This log can be later replayed and
 * analyzed for debugging.
 */

namespace tpu_driver {
namespace {

static std::atomic<int64_t> id_counter(0);

using xla::Status;

class RecordingTpuDriver;

class RecordingEvent : public Event {
 public:
  explicit RecordingEvent(std::shared_ptr<Event> event)
      : shared_event_(std::move(event)), id_(id_counter++) {}

  explicit RecordingEvent(std::shared_ptr<Event> event, int64_t id)
      : shared_event_(event), id_(id) {}

  ~RecordingEvent() override {}

  xla::Status Await() override { return shared_event_->Await(); }

  std::optional<xla::Status> AwaitWithTimeout(
      absl::Duration duration) override {
    return shared_event_->AwaitWithTimeout(duration);
  }

  void AddCallback(std::function<void(xla::Status)> callback) override {
    return shared_event_->AddCallback(callback);
  }

 private:
  std::shared_ptr<Event> shared_event_;

  int64_t id_;
  friend class RecordingTpuDriver;
};

class RecordingBufferHandle : public BufferHandle {
 public:
  explicit RecordingBufferHandle(std::unique_ptr<BufferHandle> handle)
      : handle_(std::move(handle)),
        id_(id_counter++),
        event_(std::make_shared<RecordingEvent>(handle_->OnReady(), id_)) {}
  std::shared_ptr<Event> OnReady() override { return event_; }
  int64_t size_in_bytes() override { return handle_->size_in_bytes(); }
  std::optional<xla::ShapeProto> shape() override { return handle_->shape(); }

 private:
  std::unique_ptr<BufferHandle> handle_;
  int64_t id_;
  std::shared_ptr<RecordingEvent> event_;
  friend class RecordingTpuDriver;
};

class RecordingCompiledProgramHandle : public CompiledProgramHandle {
 public:
  explicit RecordingCompiledProgramHandle(
      std::unique_ptr<CompiledProgramHandle> handle)
      : handle_(std::move(handle)),
        id_(id_counter++),
        event_(std::make_shared<RecordingEvent>(handle_->OnReady(), id_)) {}
  std::shared_ptr<Event> OnReady() override { return event_; }
  int64_t size_in_bytes() override { return handle_->size_in_bytes(); }
  xla::Status program_shape(xla::ProgramShapeProto* program_shape) override {
    return handle_->program_shape(program_shape);
  }

 private:
  std::unique_ptr<CompiledProgramHandle> handle_;
  int64_t id_;
  std::shared_ptr<RecordingEvent> event_;
  friend class RecordingTpuDriver;
};

class RecordingLoadedProgramHandle : public LoadedProgramHandle {
 public:
  explicit RecordingLoadedProgramHandle(
      std::unique_ptr<LoadedProgramHandle> handle)
      : handle_(std::move(handle)),
        id_(id_counter++),
        event_(std::make_shared<RecordingEvent>(handle_->OnReady(), id_)) {}
  std::shared_ptr<Event> OnReady() override { return event_; }
  int64_t size_in_bytes() override { return handle_->size_in_bytes(); }

 private:
  std::unique_ptr<LoadedProgramHandle> handle_;
  int64_t id_;
  std::shared_ptr<RecordingEvent> event_;
  friend class RecordingTpuDriver;
};

class RecordingTpuDriver : public TpuDriver {
 public:
  explicit RecordingTpuDriver(std::unique_ptr<TpuDriver> driver,
                              const std::string recording_path,
                              const bool flush)
      : driver_(std::move(driver)),
        recording_path_(recording_path),
        flush_(flush) {
    auto file_status = tensorflow::Env::Default()->NewAppendableFile(
        recording_path_, &log_file_);
    if (!file_status.ok()) {
      LOG(FATAL) << "Unable to open " << recording_path_
                 << " for appending. Error: " << file_status.ToString();
    }
  }
  ~RecordingTpuDriver() override {
    {
      log_file_->Flush().IgnoreError();
      log_file_->Close().IgnoreError();
      log_file_ = nullptr;
    }
  }

  void QuerySystemInfo(SystemInfo* system_info) override {
    // TODO(frankchn): Should we even save this event, since it is out-of-band.
    driver_->QuerySystemInfo(system_info);
  }

  Status Reset() override { return driver_->Reset(); }

  std::unique_ptr<BufferHandle> Allocate(
      int32_t core_id, MemoryRegion region, int64_t num_bytes,
      absl::Span<Event* const> wait_for) override {
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    auto thread_id = GetCurrentThreadId();
    auto handle =
        driver_->Allocate(core_id, region, num_bytes, unwrapped_wait_for);
    auto recording_handle =
        std::make_unique<RecordingBufferHandle>(std::move(handle));
    auto handle_id = recording_handle->id_;

    {
      StreamRequest::Entry r;
      r.mutable_alloc()->set_core_id(core_id);
      r.mutable_alloc()->set_region(region);
      r.mutable_alloc()->set_num_bytes(num_bytes);

      PopulateAndSaveEntry(&r, wait_for, handle_id, thread_id);
    }

    return recording_handle;
  }

  std::unique_ptr<BufferHandle> Allocate(
      int32_t core_id, MemoryRegion region, const xla::ShapeProto& shape,
      absl::Span<Event* const> wait_for) override {
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    auto thread_id = GetCurrentThreadId();
    auto handle = driver_->Allocate(core_id, region, shape, unwrapped_wait_for);
    auto recording_handle =
        std::make_unique<RecordingBufferHandle>(std::move(handle));
    auto handle_id = recording_handle->id_;

    {
      StreamRequest::Entry r;
      r.mutable_alloc()->set_core_id(core_id);
      r.mutable_alloc()->set_region(region);
      *(r.mutable_alloc()->mutable_shape()) = shape;

      PopulateAndSaveEntry(&r, wait_for, handle_id, thread_id);
    }

    return recording_handle;
  }

  std::unique_ptr<BufferHandle> AllocateTuple(
      int32_t core_id, MemoryRegion region,
      absl::Span<BufferHandle* const> children,
      absl::Span<Event* const> wait_for) override {
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    std::vector<BufferHandle*> unwrapped_children;
    std::vector<int64_t> child_ids;
    const auto children_size = children.size();
    unwrapped_children.reserve(children_size);
    child_ids.reserve(children_size);
    for (auto child : children) {
      BufferHandle* unwrapped_child =
          static_cast<const RecordingBufferHandle*>(child)->handle_.get();
      unwrapped_children.push_back(unwrapped_child);
      child_ids.push_back(
          static_cast<const RecordingBufferHandle*>(child)->id_);
    }

    auto thread_id = GetCurrentThreadId();
    auto handle = driver_->AllocateTuple(core_id, region, unwrapped_children,
                                         unwrapped_wait_for);
    auto recording_handle =
        std::make_unique<RecordingBufferHandle>(std::move(handle));
    auto handle_id = recording_handle->id_;

    {
      StreamRequest::Entry r;
      r.mutable_alloc_tuple()->set_core_id(core_id);
      r.mutable_alloc_tuple()->set_region(region);

      for (auto child : child_ids) {
        r.mutable_alloc_tuple()->add_children(child);
      }

      PopulateAndSaveEntry(&r, wait_for, handle_id, thread_id);
    }

    return recording_handle;
  }

  std::shared_ptr<Event> Deallocate(
      std::unique_ptr<BufferHandle> handle,
      absl::Span<Event* const> wait_for) override {
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    auto thread_id = GetCurrentThreadId();
    auto recording_handle = static_cast<RecordingBufferHandle*>(handle.get());
    int64_t recording_handle_id = recording_handle->id_;
    auto event = driver_->Deallocate(std::move(recording_handle->handle_),
                                     unwrapped_wait_for);
    auto recording_event = std::make_shared<RecordingEvent>(std::move(event));
    int64_t event_id = recording_event->id_;

    {
      StreamRequest::Entry r;
      r.mutable_dealloc()->set_handle(recording_handle_id);
      PopulateAndSaveEntry(&r, wait_for, event_id, thread_id);
    }

    return recording_event;
  }

  std::shared_ptr<Event> TransferToDevice(
      const void* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for) override {
    int64_t num_bytes = dst->size_in_bytes();
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    auto thread_id = GetCurrentThreadId();
    auto recording_handle = static_cast<RecordingBufferHandle*>(dst);
    int64_t recording_handle_id = recording_handle->id_;
    auto recording_event =
        std::make_shared<RecordingEvent>(driver_->TransferToDevice(
            src, static_cast<RecordingBufferHandle*>(dst)->handle_.get(),
            unwrapped_wait_for));
    int64_t event_id = recording_event->id_;

    {
      StreamRequest::Entry r;
      r.mutable_transfer_to()->set_target_handle(recording_handle_id);
      if (num_bytes > 0) {
        r.mutable_transfer_to()->mutable_data()->assign(
            static_cast<const char*>(src), num_bytes);
      } else {
        *r.mutable_transfer_to()->mutable_data() = "";
      }
      PopulateAndSaveEntry(&r, wait_for, event_id, thread_id);
    }

    return recording_event;
  }

  std::shared_ptr<Event> TransferFromDevice(
      const BufferHandle* src, void* dst,
      absl::Span<Event* const> wait_for) override {
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    auto thread_id = GetCurrentThreadId();
    auto src_handle_id = static_cast<const RecordingBufferHandle*>(src)->id_;
    auto recording_event =
        std::make_shared<RecordingEvent>(driver_->TransferFromDevice(
            static_cast<const RecordingBufferHandle*>(src)->handle_.get(), dst,
            unwrapped_wait_for));
    auto event_id = recording_event->id_;

    {
      StreamRequest::Entry r;
      r.mutable_transfer_from()->set_source_handle(src_handle_id);
      PopulateAndSaveEntry(&r, wait_for, event_id, thread_id);
    }

    return recording_event;
  }

  std::shared_ptr<Event> TransferFromDeviceToDevice(
      const BufferHandle* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for) override {
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    auto thread_id = GetCurrentThreadId();
    auto src_handle_id = static_cast<const RecordingBufferHandle*>(src)->id_;
    auto dst_handle_id = static_cast<const RecordingBufferHandle*>(dst)->id_;
    auto recording_event =
        std::make_shared<RecordingEvent>(driver_->TransferFromDeviceToDevice(
            static_cast<const RecordingBufferHandle*>(src)->handle_.get(),
            static_cast<const RecordingBufferHandle*>(dst)->handle_.get(),
            unwrapped_wait_for));
    auto event_id = recording_event->id_;

    {
      StreamRequest::Entry r;
      r.mutable_transfer_from_to()->set_source_handle(src_handle_id);
      r.mutable_transfer_from_to()->set_target_handle(dst_handle_id);
      PopulateAndSaveEntry(&r, wait_for, event_id, thread_id);
    }

    return recording_event;
  }

  std::unique_ptr<CompiledProgramHandle> CompileProgram(
      const xla::HloProto& source, int32_t num_replicas,
      absl::Span<Event* const> wait_for) override {
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    auto thread_id = GetCurrentThreadId();
    auto recording_handle = std::make_unique<RecordingCompiledProgramHandle>(
        driver_->CompileProgram(source, num_replicas, unwrapped_wait_for));
    auto handle_id = recording_handle->id_;

    {
      StreamRequest::Entry r;
      *r.mutable_compile()->mutable_hlo_program() = source;
      r.mutable_compile()->set_num_replicas(num_replicas);
      PopulateAndSaveEntry(&r, wait_for, handle_id, thread_id);
    }

    return recording_handle;
  }

  std::unique_ptr<LoadedProgramHandle> LoadProgram(
      int32_t core_id, const CompiledProgramHandle* handle,
      absl::Span<Event* const> wait_for) override {
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    auto thread_id = GetCurrentThreadId();
    auto compiled_handle_id =
        static_cast<const RecordingCompiledProgramHandle*>(handle)->id_;
    auto recording_handle =
        std::make_unique<RecordingLoadedProgramHandle>(driver_->LoadProgram(
            core_id,
            static_cast<const RecordingCompiledProgramHandle*>(handle)
                ->handle_.get(),
            unwrapped_wait_for));
    auto handle_id = recording_handle->id_;
    {
      StreamRequest::Entry r;
      r.mutable_load()->set_core_id(core_id);
      r.mutable_load()->set_compiled_program_handle(compiled_handle_id);
      PopulateAndSaveEntry(&r, wait_for, handle_id, thread_id);
    }

    return recording_handle;
  }

  std::shared_ptr<Event> UnloadProgram(
      std::unique_ptr<LoadedProgramHandle> handle,
      absl::Span<Event* const> wait_for) override {
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    auto thread_id = GetCurrentThreadId();
    auto loaded_handle_id =
        static_cast<RecordingLoadedProgramHandle*>(handle.get())->id_;
    auto recording_event =
        std::make_shared<RecordingEvent>(driver_->UnloadProgram(
            std::move(static_cast<RecordingLoadedProgramHandle*>(handle.get())
                          ->handle_),
            unwrapped_wait_for));
    auto event_id = recording_event->id_;

    {
      StreamRequest::Entry r;
      r.mutable_unload()->set_loaded_program_handle(loaded_handle_id);
      PopulateAndSaveEntry(&r, wait_for, event_id, thread_id);
    }

    return recording_event;
  }

  std::shared_ptr<Event> ExecuteProgram(
      LoadedProgramHandle* program, absl::Span<BufferHandle* const> inputs,
      absl::Span<BufferHandle* const> outputs,
      const xla::DeviceAssignmentProto& device_assignment,
      absl::Span<Event* const> wait_for) override {
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    auto thread_id = GetCurrentThreadId();
    auto program_handle_id =
        static_cast<RecordingLoadedProgramHandle*>(program)->id_;

    std::vector<BufferHandle*> unwrapped_inputs;
    std::vector<int64_t> input_ids;
    const auto inputs_size = inputs.size();
    unwrapped_inputs.reserve(inputs_size);
    input_ids.reserve(inputs_size);
    for (auto input : inputs) {
      BufferHandle* unwrapped_input =
          static_cast<const RecordingBufferHandle*>(input)->handle_.get();
      unwrapped_inputs.push_back(unwrapped_input);
      input_ids.push_back(
          static_cast<const RecordingBufferHandle*>(input)->id_);
    }

    std::vector<BufferHandle*> unwrapped_outputs;
    std::vector<int64_t> output_ids;
    const auto output_size = outputs.size();
    unwrapped_outputs.reserve(output_size);
    output_ids.reserve(output_size);
    for (auto output : outputs) {
      BufferHandle* unwrapped_output =
          static_cast<const RecordingBufferHandle*>(output)->handle_.get();
      unwrapped_outputs.push_back(unwrapped_output);
      output_ids.push_back(
          static_cast<const RecordingBufferHandle*>(output)->id_);
    }

    auto recording_event =
        std::make_shared<RecordingEvent>(driver_->ExecuteProgram(
            static_cast<RecordingLoadedProgramHandle*>(program)->handle_.get(),
            unwrapped_inputs, unwrapped_outputs, device_assignment,
            unwrapped_wait_for));
    auto event_id = recording_event->id_;

    {
      StreamRequest::Entry r;
      r.mutable_execute()->set_loaded_program_handle(program_handle_id);
      for (auto input_id : input_ids) {
        r.mutable_execute()->add_input_handle(input_id);
      }
      for (auto output_id : output_ids) {
        r.mutable_execute()->add_output_handle(output_id);
      }
      *r.mutable_execute()->mutable_device_assignment() = device_assignment;

      PopulateAndSaveEntry(&r, wait_for, event_id, thread_id);
    }

    return recording_event;
  }

  std::unique_ptr<TpuLinearizer> GetLinearizer() override {
    return driver_->GetLinearizer();
  }

 private:
  std::unique_ptr<TpuDriver> driver_;
  const std::string recording_path_;
  const bool flush_;

  std::unique_ptr<tensorflow::WritableFile> log_file_;

  void PopulateAndSaveEntry(StreamRequest::Entry* r,
                            absl::Span<Event* const> wait_for,
                            int64_t handle_id, int64_t thread_id) {
    for (auto event : wait_for) {
      auto recording_event = static_cast<const RecordingEvent*>(event);
      r->add_wait_for_id(recording_event->id_);
    }
    r->set_operation_id(handle_id);
    r->set_thread_id(thread_id);

    uint64_t data_size = r->ByteSizeLong();
    std::vector<char> buffer;
    buffer.resize(sizeof(data_size) + data_size);
    memcpy(buffer.data(), &data_size, sizeof(data_size));
    r->SerializeToArray(buffer.data() + sizeof(data_size), data_size);

    {
      if (log_file_ == nullptr) {
        LOG(WARNING) << "The TPU driver has been shut down before all logging "
                        "has been written.";
        return;
      }

      absl::string_view buffer_sp(buffer.data(), buffer.size());
      auto data_status = log_file_->Append(buffer_sp);
      if (!data_status.ok()) {
        LOG(WARNING) << "Unable to write data to log file. File possibly "
                        "corrupt. Error: "
                     << data_status.ToString();
      }

      if (flush_) {
        auto flush_status = log_file_->Flush();
        if (!flush_status.ok()) {
          LOG(WARNING) << "Unable to flush data to log file. File possibly "
                          "corrupt. Error: "
                       << flush_status.ToString();
        }

        auto sync_status = log_file_->Sync();
        if (!sync_status.ok()) {
          LOG(WARNING) << "Unable to sync log file. File possibly "
                          "corrupt. Error: "
                       << sync_status.ToString();
        }
      }
    }
  }

  std::vector<Event*> UnwrapWaitFor(absl::Span<Event* const> wait_for) {
    std::vector<Event*> unwrapped_events;
    for (auto event : wait_for) {
      Event* unwrapped_event =
          static_cast<RecordingEvent*>(event)->shared_event_.get();
      unwrapped_events.push_back(unwrapped_event);
    }
    return unwrapped_events;
  }

  int64_t GetCurrentThreadId() { return absl::base_internal::GetTID(); }
};

xla::StatusOr<std::unique_ptr<TpuDriver>> RegisterRecordingTpuDriver(
    const TpuDriverConfig& config) {
  std::vector<std::string> configs = absl::StrSplit(config.worker(), '|');

  std::string file;
  std::string worker;
  bool flush = false;

  for (const auto& config : configs) {
    std::vector<std::string> kv =
        absl::StrSplit(config, absl::MaxSplits('=', 1));
    if (kv[0] == "file") {
      file = kv[1];
    }
    if (kv[0] == "worker") {
      worker = kv[1];
    }
    if (kv[0] == "flush") {
      if (kv[1] == "true" || kv[1] == "1") {
        flush = true;
      }
    }
  }

  TpuDriverConfig worker_config;
  worker_config.set_worker(worker);

  auto driver_status = TpuDriverRegistry::Open(worker_config);
  if (!driver_status.ok()) return driver_status.status();
  auto driver = driver_status.ConsumeValueOrDie();

  return std::unique_ptr<TpuDriver>(
      new RecordingTpuDriver(std::move(driver), file, flush));
}

// To record a sequence of operations, set the worker configuration string to
// record://|file=<filename>|worker=grpc://1.2.3.4:8470 (for GRPC).
REGISTER_TPU_DRIVER("record://", RegisterRecordingTpuDriver);

}  // namespace
}  // namespace tpu_driver
