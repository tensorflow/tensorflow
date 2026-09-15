/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/metal/metal_profiler.h"

#import <QuartzCore/QuartzCore.h>

#include <time.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/common_runtime/metal/metal_platform.h"

namespace tensorflow {
namespace metal {
namespace {

/*** THE SMALLEST PROTOBUF WRITER THAT WILL DO ***/

// XSpace is what the C API asks for, serialized. Encoding it by hand rather
// than linking TensorFlow's generated C++ classes is deliberate: those come
// from libtensorflow_framework's C++ ABI, and a plugin that binds to that ABI
// stops loading the moment TensorFlow rebuilds it. That is precisely how
// Apple's tensorflow-metal broke. The wire format, by contrast, is fixed
// forever by the field numbers in xplane.proto.
class ProtoWriter {
 public:
  void Varint(uint64_t value) {
    while (value >= 0x80) {
      out_.push_back(static_cast<uint8_t>(value) | 0x80);
      value >>= 7;
    }
    out_.push_back(static_cast<uint8_t>(value));
  }

  void Tag(int field, int wire_type) {
    Varint((static_cast<uint64_t>(field) << 3) | wire_type);
  }

  void Int64Field(int field, int64_t value) {
    if (value == 0) return;  // proto3 omits defaults
    Tag(field, 0);
    Varint(static_cast<uint64_t>(value));
  }

  void StringField(int field, const std::string& value) {
    if (value.empty()) return;
    Tag(field, 2);
    Varint(value.size());
    out_.insert(out_.end(), value.begin(), value.end());
  }

  void MessageField(int field, const ProtoWriter& message) {
    Tag(field, 2);
    Varint(message.out_.size());
    out_.insert(out_.end(), message.out_.begin(), message.out_.end());
  }

  const std::vector<uint8_t>& bytes() const { return out_; }

 private:
  std::vector<uint8_t> out_;
};

/*** THE SESSION ***/

struct Sample {
  int64_t metadata_id = 0;
  int64_t start_ns = 0;
  int64_t duration_ns = 0;
};

// Nanoseconds since the epoch, which is the clock TensorFlow's host traces
// use. Metal reports its timestamps in CACurrentMediaTime's timebase instead,
// so one reading of both is taken when a session starts and every GPU
// timestamp is shifted by the difference. Without that the device line lands
// decades away from the host line and the trace viewer shows two islands.
int64_t RealtimeNanos() {
  struct timespec now;
  clock_gettime(CLOCK_REALTIME, &now);
  return static_cast<int64_t>(now.tv_sec) * 1000000000 + now.tv_nsec;
}

class Session {
 public:
  static Session& Global() {
    static Session* session = new Session();
    return *session;
  }

  void Start(TF_Status* status) {
    absl::MutexLock lock(&mu_);
    if (active_.load(std::memory_order_relaxed)) {
      TF_SetStatus(status, TF_FAILED_PRECONDITION,
                   "Metal: a profiling session is already collecting.");
      return;
    }
    samples_.clear();
    metadata_.clear();
    collected_ = false;
    // Both clocks read as close together as they can be, so the offset
    // between them is measured rather than assumed.
    media_base_ = CACurrentMediaTime();
    realtime_base_ns_ = RealtimeNanos();
    active_.store(true, std::memory_order_release);
    TF_SetStatus(status, TF_OK, "");
  }

  void Stop(TF_Status* status) {
    active_.store(false, std::memory_order_release);
    TF_SetStatus(status, TF_OK, "");
  }

  bool active() const { return active_.load(std::memory_order_acquire); }

  void Record(const std::string& label, double start, double end) {
    // A command buffer that did no GPU work reports zero for both, and one
    // that was still running when the session stopped would report an end
    // before its start.
    if (start <= 0.0 || end <= start) return;
    absl::MutexLock lock(&mu_);
    if (!active_.load(std::memory_order_relaxed)) return;
    Sample sample;
    sample.metadata_id = MetadataIdLocked(label.empty() ? "unnamed" : label);
    sample.start_ns =
        realtime_base_ns_ +
        static_cast<int64_t>((start - media_base_) * 1e9);
    sample.duration_ns = static_cast<int64_t>((end - start) * 1e9);
    samples_.push_back(sample);
  }

  // Serializes what was collected, and empties it: the C API says only the
  // first call after a stop returns data.
  std::vector<uint8_t> Collect() {
    absl::MutexLock lock(&mu_);
    if (collected_ || samples_.empty()) return {};
    collected_ = true;

    std::sort(samples_.begin(), samples_.end(),
              [](const Sample& a, const Sample& b) {
                return a.start_ns < b.start_ns;
              });
    const int64_t line_start_ns = samples_.front().start_ns;

    ProtoWriter line;
    line.Int64Field(1, 1);                       // id
    line.StringField(2, "Metal command buffers");
    line.Int64Field(3, line_start_ns);           // timestamp_ns
    for (const Sample& sample : samples_) {
      ProtoWriter event;
      event.Int64Field(1, sample.metadata_id);
      // Offsets are picoseconds from the line's own timestamp.
      event.Int64Field(2, (sample.start_ns - line_start_ns) * 1000);
      event.Int64Field(3, sample.duration_ns * 1000);
      line.MessageField(4, event);
    }

    ProtoWriter plane;
    plane.Int64Field(1, 1);  // id
    // The name is what the trace viewer shows as the row group. TensorFlow
    // groups device planes by the "/device:GPU:" prefix, so keeping it puts
    // this line where a CUDA trace's would be.
    plane.StringField(2, "/device:GPU:0 (Metal)");
    plane.MessageField(3, line);
    for (const auto& entry : metadata_) {
      ProtoWriter value;
      value.Int64Field(1, entry.second);
      value.StringField(2, entry.first);
      ProtoWriter pair;  // map entries are messages of key 1 and value 2
      pair.Int64Field(1, entry.second);
      pair.MessageField(2, value);
      plane.MessageField(4, pair);
    }

    ProtoWriter space;
    space.MessageField(1, plane);
    return space.bytes();
  }

 private:
  Session() = default;

  int64_t MetadataIdLocked(const std::string& name)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto it = metadata_.find(name);
    if (it != metadata_.end()) return it->second;
    // Ids start at one: zero is the proto3 default and would be dropped from
    // the wire, leaving events pointing at nothing.
    const int64_t id = static_cast<int64_t>(metadata_.size()) + 1;
    metadata_.emplace(name, id);
    return id;
  }

  // Read on every command buffer completion, so it is deliberately outside the
  // mutex: an unprofiled run must not serialise on the profiler.
  std::atomic<bool> active_{false};

  absl::Mutex mu_;
  std::vector<Sample> samples_ ABSL_GUARDED_BY(mu_);
  std::map<std::string, int64_t> metadata_ ABSL_GUARDED_BY(mu_);
  double media_base_ ABSL_GUARDED_BY(mu_) = 0.0;
  int64_t realtime_base_ns_ ABSL_GUARDED_BY(mu_) = 0;
  bool collected_ ABSL_GUARDED_BY(mu_) = false;
};

/*** THE C API ***/

void ProfilerStart(const TP_Profiler* profiler, TF_Status* status) {
  Session::Global().Start(status);
}

void ProfilerStop(const TP_Profiler* profiler, TF_Status* status) {
  Session::Global().Stop(status);
}

void ProfilerCollect(const TP_Profiler* profiler, uint8_t* buffer,
                     size_t* size_in_bytes, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  // Serialized once and held, because the caller asks for the size first and
  // for the bytes second, and the two answers have to describe the same data.
  static std::vector<uint8_t>* pending = new std::vector<uint8_t>();
  if (buffer == nullptr) {
    *pending = Session::Global().Collect();
    *size_in_bytes = pending->size();
    return;
  }
  if (*size_in_bytes < pending->size()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the profile buffer is smaller than the size that was "
                 "asked for.");
    return;
  }
  std::memcpy(buffer, pending->data(), pending->size());
  *size_in_bytes = pending->size();
  pending->clear();
}

void DestroyProfiler(TP_Profiler* profiler) {}
void DestroyProfilerFns(TP_ProfilerFns* profiler_fns) {}

}  // namespace

namespace {

// One storage behind both accessors. Thread local because the label belongs to
// whichever kernel this thread is running, and TensorFlow runs several at once.
std::string& CurrentOpNameStorage() {
  static thread_local std::string current;
  return current;
}

}  // namespace

void SetCurrentOpName(const char* name, size_t length) {
  if (name == nullptr || length == 0) {
    CurrentOpNameStorage().clear();
    return;
  }
  CurrentOpNameStorage().assign(name, length);
}

const std::string& CurrentOpName() { return CurrentOpNameStorage(); }

bool ProfilingActive() { return Session::Global().active(); }

void RecordCommandBuffer(const std::string& label, double start, double end) {
  Session::Global().Record(label, start, end);
}

void MetalInitProfiler(TF_ProfilerRegistrationParams* params,
                       TF_Status* status) {
  params->struct_size = TF_PROFILER_REGISTRATION_PARAMS_STRUCT_SIZE;
  params->major_version = TP_MAJOR;
  params->minor_version = TP_MINOR;
  params->patch_version = TP_PATCH;

  params->profiler->struct_size = TP_PROFILER_STRUCT_SIZE;
  params->profiler->device_type = kMetalDeviceType;

  params->profiler_fns->struct_size = TP_PROFILER_FNS_STRUCT_SIZE;
  params->profiler_fns->start = &ProfilerStart;
  params->profiler_fns->stop = &ProfilerStop;
  params->profiler_fns->collect_data_xspace = &ProfilerCollect;

  params->destroy_profiler = &DestroyProfiler;
  params->destroy_profiler_fns = &DestroyProfilerFns;
  TF_SetStatus(status, TF_OK, "");
}

}  // namespace metal
}  // namespace tensorflow
