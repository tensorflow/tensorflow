/* Copyright 2015 The OpenXLA Authors.

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

#include "xla/stream_executor/stream.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "Eigen/Core"  // from @eigen_archive
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/numeric_options.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/port.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_internal.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/stacktrace.h"

namespace stream_executor {

namespace {
// Code to turn parameters to functions on stream into strings that
// will be VLOG'ed. We need overloads, instead of
// e.g. BatchDescriptorToVlogString(), as the code that calls these
// functions does not know what the type of the parameter is.

std::string ToVlogString(const void *ptr) {
  if (ptr == nullptr) {
    return "null";
  }

  // StrCat does not convert pointers to text.
  std::ostringstream out;
  out << ptr;
  return out.str();
}

template <class T>
std::string ToVlogString(const std::function<T> &f) {
  return f == nullptr ? "null" : "<non-null function>";
}

template <class T>
std::string ToVlogString(const absl::AnyInvocable<T> &f) {
  return f == nullptr ? "null" : "<non-null function>";
}

std::string ToVlogString(const DeviceMemoryBase &memory) {
  return ToVlogString(memory.opaque());
}

std::string ToVlogString(const DeviceMemoryBase *memory) {
  return memory == nullptr ? "null" : ToVlogString(*memory);
}

std::string ToVlogString(uint32_t i) { return absl::StrCat(i); }

std::string ToVlogString(uint64_t i) { return absl::StrCat(i); }

std::string ToVlogString(float f) { return absl::StrCat(f); }

}  // namespace

Stream::Stream(StreamExecutor *parent)
    : parent_(parent),
      implementation_(parent->implementation()->GetStreamImplementation()),
      allocated_(false),
      status_(absl::InternalError("Uninitialized stream")) {}

Stream::~Stream() {
  // Ensure the stream is completed.
  auto status = BlockHostUntilDone();
  if (!status.ok()) {
    LOG(WARNING) << "Error blocking host until done in stream destructor: "
                 << status;
  }

  if (allocated_) {
    parent_->DeallocateStream(this);
  }
}

void Stream::SetPriority(StreamPriority priority) {
  implementation_->SetPriority(priority);
}

void Stream::SetPriority(int priority) {
  implementation_->SetPriority(priority);
}

std::variant<StreamPriority, int> Stream::priority() const {
  return implementation_->priority();
}

Stream::PlatformSpecificHandle Stream::platform_specific_handle() const {
  PlatformSpecificHandle handle;
  handle.stream = implementation_->platform_specific_stream();
  return handle;
}

absl::Status Stream::RefreshStatus() {
  absl::Status status = parent_->GetStatus(this);
  // We should not put the stream in an error state, just because the GetStatus
  // method is unimplemented.
  if (status != absl::UnimplementedError(
                    "GetStatus is not supported on this executor.")) {
    CheckStatus(status);
  }
  return status;
}

absl::Status Stream::Initialize() {
  absl::MutexLock lock(&mu_);
  if (allocated_) {
    return absl::InternalError(
        "stream appears to already have been initialized");
  }
  if (status_.ok()) {
    return absl::InternalError(
        "stream should be in !ok() state pre-initialization");
  }

  if (parent_->AllocateStream(this)) {
    // Successful initialization!
    allocated_ = true;
    status_ = absl::OkStatus();
    return absl::OkStatus();
  }

  return absl::InternalError("failed to allocate stream during initialization");
}

Stream &Stream::Init() {
  absl::Status status = Initialize();
  if (!status.ok()) {
    LOG(ERROR) << status;
  }

  return *this;
}

Stream &Stream::ThenRecordEvent(Event *event) {
  absl::Status status = RecordEvent(event);
  if (!status.ok()) {
    LOG(ERROR) << "Error recording event in stream: " << status.message()
               << "; not marking stream as bad, as the Event object may be "
               << "at fault. Monitor for further errors.";
  }

  return *this;
}

absl::Status Stream::RecordEvent(Event *event) {
  return parent_->RecordEvent(this, event);
}

absl::StatusOr<Stream *> Stream::GetOrCreateSubStream() {
  // Do not destroy bad streams when holding mu_ because ~Stream() may
  // BlockHostUntilDone and it's host callbacks might attempt to acquire mu_.
  std::vector<std::unique_ptr<Stream>> bad_streams;

  absl::MutexLock lock(&mu_);

  // Look for the first reusable sub_stream that is ok, dropping !ok sub_streams
  // we encounter along the way.
  for (size_t index = 0; index < sub_streams_.size();) {
    std::pair<std::unique_ptr<Stream>, bool> &pair = sub_streams_[index];
    if (pair.second) {
      // The sub_stream is reusable.
      Stream *sub_stream = pair.first.get();
      if (sub_stream->ok()) {
        VLOG(1) << DebugStreamPointers() << " reusing sub_stream "
                << sub_stream->DebugStreamPointers();
        pair.second = false;
        return sub_stream;
      }

      // The stream is reusable and not ok. Streams have a monotonic state
      // machine; the stream will remain in !ok forever. Swap it with the last
      // stream and pop it off.
      const int64_t last = sub_streams_.size() - 1;
      if (index != last) {
        std::swap(pair, sub_streams_[last]);
      }
      bad_streams.push_back(std::move(sub_streams_.back().first));
      sub_streams_.pop_back();
      VLOG(1) << DebugStreamPointers() << " dropped !ok sub_stream "
              << sub_stream->DebugStreamPointers();
    } else {
      // The sub_stream is not reusable, move on to the next one.
      ++index;
    }
  }

  // No streams are reusable; create a new stream.
  sub_streams_.emplace_back(std::make_unique<Stream>(parent_), false);
  Stream *sub_stream = sub_streams_.back().first.get();
  TF_RETURN_IF_ERROR(sub_stream->Initialize());
  VLOG(1) << DebugStreamPointers() << " created new sub_stream "
          << sub_stream->DebugStreamPointers();

  return sub_stream;
}

void Stream::ReturnSubStream(Stream *sub_stream) {
  // Do not destroy bad streams when holding mu_ because ~Stream() may
  // BlockHostUntilDone and it's host callbacks might attempt to acquire mu_.
  std::unique_ptr<Stream> bad_stream;

  absl::MutexLock lock(&mu_);

  // Look for the sub-stream.
  for (int64_t index = 0, end = sub_streams_.size(); index < end; ++index) {
    std::pair<std::unique_ptr<Stream>, bool> &pair = sub_streams_[index];
    if (pair.first.get() != sub_stream) {
      continue;
    }

    // Found the sub_stream.
    if (sub_stream->ok()) {
      VLOG(1) << DebugStreamPointers() << " returned ok sub_stream "
              << sub_stream->DebugStreamPointers();
      pair.second = true;
    } else {
      // The returned stream is not ok. Streams have a monotonic state
      // machine; the stream will remain in !ok forever. Swap it with the last
      // stream and pop it off.
      VLOG(1) << DebugStreamPointers() << " returned !ok sub_stream "
              << sub_stream->DebugStreamPointers();
      const int64_t last = sub_streams_.size() - 1;
      if (index != last) {
        std::swap(pair, sub_streams_[last]);
      }
      std::swap(bad_stream, sub_streams_.back().first);
      sub_streams_.pop_back();
    }
    return;
  }

  LOG(FATAL) << DebugStreamPointers()
             << " did not create the returned sub-stream "
             << sub_stream->DebugStreamPointers();
}

absl::Status Stream::WaitFor(Stream *other) {
  if (this == other) {
    return absl::InternalError("stream cannot wait for itself");
  }
  if (parent_->CreateStreamDependency(this, other)) {
    return absl::OkStatus();
  }
  return absl::InternalError("stream cannot wait for other");
}

absl::Status Stream::WaitFor(Event *event) {
  return parent_->WaitForEvent(this, event);
}

Stream &Stream::ThenMemcpy(void *host_dst, const DeviceMemoryBase &gpu_src,
                           uint64_t size) {
  CheckStatus(Memcpy(host_dst, gpu_src, size));
  return *this;
}

absl::Status Stream::Memcpy(void *host_dst, const DeviceMemoryBase &gpu_src,
                            uint64_t size) {
  if (parent_->Memcpy(this, host_dst, gpu_src, size)) {
    return absl::OkStatus();
  }
  return absl::InternalError("failed to memcpy");
}

Stream &Stream::ThenMemcpy(DeviceMemoryBase *gpu_dst, const void *host_src,
                           uint64_t size) {
  CheckStatus(Memcpy(gpu_dst, host_src, size));
  return *this;
}

absl::Status Stream::Memcpy(DeviceMemoryBase *gpu_dst, const void *host_src,
                            uint64_t size) {
  if (parent_->Memcpy(this, gpu_dst, host_src, size)) {
    return absl::OkStatus();
  }
  return absl::InternalError("failed to memcpy");
}

Stream &Stream::ThenMemcpy(DeviceMemoryBase *gpu_dst,
                           const DeviceMemoryBase &gpu_src, uint64_t size) {
  CheckStatus(Memcpy(gpu_dst, gpu_src, size));
  return *this;
}

absl::Status Stream::Memcpy(DeviceMemoryBase *gpu_dst,
                            const DeviceMemoryBase &gpu_src, uint64_t size) {
  if (parent_->MemcpyDeviceToDevice(this, gpu_dst, gpu_src, size)) {
    return absl::OkStatus();
  }
  return absl::InternalError("failed to memcpy");
}

absl::Status Stream::MemZero(DeviceMemoryBase *location, uint64_t size) {
  return parent_->MemZero(this, location, size);
}

absl::Status Stream::Memset32(DeviceMemoryBase *location, uint32_t pattern,
                              uint64_t size) {
  return parent_->Memset32(this, location, pattern, size);
}

absl::Status Stream::DoHostCallback(absl::AnyInvocable<void() &&> callback) {
  return DoHostCallbackWithStatus([cb = std::move(callback)]() mutable {
    std::move(cb)();
    return absl::OkStatus();
  });
}

absl::Status Stream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback) {
  if (parent_->HostCallback(this, std::move(callback))) {
    return absl::OkStatus();
  }
  return absl::InternalError("failed to host callback");
}

absl::Status Stream::BlockHostUntilDone() {
  if (!ok()) {
    absl::MutexLock lock(&mu_);
    LOG(INFO) << status_.ToString();
    absl::Status status = absl::InternalError(
        "stream did not block host until done; was already in an error state");
    LOG(INFO) << DebugStreamPointers() << " " << status;
    return status;
  }

  return parent_->BlockHostUntilDone(this);
}

std::string Stream::DebugStreamPointers() const {
  // Relies on the ToVlogString(const void*) overload above.
  return absl::StrCat("[stream=", ToVlogString(this),
                      ",impl=", ToVlogString(implementation_.get()), "]");
}

void Stream::CheckStatus(absl::Status status) {
  if (status.ok()) {
    return;
  }
  LOG(ERROR) << status;
  absl::MutexLock lock(&mu_);
  status_ = status;
}

}  // namespace stream_executor
