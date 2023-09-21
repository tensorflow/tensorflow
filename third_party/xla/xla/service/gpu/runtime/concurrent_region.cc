/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/runtime/concurrent_region.h"

#include <algorithm>
#include <utility>

#include "xla/runtime/custom_call.h"
#include "xla/runtime/executable.h"
#include "xla/service/gpu/runtime/support.h"
#include "xla/service/stream_pool.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

//===----------------------------------------------------------------------===//
// Definitions for ConcurrentRegionStatus.
//===----------------------------------------------------------------------===//

ConcurrentRegionStatus::ConcurrentRegionStatus(
    const ServiceExecutableRunOptions* run_options, int num_borrowed_streams)
    : num_borrowed_streams_(num_borrowed_streams),
      run_options_(run_options),
      stream_index_(0),
      capture_stream_(nullptr) {}

ConcurrentRegionStatus::~ConcurrentRegionStatus() {
  DCHECK(!IsInConcurrentRegion());
}

// Assign a stream in a round-robin fashion. Either the capture stream or one of
// the borrowed streams is returned.
se::Stream* ConcurrentRegionStatus::GetNextStream() {
  DCHECK(IsInConcurrentRegion());
  if (borrowed_streams_.empty()) {
    return nullptr;
  }

  int index = stream_index_ % (borrowed_streams_.size() + 1);
  stream_index_++;

  if (index == 0) {
    return capture_stream_;
  }

  return borrowed_streams_[index - 1].get();
}

absl::StatusOr<se::Stream*> ConcurrentRegionStatus::GetStream(int index) {
  DCHECK(IsInConcurrentRegion());

  if (index < 0 || index >= region_size_) {
    return absl::OutOfRangeError("Invalid stream index");
  }

  if (index == 0) {
    return capture_stream_;
  }

  return borrowed_streams_[index - 1].get();
}

absl::Status ConcurrentRegionStatus::StartConcurrentRegion(
    se::Stream* capture_stream, int64_t size) {
  if (disabled_) {
    return absl::OkStatus();
  }

  DCHECK(!IsInConcurrentRegion());
  se::StreamExecutor* executor = run_options_->stream()->parent();

  // Stream borrowing should only happen in the first call to this function.
  if (borrowed_streams_.empty()) {
    TF_ASSIGN_OR_RETURN(std::vector<StreamPool::Ptr> borrowed_streams,
                        run_options_->BorrowStreams(executor->device_ordinal(),
                                                    num_borrowed_streams_));
    for (StreamPool::Ptr& stream : borrowed_streams) {
      borrowed_streams_.push_back(std::move(stream));
    }
  }

  // Switch borrowed streams into capture mode. We only synchronize enough
  // streams to run the kernels.
  for (int i = 0; i < std::min<size_t>(size - 1, num_borrowed_streams_); ++i) {
    borrowed_streams_[i]->ThenWaitFor(capture_stream);
  }

  region_size_ = size;
  capture_stream_ = capture_stream;
  return absl::OkStatus();
}

void ConcurrentRegionStatus::EndConcurrentRegion() {
  if (disabled_) {
    return;
  }

  DCHECK(IsInConcurrentRegion());

  // Synchronize main capture stream with all borrowed streams in capture mode.
  for (int i = 0; i < std::min<size_t>(region_size_ - 1, num_borrowed_streams_);
       ++i) {
    capture_stream_->ThenWaitFor(borrowed_streams_[i].get());
  }

  stream_index_ = 0;
  capture_stream_ = nullptr;
}

bool ConcurrentRegionStatus::IsInConcurrentRegion() {
  return capture_stream_ != nullptr;
}

//===----------------------------------------------------------------------===//
// Define custom calls that mark the concurrent region in CUDA graphs.
//===----------------------------------------------------------------------===//

using xla::runtime::CustomCall;

static absl::Status RegionBegin(const ServiceExecutableRunOptions* run_options,
                                ConcurrentRegionStatus* region_status,
                                int64_t size) {
  se::Stream* capture_stream = run_options->stream();
  return region_status->StartConcurrentRegion(capture_stream, size);
}

static absl::Status RegionEnd(ConcurrentRegionStatus* region_status) {
  region_status->EndConcurrentRegion();
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    Begin, FunctionWrapper<RegionBegin>(), checks,
    CustomCall::Bind("xla.gpu.concurrent_region.begin")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<ConcurrentRegionStatus*>()
        .Attr<int64_t>("size"));

XLA_RUNTIME_DEFINE_CUSTOM_CALL(End, FunctionWrapper<RegionEnd>(), checks,
                               CustomCall::Bind("xla.gpu.concurrent_region.end")
                                   .UserData<ConcurrentRegionStatus*>());

void RegisterConcurrentRegionCustomCalls(
    runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.concurrent_region.begin", Begin);
  registry.Register("xla.gpu.concurrent_region.end", End);
}

}  // namespace gpu
}  // namespace xla
