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

#include "tensorflow/compiler/xla/service/gpu/runtime/concurrent_region.h"

#include <algorithm>
#include <utility>

#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/stream_pool.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"
#include "tensorflow/tsl/platform/statusor.h"

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

se::Stream* ConcurrentRegionStatus::GetNextStream() {
  DCHECK(IsInConcurrentRegion());
  if (borrowed_streams_.empty()) {
    return nullptr;
  }
  int index = stream_index_ % borrowed_streams_.size();
  stream_index_++;
  return borrowed_streams_[index].get();
}

absl::Status ConcurrentRegionStatus::StartConcurrentRegion(
    se::Stream* capture_stream, int64_t size) {
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

  // Switch borrowed streams into capture mode. If the number of kernel launches
  // in the region is less than the number of borrowed streams, only synchronize
  // enough streams to run the kernels.
  for (int i = 0; i < std::min<size_t>(size, num_borrowed_streams_); ++i) {
    borrowed_streams_[i]->ThenWaitFor(capture_stream);
  }

  region_size_ = size;
  capture_stream_ = capture_stream;
  return absl::OkStatus();
}

void ConcurrentRegionStatus::EndConcurrentRegion() {
  DCHECK(IsInConcurrentRegion());

  // Synchronize main capture stream with all borrowed streams in capture mode.
  for (int i = 0; i < std::min<size_t>(region_size_, num_borrowed_streams_);
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
