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

#include <utility>

#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

//===----------------------------------------------------------------------===//
// Definitions for ConcurrentRegionStatus.
//===----------------------------------------------------------------------===//

ConcurrentRegionStatus::ConcurrentRegionStatus(
    const ServiceExecutableRunOptions* run_options, int num_borrowed_streams)
    : num_borrowed_streams_(num_borrowed_streams),
      is_in_concurrent_region_(false),
      stream_index_(0),
      run_options_(run_options) {}

se::Stream* ConcurrentRegionStatus::GetNextStream() {
  if (borrowed_streams_.empty()) {
    return nullptr;
  }
  int index = stream_index_ % borrowed_streams_.size();
  stream_index_++;
  return borrowed_streams_[index].get();
}

bool ConcurrentRegionStatus::is_in_concurrent_region() {
  return is_in_concurrent_region_;
}

absl::Status ConcurrentRegionStatus::StartConcurrentRegion() {
  DCHECK(!is_in_concurrent_region_);
  se::StreamExecutor* executor = run_options_->stream()->parent();

  // Stream borrowing should only happen in the first call to this function.
  for (int i = borrowed_streams_.size(); i < num_borrowed_streams_; i++) {
    TF_ASSIGN_OR_RETURN(StreamPool::Ptr ptr,
                        run_options_->BorrowStream(executor->device_ordinal()));
    borrowed_streams_.push_back(std::move(ptr));
  }
  is_in_concurrent_region_ = true;
  return absl::OkStatus();
}

void ConcurrentRegionStatus::EndConcurrentRegion() {
  DCHECK(is_in_concurrent_region_);
  stream_index_ = 0;
  is_in_concurrent_region_ = false;
}

//===----------------------------------------------------------------------===//
// Define custom calls that mark the concurrent region in CUDA graphs.
//===----------------------------------------------------------------------===//

using xla::runtime::CustomCall;

static absl::Status RegionBegin(ConcurrentRegionStatus* region_status) {
  return region_status->StartConcurrentRegion();
}

static absl::Status RegionEnd(ConcurrentRegionStatus* region_status) {
  region_status->EndConcurrentRegion();
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    Begin, FunctionWrapper<RegionBegin>(), checks,
    CustomCall::Bind("xla.gpu.concurrent_region.begin")
        .UserData<ConcurrentRegionStatus*>());

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
