/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/conditional_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/host_memory_pool.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/overload.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

ConditionalThunk::ConditionalThunk(
    ThunkInfo thunk_info, ConditionalThunkConfig config,
    const BufferAllocation::Slice& branch_index_buffer_index)
    : Thunk(Kind::kConditional, thunk_info),
      config_(std::move(config)),
      branch_index_buffer_index_(branch_index_buffer_index) {}

absl::Status ConditionalThunk::Prepare(
    const PrepareParams& params, ResourceRequestsInterface& resource_requests) {
  if (config_.branch_index_is_bool) {
    TF_RET_CHECK(config_.branch_thunks.size() == 2);
  } else {
    TF_RET_CHECK(!config_.branch_thunks.empty());
  }
  for (auto& branch_thunk : config_.branch_thunks) {
    TF_RETURN_IF_ERROR(branch_thunk->Prepare(params, resource_requests));
  }
  return absl::OkStatus();
}

absl::Status ConditionalThunk::Initialize(const InitializeParams& params) {
  if (config_.branch_index_is_bool) {
    TF_RET_CHECK(config_.branch_thunks.size() == 2);
  } else {
    TF_RET_CHECK(!config_.branch_thunks.empty());
  }
  for (auto& branch_thunk : config_.branch_thunks) {
    TF_RETURN_IF_ERROR(branch_thunk->Initialize(params));
  }

  absl::MutexLock lock(&mutex_);

  if (!host_memory_pools_.contains(params.executor)) {
    PrimitiveType type =
        config_.branch_index_is_bool ? PrimitiveType::PRED : PrimitiveType::S32;
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HostMemoryPool> pool,
                        HostMemoryPool::Create(params.executor, type));
    host_memory_pools_[params.executor] = std::move(pool);
  }

  return absl::OkStatus();
}

absl::Status ConditionalThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto& stream = *params.stream;

  HostMemoryPool* pool;
  {
    absl::MutexLock lock(&mutex_);
    pool = host_memory_pools_.at(stream.parent()).get();
  }
  TF_ASSIGN_OR_RETURN(HostMemoryPool::Handle handle, pool->Acquire());

  // Copy the predicate value from device.
  auto branch_index_or_pred = [&]() -> std::variant<int32_t*, bool*> {
    if (config_.branch_index_is_bool) {
      return handle.get<bool>();
    } else {
      return handle.get<int32_t>();
    }
  }();

  se::DeviceMemoryBase branch_index_address =
      params.buffer_allocations->GetDeviceAddress(branch_index_buffer_index_);
  if (config_.branch_index_is_bool) {
    TF_RETURN_IF_ERROR(stream.Memcpy(std::get<bool*>(branch_index_or_pred),
                                     branch_index_address, sizeof(bool)));
  } else {
    TF_RETURN_IF_ERROR(stream.Memcpy(std::get<int32_t*>(branch_index_or_pred),
                                     branch_index_address, sizeof(int32_t)));
  }

  if (absl::Status blocked = stream.BlockHostUntilDone(); !blocked.ok()) {
    return Internal("Failed to retrieve branch_index value on stream %p: %s.",
                    &stream, blocked.message());
  }

  int32_t branch_index =
      std::visit(Overload{[](int32_t* branch_index) { return *branch_index; },
                          [](bool* pred) { return *pred ? 0 : 1; }},
                 branch_index_or_pred);

  absl::string_view branch_kind = std::visit(
      Overload{[](int32_t*) { return "index"; }, [](bool*) { return "pred"; }},
      branch_index_or_pred);

  VLOG(3) << "ConditionalThunk: branch_index=" << branch_index
          << " (kind: " << branch_kind << ")";

  // Handle default scenario for branch_index not in [0, num_branches).
  if (branch_index < 0 || branch_index >= config_.branch_count) {
    branch_index = config_.branch_count - 1;
  }

  // Execute the branch computation corresponding to the value of branch_index.
  TF_RETURN_IF_ERROR(
      config_.branch_thunks[branch_index]->ExecuteOnStream(params));

  return absl::OkStatus();
}

void ConditionalThunk::ForAllThunks(
    absl::FunctionRef<void(const Thunk*)> fn) const {
  fn(this);
  for (const std::unique_ptr<SequentialThunk>& branch_thunk :
       config_.branch_thunks) {
    branch_thunk->ForAllThunks(fn);
  }
}

}  // namespace gpu
}  // namespace xla
