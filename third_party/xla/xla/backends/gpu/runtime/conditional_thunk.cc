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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/functional/overload.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/host_memory_pool.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/shaped_slice.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

ConditionalThunk::ConditionalThunk(
    ThunkInfo thunk_info, const ShapedSlice& branch_index_buffer_index,
    std::vector<std::unique_ptr<SequentialThunk>>&& branch_thunks)
    : Thunk(Kind::kConditional, thunk_info),
      branch_index_buffer_index_(branch_index_buffer_index),
      branch_thunks_(std::move(branch_thunks)),
      branch_index_is_bool_(branch_index_buffer_index.shape.element_type() ==
                            PRED) {
  PrimitiveType element_type = branch_index_buffer_index.shape.element_type();
  CHECK(element_type == PRED || element_type == S32);
  CHECK_EQ(branch_index_buffer_index.shape.dimensions(),
           std::vector<int64_t>{});
}

absl::Status ConditionalThunk::Prepare(const PrepareParams& params) {
  if (branch_index_is_bool_) {
    TF_RET_CHECK(branch_thunks_.size() == 2);
  } else {
    TF_RET_CHECK(!branch_thunks_.empty());
  }
  for (auto& branch_thunk : branch_thunks_) {
    TF_RETURN_IF_ERROR(branch_thunk->Prepare(params));
  }
  return absl::OkStatus();
}

absl::Status ConditionalThunk::Initialize(const InitializeParams& params) {
  if (branch_index_is_bool_) {
    TF_RET_CHECK(branch_thunks_.size() == 2);
  } else {
    TF_RET_CHECK(!branch_thunks_.empty());
  }
  for (auto& branch_thunk : branch_thunks_) {
    TF_RETURN_IF_ERROR(branch_thunk->Initialize(params));
  }

  absl::MutexLock lock(mutex_);

  if (!host_memory_pools_.contains(params.executor)) {
    PrimitiveType type =
        branch_index_is_bool_ ? PrimitiveType::PRED : PrimitiveType::S32;
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
    absl::MutexLock lock(mutex_);
    pool = host_memory_pools_.at(stream.parent()).get();
  }
  TF_ASSIGN_OR_RETURN(HostMemoryPool::Handle handle, pool->Acquire());

  // Copy the predicate value from device.
  auto branch_index_or_pred = [&]() -> std::variant<int32_t*, bool*> {
    if (branch_index_is_bool_) {
      return handle.get<bool>();
    }
    return handle.get<int32_t>();
  }();

  se::DeviceAddressBase branch_index_address =
      params.buffer_allocations->GetDeviceAddress(
          branch_index_buffer_index_.slice);
  if (branch_index_is_bool_) {
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

  int32_t branch_index = std::visit(
      absl::Overload([](int32_t* branch_index) { return *branch_index; },
                     [](bool* pred) { return *pred ? 0 : 1; }),
      branch_index_or_pred);

  absl::string_view branch_kind =
      std::visit(absl::Overload([](int32_t*) { return "index"; },
                                [](bool*) { return "pred"; }),
                 branch_index_or_pred);

  VLOG(3) << "ConditionalThunk: branch_index=" << branch_index
          << " (kind: " << branch_kind << ")";

  // Handle default scenario for branch_index not in [0, num_branches).
  if (branch_index < 0 || branch_index >= branch_thunks_.size()) {
    branch_index = static_cast<int32_t>(branch_thunks_.size()) - 1;
  }

  // Execute the branch computation corresponding to the value of branch_index.
  TF_RETURN_IF_ERROR(branch_thunks_[branch_index]->ExecuteOnStream(params));

  return absl::OkStatus();
}

absl::Status ConditionalThunk::WalkNested(Walker callback) {
  for (const std::unique_ptr<SequentialThunk>& branch_thunk : branch_thunks_) {
    TF_RETURN_IF_ERROR(branch_thunk->Walk(callback));
  }
  return absl::OkStatus();
}

absl::Status ConditionalThunk::TransformNested(Transformer callback) {
  for (std::unique_ptr<SequentialThunk>& branch_thunk : branch_thunks_) {
    TF_RETURN_IF_ERROR(branch_thunk->TransformNested(callback));
    TF_ASSIGN_OR_RETURN(auto thunk, callback(std::move(branch_thunk)));
    branch_thunk = SequentialThunk::FromThunk(std::move(thunk));
  }
  return absl::OkStatus();
}

absl::StatusOr<ThunkProto> ConditionalThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  auto* conditional_thunk_proto = proto.mutable_conditional_thunk();
  TF_ASSIGN_OR_RETURN(*conditional_thunk_proto->mutable_branch_index_buffer(),
                      branch_index_buffer_index_.ToProto());

  for (const auto& seq_thunk : branch_thunks_) {
    TF_ASSIGN_OR_RETURN(ThunkProto seq_thunk_proto, seq_thunk->ToProto());
    *conditional_thunk_proto->add_branch_thunks() =
        std::move(seq_thunk_proto).sequential_thunk();
  }
  return proto;
}

absl::StatusOr<std::unique_ptr<ConditionalThunk>> ConditionalThunk::FromProto(
    ThunkInfo thunk_info, const ConditionalThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    const Deserializer& deserializer) {
  TF_ASSIGN_OR_RETURN(ShapedSlice branch_index_buffer_index,
                      ShapedSlice::FromProto(thunk_proto.branch_index_buffer(),
                                             buffer_allocations));

  std::vector<std::unique_ptr<SequentialThunk>> branch_thunks;
  branch_thunks.reserve(thunk_proto.branch_thunks_size());
  for (const auto& seq_thunk_proto : thunk_proto.branch_thunks()) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<SequentialThunk> seq_thunk,
        SequentialThunk::FromProto(thunk_info, seq_thunk_proto, deserializer));
    branch_thunks.push_back(std::move(seq_thunk));
  }
  return std::make_unique<ConditionalThunk>(std::move(thunk_info),
                                            branch_index_buffer_index,
                                            std::move(branch_thunks));
}

std::string ConditionalThunk::ToString(int indent) const {
  std::string indent_str(indent * 2, ' ');
  std::string result;
  absl::StrAppend(&result, indent_str, "\n");
  if (branch_index_is_bool_) {
    CHECK_EQ(branch_thunks_.size(), 2);
    absl::StrAppend(&result, indent_str, "false_branch:\n",
                    branch_thunks_[0]->ToString(indent + 1));
    absl::StrAppend(&result, indent_str, "true_branch:\n",
                    branch_thunks_[1]->ToString(indent + 1));
  } else {
    for (size_t i = 0; i < branch_thunks_.size(); ++i) {
      absl::StrAppend(&result, indent_str, "branch_", i, ":\n",
                      branch_thunks_[i]->ToString(indent + 1));
    }
  }
  return result;
}

}  // namespace gpu
}  // namespace xla
