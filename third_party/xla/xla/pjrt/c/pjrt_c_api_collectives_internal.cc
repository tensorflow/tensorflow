/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/c/pjrt_c_api_collectives_internal.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/backends/cpu/collectives/cpu_clique_key.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/executable_run_options.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_collectives_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/runtime/device_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/tsl/concurrency/future.h"

typedef struct PJRT_Collectives_Communicator {
  std::unique_ptr<xla::Communicator> communicator;
} PJRT_Collectives_Communicator;

typedef struct PJRT_Collectives_Communicators {
  std::vector<PJRT_Collectives_Communicator*> communicators;
} PJRT_Collectives_Communicators;

typedef struct PJRT_Collectives_ToString_Holder {
  std::string str;
} PJRT_Collectives_ToString_Holder;

namespace pjrt {

namespace {

void CommunicatorsDeleter(PJRT_Collectives_Communicators* ptr) { delete ptr; }

void ToStringHolderDeleter(PJRT_Collectives_ToString_Holder* ptr) {
  delete ptr;
}

std::vector<xla::GlobalDeviceId> ConvertDeviceIds(const int64_t* device_ids,
                                                  size_t num_device_ids) {
  std::vector<xla::GlobalDeviceId> device_ids_vec;
  device_ids_vec.reserve(num_device_ids);
  for (size_t i = 0; i < num_device_ids; ++i) {
    device_ids_vec.emplace_back(device_ids[i]);
  }
  return device_ids_vec;
}

xla::RendezvousKey::CollectiveOpKind ConvertFromPjRtCollectiveOpKind(
    PJRT_Collectives_CollectiveOpKind collective_op_kind) {
  switch (collective_op_kind) {
    case PJRT_COLLECTIVES_COLLECTIVE_OP_KIND_CROSS_MODULE:
      return xla::RendezvousKey::CollectiveOpKind::kCrossModule;
    case PJRT_COLLECTIVES_COLLECTIVE_OP_KIND_CROSS_REPLICA:
      return xla::RendezvousKey::CollectiveOpKind::kCrossReplica;
  }
}

xla::ReductionKind ConvertFromPjRtReductionKind(
    PJRT_Collectives_ReductionKind reduction_kind) {
  switch (reduction_kind) {
    case PJRT_COLLECTIVES_REDUCTION_SUM:
      return xla::ReductionKind::SUM;
    case PJRT_COLLECTIVES_REDUCTION_PRODUCT:
      return xla::ReductionKind::PRODUCT;
    case PJRT_COLLECTIVES_REDUCTION_MIN:
      return xla::ReductionKind::MIN;
    case PJRT_COLLECTIVES_REDUCTION_MAX:
      return xla::ReductionKind::MAX;
  }
}

std::unique_ptr<xla::cpu::CpuCollectives::Executor> CreateCpuExecutor(
    PJRT_Collectives_CpuExecutor* cpu_executor) {
  return std::make_unique<xla::cpu::CpuCollectives::Executor>(
      xla::RendezvousKey(
          xla::RunId(cpu_executor->run_id),
          ConvertDeviceIds(cpu_executor->global_device_ids,
                           cpu_executor->num_global_device_ids),
          cpu_executor->num_local_participants,
          ConvertFromPjRtCollectiveOpKind(cpu_executor->collective_op_kind),
          cpu_executor->op_id),
      absl::Nanoseconds(cpu_executor->timeout_in_ns));
}

template <typename Args>
absl::StatusOr<std::unique_ptr<xla::Communicator::Executor>> CreateExecutor(
    Args* args) {
  if (args->cpu_executor != nullptr) {
    return CreateCpuExecutor(args->cpu_executor);
  }
  return absl::InvalidArgumentError("cpu_executor is null");
}

PJRT_Error* CollectivesDestroy(PJRT_Collectives_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Collectives_Destroy_Args",
      PJRT_Collectives_Destroy_Args_STRUCT_SIZE, args->struct_size));
  delete args->collectives;
  return nullptr;
}

PJRT_Error* CollectivesCreateCommunicators(
    PJRT_Collectives_CreateCommunicators_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Collectives_CreateCommunicators_Args",
      PJRT_Collectives_CreateCommunicators_Args_STRUCT_SIZE,
      args->struct_size));

  std::vector<xla::GlobalDeviceId> device_ids =
      ConvertDeviceIds(args->device_ids, args->num_device_ids);

  xla::cpu::CpuCliqueKey clique_key(device_ids);

  std::optional<xla::CliqueIds> clique_ids = std::nullopt;
  if (args->clique_ids != nullptr) {
    clique_ids.emplace();
    for (size_t i = 0; i < args->num_clique_ids; ++i) {
      clique_ids->Add(xla::CliqueId(
          absl::string_view(args->clique_ids[i], args->clique_id_sizes[i])));
    }
  }

  std::vector<xla::Collectives::DeviceRank> device_ranks;
  device_ranks.reserve(args->num_device_ranks);
  for (size_t i = 0; i < args->num_device_ranks; ++i) {
    device_ranks.emplace_back(nullptr, xla::RankId(args->rank_ids[i]));
  }

  xla::Collectives::Config config;

  PJRT_ASSIGN_OR_RETURN(auto communicators,
                        args->collectives->collectives->CreateCommunicators(
                            clique_key, clique_ids, device_ranks, config));

  PJRT_Collectives_Communicators* communicators_holder =
      new PJRT_Collectives_Communicators;
  communicators_holder->communicators.reserve(communicators.size());
  for (auto& communicator : communicators) {
    PJRT_Collectives_Communicator* c_communicator =
        new PJRT_Collectives_Communicator;
    c_communicator->communicator = std::move(communicator);
    communicators_holder->communicators.push_back(c_communicator);
  }

  args->num_communicators = communicators_holder->communicators.size();
  args->communicators = communicators_holder->communicators.data();
  args->communicators_holder = communicators_holder;
  args->communicators_holder_deleter = CommunicatorsDeleter;
  return nullptr;
}

PJRT_Error* CommunicatorDestroy(
    PJRT_Collectives_Communicator_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Collectives_Communicator_Destroy_Args",
      PJRT_Collectives_Communicator_Destroy_Args_STRUCT_SIZE,
      args->struct_size));
  delete args->communicator;
  return nullptr;
}

PJRT_Error* CommunicatorAllReduce(
    PJRT_Collectives_Communicator_AllReduce_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Collectives_Communicator_AllReduce_Args",
      PJRT_Collectives_Communicator_AllReduce_Args_STRUCT_SIZE,
      args->struct_size));

  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::Communicator::Executor> executor,
                        CreateExecutor(args));
  stream_executor::DeviceAddressBase send_buffer(args->send_buffer_ptr,
                                                 args->send_buffer_size);
  stream_executor::DeviceAddressBase recv_buffer(args->recv_buffer_ptr,
                                                 args->recv_buffer_size);

  tsl::Future<> result = args->communicator->communicator->AllReduce(
      send_buffer, recv_buffer, ConvertFromPjRtBufferType(args->primitive_type),
      args->count, ConvertFromPjRtReductionKind(args->reduction_kind),
      *executor);
  args->event = new PJRT_Event{std::move(result)};

  return nullptr;
}

PJRT_Error* CommunicatorReduceScatter(
    PJRT_Collectives_Communicator_ReduceScatter_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Collectives_Communicator_ReduceScatter_Args",
      PJRT_Collectives_Communicator_ReduceScatter_Args_STRUCT_SIZE,
      args->struct_size));

  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::Communicator::Executor> executor,
                        CreateExecutor(args));
  stream_executor::DeviceAddressBase send_buffer(args->send_buffer_ptr,
                                                 args->send_buffer_size);
  stream_executor::DeviceAddressBase recv_buffer(args->recv_buffer_ptr,
                                                 args->recv_buffer_size);

  tsl::Future<> result = args->communicator->communicator->ReduceScatter(
      send_buffer, recv_buffer, ConvertFromPjRtBufferType(args->primitive_type),
      args->count, ConvertFromPjRtReductionKind(args->reduction_kind),
      *executor);
  args->event = new PJRT_Event{std::move(result)};

  return nullptr;
}

PJRT_Error* CommunicatorAllGather(
    PJRT_Collectives_Communicator_AllGather_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Collectives_Communicator_AllGather_Args",
      PJRT_Collectives_Communicator_AllGather_Args_STRUCT_SIZE,
      args->struct_size));

  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::Communicator::Executor> executor,
                        CreateExecutor(args));
  stream_executor::DeviceAddressBase send_buffer(args->send_buffer_ptr,
                                                 args->send_buffer_size);
  stream_executor::DeviceAddressBase recv_buffer(args->recv_buffer_ptr,
                                                 args->recv_buffer_size);

  tsl::Future<> result = args->communicator->communicator->AllGather(
      send_buffer, recv_buffer, ConvertFromPjRtBufferType(args->primitive_type),
      args->count, *executor);
  args->event = new PJRT_Event{std::move(result)};
  return nullptr;
}

PJRT_Error* CommunicatorCollectivePermute(
    PJRT_Collectives_Communicator_CollectivePermute_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Collectives_Communicator_CollectivePermute_Args",
      PJRT_Collectives_Communicator_CollectivePermute_Args_STRUCT_SIZE,
      args->struct_size));

  stream_executor::DeviceAddressBase send_buffer(args->send_buffer_ptr,
                                                 args->send_buffer_size);
  stream_executor::DeviceAddressBase recv_buffer(args->recv_buffer_ptr,
                                                 args->recv_buffer_size);

  std::optional<xla::RankId> source_rank_opt;
  if (args->has_source_rank) {
    source_rank_opt = xla::RankId(args->source_rank);
  }
  std::vector<xla::RankId> target_ranks;
  target_ranks.reserve(args->num_target_ranks);
  for (size_t i = 0; i < args->num_target_ranks; ++i) {
    target_ranks.emplace_back(args->target_ranks[i]);
  }

  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::Communicator::Executor> executor,
                        CreateExecutor(args));

  tsl::Future<> result = args->communicator->communicator->CollectivePermute(
      send_buffer, recv_buffer, ConvertFromPjRtBufferType(args->primitive_type),
      args->count, source_rank_opt, target_ranks, *executor);
  args->event = new PJRT_Event{std::move(result)};

  return nullptr;
}

PJRT_Error* CommunicatorAllToAll(
    PJRT_Collectives_Communicator_AllToAll_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Collectives_Communicator_AllToAll_Args",
      PJRT_Collectives_Communicator_AllToAll_Args_STRUCT_SIZE,
      args->struct_size));

  absl::InlinedVector<stream_executor::DeviceAddressBase, 4> send_buffers;
  send_buffers.reserve(args->num_send_buffers);
  for (size_t i = 0; i < args->num_send_buffers; ++i) {
    send_buffers.emplace_back(args->send_buffers_ptrs[i],
                              args->send_buffers_sizes[i]);
  }
  absl::InlinedVector<stream_executor::DeviceAddressBase, 4> recv_buffers;
  recv_buffers.reserve(args->num_recv_buffers);
  for (size_t i = 0; i < args->num_recv_buffers; ++i) {
    recv_buffers.emplace_back(args->recv_buffers_ptrs[i],
                              args->recv_buffers_sizes[i]);
  }

  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::Communicator::Executor> executor,
                        CreateExecutor(args));
  tsl::Future<> result = args->communicator->communicator->AllToAll(
      send_buffers, recv_buffers,
      ConvertFromPjRtBufferType(args->primitive_type), args->count, *executor);
  args->event = new PJRT_Event{std::move(result)};
  return nullptr;
}

PJRT_Error* CommunicatorToString(
    PJRT_Collectives_Communicator_ToString_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Collectives_Communicator_ToString_Args",
      PJRT_Collectives_Communicator_ToString_Args_STRUCT_SIZE,
      args->struct_size));

  PJRT_Collectives_ToString_Holder* str_holder =
      new PJRT_Collectives_ToString_Holder;
  args->str_holder = str_holder;
  args->str_holder_deleter = ToStringHolderDeleter;

  str_holder->str = args->communicator->communicator->ToString();
  args->str = str_holder->str.c_str();
  args->str_size = str_holder->str.size();

  return nullptr;
}

}  // namespace

PJRT_Collectives_Extension CreateCollectivesExtension(
    PJRT_Extension_Base* next) {
  return PJRT_Collectives_Extension{
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_Collectives_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type_Collectives,
          /*next=*/next,
      },
      /*collectives_destroy=*/CollectivesDestroy,
      /*collectives_create_communicators=*/CollectivesCreateCommunicators,
      /*communicator_destroy=*/CommunicatorDestroy,
      /*communicator_all_reduce=*/CommunicatorAllReduce,
      /*communicator_reduce_scatter=*/CommunicatorReduceScatter,
      /*communicator_all_gather=*/CommunicatorAllGather,
      /*communicator_collective_permute=*/CommunicatorCollectivePermute,
      /*communicator_all_to_all=*/CommunicatorAllToAll,
      /*communicator_to_string=*/CommunicatorToString,
  };
}

}  // namespace pjrt
