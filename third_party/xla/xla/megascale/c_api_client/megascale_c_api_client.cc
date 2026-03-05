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

#include "xla/megascale/c_api_client/megascale_c_api_client.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/future.h"
#include "xla/megascale/c_api_client/megascale_types.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_collectives_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_megascale_extension.h"
#include "xla/pjrt/c/pjrt_c_api_multi_slice_extension.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_multi_slice_config.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/plugin_names.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"

// Return error future if not success and frees the PJRT_Error returned by
// `expr`.
#define RETURN_FUTURE_IF_ERROR(expr, c_api)                              \
  do {                                                                   \
    PJRT_Error* error = (expr);                                          \
    std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> _error(         \
        error, pjrt::MakeErrorDeleter(c_api));                           \
    absl::Status _status = pjrt::PjrtErrorToStatus(_error.get(), c_api); \
    if (!_status.ok()) {                                                 \
      return Future<>(_status);                                          \
    }                                                                    \
  } while (false)

#define ASSIGN_OR_RETURN_FUTURE(lhs, rexpr) \
  ASSIGN_OR_RETURN_FUTURE_IMPL(             \
      TF_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, rexpr)

#define ASSIGN_OR_RETURN_FUTURE_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                                 \
  if (TF_PREDICT_FALSE(!statusor.ok())) {                  \
    return Future<>(statusor.status());                    \
  }                                                        \
  lhs = std::move(statusor).value()

namespace xla {
namespace megascale {
namespace c_api_client {
namespace {

PJRT_Collectives_ReductionKind ConvertToPjRtReductionKind(
    xla::ReductionKind reduction_kind) {
  switch (reduction_kind) {
    case xla::ReductionKind::SUM:
      return PJRT_COLLECTIVES_REDUCTION_SUM;
    case xla::ReductionKind::PRODUCT:
      return PJRT_COLLECTIVES_REDUCTION_PRODUCT;
    case xla::ReductionKind::MIN:
      return PJRT_COLLECTIVES_REDUCTION_MIN;
    case xla::ReductionKind::MAX:
      return PJRT_COLLECTIVES_REDUCTION_MAX;
  }
}

PJRT_Collectives_CollectiveOpKind ConvertToPjRtCollectiveOpKind(
    xla::RendezvousKey::CollectiveOpKind collective_op_kind) {
  switch (collective_op_kind) {
    case xla::RendezvousKey::CollectiveOpKind::kCrossModule:
      return PJRT_COLLECTIVES_COLLECTIVE_OP_KIND_CROSS_MODULE;
    case xla::RendezvousKey::CollectiveOpKind::kCrossReplica:
      return PJRT_COLLECTIVES_COLLECTIVE_OP_KIND_CROSS_REPLICA;
  }
}

void FillCpuExecutor(const xla::cpu::CpuCollectives::Executor& executor,
                     PJRT_Collectives_CpuExecutor* c_executor,
                     absl::InlinedVector<int64_t, 4>* global_device_ids) {
  global_device_ids->clear();
  global_device_ids->reserve(executor.rendezvous_key().global_devices.size());
  for (const auto& device_id : executor.rendezvous_key().global_devices) {
    global_device_ids->push_back(device_id.value());
  }

  c_executor->run_id = executor.rendezvous_key().run_id.ToInt();
  c_executor->global_device_ids = global_device_ids->data();
  c_executor->num_global_device_ids = global_device_ids->size();
  c_executor->num_local_participants =
      executor.rendezvous_key().num_local_participants;
  c_executor->collective_op_kind = ConvertToPjRtCollectiveOpKind(
      executor.rendezvous_key().collective_op_kind);
  c_executor->op_id = executor.rendezvous_key().op_id;
  c_executor->timeout_in_ns = absl::ToInt64Nanoseconds(executor.timeout());
}

class MegascaleCApiCommunicator : public Communicator {
 public:
  MegascaleCApiCommunicator(PJRT_Collectives_Communicator* communicator,
                            const PJRT_Api* c_api,
                            const PJRT_Collectives_Extension* extension)
      : communicator_(communicator), c_api_(c_api), extension_(extension) {}

  ~MegascaleCApiCommunicator() override {
    PJRT_Collectives_Communicator_Destroy_Args args;
    args.struct_size = PJRT_Collectives_Communicator_Destroy_Args_STRUCT_SIZE;
    args.communicator = communicator_;
    std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> error(
        extension_->communicator_destroy(&args),
        pjrt::MakeErrorDeleter(c_api_));
    if (error) {
      LOG(ERROR) << "Failed to destroy MegascaleCApiCommunicator: "
                 << pjrt::PjrtErrorToStatus(error.get(), c_api_);
    }
  }

  xla::Future<> AllReduce(stream_executor::DeviceAddressBase send_buffer,
                          stream_executor::DeviceAddressBase recv_buffer,
                          xla::PrimitiveType dtype, size_t count,
                          xla::ReductionKind reduction_kind,
                          const Executor& executor) override {
    absl::InlinedVector<int64_t, 4> global_device_ids;
    PJRT_Collectives_CpuExecutor c_cpu_executor;

    ASSIGN_OR_RETURN_FUTURE(
        const xla::cpu::CpuCollectives::Executor* cpu_executor,
        xla::cpu::CpuCollectives::TryCast(&executor));
    FillCpuExecutor(*cpu_executor, &c_cpu_executor, &global_device_ids);

    PJRT_Collectives_Communicator_AllReduce_Args args;
    args.struct_size = PJRT_Collectives_Communicator_AllReduce_Args_STRUCT_SIZE;
    args.communicator = communicator_;

    args.send_buffer_ptr = send_buffer.opaque();
    args.send_buffer_size = send_buffer.size();

    args.recv_buffer_ptr = recv_buffer.opaque();
    args.recv_buffer_size = recv_buffer.size();

    args.primitive_type = pjrt::ConvertToPjRtBufferType(dtype);
    args.count = count;
    args.reduction_kind = ConvertToPjRtReductionKind(reduction_kind);
    args.cpu_executor = &c_cpu_executor;
    args.event = nullptr;

    RETURN_FUTURE_IF_ERROR(extension_->communicator_all_reduce(&args), c_api_);

    return pjrt::ConvertCEventToCppFuture(args.event, c_api_);
  }

  xla::Future<> ReduceScatter(stream_executor::DeviceAddressBase send_buffer,
                              stream_executor::DeviceAddressBase recv_buffer,
                              xla::PrimitiveType dtype, size_t count,
                              xla::ReductionKind reduction_kind,
                              const Executor& executor) override {
    absl::InlinedVector<int64_t, 4> global_device_ids;
    PJRT_Collectives_CpuExecutor c_cpu_executor;
    ASSIGN_OR_RETURN_FUTURE(
        const xla::cpu::CpuCollectives::Executor* cpu_executor,
        xla::cpu::CpuCollectives::TryCast(&executor));
    FillCpuExecutor(*cpu_executor, &c_cpu_executor, &global_device_ids);

    PJRT_Collectives_Communicator_ReduceScatter_Args args;
    args.struct_size =
        PJRT_Collectives_Communicator_ReduceScatter_Args_STRUCT_SIZE;
    args.communicator = communicator_;

    args.send_buffer_ptr = send_buffer.opaque();
    args.send_buffer_size = send_buffer.size();

    args.recv_buffer_ptr = recv_buffer.opaque();
    args.recv_buffer_size = recv_buffer.size();

    args.primitive_type = pjrt::ConvertToPjRtBufferType(dtype);
    args.count = count;
    args.reduction_kind = ConvertToPjRtReductionKind(reduction_kind);
    args.cpu_executor = &c_cpu_executor;
    args.event = nullptr;

    RETURN_FUTURE_IF_ERROR(extension_->communicator_reduce_scatter(&args),
                           c_api_);
    return pjrt::ConvertCEventToCppFuture(args.event, c_api_);
  }

  xla::Future<> AllGather(stream_executor::DeviceAddressBase send_buffer,
                          stream_executor::DeviceAddressBase recv_buffer,
                          xla::PrimitiveType dtype, size_t count,
                          const Executor& executor) override {
    absl::InlinedVector<int64_t, 4> global_device_ids;
    PJRT_Collectives_CpuExecutor c_cpu_executor;
    ASSIGN_OR_RETURN_FUTURE(
        const xla::cpu::CpuCollectives::Executor* cpu_executor,
        xla::cpu::CpuCollectives::TryCast(&executor));
    FillCpuExecutor(*cpu_executor, &c_cpu_executor, &global_device_ids);

    PJRT_Collectives_Communicator_AllGather_Args args;
    args.struct_size = PJRT_Collectives_Communicator_AllGather_Args_STRUCT_SIZE;
    args.communicator = communicator_;

    args.send_buffer_ptr = send_buffer.opaque();
    args.send_buffer_size = send_buffer.size();

    args.recv_buffer_ptr = recv_buffer.opaque();
    args.recv_buffer_size = recv_buffer.size();

    args.primitive_type = pjrt::ConvertToPjRtBufferType(dtype);
    args.count = count;
    args.cpu_executor = &c_cpu_executor;
    args.event = nullptr;

    RETURN_FUTURE_IF_ERROR(extension_->communicator_all_gather(&args), c_api_);
    return pjrt::ConvertCEventToCppFuture(args.event, c_api_);
  }

  xla::Future<> CollectivePermute(
      stream_executor::DeviceAddressBase send_buffer,
      stream_executor::DeviceAddressBase recv_buffer, xla::PrimitiveType dtype,
      size_t count, std::optional<xla::RankId> source_rank,
      absl::Span<const xla::RankId> target_ranks,
      const Executor& executor) override {
    absl::InlinedVector<int64_t, 4> global_device_ids;
    PJRT_Collectives_CpuExecutor c_cpu_executor;
    ASSIGN_OR_RETURN_FUTURE(
        const xla::cpu::CpuCollectives::Executor* cpu_executor,
        xla::cpu::CpuCollectives::TryCast(&executor));
    FillCpuExecutor(*cpu_executor, &c_cpu_executor, &global_device_ids);

    PJRT_Collectives_Communicator_CollectivePermute_Args args;
    args.struct_size =
        PJRT_Collectives_Communicator_CollectivePermute_Args_STRUCT_SIZE;
    args.communicator = communicator_;

    args.send_buffer_ptr = send_buffer.opaque();
    args.send_buffer_size = send_buffer.size();

    args.recv_buffer_ptr = recv_buffer.opaque();
    args.recv_buffer_size = recv_buffer.size();

    args.primitive_type = pjrt::ConvertToPjRtBufferType(dtype);
    args.count = count;

    if (source_rank.has_value()) {
      args.source_rank = source_rank->value();
      args.has_source_rank = true;
    } else {
      args.source_rank = 0;
      args.has_source_rank = false;
    }

    absl::InlinedVector<int64_t, 4> target_ranks_val;
    target_ranks_val.reserve(target_ranks.size());
    for (const auto& rank : target_ranks) {
      target_ranks_val.push_back(rank.value());
    }
    args.target_ranks = target_ranks_val.data();
    args.num_target_ranks = target_ranks_val.size();
    args.cpu_executor = &c_cpu_executor;
    args.event = nullptr;

    RETURN_FUTURE_IF_ERROR(extension_->communicator_collective_permute(&args),
                           c_api_);
    return pjrt::ConvertCEventToCppFuture(args.event, c_api_);
  }

  xla::Future<> AllToAll(
      absl::InlinedVector<stream_executor::DeviceAddressBase, 4> send_buffers,
      absl::InlinedVector<stream_executor::DeviceAddressBase, 4> recv_buffers,
      xla::PrimitiveType dtype, size_t count,
      const Executor& executor) override {
    absl::InlinedVector<int64_t, 4> global_device_ids;
    PJRT_Collectives_CpuExecutor c_cpu_executor;
    ASSIGN_OR_RETURN_FUTURE(
        const xla::cpu::CpuCollectives::Executor* cpu_executor,
        xla::cpu::CpuCollectives::TryCast(&executor));
    FillCpuExecutor(*cpu_executor, &c_cpu_executor, &global_device_ids);

    PJRT_Collectives_Communicator_AllToAll_Args args;
    args.struct_size = PJRT_Collectives_Communicator_AllToAll_Args_STRUCT_SIZE;
    args.communicator = communicator_;

    std::vector<void*> send_ptrs;
    std::vector<size_t> send_sizes;
    send_ptrs.reserve(send_buffers.size());
    send_sizes.reserve(send_buffers.size());
    for (const auto& buffer : send_buffers) {
      send_ptrs.push_back(buffer.opaque());
      send_sizes.push_back(buffer.size());
    }
    args.send_buffers_ptrs = send_ptrs.data();
    args.send_buffers_sizes = send_sizes.data();
    args.num_send_buffers = send_buffers.size();

    std::vector<void*> recv_ptrs;
    std::vector<size_t> recv_sizes;
    recv_ptrs.reserve(recv_buffers.size());
    recv_sizes.reserve(recv_buffers.size());
    for (const auto& buffer : recv_buffers) {
      recv_ptrs.push_back(buffer.opaque());
      recv_sizes.push_back(buffer.size());
    }
    args.recv_buffers_ptrs = recv_ptrs.data();
    args.recv_buffers_sizes = recv_sizes.data();
    args.num_recv_buffers = recv_buffers.size();

    args.primitive_type = pjrt::ConvertToPjRtBufferType(dtype);
    args.count = count;
    args.cpu_executor = &c_cpu_executor;
    args.event = nullptr;

    RETURN_FUTURE_IF_ERROR(extension_->communicator_all_to_all(&args), c_api_);
    return pjrt::ConvertCEventToCppFuture(args.event, c_api_);
  }

  Future<> Broadcast(stream_executor::DeviceAddressBase,
                     stream_executor::DeviceAddressBase, PrimitiveType, size_t,
                     RankId, const Executor&) override {
    return Unimplemented("Broadcast is not implemented");
  }

  Future<> Send(stream_executor::DeviceAddressBase, PrimitiveType, size_t,
                RankId, const Executor&) override {
    return Unimplemented("Send is not implemented");
  }

  Future<> Recv(stream_executor::DeviceAddressBase, PrimitiveType, size_t,
                RankId, const Executor&) override {
    return Unimplemented("Recv is not implemented");
  }

  absl::StatusOr<size_t> NumRanks() const override {
    return Unimplemented("NumRanks is not implemented");
  }

  std::string ToString() const override {
    PJRT_Collectives_Communicator_ToString_Args args;
    args.struct_size = PJRT_Collectives_Communicator_ToString_Args_STRUCT_SIZE;
    args.communicator = communicator_;
    pjrt::LogFatalIfPjrtError(extension_->communicator_to_string(&args),
                              c_api_);
    std::string result = std::string(args.str, args.str_size);
    args.str_holder_deleter(args.str_holder);
    return result;
  }

 private:
  PJRT_Collectives_Communicator* communicator_;
  const PJRT_Api* c_api_;
  const PJRT_Collectives_Extension* extension_;
};

class MegascaleCApiCollectives : public cpu::CpuCollectives {
 public:
  MegascaleCApiCollectives(PJRT_Collectives* collectives, const PJRT_Api* c_api,
                           const PJRT_Collectives_Extension* extension)
      : collectives_(collectives), c_api_(c_api), extension_(extension) {}

  ~MegascaleCApiCollectives() override {
    PJRT_Collectives_Destroy_Args args;
    args.struct_size = PJRT_Collectives_Destroy_Args_STRUCT_SIZE;
    args.collectives = collectives_;
    std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> error(
        extension_->collectives_destroy(&args), pjrt::MakeErrorDeleter(c_api_));
    if (error) {
      LOG(ERROR) << "Failed to destroy MegascaleCApiCollectives: "
                 << pjrt::PjrtErrorToStatus(error.get(), c_api_);
    }
  }

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
  CreateCommunicators(const xla::CliqueKey& clique_key,
                      const std::optional<CliqueIds>& clique_ids,
                      absl::Span<const DeviceRank> ranks,
                      const Config& config) override {
    PJRT_Collectives_CreateCommunicators_Args args;
    args.struct_size = PJRT_Collectives_CreateCommunicators_Args_STRUCT_SIZE;
    args.collectives = collectives_;

    absl::InlinedVector<int64_t, 4> device_ids;
    device_ids.reserve(clique_key.devices().size());
    for (const auto& device_id : clique_key.devices()) {
      device_ids.push_back(device_id.value());
    }
    args.device_ids = device_ids.data();
    args.num_device_ids = device_ids.size();

    absl::InlinedVector<const char*, 4> c_clique_ids;
    absl::InlinedVector<size_t, 4> c_clique_id_sizes;
    if (clique_ids.has_value()) {
      c_clique_ids.reserve(clique_ids->size());
      for (const auto& clique_id : clique_ids->data()) {
        c_clique_ids.push_back(clique_id.data().data());
        c_clique_id_sizes.push_back(clique_id.size());
      }
      args.clique_ids = c_clique_ids.data();
      args.clique_id_sizes = c_clique_id_sizes.data();
      args.num_clique_ids = c_clique_ids.size();
    } else {
      args.clique_ids = nullptr;
      args.clique_id_sizes = nullptr;
      args.num_clique_ids = 0;
    }

    absl::InlinedVector<int64_t, 4> rank_ids;
    rank_ids.reserve(ranks.size());
    for (const auto& rank : ranks) {
      rank_ids.push_back(rank.rank.value());
    }
    args.rank_ids = rank_ids.data();
    args.num_device_ranks = rank_ids.size();

    RETURN_STATUS_IF_PJRT_ERROR(
        extension_->collectives_create_communicators(&args), c_api_);

    std::vector<std::unique_ptr<Communicator>> communicators;
    communicators.reserve(args.num_communicators);
    for (size_t i = 0; i < args.num_communicators; ++i) {
      communicators.push_back(std::make_unique<MegascaleCApiCommunicator>(
          args.communicators[i], c_api_, extension_));
    }
    // Delete the array of pointers, but not the pointers themselves.
    args.communicators_holder_deleter(args.communicators_holder);

    return communicators;
  }

 private:
  PJRT_Collectives* collectives_;
  const PJRT_Api* c_api_;
  const PJRT_Collectives_Extension* extension_;
};

absl::StatusOr<PJRT_Megascale_Extension*> GetMegascaleExtension(
    const PJRT_Api* c_api) {
  PJRT_Megascale_Extension* extension =
      pjrt::FindExtension<PJRT_Megascale_Extension>(
          c_api, PJRT_Extension_Type_Megascale);
  if (extension == nullptr) {
    return absl::InternalError("Megascale extension is not available.");
  }
  return extension;
}

absl::StatusOr<PJRT_Collectives_Extension*> GetCollectivesExtension(
    const PJRT_Api* c_api) {
  PJRT_Collectives_Extension* extension =
      pjrt::FindExtension<PJRT_Collectives_Extension>(
          c_api, PJRT_Extension_Type_Collectives);
  if (extension == nullptr) {
    return absl::InternalError("Collectives extension is not available.");
  }
  return extension;
}

absl::StatusOr<PJRT_MultiSlice_Extension*> GetMultiSliceExtension(
    const PJRT_Api* c_api) {
  PJRT_MultiSlice_Extension* extension =
      pjrt::FindExtension<PJRT_MultiSlice_Extension>(
          c_api, PJRT_Extension_Type_MultiSlice);
  if (extension == nullptr) {
    return absl::InternalError("MultiSlice extension is not available.");
  }
  return extension;
}

}  // namespace

absl::StatusOr<std::unique_ptr<xla::MultiSliceConfig>> CreateAoTMegascaleConfig(
    const xla::PjRtTopologyDescription& topology_description, int num_slices) {
  TF_ASSIGN_OR_RETURN(const PJRT_Api* c_api, pjrt::PjrtApi(kTpuPjrtName));
  TF_ASSIGN_OR_RETURN(PJRT_Megascale_Extension * extension,
                      GetMegascaleExtension(c_api));
  TF_ASSIGN_OR_RETURN(PJRT_MultiSlice_Extension * multi_slice_extension,
                      GetMultiSliceExtension(c_api));

  PJRT_Megascale_CreateAoTConfig_Args args;
  args.struct_size = PJRT_Megascale_CreateAoTConfig_Args_STRUCT_SIZE;
  args.topology = tsl::down_cast<const xla::PjRtCApiTopologyDescription&>(
                      topology_description)
                      .c_topology();
  args.num_slices = num_slices;
  args.multi_slice_config = nullptr;

  RETURN_STATUS_IF_PJRT_ERROR(extension->create_aot_config(&args), c_api);

  return std::make_unique<pjrt::PjRtCApiMultiSliceConfig>(
      args.multi_slice_config, c_api, multi_slice_extension);
}

absl::StatusOr<std::unique_ptr<const xla::MultiSliceConfig>>
CreateMultiSliceMegascaleConfig(
    const xla::PjRtTopologyDescription& topology_description, int num_slices,
    int32_t local_slice_id, int32_t local_host_id,
    const xla::megascale::runtime::EndpointAddresses& endpoint_addresses,
    const xla::megascale::runtime::DCNTopology& dcn_topology,
    std::shared_ptr<CApiPjRtClientContext> megascale_client_ctx) {
  const PJRT_Api* c_api = megascale_client_ctx->c_api();
  const PJRT_Megascale_Extension* extension = megascale_client_ctx->extension();
  TF_ASSIGN_OR_RETURN(PJRT_MultiSlice_Extension * multi_slice_extension,
                      GetMultiSliceExtension(c_api));

  std::string endpoint_addresses_str = endpoint_addresses.SerializeAsString();
  std::string dcn_topology_str = dcn_topology.SerializeAsString();

  PJRT_Megascale_CreateMultiSliceConfig_Args args;
  args.struct_size = PJRT_Megascale_CreateMultiSliceConfig_Args_STRUCT_SIZE;
  args.topology = tsl::down_cast<const xla::PjRtCApiTopologyDescription&>(
                      topology_description)
                      .c_topology();
  args.num_slices = num_slices;
  args.local_slice_id = local_slice_id;
  args.local_host_id = local_host_id;
  args.endpoint_addresses = endpoint_addresses_str.data();
  args.endpoint_addresses_size = endpoint_addresses_str.size();
  args.dcn_topology = dcn_topology_str.data();
  args.dcn_topology_size = dcn_topology_str.size();
  args.client_context = megascale_client_ctx->get();
  args.multi_slice_config = nullptr;

  RETURN_STATUS_IF_PJRT_ERROR(extension->create_multi_slice_config(&args),
                              c_api);

  CHECK(args.multi_slice_config != nullptr);
  return std::make_unique<pjrt::PjRtCApiMultiSliceConfig>(
      args.multi_slice_config, c_api, multi_slice_extension);
}

absl::StatusOr<std::shared_ptr<CApiPjRtClientContext>>
MegaScaleClientContextFromClient(xla::PjRtClient* client) {
  xla::PjRtCApiClient* c_api_client =
      tsl::down_cast<xla::PjRtCApiClient*>(client);
  const PJRT_Api* c_api = c_api_client->pjrt_c_api();
  const PJRT_Megascale_Extension* extension =
      c_api_client->FindExtension<PJRT_Megascale_Extension>(
          PJRT_Extension_Type_Megascale);
  PJRT_Megascale_CreateClientContextFromPjRtClient_Args args;
  args.struct_size =
      PJRT_Megascale_CreateClientContextFromPjRtClient_Args_STRUCT_SIZE;

  args.client = c_api_client->pjrt_c_client();
  args.client_context = nullptr;

  RETURN_STATUS_IF_PJRT_ERROR(
      extension->create_client_context_from_pjrt_client(&args), c_api);

  CHECK(args.client_context != nullptr);
  return std::make_shared<CApiPjRtClientContext>(args.client_context, c_api,
                                                 extension);
}

absl::StatusOr<std::shared_ptr<CApiPjRtClientContext>>
CreateDefaultMegaScaleClientContext() {
  TF_ASSIGN_OR_RETURN(const PJRT_Api* c_api, pjrt::PjrtApi(kTpuPjrtName));
  TF_ASSIGN_OR_RETURN(PJRT_Megascale_Extension * extension,
                      GetMegascaleExtension(c_api));

  PJRT_Megascale_CreateDefaultClientContext_Args args;
  args.struct_size = PJRT_Megascale_CreateDefaultClientContext_Args_STRUCT_SIZE;
  args.client_context = nullptr;

  RETURN_STATUS_IF_PJRT_ERROR(extension->create_default_client_context(&args),
                              c_api);

  return std::make_shared<CApiPjRtClientContext>(args.client_context, c_api,
                                                 extension);
}

absl::StatusOr<std::unique_ptr<xla::cpu::CpuCollectives>>
CreateMegascaleCollectives(
    const CApiPjRtClientContext& megascale_client_ctx,
    ProcessesInfo&& processes_info,
    std::optional<xla::megascale::runtime::DCNTopology>&& dcn_topology) {
  const PJRT_Api* c_api = megascale_client_ctx.c_api();
  const PJRT_Megascale_Extension* extension = megascale_client_ctx.extension();
  TF_ASSIGN_OR_RETURN(PJRT_Collectives_Extension * collectives_extension,
                      GetCollectivesExtension(c_api));

  std::vector<const char*> addresses_ptrs;
  std::vector<size_t> address_sizes;
  for (const auto& addr : processes_info.addresses) {
    addresses_ptrs.push_back(addr.c_str());
    address_sizes.push_back(addr.size());
  }

  PJRT_Megascale_ProcessesInfo c_processes_info;
  c_processes_info.addresses = addresses_ptrs.data();
  c_processes_info.address_sizes = address_sizes.data();
  c_processes_info.num_addresses = addresses_ptrs.size();

  if (processes_info.slice_indexes) {
    c_processes_info.slice_indexes = processes_info.slice_indexes->data();
    c_processes_info.num_slice_indexes = processes_info.slice_indexes->size();
  } else {
    c_processes_info.slice_indexes = nullptr;
    c_processes_info.num_slice_indexes = 0;
  }

  if (processes_info.per_slice_indexes) {
    c_processes_info.per_slice_indexes =
        processes_info.per_slice_indexes->data();
    c_processes_info.num_per_slice_indexes =
        processes_info.per_slice_indexes->size();
  } else {
    c_processes_info.per_slice_indexes = nullptr;
    c_processes_info.num_per_slice_indexes = 0;
  }
  c_processes_info.num_devices_per_process =
      processes_info.num_devices_per_process;

  PJRT_Megascale_CreateMegascaleCollectives_Args args;
  args.struct_size = PJRT_Megascale_CreateMegascaleCollectives_Args_STRUCT_SIZE;
  args.client_context = megascale_client_ctx.get();
  args.processes_info = &c_processes_info;

  std::string dcn_topology_str;
  if (dcn_topology) {
    dcn_topology_str = dcn_topology->SerializeAsString();
    args.dcn_topology = dcn_topology_str.data();
    args.dcn_topology_size = dcn_topology_str.size();
  } else {
    args.dcn_topology = nullptr;
    args.dcn_topology_size = 0;
  }

  RETURN_STATUS_IF_PJRT_ERROR(extension->create_megascale_collectives(&args),
                              c_api);

  return std::make_unique<MegascaleCApiCollectives>(args.collectives, c_api,
                                                    collectives_extension);
}

}  // namespace c_api_client
}  // namespace megascale
}  // namespace xla
