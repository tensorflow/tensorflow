/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/pjrt/cpu/gloo_collectives.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "gloo/algorithm.h"
#include "gloo/allgather.h"
#include "gloo/allreduce.h"
#include "gloo/context.h"
#include "gloo/math.h"
#include "gloo/reduce_scatter.h"
#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/prefix_store.h"
#include "gloo/rendezvous/store.h"
#include "gloo/transport/device.h"
#include "gloo/transport/unbound_buffer.h"
#include "gloo/types.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/cpu/collectives_interface.h"
#include "xla/service/global_device_id.h"
#include "xla/status_macros.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla::cpu {

GlooCollectivesCommunicator::GlooCollectivesCommunicator(
    std::shared_ptr<gloo::Context> context)
    : context_(std::move(context)) {}
GlooCollectivesCommunicator::~GlooCollectivesCommunicator() = default;

template <typename T>
static absl::Status SetAllReduceOptions(ReductionKind reduction_kind,
                                        const void* input_buffer,
                                        void* output_buffer,
                                        size_t num_elements,
                                        gloo::AllreduceOptions& options) {
  options.setInput(reinterpret_cast<T*>(const_cast<void*>(input_buffer)),
                   num_elements);
  options.setOutput(reinterpret_cast<T*>(const_cast<void*>(output_buffer)),
                    num_elements);

  using ReductionFn = void (*)(void*, const void*, const void*, size_t);

  switch (reduction_kind) {
    case ReductionKind::SUM:
      options.setReduceFunction(static_cast<ReductionFn>(&gloo::sum<T>));
      break;
    case ReductionKind::PRODUCT:
      options.setReduceFunction(static_cast<ReductionFn>(&gloo::product<T>));
      break;
    case ReductionKind::MIN:
      if constexpr (!is_complex_v<T>) {
        options.setReduceFunction(static_cast<ReductionFn>(&gloo::min<T>));
      } else {
        return absl::InvalidArgumentError(
            "MIN reduction not supported for complex types");
      }
      break;
    case ReductionKind::MAX:
      if constexpr (!is_complex_v<T>) {
        options.setReduceFunction(static_cast<ReductionFn>(&gloo::max<T>));
      } else {
        return absl::InvalidArgumentError(
            "MAX reduction not supported for complex types");
      }
      break;
  }
  return absl::OkStatus();
}

absl::Status GlooCollectivesCommunicator::AllReduce(
    const RendezvousKey& key, ReductionKind reduction_kind,
    PrimitiveType element_type, size_t num_elements, const void* input_buffer,
    void* output_buffer, absl::Duration timeout) {
  gloo::AllreduceOptions options(context_);
  // TODO(phawkins): how to do tags?
  // options.setTag(tag);
  switch (element_type) {
    case S8:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<int8_t>(
          reduction_kind, input_buffer, output_buffer, num_elements, options));
      break;
    case PRED:
    case U8:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<uint8_t>(
          reduction_kind, input_buffer, output_buffer, num_elements, options));
      break;
    case S16:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<int16_t>(
          reduction_kind, input_buffer, output_buffer, num_elements, options));
      break;
    case U16:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<uint16_t>(
          reduction_kind, input_buffer, output_buffer, num_elements, options));
      break;
    case S32:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<int32_t>(
          reduction_kind, input_buffer, output_buffer, num_elements, options));
      break;
    case U32:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<uint32_t>(
          reduction_kind, input_buffer, output_buffer, num_elements, options));
      break;
    case S64:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<int64_t>(
          reduction_kind, input_buffer, output_buffer, num_elements, options));
      break;
    case U64:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<uint64_t>(
          reduction_kind, input_buffer, output_buffer, num_elements, options));
      break;
    case F16:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<gloo::float16>(
          reduction_kind, input_buffer, output_buffer, num_elements, options));
      break;
    case BF16:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<bfloat16>(
          reduction_kind, input_buffer, output_buffer, num_elements, options));
      break;
    case F32:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<float>(
          reduction_kind, input_buffer, output_buffer, num_elements, options));
      break;
    case F64:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<double>(
          reduction_kind, input_buffer, output_buffer, num_elements, options));
      break;
    case C64:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<std::complex<float>>(
          reduction_kind, input_buffer, output_buffer, num_elements, options));
      break;
    case C128:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<std::complex<double>>(
          reduction_kind, input_buffer, output_buffer, num_elements, options));
      break;
    default:
      return absl::InvalidArgumentError("Unknown datatype in allreduce");
  }
  options.setAlgorithm(gloo::AllreduceOptions::Algorithm::RING);
  options.setTimeout(absl::ToChronoMilliseconds(timeout));

  try {
    gloo::allreduce(options);
  } catch (std::exception& e) {
    return absl::UnknownError(
        absl::StrCat("Gloo all-reduce failed: ", e.what()));
  }
  return absl::OkStatus();
}

static constexpr uint8_t kCollectivePermuteSlotPrefix = 0x40;

absl::Status GlooCollectivesCommunicator::CollectivePermute(
    const RendezvousKey& key, size_t num_bytes, std::optional<int> source_rank,
    absl::Span<int const> target_ranks, const void* input_buffer,
    void* output_buffer, absl::Duration timeout) {
  uint32_t tag = 0;  // TODO(phawkins): come up with better tags.
  const auto slot = gloo::Slot::build(kCollectivePermuteSlotPrefix, tag);
  try {
    std::unique_ptr<gloo::transport::UnboundBuffer> in;
    std::unique_ptr<gloo::transport::UnboundBuffer> out;
    for (int target : target_ranks) {
      if (target != context_->rank) {
        VLOG(1) << "send from " << context_->rank << " to " << target;
        if (!in) {
          in = context_->createUnboundBuffer(const_cast<void*>(input_buffer),
                                             num_bytes);
        }
        in->send(target, slot);
      }
    }
    if (source_rank) {
      if (*source_rank == context_->rank) {
        std::memcpy(output_buffer, input_buffer, num_bytes);
      } else {
        VLOG(1) << "recv at " << context_->rank << " from " << *source_rank;
        out = context_->createUnboundBuffer(output_buffer, num_bytes);
        out->recv(*source_rank, slot);
      }
    } else {
      std::memset(output_buffer, 0, num_bytes);
    }
    VLOG(1) << "wait for send at " << context_->rank;
    auto deadline = absl::ToChronoTime(absl::Now() + timeout);
    if (in) {
      in->waitSend(deadline);
    }
    VLOG(1) << "wait for recv at " << context_->rank;
    if (out) {
      out->waitRecv(deadline);
    }
    VLOG(1) << "done waiting at " << context_->rank;
  } catch (std::exception& e) {
    return absl::UnknownError(
        absl::StrCat("Gloo collective permute failed: ", e.what()));
  }
  return absl::OkStatus();
}

absl::Status GlooCollectivesCommunicator::AllToAll(
    const RendezvousKey& key, size_t chunk_bytes,
    absl::Span<const void* const> input_buffers,
    absl::Span<void* const> output_buffers, absl::Duration timeout) {
  // We can't use Gloo's all-to-all implementation directly because it assumes
  // that the inputs and outputs are contiguous. No big deal; it's just built
  // on top of send/recv and we can do the same as it.
  uint32_t tag = 0;  // TODO(phawkins): use better tags.
  int my_rank = context_->rank;
  int world_size = context_->size;

  TF_RET_CHECK(world_size == input_buffers.size());
  TF_RET_CHECK(world_size == output_buffers.size());

  try {
    const auto slot = gloo::Slot::build(gloo::kAlltoallSlotPrefix, tag);
    std::vector<std::unique_ptr<gloo::transport::UnboundBuffer>> ins(
        context_->size);
    std::vector<std::unique_ptr<gloo::transport::UnboundBuffer>> outs(
        context_->size);
    for (size_t i = 0; i < world_size; ++i) {
      if (i != my_rank) {
        ins[i] = context_->createUnboundBuffer(
            const_cast<void*>(input_buffers[i]), chunk_bytes);
        outs[i] = context_->createUnboundBuffer(output_buffers[i], chunk_bytes);
      }
    }

    for (int i = 1; i < world_size; i++) {
      int send_rank = (my_rank + i) % world_size;
      int recv_rank = (my_rank + world_size - i) % world_size;
      ins[send_rank]->send(send_rank, slot);
      outs[recv_rank]->recv(recv_rank, slot);
    }

    std::memcpy(output_buffers[my_rank], input_buffers[my_rank], chunk_bytes);

    auto deadline = absl::ToChronoTime(absl::Now() + timeout);
    for (int i = 0; i < world_size; i++) {
      if (i != my_rank) {
        ins[i]->waitSend(deadline);
        outs[i]->waitRecv(deadline);
      }
    }
  } catch (std::exception& e) {
    return absl::UnknownError(
        absl::StrCat("Gloo all-to-all failed: ", e.what()));
  }
  return absl::OkStatus();
}

absl::Status GlooCollectivesCommunicator::AllGather(const RendezvousKey& key,
                                                    size_t chunk_bytes,
                                                    const void* input_buffer,
                                                    void* output_buffer,
                                                    absl::Duration timeout) {
  uint32_t tag = 0;  // TODO(phawkins): use better tags.

  gloo::AllgatherOptions options(context_);
  options.setTag(tag);
  options.setTimeout(absl::ToChronoMilliseconds(timeout));
  options.setInput(reinterpret_cast<char*>(const_cast<void*>(input_buffer)),
                   chunk_bytes);
  options.setOutput(reinterpret_cast<char*>(output_buffer),
                    chunk_bytes * context_->size);

  try {
    gloo::allgather(options);
  } catch (std::exception& e) {
    return absl::UnknownError(
        absl::StrCat("Gloo AllGather failed: ", e.what()));
  }
  return absl::OkStatus();
}

template <typename T>
absl::Status ReduceScatterHelper(std::shared_ptr<gloo::Context> context,
                                 ReductionKind reduction_kind, void* buffer,
                                 size_t chunk_elems) {
  const gloo::ReductionFunction<T>* reduction_function = nullptr;
  if constexpr (is_complex_v<T>) {
    switch (reduction_kind) {
      case ReductionKind::SUM:
        reduction_function = gloo::ReductionFunction<T>::sum;
        break;
      case ReductionKind::PRODUCT:
        reduction_function = gloo::ReductionFunction<T>::product;
        break;
      default:
        return absl::InvalidArgumentError(absl::StrCat(
            "Unsupported reduction kind: ", static_cast<int>(reduction_kind)));
    }
  } else {
    switch (reduction_kind) {
      case ReductionKind::SUM:
        reduction_function = gloo::ReductionFunction<T>::sum;
        break;
      case ReductionKind::PRODUCT:
        reduction_function = gloo::ReductionFunction<T>::product;
        break;
      case ReductionKind::MAX:
        reduction_function = gloo::ReductionFunction<T>::max;
        break;
      case ReductionKind::MIN:
        reduction_function = gloo::ReductionFunction<T>::min;
        break;
      default:
        return absl::InvalidArgumentError(absl::StrCat(
            "Unsupported reduction kind: ", static_cast<int>(reduction_kind)));
    }
  }
  try {
    std::vector<int> recv_elems(context->size, chunk_elems);
    gloo::ReduceScatterHalvingDoubling<T> algorithm(
        context, std::vector<T*>{reinterpret_cast<T*>(buffer)},
        chunk_elems * context->size, recv_elems, reduction_function);
    algorithm.run();
  } catch (std::exception& e) {
    return absl::UnknownError(
        absl::StrCat("Gloo ReduceScatter failed: ", e.what()));
  }
  return absl::OkStatus();
}

absl::Status GlooCollectivesCommunicator::ReduceScatter(
    const RendezvousKey& key, ReductionKind reduction_kind,
    PrimitiveType element_type, size_t chunk_elems, const void* input_buffer,
    void* output_buffer, absl::Duration timeout) {
  size_t chunk_bytes = chunk_elems * primitive_util::ByteWidth(element_type);
  std::unique_ptr<char[]> temp(new char[chunk_bytes * context_->size]);
  std::memcpy(temp.get(), input_buffer, chunk_bytes * context_->size);
  switch (element_type) {
    case S8:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<int8_t>(context_, reduction_kind,
                                                     temp.get(), chunk_elems));
      break;
    case PRED:
    case U8:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<uint8_t>(context_, reduction_kind,
                                                      temp.get(), chunk_elems));
      break;
    case S16:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<int16_t>(context_, reduction_kind,
                                                      temp.get(), chunk_elems));
      break;
    case U16:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<uint16_t>(
          context_, reduction_kind, temp.get(), chunk_elems));
      break;
    case S32:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<int32_t>(context_, reduction_kind,
                                                      temp.get(), chunk_elems));
      break;
    case U32:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<uint32_t>(
          context_, reduction_kind, temp.get(), chunk_elems));
      break;
    case S64:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<int64_t>(context_, reduction_kind,
                                                      temp.get(), chunk_elems));
      break;
    case U64:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<uint64_t>(
          context_, reduction_kind, temp.get(), chunk_elems));
      break;
    case BF16:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<bfloat16>(
          context_, reduction_kind, temp.get(), chunk_elems));
      break;
    case F16:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<gloo::float16>(
          context_, reduction_kind, temp.get(), chunk_elems));
      break;
    case F32:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<float>(context_, reduction_kind,
                                                    temp.get(), chunk_elems));
      break;
    case F64:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<double>(context_, reduction_kind,
                                                     temp.get(), chunk_elems));
      break;
    case C64:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<std::complex<float>>(
          context_, reduction_kind, temp.get(), chunk_elems));
      break;
    case C128:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<std::complex<double>>(
          context_, reduction_kind, temp.get(), chunk_elems));
      break;
    default:
      return absl::InvalidArgumentError("Unknown datatype in reducescatter");
  }
  std::memcpy(output_buffer, temp.get(), chunk_bytes);
  return absl::OkStatus();
}

GlooCollectives::GlooCollectives(
    std::unique_ptr<gloo::rendezvous::Store> store,
    std::shared_ptr<gloo::transport::Device> device)
    : store_(std::move(store)), device_(std::move(device)) {}

GlooCollectives::~GlooCollectives() = default;

absl::StatusOr<std::shared_ptr<CollectivesCommunicator>>
GlooCollectives::GetCommunicator(
    absl::Span<GlobalDeviceId const> global_devices, int rank) {
  Context* context;
  {
    absl::MutexLock lock(&mu_);
    auto& context_ref = contexts_[std::make_tuple(
        std::vector<GlobalDeviceId>(global_devices.begin(),
                                    global_devices.end()),
        rank)];
    if (!context_ref) {
      context_ref = std::make_unique<Context>();
    }
    context = context_ref.get();
  }
  absl::MutexLock context_lock(&context->mu);
  if (context->communicator) {
    return context->communicator;
  }
  auto gloo_context =
      std::make_shared<gloo::rendezvous::Context>(rank, global_devices.size());
  auto prefix_store = gloo::rendezvous::PrefixStore(
      absl::StrCat("gloo/",
                   absl::StrJoin(global_devices, ",",
                                 [](std::string* out, GlobalDeviceId id) {
                                   absl::StrAppend(out, id.value());
                                 })),
      *store_);
  try {
    gloo_context->connectFullMesh(prefix_store, device_);
  } catch (std::exception& e) {
    return absl::UnknownError(
        absl::StrCat("Gloo context initialization failed: ", e.what()));
  }
  context->communicator =
      std::make_shared<GlooCollectivesCommunicator>(std::move(gloo_context));
  return context->communicator;
}

}  // namespace xla::cpu
