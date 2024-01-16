/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/nccl_api.h"

#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "third_party/nccl/nccl.h"
#include "xla/service/gpu/nccl_clique_key.h"
#include "tsl/platform/logging.h"

namespace xla::gpu {

//==-----------------------------------------------------------------------===//
// Macros to return or warn on NCCL errors.
//==-----------------------------------------------------------------------===//

static absl::Status ToStatus(ncclResult_t s, const char* file, int64_t line,
                             const char* expr) {
  if (s == ncclSuccess) return absl::OkStatus();

  return absl::InternalError(absl::StrFormat(
      "%s:%d: NCCL operation %s failed: %s."
      " Last NCCL warning(error) log entry (may be unrelated) '%s'.",
      file, line, expr, ncclGetErrorString(s), ncclGetLastError(nullptr)));
}

#define XLA_NCCL_STATUS(expr) \
  xla::gpu::ToStatus(expr, __FILE__, __LINE__, #expr)

#define XLA_NCCL_RETURN_IF_ERROR(expr)      \
  do {                                      \
    absl::Status s = XLA_NCCL_STATUS(expr); \
    if (!s.ok()) {                          \
      return s;                             \
    }                                       \
  } while (0)

#define XLA_NCCL_LOG_IF_ERROR(expr)         \
  do {                                      \
    absl::Status s = XLA_NCCL_STATUS(expr); \
    if (!s.ok()) {                          \
      LOG(ERROR) << s.ToString();           \
    }                                       \
  } while (0)

//==-----------------------------------------------------------------------===//
// NcclApi
//==-----------------------------------------------------------------------===//

static_assert(NCCL_UNIQUE_ID_BYTES == NcclCliqueId::kSize,
              "size of nccl unique id must match the clique id size");

static NcclCommHandle Cast(ncclComm_t comm) {
  return reinterpret_cast<NcclCommHandle>(comm);
}

static ncclComm_t Cast(NcclCommHandle comm) {
  return reinterpret_cast<ncclComm_t>(comm);
}

static ncclUniqueId AsNcclUniqueId(const NcclCliqueId& clique_id) {
  ncclUniqueId id;
  absl::c_copy(clique_id.data(), id.internal);
  return id;
}

absl::StatusOr<NcclCliqueId> NcclApi::GetUniqueId() {
  ncclUniqueId id;
  XLA_NCCL_RETURN_IF_ERROR(ncclGetUniqueId(&id));
  return NcclCliqueId(id.internal);
}

absl::StatusOr<NcclCommHandle> NcclApi::CommInitRank(
    int32_t nranks, const NcclCliqueId& clique_id, int32_t rank) {
  VLOG(3) << "Initialize NCCL communicator for rank #" << rank << " of "
          << nranks << "; hash(id)=" << absl::HashOf(clique_id.data());

  if (rank < 0 || rank >= nranks)
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid rank %d, it must be in [0, %d) range", rank, nranks));

  ncclComm_t comm = nullptr;
  absl::Status status = XLA_NCCL_STATUS(
      ncclCommInitRank(&comm, nranks, AsNcclUniqueId(clique_id), rank));

  return Cast(comm);
}

absl::Status NcclApi::CommAbort(NcclCommHandle comm) {
  VLOG(3) << "Abort NCCL communicator: " << comm;
  return XLA_NCCL_STATUS(ncclCommAbort(Cast(comm)));
}

absl::Status NcclApi::CommGetAsyncError(NcclCommHandle comm) {
  VLOG(3) << "Get last async error for NCCL communicator: " << comm;

  ncclResult_t async_err;
  XLA_NCCL_RETURN_IF_ERROR(ncclCommGetAsyncError(Cast(comm), &async_err));
  if (async_err == ncclSuccess) return absl::OkStatus();

  return absl::InternalError(absl::StrCat(
      ncclGetErrorString(async_err),
      ". Last NCCL error (maybe unrelated): ", ncclGetLastError(Cast(comm))));
}

}  // namespace xla::gpu
