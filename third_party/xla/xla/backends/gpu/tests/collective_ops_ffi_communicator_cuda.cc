/* Copyright 2025 The OpenXLA Authors.

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

#include <cstdint>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "third_party/nccl/nccl.h"
#include "xla/ffi/api/collectives_c_api.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

absl::Status CommunicatorAllReduceU32(stream_executor::Stream* stream,
                                      XLA_FFI_Communicator* communicator,
                                      const void* send_buffer,
                                      void* recv_buffer, int64_t count) {
  ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(communicator);
  cudaStream_t cuda_stream =
      absl::bit_cast<cudaStream_t>(stream->platform_specific_handle().stream);

  ncclResult_t result =
      ncclAllReduce(send_buffer, recv_buffer, count, ncclUint32, ncclSum,
                    nccl_comm, cuda_stream);
  TF_RET_CHECK(result == ncclSuccess)
      << "ncclAllReduce failed: " << ncclGetErrorString(result);
  return stream->BlockHostUntilDone();
}

}  // namespace xla::gpu
