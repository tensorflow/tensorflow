/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_ALL_REDUCE_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_ALL_REDUCE_THUNK_H_

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace gpu {

// Thunk that performs a NCCL-based All-Reduce among CUDA GPU-based replicas.
class NcclAllReduceThunk : public Thunk {
 public:
  // Returns whether NCCL operations appear possible to perform; e.g. if we
  // haven't done a build with the CUDA compiler enabled, we can't compile the
  // NCCL header, and thus this will be false.
  //
  // When this is false, the ExecuteOnStream() call will simply return a status
  // error.
  static bool NcclIsEnabled();

  // Gets the set of devices that have a NCCL channel open.  This is primarily
  // for testing.
  //
  // (Indeed, because the NCCL channels are a global variable, in the real
  // world, the value returned here is stale as soon as you read it, so it's not
  // clear how you *could* use it for anything other than tests.)
  static absl::flat_hash_set<GlobalDeviceId> DevicesWithOpenNcclChannels();

  // TODO(b/125951860): Support all-reduces with replica groups, i.e.
  // all-reduces that compute multiple sums across subsets of all replicas.
  struct Buffer {
    int64 element_count;
    BufferAllocation::Slice source_buffer;
    BufferAllocation::Slice destination_buffer;
  };
  NcclAllReduceThunk(int64 replica_count, std::vector<Buffer> buffers,
                     const HloInstruction* all_reduce);
  ~NcclAllReduceThunk() override;

  Status ExecuteOnStream(const ExecuteParams& params) override;

  // Returns whether the given instruction can be lowered to a nccl all-reduce
  // call.
  static bool CanImplement(const HloInstruction* crs);

 private:
  // Extra data stored in NcclAllReduceThunk whose types we don't want exposed
  // in the header file.  (This is mainly because the implementation of
  // NcclAllReduceThunk is different depending on whether CUDA is enabled in the
  // build, and we don't want to expose *that* mess in the header.)
  struct AuxData;

  const int64 replica_count_;
  const std::vector<Buffer> buffers_;
  std::unique_ptr<AuxData> aux_data_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_ALL_REDUCE_THUNK_H_
