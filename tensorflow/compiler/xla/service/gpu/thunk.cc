/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/thunk.h"

#include <functional>
#include <memory>
#include <ostream>
#include <string>

namespace xla {
namespace gpu {

Thunk::ExecuteParams::ExecuteParams(
    const ServiceExecutableRunOptions& run_options,
    const BufferAllocations& buffer_allocations, se::Stream* stream,
    se::Stream* async_comms_stream)
    : buffer_allocations(&buffer_allocations),
      stream(stream),
      async_comms_stream(async_comms_stream),
      nccl_params(run_options, stream->parent()) {}

/*static*/ absl::string_view Thunk::KindToString(Thunk::Kind kind) {
#define CASE(x)  \
  case Thunk::x: \
    return #x
  switch (kind) {
    CASE(kCholesky);
    CASE(kConditional);
    CASE(kConvolution);
    CASE(kConvolutionReorder);
    CASE(kCopy);
    CASE(kCublasLtMatmul);
    CASE(kCustomCall);
    CASE(kNcclAllGather);
    CASE(kNcclAllGatherStart);
    CASE(kNcclAllGatherDone);
    CASE(kNcclAllReduce);
    CASE(kNcclAllReduceStart);
    CASE(kNcclAllReduceDone);
    CASE(kNcclCollectivePermute);
    CASE(kNcclCollectivePermuteStart);
    CASE(kNcclCollectivePermuteDone);
    CASE(kNcclReduceScatter);
    CASE(kNcclReduceScatterStart);
    CASE(kNcclReduceScatterDone);
    CASE(kNcclAllToAll);
    CASE(kNcclAllToAllStart);
    CASE(kNcclAllToAllDone);
    CASE(kFft);
    CASE(kFor);
    CASE(kGemm);
    CASE(kInfeed);
    CASE(kKernel);
    CASE(kMemset32BitValue);
    CASE(kMemzero);
    CASE(kOutfeed);
    CASE(kReplicaId);
    CASE(kPartitionId);
    CASE(kSequential);
    CASE(kTriangularSolve);
    CASE(kWhile);
    CASE(kFusedMHA);
  }
}

std::ostream& operator<<(std::ostream& os, Thunk::Kind kind) {
  return os << Thunk::KindToString(kind);
}

std::string ThunkSequence::ToString(
    int indent,
    std::function<std::string(const Thunk*)> get_thunk_annotation) const {
  const std::string indent_str(indent * 2, ' ');
  if (empty()) return indent_str + "No thunks.";

  auto thunk_with_longest_kind = absl::c_max_element(
      *this,
      [](const std::unique_ptr<Thunk>& a, const std::unique_ptr<Thunk>& b) {
        return Thunk::KindToString(a->kind()).length() <
               Thunk::KindToString(b->kind()).length();
      });
  int64_t max_thunk_kind_len =
      Thunk::KindToString(thunk_with_longest_kind->get()->kind()).length();
  std::string result;
  for (const std::unique_ptr<Thunk>& thunk : *this) {
    // Write out the thunk kind, padded out to max_thunk_kind_len.
    absl::string_view kind_str = Thunk::KindToString(thunk->kind());
    absl::StrAppend(&result, indent_str, kind_str,
                    std::string(max_thunk_kind_len - kind_str.length(), ' '),
                    "\t");
    if (get_thunk_annotation) {
      absl::StrAppend(&result, get_thunk_annotation(thunk.get()));
    }
    absl::StrAppend(&result, thunk->ToStringExtra(indent));
    absl::StrAppend(&result, "\n");
  }
  return result;
}

bool IsReductionCollective(Thunk::Kind kind) {
  return kind == Thunk::kNcclAllReduce || kind == Thunk::kNcclAllReduceStart ||
         kind == Thunk::kNcclReduceScatter ||
         kind == Thunk::kNcclReduceScatterStart;
}

}  // namespace gpu
}  // namespace xla
