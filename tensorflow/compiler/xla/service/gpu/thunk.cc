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

namespace xla {
namespace gpu {

Thunk::ExecuteParams::ExecuteParams(
    const ServiceExecutableRunOptions& run_options,
    const BufferAllocations& buffer_allocations, se::Stream* stream,
    se::Stream* async_comms_stream)
    : buffer_allocations(&buffer_allocations),
      stream(stream),
      async_comms_stream(async_comms_stream),
      nccl_params(run_options, stream) {}

/*static*/ absl::string_view Thunk::KindToString(Thunk::Kind kind) {
  switch (kind) {
    case Thunk::kCholesky:
      return "kCholesky";
    case Thunk::kCollectivePermute:
      return "kCollectivePermute";
    case Thunk::kConditional:
      return "kConditional";
    case Thunk::kConvolution:
      return "kConvolution";
    case Thunk::kCopy:
      return "kCopy";
    case Thunk::kCublasLtMatmul:
      return "kCublasLtMatmul";
    case Thunk::kCustomCall:
      return "kCustomCall";
    case Thunk::kNcclAllGather:
      return "kNcclAllGather";
    case Thunk::kNcclAllReduce:
      return "kNcclAllReduce";
    case Thunk::kNcclAllReduceStart:
      return "kNcclAllReduceStart";
    case Thunk::kNcclAllReduceDone:
      return "kNcclAllReduceDone";
    case Thunk::kNcclReduceScatter:
      return "kNcclReduceScatter";
    case Thunk::kNcclAllToAll:
      return "kNcclAllToAll";
    case Thunk::kFft:
      return "kFft";
    case Thunk::kFor:
      return "kFor";
    case Thunk::kGemm:
      return "kGemm";
    case Thunk::kInfeed:
      return "kInfeed";
    case Thunk::kKernel:
      return "kKernel";
    case Thunk::kMemset32BitValue:
      return "kMemset32BitValue";
    case Thunk::kMemzero:
      return "kMemzero";
    case Thunk::kOutfeed:
      return "kOutfeed";
    case Thunk::kReplicaId:
      return "kReplicaId";
    case Thunk::kPartitionId:
      return "kPartitionId";
    case Thunk::kSequential:
      return "kSequential";
    case Thunk::kTriangularSolve:
      return "kTriangularSolve";
    case Thunk::kWhile:
      return "kWhile";
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

}  // namespace gpu
}  // namespace xla
