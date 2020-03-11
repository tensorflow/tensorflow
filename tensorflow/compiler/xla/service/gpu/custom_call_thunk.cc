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

#include "tensorflow/compiler/xla/service/gpu/custom_call_thunk.h"

#include "absl/strings/str_format.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"

namespace xla {
namespace gpu {

CustomCallThunk::CustomCallThunk(
    void* call_target,
    std::vector<ShapeTree<BufferAllocation::Slice>> operand_slices,
    ShapeTree<BufferAllocation::Slice> result_slices, std::string opaque,
    const HloInstruction* instr)
    : Thunk(Thunk::kCustomCall, instr),
      call_target_(call_target),
      operand_slices_(std::move(operand_slices)),
      result_slices_(std::move(result_slices)),
      opaque_(std::move(opaque)) {
  CHECK_EQ(instr->operand_count(), operand_slices_.size());
  for (int64 i = 0; i < instr->operand_count(); ++i) {
    const auto& s1 = operand_slices_[i].shape();
    const auto& s2 = instr->operand(i)->shape();
    CHECK(ShapeUtil::Equal(s1, s2)) << absl::StreamFormat(
        "Shape mismatch between instr->operand(%d) and "
        "operand_slices[%d].shape(): %s vs %s",
        i, i, s1.ToString(), s2.ToString());
  }
  CHECK(ShapeUtil::Equal(instr->shape(), result_slices.shape()))
      << absl::StreamFormat(
             "Shape mismatch between instr->shape() and result_slices.shape(): "
             "%s vs %s.",
             instr->shape().ToString(), result_slices.shape().ToString());
}

// For each leaf in a preorder traversal of `slices`, appends its device address
// to `buffers`.
//
// In the common case, this is trivial; simply iterate over the ShapeTree and
// add every leaf to `buffers`.  But under some circumstances XLA doesn't
// statically know the address of a leaf buffer and has to derive it by walking
// the on-device tuple.
static Status AppendBuffersFor(const ShapeTree<BufferAllocation::Slice>& slices,
                               const BufferAllocations* buffer_allocations,
                               se::Stream* stream,
                               std::vector<void*>* buffers) {
  // Buffer addresses we've retrieved by following device tuples.
  ShapeTree<void*> retrieved_addrs(slices.shape());

  // We make this lambda an std::function so it can capture itself.
  std::function<StatusOr<void*>(const ShapeIndexView&)> get_addr_for =
      [&](ShapeIndexView index) -> StatusOr<void*> {
    auto slice = slices.element(index);

    // If we know the address of this sub-buffer statically, return it.
    if (slice.allocation() != nullptr) {
      return buffer_allocations->GetDeviceAddress(slice).opaque();
    }
    // If we've already pulled the address for this sub-buffer down from the
    // GPU, return it.
    if (retrieved_addrs.element(index) != nullptr) {
      return retrieved_addrs.element(index);
    }

    // Recurse to get the address of the parent sub-buffer.
    CHECK(!index.empty()) << "Address of tuple root cannot be unknown!";
    TF_ASSIGN_OR_RETURN(void* parent_buffer, get_addr_for(index.ConsumeBack()));

    // Pull down the entirety of parent_buffer from the GPU, getting the address
    // we're interested in plus all of its siblings.  (Perhaps only some of the
    // siblings are unknown and we could get away without retrieving all of
    // them.  But in practice, getting them all in one fell swoop should be just
    // as fast as getting just one.)
    //
    // TODO(jlebar): This is not as efficient as possible.  In particular, at
    // the expense of some complexity we could batch up multiple parallel D2H
    // copies (say for multiple unrelated sub-buffers, maybe even across
    // different parameters) and do just one BlockHostUntilDone.  Hopefully the
    // case when we have to do any copies at all is uncommon.
    int64 num_siblings =
        ShapeUtil::GetSubshape(slices.shape(), index.ConsumeBack())
            .tuple_shapes_size();
    std::vector<void*> sibling_addrs(num_siblings);
    TF_RETURN_IF_ERROR(
        stream
            ->ThenMemcpy(sibling_addrs.data(),
                         se::DeviceMemoryBase(parent_buffer, sizeof(void*)),
                         num_siblings * sizeof(void*))
            .BlockHostUntilDone());

    // Save the data we retrieved into retrieved_addrs.
    for (int64 i = 0; i < num_siblings; ++i) {
      ShapeIndex sibling_index(index.ConsumeBack());
      sibling_index.push_back(i);
      *retrieved_addrs.mutable_element(sibling_index) = sibling_addrs[i];
    }
    return sibling_addrs[index.back()];
  };

  return slices.ForEachElementWithStatus(
      [&](const ShapeIndex& index, const BufferAllocation::Slice&) {
        if (slices.IsLeaf(index)) {
          TF_ASSIGN_OR_RETURN(void* addr, get_addr_for(index));
          buffers->push_back(addr);
        }
        return Status::OK();
      });
}

Status CustomCallThunk::ExecuteOnStream(const ExecuteParams& params) {
  // gpu_stream is CUstream or e.g. the equivalent type in ROCm.
  se::Stream* stream = params.stream;
  auto gpu_stream = se::gpu::AsGpuStreamValue(params.stream);
  auto typed_call_target =
      reinterpret_cast<void (*)(decltype(gpu_stream), void** /*buffers*/,
                                const char* /*opaque*/, size_t /*opaque_len*/)>(
          call_target_);

  std::vector<void*> buffers;
  for (const auto& slices : operand_slices_) {
    TF_RETURN_IF_ERROR(
        AppendBuffersFor(slices, params.buffer_allocations, stream, &buffers));
  }
  TF_RETURN_IF_ERROR(AppendBuffersFor(result_slices_, params.buffer_allocations,
                                      stream, &buffers));

  typed_call_target(gpu_stream, buffers.data(), opaque_.data(), opaque_.size());

  // If the custom-call returns a tuple, populate the result tuple index
  // buffers.
  return result_slices_.ForEachElementWithStatus(
      [&](const ShapeIndex& index, const BufferAllocation::Slice& slice) {
        const Shape& subshape =
            ShapeUtil::GetSubshape(result_slices_.shape(), index);
        auto n = subshape.tuple_shapes_size();
        if (!subshape.IsTuple() || n == 0) {
          return Status::OK();
        }
        auto tuple_ptrs = absl::make_unique<void*[]>(n);
        ShapeIndex subindex(index);
        for (int i = 0; i < n; ++i) {
          subindex.push_back(i);
          tuple_ptrs[i] =
              params.buffer_allocations
                  ->GetDeviceAddress(result_slices_.element(subindex))
                  .opaque();
          subindex.pop_back();
        }
        SafeH2DMemcpy(se::DeviceMemory<void*>(
                          params.buffer_allocations->GetDeviceAddress(slice)),
                      std::move(tuple_ptrs), n, stream,
                      params.deferred_host_callbacks);
        return Status::OK();
      });
}

}  // namespace gpu
}  // namespace xla
