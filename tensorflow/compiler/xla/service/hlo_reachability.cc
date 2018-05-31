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

#include "tensorflow/compiler/xla/service/hlo_reachability.h"

namespace xla {

HloReachabilityMap::HloReachabilityMap(
    const std::list<HloInstruction*>& instructions)
    : size_(instructions.size()) {
  bit_vectors_.reserve(size_);
  for (const HloInstruction* hlo : instructions) {
    indices_[hlo] = bit_vectors_.size();
    bit_vectors_.emplace_back(size_);
  }
  CHECK_EQ(size_, indices_.size());  // instructions should be unique
}

bool HloReachabilityMap::SetReachabilityToUnion(
    tensorflow::gtl::ArraySlice<const HloInstruction*> inputs,
    const HloInstruction* instruction) {
  BitVector& bit_vector = GetBitVector(instruction);
  tmp_bit_vector_ = bit_vector;
  SetReachabilityToUnionHelper(inputs, instruction, &bit_vector);
  return bit_vector != tmp_bit_vector_;
}

void HloReachabilityMap::FastSetReachabilityToUnion(
    tensorflow::gtl::ArraySlice<const HloInstruction*> inputs,
    const HloInstruction* instruction) {
  SetReachabilityToUnionHelper(inputs, instruction, &GetBitVector(instruction));
}

void HloReachabilityMap::SetReachabilityToUnionHelper(
    tensorflow::gtl::ArraySlice<const HloInstruction*> inputs,
    const HloInstruction* instruction, BitVector* bit_vector) {
  // If instruction is part of inputs, don't reset the bit_vector.
  if (std::find(inputs.begin(), inputs.end(), instruction) == inputs.end()) {
    bit_vector->SetToZero();
  }
  bit_vector->Set(GetIndex(instruction));
  for (const HloInstruction* input : inputs) {
    bit_vector->OrWith(GetBitVector(input));
  }
}

void HloReachabilityMap::SetReachable(const HloInstruction* a,
                                      const HloInstruction* b) {
  GetBitVector(b).Set(GetIndex(a));
}

bool HloReachabilityMap::IsReachable(const HloInstruction* a,
                                     const HloInstruction* b) const {
  return GetBitVector(b).Get(GetIndex(a));
}

bool HloReachabilityMap::IsConnected(const HloInstruction* a,
                                     const HloInstruction* b) const {
  return IsReachable(a, b) || IsReachable(b, a);
}

}  // namespace xla
