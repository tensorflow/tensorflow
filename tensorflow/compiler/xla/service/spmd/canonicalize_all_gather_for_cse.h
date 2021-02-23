/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_CANONICALIZE_ALL_GATHER_FOR_CSE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_CANONICALIZE_ALL_GATHER_FOR_CSE_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// Performs canonicalizations on AllGather for CSE.
class CanonicalizeAllGatherForCSE : public HloModulePass {
 public:
  CanonicalizeAllGatherForCSE() : next_channel_id_(0) {}

  ~CanonicalizeAllGatherForCSE() override = default;
  absl::string_view name() const override { return "canon-all-gather-for-cse"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  StatusOr<bool> RunOnComputation(HloComputation* comp);
  int64 NextChannelId() { return next_channel_id_++; }

  int64 next_channel_id_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_CANONICALIZE_ALL_GATHER_FOR_CSE_H_
