/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DOMAIN_ISOLATOR_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DOMAIN_ISOLATOR_H_

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// Domain isolation is the task of placing kDomain instructions between HLO
// instructions having different sharding. A kDomain instruction is essentially
// used to break an HLO graph edge connecting two instructions with different
// sharding. If a set of connected instructions have all the same sharding, no
// kDomain instruction will be placed.
class HloDomainIsolator : public HloModulePass {
 public:
  // Creates a new kDomain instruction for the edge between the use instruction
  // (the first HloInstruction argument), and the operand instruction (the
  // third HloInstruction argument) if the interesting attribute of the
  // instruction differences from the attribute of the root (the second
  // HloInstruction argument).
  // Returns nullptr in case no domain separation is necessary.
  using DomainCreator = std::function<HloInstruction*(
      HloInstruction*, HloInstruction*, HloInstruction*)>;
  using DomainCreatorFactory = std::function<DomainCreator()>;
  explicit HloDomainIsolator(DomainCreatorFactory creator_factory_);

  absl::string_view name() const override { return "domain_isolator"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  DomainCreatorFactory creator_factory_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DOMAIN_ISOLATOR_H_
