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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DOMAIN_REMOVER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DOMAIN_REMOVER_H_

#include "tensorflow/compiler/xla/service/hlo_domain_metadata.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

// Removes all the kDomain instructions of a given kind from the input module,
// and calls the normalizer to propagate the properties on the possibly new born
// instructions.
class HloDomainRemover : public HloPassInterface {
 public:
  // Creates a new HloDomainRemover object tasked at removing all the kDomain
  // instructions of a given kind.
  // In case a reachable set (the set of instructions within a computation,
  // which are mutually reachable via operand/user pathways) has all the
  // instructions in it with the same attributes (ie, sharding), a normalizer
  // function is tasked at applying attribute normalization on the instructions
  // within such domain.
  HloDomainRemover(
      tensorflow::StringPiece kind,
      std::function<Status(const DomainMetadata::Domain&)> normalizer)
      : kind_(kind.ToString()), normalizer_(std::move(normalizer)) {}

  tensorflow::StringPiece name() const override { return "domain_remover"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  class RunContext;

  string kind_;
  std::function<Status(const DomainMetadata::Domain&)> normalizer_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DOMAIN_REMOVER_H_
