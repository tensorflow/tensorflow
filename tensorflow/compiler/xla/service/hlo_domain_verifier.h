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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DOMAIN_VERIFIER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DOMAIN_VERIFIER_H_

#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_domain_map.h"
#include "tensorflow/compiler/xla/service/hlo_domain_metadata.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

// Verifies that the domain instructions are consistent, and the each domain is
// surrounded by the same metadata.
class HloDomainVerifier : public HloPassInterface {
 public:
  HloDomainVerifier(std::vector<string> kinds) : kinds_(std::move(kinds)) {}

  absl::string_view name() const override { return "domain_verifier"; }

  StatusOr<bool> Run(HloModule* module) override;

  // Verify that the whole kDomain frontier bounding the instruction reach set,
  // has matching metadata.
  // A kDomain instruction has two sides of metadata, a user facing and an
  // operand facing.
  // A reachable instruction set can make contact with a kDomain instruction on
  // a user facing side (the kDomain is operand of the instruction), or on a
  // operand facing side (the kDomain is user of the instruction).
  // And depending on the contact side, the proper metadata object
  // (user_side_metadata() vs. operand_side_metadata()) needs to be used for
  // consistency checks.
  // Returns the DomainMetadata pointer which surrounds the domain, and
  // represents the common metadata within such domain. If the returned
  // DomainMetadata pointer is nullptr, the input domain had no kDomain
  // boundary.
  static StatusOr<const DomainMetadata*> VerifyDomain(
      const DomainMetadata::Domain& domain);

 private:
  class RunContext;

  std::vector<string> kinds_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DOMAIN_VERIFIER_H_
