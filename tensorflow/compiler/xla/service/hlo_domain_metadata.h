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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DOMAIN_METADATA_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DOMAIN_METADATA_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

// Cannot include hlo_instruction.h as this file is included from there.
class HloInstruction;

// The DomainMetadata represents the base class for metadata which can be
// attached to kDomain HLO instructions.
class DomainMetadata {
 public:
  // A Domain data structure captures all the information about a kDomain
  // bounded instruction set.
  struct Domain {
    // The set of instructions which are reachable from each other via
    // operand/user pathways, without crossing a kDomain instruction of a given
    // kind. The reach_set can contain kDomain instructions of other kinds, if
    // two domains of different kind intersect each other.
    absl::flat_hash_set<HloInstruction*> reach_set;

    // The same instructions in reach_set, but purged from kDomain instructions
    // and ordered according to their computation graph post-order, i.e.
    // if instructions[pos_a] depends on instructions[pos_b], then pos_a >
    // pos_b.
    std::vector<HloInstruction*> instructions;

    // If we consider a graph edge as an arrow oriented from the operand to the
    // user, the enter_domains will contain the set of kDomain instructions
    // whose dataflow enters the reach set (domain), while the exit_domains
    // contains the set of kDomain instructions whose dataflow exit the reach
    // set.
    absl::flat_hash_set<HloInstruction*> enter_domains;
    absl::flat_hash_set<HloInstruction*> exit_domains;
  };

  virtual ~DomainMetadata() = default;

  // Clones the metadata object.
  virtual std::unique_ptr<DomainMetadata> Clone() const = 0;

  // Returns the metadata type. A unique identifier which describes the real
  // metadata type.
  virtual absl::string_view Kind() const = 0;

  // Compares the metadata object with another one and returns true if the
  // two matches.
  virtual bool Matches(const DomainMetadata& other) const = 0;

  // Returns the hash value of the metadata.
  virtual size_t Hash() const = 0;

  // Returns a string representation of the metadata.
  virtual std::string ToString() const = 0;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DOMAIN_METADATA_H_
