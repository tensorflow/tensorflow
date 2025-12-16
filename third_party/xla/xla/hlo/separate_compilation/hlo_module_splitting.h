/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_HLO_SEPARATE_COMPILATION_HLO_MODULE_SPLITTING_H_
#define XLA_HLO_SEPARATE_COMPILATION_HLO_MODULE_SPLITTING_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/separate_compilation/hlo_linking_manifest.h"

namespace xla::separate_compilation {

// Returns a list of sets of computations that can be split into separate
// modules. Adjacent computations in the same set can be compiled together.
absl::StatusOr<std::vector<absl::flat_hash_set<const HloComputation*>>>
GroupComputationsForSplitting(const HloModule& module);

// Represents one part of a module that was split into multiple parts for
// separate compilation.
struct HloModuleSplit {
  using StubComputation = HloComputation;
  using OriginalComputation = HloComputation;
  using ClonedComputation = HloComputation;

  // The original `HloModule` that this split originated from.
  const HloModule& module;
  // An `HloModule` containing computations belonging to this split. If any
  // computation in `submodule` calls a computation which is part of another
  // split, that call is replaced with a call to a stub computation.
  std::unique_ptr<HloModule> submodule;
  // Maps stub computations defined in `submodule` to the original computations
  // in `module` which they replace.
  absl::flat_hash_map<const StubComputation*, const OriginalComputation*>
      stub_map;
  // Maps computations from `module` to their cloned versions in `submodule`.
  absl::flat_hash_map<const OriginalComputation*, const ClonedComputation*>
      computation_map;
  // All `kCall` instructions in `submodule` which originally belonged to
  // computations cloned into this split. This includes calls to stubbed out
  // computations, and calls to computations within this split.
  std::vector<const HloInstruction*> call_sites;

  HloModuleSplit(
      const HloModule& module, std::unique_ptr<HloModule> submodule,
      absl::flat_hash_map<const StubComputation*, const OriginalComputation*>
          stub_map,
      absl::flat_hash_map<const OriginalComputation*, const ClonedComputation*>
          computation_map,
      std::vector<const HloInstruction*> call_sites)
      : module{std::move(module)},
        submodule{std::move(submodule)},
        stub_map{std::move(stub_map)},
        computation_map{std::move(computation_map)},
        call_sites{std::move(call_sites)} {}
};

// Creates an `HloModule` that only contains the requested computations
// from the original module and, potentially, insert callee stubs.
absl::StatusOr<std::unique_ptr<HloModuleSplit>> CreateHloModuleSplit(
    const HloModule& module, absl::flat_hash_set<const HloComputation*> split);

// Represents a group of `HloModuleSplit`s.
struct HloModuleSplitGroup {
  absl::flat_hash_map<const HloComputation*, const HloModuleSplit*>
      address_book;
  std::vector<std::unique_ptr<HloModuleSplit>> module_splits;
  HloLinkingManifest linking_manifest;

  HloModuleSplitGroup(
      absl::flat_hash_map<const HloComputation*, const HloModuleSplit*>&&
          address_book,
      std::vector<std::unique_ptr<HloModuleSplit>>&& module_splits,
      HloLinkingManifest&& linking_manifest)
      : address_book(std::move(address_book)),
        module_splits(std::move(module_splits)),
        linking_manifest(std::move(linking_manifest)) {}

  // Returns the cloned version of the given original computation, or
  // an error if the computation is not part of this split group.
  absl::StatusOr<const HloComputation*> GetClonedComputation(
      const HloComputation* original_computation) const;
};

// Split the given module. Returns a mapping from `HloComputation*` to
// the `ModulePartition` data where that computation was assigned. If multiple
// computations are assigned to the same module there are multiple keys pointing
// to the same `ModulePartition` structure.
absl::StatusOr<std::unique_ptr<HloModuleSplitGroup>> CreateHloModuleSplitGroup(
    const HloModule& module);

}  // namespace xla::separate_compilation
#endif  // XLA_HLO_SEPARATE_COMPILATION_HLO_MODULE_SPLITTING_H_
