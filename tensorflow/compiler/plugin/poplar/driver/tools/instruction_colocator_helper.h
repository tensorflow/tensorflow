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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_INSTRUCTION_COLLOCATOR_HELPER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_INSTRUCTION_COLLOCATOR_HELPER_H_

#include "tensorflow/compiler/plugin/poplar/driver/compiler_information.h"

#include "tensorflow/compiler/xla/types.h"

#include "absl/types/optional.h"

#include <vector>

namespace xla {

class HloInstruction;

namespace poplarplugin {

class InstructionColocatorHelper {
 protected:
  InstructionColocatorHelper();
  // Function which can be specialized for each collocator.
  virtual bool CanColocateExtra(const HloInstruction* a,
                                const HloInstruction* b) const;

 public:
  // Return true if this Colocator can be used for this instruction.
  virtual bool CanColocate(const HloInstruction* inst) const = 0;
  // Return true iff a and b can be used colocated together.
  bool CanColocate(const HloInstruction* a, const HloInstruction* b) const;
  // Returns how many bytes to colocate.
  virtual int64 GetColocateBufferSize(
      const CompilerInformation& information) const = 0;

  int64 GetID() const;

 private:
  // ID used for determinism in scheduling.
  static int64 GetNextID();
  int64 id_;
};

struct InstructionColocatorHelperPtrComparator {
  bool operator()(const InstructionColocatorHelper* const& lhs,
                  const InstructionColocatorHelper* const& rhs) const;
};

// Get all registered collocators.
const std::vector<const InstructionColocatorHelper*>&
GetAllInstructionColocatorHelpers();

// Return whether two instructions can be colocated.
bool CanColocate(const HloInstruction* a, const HloInstruction* b);

// Get a collocator given an instruction.
absl::optional<const InstructionColocatorHelper*> GetInstructionColocatorHelper(
    const HloInstruction* inst);

// Helper class for schedulers to create a cluster of collocated instructions.
template <typename ClusterType>
class ColocatorCluster {
 public:
  ColocatorCluster(const InstructionColocatorHelper* collocator_type,
                   const CompilerInformation& information)
      : collocator_type_(collocator_type),
        information_(information),
        cluster_size_(0) {}
  ColocatorCluster() = delete;

  // Add an element to cluster. Returns whether all the clusters should be
  // scheduled given the collocator type.
  bool Add(ClusterType cluster, int64 size) {
    cluster_.push_back(cluster);
    cluster_size_ += size;
    return cluster_size_ >=
           collocator_type_->GetColocateBufferSize(information_);
  }

  // Get all elements from the cluster and clear it.
  std::vector<ClusterType> GetAll() {
    std::vector<ClusterType> out = cluster_;
    cluster_.clear();
    cluster_size_ = 0;
    return out;
  }

 private:
  std::vector<ClusterType> cluster_;
  const InstructionColocatorHelper* collocator_type_;
  const CompilerInformation& information_;
  int64 cluster_size_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_INSTRUCTION_COLLOCATOR_HELPER_H_
