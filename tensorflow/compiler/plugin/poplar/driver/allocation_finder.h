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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_ALLOCATION_FINDER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_ALLOCATION_FINDER_H_

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include <vector>

namespace xla {

class HloModule;
class HloInstruction;

namespace poplarplugin {

struct TensorTarget {
  const HloInstruction* tgt;
  int64 input_index;
  std::vector<const HloInstruction*> path;
  TensorTarget(const HloInstruction* tgt,
               int64 input_index,
               const std::vector<const HloInstruction*>& path)  :
      tgt(tgt),
      input_index(input_index),
      path(path) {}
};

using TensorSource = std::pair<const HloInstruction*,int64>;
using TensorAllocationMap = std::map<TensorSource, TensorTarget>;

/**
 * This class finds all instructions that explicitly add tensors to the
 * graph.  For each one of them, it locates the downstream consumers of that
 * tensor, and if any of those instructions require a specific tensor allocation
 * method (e.g. convolution), then it notes the downstream instruction
 */
class AllocationFinder {
public:
  AllocationFinder() {}

  ~AllocationFinder() = default;

  Status CreateAllocationMap(HloModule* module);

  TensorAllocationMap tensor_allocation_map;

private:
  void FindConsumers(const TensorSource&, const HloInstruction* tgt, int64);

  // Should return true when target 'a' should be used over 'b'
  bool CompareConvolutionTargets(const TensorTarget& a, const TensorTarget& b);
  bool CompareDotTargets(const TensorTarget& a, const TensorTarget& b);

  std::set<HloInstruction*> visited;
  std::vector<const HloInstruction*> path;
};

}
}

#endif
