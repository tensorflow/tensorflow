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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_BUFFER_LIVENESS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_BUFFER_LIVENESS_H_

#include <memory>
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"

namespace xla {

// Class which computes liveness of the output buffers of HLOs and their
// interference.
class BufferLiveness {
 public:
  // Constructs a buffer liveness object for the given module assuming the given
  // HLO instruction ordering.
  static StatusOr<std::unique_ptr<BufferLiveness>> Run(
      const HloModule* module, std::unique_ptr<HloOrdering> hlo_ordering);

  // Returns true if the live range of the buffer containing the output of 'a'
  // may overlap with the live range of the buffer of 'b'. If instruction 'a'
  // interferes with instruction 'b' then they cannot share the same buffer.
  bool MayInterfere(const LogicalBuffer& a, const LogicalBuffer& b) const;

  // Returns true if the buffer for the given instruction may be live out of the
  // module. That is, the instruction's buffer may be included in the output of
  // the entry computation.
  bool MaybeLiveOut(const LogicalBuffer& buffer) const;

  // Returns the underlying points-to analysis used for this liveness analysis.
  const TuplePointsToAnalysis& points_to_analysis() const {
    return *points_to_analysis_;
  }

  string ToString() const;

 private:
  explicit BufferLiveness(const HloModule* module,
                          std::unique_ptr<HloOrdering> hlo_ordering)
      : module_(module), hlo_ordering_(std::move(hlo_ordering)) {}

  // Perform buffer liveness analysis. This method must be called prior to
  // MayInterfere or MaybeLiveOut.
  tensorflow::Status Analyze();

  // Returns true if the live range of the buffer of 'a' is strictly before the
  // live range of the buffer of 'b' (they do not overlap).
  bool live_range_strictly_before(const LogicalBuffer& a,
                                  const LogicalBuffer& b) const;

  const HloModule* module_;
  std::unique_ptr<HloOrdering> hlo_ordering_;

  // Set of LogicalBuffers which are aliased in the output of other
  // instructions. For example, a LogicalBuffer which is inserted into a tuple
  // is considered to be aliased and will be in this set.
  tensorflow::gtl::FlatSet<const LogicalBuffer*> aliased_buffers_;

  // LogicalBuffers that may be live out of the entry computation.
  tensorflow::gtl::FlatSet<const LogicalBuffer*> maybe_live_out_buffers_;

  std::unique_ptr<TuplePointsToAnalysis> points_to_analysis_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_BUFFER_LIVENESS_H_
