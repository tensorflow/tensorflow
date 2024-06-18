/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SPMD_CONVOLUTION_HANDLER_H_
#define XLA_SERVICE_SPMD_CONVOLUTION_HANDLER_H_

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/dot_as_convolution_util.h"
#include "xla/service/spmd/spmd_partitioner.h"

namespace xla {
namespace spmd {

// Partition convolution.
absl::StatusOr<HloInstruction*> PartitionConvolution(
    const PartitionedHlo& lhs, const PartitionedHlo& rhs,
    const Shape& output_base_shape, const HloSharding& output_sharding,
    const dot_as_convolution_util::DotConvolutionDimsInfo& dims_mapping,
    absl::FunctionRef<absl::StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>
        create_sharded_conv,
    const Window& conv_window, HloInstruction* original_hlo,
    int64_t num_partitions, const SpmdPartitionerOptions& options,
    HloInstruction* partition_id, HloModule* module, SpmdBuilder* b);

}  // namespace spmd
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_CONVOLUTION_HANDLER_H_
