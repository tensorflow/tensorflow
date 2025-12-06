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

#include <cstdint>

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/dot_as_convolution_util.h"
#include "xla/service/spmd/dot_handler.h"
#include "xla/service/spmd/spmd_partitioner.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace spmd {

// Partition convolution.
absl::StatusOr<HloInstruction*> PartitionConvolution(
    const PartitionedHlo& lhs, const PartitionedHlo& rhs,
    const Shape& output_base_shape, const HloSharding& output_sharding,
    const dot_as_convolution_util::DotConvolutionDimsInfo& dims_mapping,
    CreateShardedConvolutionFunctor& create_sharded_conv,
    const Window& conv_window, HloInstruction* original_hlo,
    int64_t num_partitions, const SpmdPartitionerOptions& options,
    HloInstruction* partition_id, HloModule* module, SpmdBuilder* b);

absl::StatusOr<std::unique_ptr<HloInstruction>> CreateShardedConvolution(
    const HloInstruction& conv,
    const dot_as_convolution_util::DotConvolutionDimsInfo& dot_dnums,
    HloInstruction* sharded_lhs_hlo, HloInstruction* sharded_rhs_hlo,
    const Window& conv_window);

// Functor class for creating sharded convolutions with operands of type
// PartitionedHlo.
class CreateShardedConvolutionFunctor final
    : public CreateShardedFunctorBase<PartitionedHlo> {
 public:
  CreateShardedConvolutionFunctor(
      HloInstruction* conv,
      const dot_as_convolution_util::DotConvolutionDimsInfo& dims_info)
      : conv_(conv), dims_info_(dims_info) {}

  // Implements the creation of sharded convolutions.
  absl::StatusOr<HloInstruction*> CreateSharded(
      const PartitionedHlo& ll, const PartitionedHlo& rr, spmd::SpmdBuilder* b,
      const Window& conv_window) const override {
    HloInstruction* l = ll.hlo();
    HloInstruction* r = rr.hlo();
    if (dims_info_.conv_spatial_dims.empty() &&
        conv_->feature_group_count() == 1 && conv_->batch_group_count() == 1) {
      TF_ASSIGN_OR_RETURN(
          auto sharded_conv,
          dot_as_convolution_util::CreateShardedConvForDotGeneralConvolution(
              *conv_, dims_info_, l, r));
      return b->AddInstruction(std::move(sharded_conv));
    } else {
      TF_ASSIGN_OR_RETURN(
          auto sharded_conv,
          CreateShardedConvolution(*conv_, dims_info_, l, r, conv_window));
      return b->AddInstruction(std::move(sharded_conv));
    }
  }

 private:
  HloInstruction* conv_;
  const dot_as_convolution_util::DotConvolutionDimsInfo& dims_info_;
};

}  // namespace spmd
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_CONVOLUTION_HANDLER_H_
