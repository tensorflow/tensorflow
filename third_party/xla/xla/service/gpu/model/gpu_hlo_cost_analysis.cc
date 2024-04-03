/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/map_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/elemental_ir_emitter.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/model/hlo_op_profiles.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

// Use the "reserved" keys for these properties so lookups are fast.
static constexpr absl::string_view kIRSizeKey = HloCostAnalysis::kReserved0Key;
static constexpr absl::string_view kBasicBlockSplitCountKey =
    HloCostAnalysis::kReserved1Key;

// TODO TJ consider adding these in the fast lookup path
static constexpr absl::string_view kCollAlgoScaleRatioKey =
    "Collective algorithm's scaling ratio";
static constexpr absl::string_view kCollNumDevicesKey =
    "Number of devices of a collective group";

// We use static tables to look up system bandwidths for different
// type of hardware below.
// TODO TJ this needs to be hosted somewhere more centralized.

absl::Status GpuHloCostAnalysis::Preprocess(const HloInstruction* hlo) {
  TF_RETURN_IF_ERROR(HloCostAnalysis::Preprocess(hlo));

  current_properties_[kIRSizeKey] = 1;
  current_properties_[kBasicBlockSplitCountKey] =
      ElementalIrEmitter::OpInvalidatesCache(hlo);

  return absl::OkStatus();
}

float GpuHloCostAnalysis::ScalingRatio(const HloInstruction& hlo) const {
  return GetPropertyForHlo(hlo, kCollAlgoScaleRatioKey, hlo_properties_);
}

int64_t GpuHloCostAnalysis::NumOfDevices(const HloInstruction& hlo) const {
  return GetPropertyForHlo(hlo, kCollNumDevicesKey, hlo_properties_);
}

int64_t GpuHloCostAnalysis::FusionParameterReadBytes(
    const HloInstruction* hlo) const {
  CHECK(hlo->IsFused() && (hlo->opcode() == HloOpcode::kParameter ||
                           hlo->opcode() == HloOpcode::kGetTupleElement));
  float utilization = hlo_properties_.at(hlo)[kUtilizationKey];
  if (!options_.count_multiple_input_accesses) {
    utilization = fmin(utilization, 1.0);
  }
  return std::llround(GetShapeSize(hlo->shape()) * utilization);
}

absl::Status GpuHloCostAnalysis::FusionCalculateUtilizations(
    const HloInstruction* fusion) {
  const HloInstruction* root = fusion->fused_expression_root();
  // Traverse through the computation from the root till parameters propagating
  // the utilization of operands; store utilization of each node in
  // hlo_properties_. All consumers of an instruction are processed before the
  // instruction itself.
  std::vector<HloInstruction*> instructions =
      fusion->fused_instructions_computation()->MakeInstructionPostOrder();
  absl::c_reverse(instructions);

  // Whenever we account a non-element-wise operation we forget about
  // element-wise roots encountered so far and provisionally set its operands
  // as new element-wise roots.
  absl::flat_hash_map<const HloInstruction*, int64_t> root_ir_sizes;

  for (const HloInstruction* instr : instructions) {
    hlo_properties_[instr][kUtilizationKey] = 0;
    hlo_properties_[instr][kIRSizeKey] = 0;
    elementwise_use_roots_[instr].clear();
    root_utilizations_[instr] = 0;
  }

  // For the purpose of operand utilization analysis, no matter how the fusion
  // outputs are used, we assume that fusion is always executed completely
  // producing 100% of its outputs.
  root_utilizations_[root] = 1.0;
  root_ir_sizes[root] = 1;
  elementwise_use_roots_[root].insert(root);

  current_properties_[kFlopsKey] = 0;
  current_properties_[kBasicBlockSplitCountKey] = 0;
  current_properties_[kIRSizeKey] = 0;

  for (const HloInstruction* instr : instructions) {
    VLOG(8) << instr->name() << ":";
    VLOG(9) << "Elementwise use roots:";
    Properties& instr_props = hlo_properties_[instr];
    for (const HloInstruction* r : elementwise_use_roots_[instr]) {
      VLOG(9) << "\t" << r->name() << ": " << root_utilizations_[r];
      instr_props[kUtilizationKey] += root_utilizations_[r];
      instr_props[kIRSizeKey] += root_ir_sizes[r];
    }

    float cur_instr_utilization = instr_props[kUtilizationKey];
    VLOG(8) << "Total utilization: " << cur_instr_utilization;
    float cur_instr_times_emitted = instr_props[kIRSizeKey];
    VLOG(8) << "Times emitted: " << cur_instr_times_emitted;

    current_properties_[kFlopsKey] +=
        cur_instr_utilization * instr_props[kFlopsKey];
    current_properties_[kIRSizeKey] += cur_instr_times_emitted;
    current_properties_[kBasicBlockSplitCountKey] +=
        cur_instr_times_emitted * ElementalIrEmitter::OpInvalidatesCache(instr);

    for (int operand_idx = 0; operand_idx < instr->operand_count();
         ++operand_idx) {
      const HloInstruction* operand = instr->operand(operand_idx);
      if ((instr->IsElementwise()) || instr->opcode() == HloOpcode::kTuple ||
          instr->opcode() == HloOpcode::kGetTupleElement) {
        for (const HloInstruction* r : elementwise_use_roots_[instr]) {
          elementwise_use_roots_[operand].insert(r);
        }
      } else {
        elementwise_use_roots_[operand].insert(operand);
        float cur_operand_utilization =
            cur_instr_utilization * operand_utilization(*instr, operand_idx);
        // The utilization is always a best-effort estimate, but in some cases
        // cannot be precise due to dynamic nature of operations - dynamic
        // slice is one such example. We do an average estimate in these
        // cases and this can sometimes produce fractional utilizations which
        // should be at least rounded up to a whole number of produced elements
        // to be more realistic.
        int64_t operand_elements =
            ShapeUtil::ElementsInRecursive(operand->shape());

        if (operand_elements == 0) {
          // Element count should not be 0 in any production use case, but there
          // are valid HLO inputs that occur in tests.
          cur_operand_utilization = 0;
        } else {
          cur_operand_utilization =
              ceil(cur_operand_utilization * operand_elements) /
              operand_elements;
        }
        root_utilizations_[operand] += cur_operand_utilization;
        root_ir_sizes[operand] += cur_instr_times_emitted;
      }
    }
  }

  return absl::OkStatus();
}

float GpuHloCostAnalysis::CommonElementwiseUtilization(
    const HloInstruction* a, const HloInstruction* b) const {
  float ret = 0;
  for (auto r : elementwise_use_roots_.at(a)) {
    if (elementwise_use_roots_.at(b).count(r)) {
      ret += root_utilizations_.at(r);
    }
  }
  return ret;
}

bool GpuHloCostAnalysis::ProducerConsumerMergedTooLarge(
    const HloInstruction& producer, const HloInstruction& consumer) {
  int64_t producer_replication = 1;
  // Fusing 'producer' into 'consumer' fusion currently results in replicating
  // its IR the number of times the consumer replicates the access
  // to the parameter corresponding to the producer.
  if (consumer.opcode() == HloOpcode::kFusion) {
    producer_replication =
        IrSize(*consumer.fused_parameter(consumer.operand_index(&producer)));
  }
  VLOG(5) << producer.name() << " would be emitted by " << consumer.name()
          << " x" << producer_replication;
  int64_t n_splits = producer_replication * IrBasicBlockSplitCount(producer) +
                     IrBasicBlockSplitCount(consumer);
  VLOG(5) << "Basic block split counts: " << IrBasicBlockSplitCount(producer)
          << ", " << IrBasicBlockSplitCount(consumer) << " -> " << n_splits;
  if (n_splits > kMaxBasicBlockSplitsPerFusion) {
    return true;
  }
  int64_t merged_ir_size =
      (IrSize(producer) * producer_replication + IrSize(consumer)) *
      (1 << n_splits);
  VLOG(5) << "IR sizes: " << IrSize(producer) << ", " << IrSize(consumer)
          << " -> " << merged_ir_size;
  return merged_ir_size > kMaxIRSize;
}

absl::Status GpuHloCostAnalysis::HandleCustomCall(
    const HloInstruction* custom_call) {
  if (IsCublasGemm(*custom_call)) {
    // The naming conventions and meanings of gemm parameters are documented
    // here:
    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
    TF_ASSIGN_OR_RETURN(auto gpu_config,
                        custom_call->backend_config<gpu::GpuBackendConfig>());
    const gpu::GemmBackendConfig& gemm_config =
        gpu_config.gemm_backend_config();
    // Technically, in addition to the dot product (A * B), cuBLAS gemm also
    // performs additional scaling (by factor 'alpha') and addition with a
    // scaled third matrix (beta * C), which will introduce additional
    // multiplications and additions. But total FLOPS will be dominated by the
    // dot product, so we don't include these extra multiplications and
    // additions in the FLOPS calculation.

    // Also, this calculation assumes that the strides for the gemm are
    // properly set such that none of the inputs in a batch overlap with any
    // other batches. If they do, this will undercount the FLOPS, because it
    // assumes that the strides are implicit in the sizes of the batch
    // dimensions.

    // Finally, this is technically incorrect if the element type of this
    // gemm is an integer type, because in that case no floating point
    // operations are involved at all! But we still calculate FLOPS because the
    // number is sometimes required for ad-hoc calculations.

    // cublasLt supports auxiliary outputs, so output may be tuple.
    const Shape& output_shape = custom_call->shape().IsTuple()
                                    ? custom_call->shape().tuple_shapes(0)
                                    : custom_call->shape();

    current_properties_[kFlopsKey] =
        GetDotFlops(custom_call->operand(0)->shape(), output_shape,
                    gemm_config.dot_dimension_numbers());
    return absl::OkStatus();
  }

  if (IsCustomCallToDnnConvolution(*custom_call)) {
    // As with dots, this flops calculation has the following inaccuracies.
    //
    //  - We may have a fused conv which does additional ops (multiplying by a
    //    scalar `alpha`, adding a bias or side-input, doing a relu, etc).  But
    //    we can safely ignore this because the overall computation is dominated
    //    by the convolution itself.
    //
    //  - cudnn may use complex conv algorithms that do fewer (or more!) flops
    //    than we calculate.
    //
    //  - for int8_t convs, these aren't *fl*ops, but we fudge it.
    current_properties_[kFlopsKey] = GetConvolutionFlops(custom_call);

    // conv custom-calls return a tuple (real_output, temp_bytes).  Count just
    // the real_output in output bytes accessed.  The main purpose of
    // hlo_cost_analysis is to figure out if ops are running "as fast as
    // possible", and if we were to include temp memory in here, we'd
    // essentially be *rewarding* convs that use additional temp memory!
    if (custom_call->shape().IsTuple()) {
      float output_size =
          options_.shape_size(custom_call->shape().tuple_shapes(0));
      // 'Bytes accessed' are estimated in HloCostAnalysis::Preprocess() as
      // input + output. As the output size is being adjusted here it has
      // to propagate to the total bytes accessed.
      current_properties_[kBytesAccessedKey] -=
          current_properties_.output_bytes_accessed();
      current_properties_[kBytesAccessedKey] += output_size;
      current_properties_.set_output_bytes_accessed(output_size);
    }
    return absl::OkStatus();
  }

  return HloCostAnalysis::HandleCustomCall(custom_call);
}

int64_t GpuHloCostAnalysis::GetConvolutionFlops(
    const HloInstruction* convolution) {
  auto lhs = convolution->operand(0);
  auto rhs = convolution->operand(1);
  const Shape& lhs_shape = lhs->shape();
  const Shape& rhs_shape = rhs->shape();
  const Shape& result_shape = [&]() -> const Shape& {
    // convolution custom-calls return a tuple of (actual_result, temp_buffer).
    const Shape& shape = convolution->shape();
    if (IsCustomCallToDnnConvolution(*convolution) &&
        convolution->shape().IsTuple()) {
      return shape.tuple_shapes(0);
    }
    return shape;
  }();

  return HloCostAnalysis::GetConvolutionFlops(convolution, lhs_shape, rhs_shape,
                                              result_shape);
}

int64_t FlopsPerElement(const se::DeviceDescription* device_info,
                        const PrimitiveType type, const HloOpcode opcode) {
  auto device_profile = HloOpProfiles::Singleton().GetProfile(device_info);
  // Elementwise instructions typically take at least a few clock cycles.
  constexpr int64_t kDefaultFlopsPerElement = 3;
  return FindOrDefault(device_profile, std::make_pair(opcode, type),
                       kDefaultFlopsPerElement);
}

int64_t GetFlopsForElementwiseOp(const se::DeviceDescription* gpu_device_info,
                                 const HloOpcode op_code, const Shape& shape) {
  int64_t flop_per_element =
      FlopsPerElement(gpu_device_info, shape.element_type(), op_code);
  return flop_per_element * ShapeUtil::ElementsInRecursive(shape);
}

int64_t GetFlopsForElementwiseOp(const se::DeviceDescription* gpu_device_info,
                                 const HloInstruction* instr) {
  return GetFlopsForElementwiseOp(gpu_device_info, instr->opcode(),
                                  instr->shape());
}

absl::Status GpuHloCostAnalysis::HandleAllReduce(
    const HloInstruction* allreduce) {
  const HloModuleConfig& config = allreduce->GetModule()->config();
  TF_ASSIGN_OR_RETURN(
      CollectiveOpGroupMode group_mode,
      GetCollectiveOpGroupMode(
          allreduce->channel_id().has_value(),
          Cast<HloAllReduceInstruction>(allreduce)->use_global_device_ids()));

  // Get number of ranks for this instruction based on replica groups and mode.
  int64_t num_devices = config.num_partitions();
  int64_t num_replicas = config.replica_count();
  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> participant_counts,
      GetPariticipantCountsForReplicaGroups(
          num_replicas, num_devices, allreduce->replica_groups(), group_mode));
  int64_t num_ranks = 1;

  for (auto count : participant_counts) {
    num_ranks = std::max(num_ranks, count);
  }

  VLOG(5) << "Computing cost for " << num_ranks << " ranks in "
          << allreduce->ToString();

  int64_t output_bytes_accessed = 0;
  // Since for allreduces, the input shape is the same as output shape and can
  // be done in-place, we calculate output_bytes_accessed based on just the
  // output size.
  ShapeUtil::ForEachSubshape(
      allreduce->shape(), [&](const Shape& subshape, const ShapeIndex&) {
        if (subshape.IsArray()) {
          output_bytes_accessed += GetShapeSize(subshape);
        }
      });
  int64_t bytes_accessed = output_bytes_accessed;
  for (const HloInstruction* operand : allreduce->operands()) {
    bytes_accessed += GetShapeSize(operand->shape());
  }
  current_properties_.set_output_bytes_accessed(output_bytes_accessed);
  current_properties_[kBytesAccessedKey] = bytes_accessed;
  current_properties_[kCollNumDevicesKey] = num_ranks;
  // Since allreduce has compute, we need to get flops for the compute
  // part which is an elementwise op.
  current_properties_[kFlopsKey] = GetFlopsForElementwiseOp(
      device_info_, allreduce->to_apply()->root_instruction()->opcode(),
      allreduce->shape());

  // TODO TJ support multi-node case, we need to know how many nodes there are.
  int num_intra_steps = 2 * (num_ranks - 1);
  // Compute algorithmic scaling ratio, this can be used to be multiplied with
  // bus bandwidth to get the effective bandwidth of the algorithm. The scaling
  // ratio differs based on what algorithm NCCL chooses to use. This is the
  // scaling factor for ring since NCCL will only use ring for single-node, need
  // to add support for tree algo in multi-node case.
  float scaling_ratio = (1.0 * num_ranks) / num_intra_steps;
  current_properties_[kCollAlgoScaleRatioKey] = scaling_ratio;

  return absl::OkStatus();
}

absl::Status GpuHloCostAnalysis::HandleConcatenate(const HloInstruction* hlo) {
  // Concat turns into a compare plus branch instruction.
  int64_t flop_per_element = 6;
  // If a warp crosses the operands boundary, both branches are executed. This
  // depends on the tiling of the final fusion and is therefore hard to predict
  // at this level. Executing both branches drives up the flops, but not the
  // bandwidth. So it might seem like a good idea to fuse a concat into a
  // memory-bound consumer. However, the divergent warps increase the cost of
  // compute-heavy producers that might be fused later. We see this issue in
  // some important LLM models that fuse a concat into a column reduction (see
  // PriorityFusionTest.DontFuseConcat test). To prevent this particular fusion,
  // we add large number of flops to the concat. Both the condition and the flop
  // count are tuned to this particular case.
  // TODO(b/315776282): Model this more accurately once we can reason about
  // tiling patterns.
  int64_t dim = Cast<HloConcatenateInstruction>(hlo)->concatenate_dimension();
  if (dim > 0 && hlo->operand(0)->shape().dimensions()[dim] & 31) {
    flop_per_element = 400;
  }
  current_properties_[kFlopsKey] =
      flop_per_element * ShapeUtil::ElementsInRecursive(hlo->shape());
  return absl::OkStatus();
}

absl::Status GpuHloCostAnalysis::HandleReduce(const HloInstruction* hlo) {
  // HloCostAnalysis::HandleReduce computes FLOPs for the computation correctly,
  // but `bytes_accessed` estimates are different for GPU.
  TF_RETURN_IF_ERROR(HloCostAnalysis::HandleReduce(hlo));

  const HloReduceInstruction* reduce = DynCast<HloReduceInstruction>(hlo);
  auto output_shape = reduce->shape().IsArray()
                          ? reduce->shape()
                          : reduce->shape().tuple_shapes(0);

  int64_t output_bytes_accessed = 0;
  ShapeUtil::ForEachLeafShape(
      reduce->shape(), [&](const Shape& sub_shape, const ShapeIndex& index) {
        output_bytes_accessed += GetShapeSize(sub_shape);
      });

  current_properties_.set_output_bytes_accessed(output_bytes_accessed);

  int64_t bytes_accessed = output_bytes_accessed;
  for (int64_t input_operand_id = 0; input_operand_id < reduce->input_count();
       ++input_operand_id) {
    bytes_accessed +=
        current_properties_.operand_bytes_accessed(input_operand_id);
  }

  int64_t output_shape_size = ShapeUtil::ElementsIn(output_shape);
  for (int64_t init_operand_id = reduce->input_count();
       init_operand_id < reduce->operand_count(); ++init_operand_id) {
    auto init_operand = reduce->operand(init_operand_id);

    int64_t operand_bytes_accessed =
        output_shape_size * GetShapeSize(init_operand->shape());
    current_properties_.set_operand_bytes_accessed(init_operand_id,
                                                   operand_bytes_accessed);
    current_properties_.set_operand_utilization(init_operand_id,
                                                output_shape_size);

    bytes_accessed += operand_bytes_accessed;
  }

  current_properties_[kBytesAccessedKey] = bytes_accessed;

  return absl::OkStatus();
}

absl::Status GpuHloCostAnalysis::HandleElementwiseOp(
    const HloInstruction* hlo) {
  current_properties_[kFlopsKey] = GetFlopsForElementwiseOp(device_info_, hlo);
  return absl::OkStatus();
}

absl::Status GpuHloCostAnalysis::HandleElementwiseUnary(
    const HloInstruction* hlo) {
  return HandleElementwiseOp(hlo);
}

absl::Status GpuHloCostAnalysis::HandleElementwiseBinary(
    const HloInstruction* hlo) {
  return HandleElementwiseOp(hlo);
}

std::unique_ptr<HloCostAnalysis>
GpuHloCostAnalysis::CreateNestedCostAnalysis() {
  return std::make_unique<GpuHloCostAnalysis>(options_, device_info_);
}

bool GpuHloCostAnalysis::KeyToCopyFromSubcomputation(
    absl::string_view key) const {
  return !absl::StartsWith(key, kBytesAccessedKey) &&
         !absl::StartsWith(key, kUtilizationKey) &&
         !absl::StartsWith(key, kIRSizeKey) &&
         !absl::StartsWith(key, kBasicBlockSplitCountKey);
}

float GpuHloCostAnalysis::IrBasicBlockSplitCount(
    const HloInstruction& hlo) const {
  return GetPropertyForHlo(hlo, kBasicBlockSplitCountKey, hlo_properties_);
}

float GpuHloCostAnalysis::IrSize(const HloInstruction& hlo) const {
  return GetPropertyForHlo(hlo, kIRSizeKey, hlo_properties_);
}

}  // namespace gpu
}  // namespace xla
