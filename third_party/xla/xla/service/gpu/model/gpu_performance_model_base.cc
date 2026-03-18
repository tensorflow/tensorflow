/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/model/gpu_performance_model_base.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/emitters/emitter_base.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/backends/gpu/codegen/triton/fusion.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "xla/codegen/emitters/transforms/pass_pipelines.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/backends/gpu/codegen/emitters/emitter_base.h"

namespace xla {
namespace gpu {

namespace {

// Returns whether a fusion uses the parameter at the given index elementwise
// from its root. Also works if 'fusion' is a multi-output fusion.
bool FusionUsesParameterElementwiseFromRoot(
    const HloInstruction* fusion, int parameter_index,
    const GpuHloCostAnalysis* cost_analysis) {
  // This checks whether there is a path from fused_expression_root() to the
  // parameter that only goes through elementwise, Tuple and GetTupleElement
  // ops.
  return cost_analysis->CommonElementwiseUtilization(
             fusion->fused_parameter(parameter_index),
             fusion->fused_expression_root()) == 1.f;
}

// Limit the bandwidth for low occupancy cases. Each SM can issue at most
// one 32B memory transaction per clock. H100 needs at least 56.8 active SMs
// (1830 MHz) to saturate the memory bandwidth (3.35 TB/s).
float AdjustBandwidth(const se::DeviceDescription& gpu_device_info,
                      float bandwidth, int64_t num_blocks) {
  float per_block_bandwidth = gpu_device_info.clock_rate_ghz() * 1.0e9f *
                              gpu_device_info.memory_transactions_per_clock();
  float max_bandwidth = num_blocks * per_block_bandwidth;

  return std::min(bandwidth, max_bandwidth);
}

}  // namespace

std::optional<EstimateRunTimeData> GpuPerformanceModelCache::Get(
    const HloInstruction& instruction) {
  auto it = instruction_runtime_data_.find(&instruction);
  if (it != instruction_runtime_data_.end()) {
    return it->second;
  }
  return std::nullopt;
}

std::optional<absl::Duration> GpuPerformanceModelCache::Get(
    const HloInstruction& producer, const HloInstruction& consumer) {
  absl::MutexLock lock(mutex_);

  auto it = fusion_runtime_data_.find(&producer);
  if (it != fusion_runtime_data_.end()) {
    auto jt = it->second.find(&consumer);
    if (jt != it->second.end()) {
      return jt->second;
    }
  }
  return std::nullopt;
}

const absl::flat_hash_map<const HloInstruction*, absl::Duration>&
GpuPerformanceModelCache::GetAllConsumers(const HloInstruction& producer) {
  return fusion_runtime_data_[&producer];
}

bool GpuPerformanceModelCache::ContainsConsumers(
    const HloInstruction& producer) {
  return fusion_runtime_data_.contains(&producer);
}

void GpuPerformanceModelCache::Set(const HloInstruction& instruction,
                                   const EstimateRunTimeData& runtime_data) {
  instruction_runtime_data_[&instruction] = runtime_data;
}

void GpuPerformanceModelCache::Set(const HloInstruction& producer,
                                   const HloInstruction& consumer,
                                   absl::Duration runtime) {
  absl::MutexLock lock(mutex_);
  fusion_runtime_data_[&producer][&consumer] = runtime;
}

void GpuPerformanceModelCache::Invalidate(const HloInstruction& instruction) {
  // Remove runtime data for the instruction.
  instruction_runtime_data_.erase(&instruction);

  // Remove cache for all producer-consumer pairs where the instruction is
  // producer.
  fusion_runtime_data_.erase(&instruction);

  // Remove register usage for the instruction.
  register_usage_.erase(&instruction);

  // Iterate through operands to find all producer-consumer pairs where
  // instruction is consumer and remove them from cache.
  for (auto* operand : instruction.operands()) {
    if (operand->opcode() == HloOpcode::kGetTupleElement) {
      operand = operand->mutable_operand(0);
    }
    auto it = fusion_runtime_data_.find(operand);
    if (it != fusion_runtime_data_.end()) {
      it->second.erase(&instruction);
    }
  }
}

/*static*/ std::unique_ptr<mlir::MLIRContext>
GpuPerformanceModelBase::CreateMlirContext() {
  // We are already in a multi-threaded context, so we can disable MLIR threading.
  auto mlir_context = std::make_unique<mlir::MLIRContext>(mlir::MLIRContext::Threading::DISABLED);
  mlir::DialectRegistry registry = EmitterBase::GetDialectRegistry();
  mlir_context->appendDialectRegistry(registry);
  for (mlir::StringRef name : registry.getDialectNames()) {
    mlir_context->getOrLoadDialect(name);
  }
  RegisterSymbolicExprStorage(mlir_context.get());
  return mlir_context;
}

/*static*/
LaunchDimensions GpuPerformanceModelBase::EstimateFusionLaunchDimensions(
    const HloFusionAnalysis& fusion_analysis, mlir::MLIRContext* mlir_context) {
  auto emitter = GetFusionEmitter(
      PreBufferAssignmentFusionInfo{fusion_analysis}, mlir_context);
  if (const auto* kernel_emitter =
          dynamic_cast<const KernelFusionInterface*>(emitter.get())) {
    return kernel_emitter->launch_dimensions();
  }

  // TritonFusion does not implement KernelFusionInterface, because it provides
  // launch dimensions only for SoftMax fusions.
  if (const auto* triton_emitter =
          dynamic_cast<const TritonFusion*>(emitter.get())) {
    if (auto launch_config = triton_emitter->GetLaunchConfig()) {
      return launch_config->launch_dimensions;
    }
  }

  // This estimate should never be reached in fusion code. Fusions that don't
  // implement KernelFusionInterface, don't generate GPU kernels, so there is
  // nothing to fuse. Keep this estimate as a simple fallback.
  //
  // We assume that the kernel launches 1 thread per output element and 128
  // threads per block. In multi-output fusions, only look at one root.
  VLOG(5) << "Using fallback launch dimensions estimate for "
          << fusion_analysis.fusion().ToString();
  int64_t num_threads_per_block = 128;
  int64_t estimated_num_threads =
      ShapeUtil::ElementsInRecursive(fusion_analysis.fusion_root(0).shape());
  int64_t num_blocks =
      CeilOfRatio(estimated_num_threads, num_threads_per_block);
  return LaunchDimensions(num_blocks, num_threads_per_block);
}

/*static*/
int64_t GpuPerformanceModelBase::GetOperandBytesAccessed(
    const GpuHloCostAnalysis* cost_analysis, const HloInstruction* instr,
    const HloInstruction* operand) {
  // When called for a producer-consumer fusion, the operand can be from a
  // different instruction. GpuHloCostAnalysis can't fail gracefully in this
  // case, so we need an explicit check.
  if (!instr->IsUserOf(operand)) {
    return 0;
  }

  return cost_analysis->operand_bytes_accessed(*instr,
                                               instr->operand_index(operand));
}

/*static*/
float GpuPerformanceModelBase::GetOperandUtilization(
    const GpuHloCostAnalysis* cost_analysis, const HloInstruction* instr,
    const HloInstruction* operand) {
  if (operand->IsMultiOutputFusion()) {
    // If 'operand' is a multi-output fusion, we need to check which of its
    // outputs are used by 'instr'.
    float res = 0.f;
    for (int64_t i = 0; i < instr->operand_count(); ++i) {
      if (instr->operand(i)->opcode() == HloOpcode::kGetTupleElement &&
          instr->operand(i)->operand(0) == operand) {
        res += cost_analysis->operand_utilization(*instr, i);
      }
    }
    return res;
  }
  // When called for a producer-consumer fusion, the operand can be from a
  // different instruction. GpuHloCostAnalysis can't fail gracefully in this
  // case, so we need an explicit check.
  if (!instr->IsUserOf(operand)) {
    return 0.f;
  }

  return cost_analysis->operand_utilization(*instr,
                                            instr->operand_index(operand));
}

/*static*/
float GpuPerformanceModelBase::GetCommonUtilization(
    const GpuHloCostAnalysis* cost_analysis, const HloInstruction* producer,
    int64_t producer_idx_of_operand, const HloInstruction* consumer) {
  const auto* operand = producer->operand(producer_idx_of_operand);

  if (!consumer || !consumer->IsUserOf(operand)) {
    return 0.f;
  }

  if (producer->IsElementwise() ||
      (producer->opcode() == HloOpcode::kFusion &&
       FusionUsesParameterElementwiseFromRoot(producer, producer_idx_of_operand,
                                              cost_analysis))) {
    if (consumer->opcode() == HloOpcode::kFusion) {
      int64_t consumer_idx_of_common_operand = consumer->operand_index(operand);
      float res = 0.f;
      std::vector<int64_t> consumer_indices_of_producer;
      if (producer->IsMultiOutputFusion()) {
        for (int64_t i = 0; i < consumer->operand_count(); ++i) {
          if (consumer->operand(i)->opcode() == HloOpcode::kGetTupleElement &&
              consumer->operand(i)->operand(0) == producer) {
            consumer_indices_of_producer.push_back(i);
          }
        }
      } else {
        consumer_indices_of_producer.push_back(
            consumer->operand_index(producer));
      }
      for (int64_t consumer_idx_of_producer : consumer_indices_of_producer) {
        res += cost_analysis->CommonElementwiseUtilization(
            consumer->fused_parameter(consumer_idx_of_common_operand),
            consumer->fused_parameter(consumer_idx_of_producer));
      }
      return res;
    } else if (consumer->IsElementwise()) {
      return 1.f;
    }
  }
  return 0.f;
}

/*static*/
int64_t GpuPerformanceModelBase::GetSharedOperandBytesAccessed(
    const GpuHloCostAnalysis* cost_analysis, const HloInstruction* producer,
    const HloInstruction* consumer, const HloInstruction* operand) {
  float producer_utilization_by_consumer =
      GetOperandUtilization(cost_analysis, consumer, producer);

  int64_t bytes_accessed_by_producer =
      GetOperandBytesAccessed(cost_analysis, producer, operand);

  int64_t bytes_accessed_by_consumer =
      GetOperandBytesAccessed(cost_analysis, consumer, operand);

  float common_utilization =
      producer->IsUserOf(operand)
          ? GetCommonUtilization(cost_analysis, producer,
                                 producer->operand_index(operand), consumer)
          : 0.f;

  int64_t operand_size = cost_analysis->GetShapeSize(operand->shape());
  int64_t common_bytes_accessed =
      std::llround(operand_size * common_utilization);

  return std::llround(bytes_accessed_by_producer *
                      producer_utilization_by_consumer) +
         bytes_accessed_by_consumer - common_bytes_accessed;
}

/*static*/
absl::Duration GpuPerformanceModelBase::ReadTimeWithDRAMHeuristic(
    const se::DeviceDescription& gpu_device_info, int64_t num_blocks,
    int64_t n_bytes_net, int64_t n_bytes_total, PrimitiveType element_type,
    double hbm_bandwidth_utilization_rate) {
  // The first read of the input buffer always happens from DRAM. If reads are
  // no coaleced, bandwidth is reduced by the waste factor.
  float dram_bandwidth =
      gpu_device_info.memory_bandwidth() * hbm_bandwidth_utilization_rate;

  // Two things can happed on re-reading the buffer:
  //   - If the buffer fits into cache, the L1/L2 cache speedup is applied.
  //   - If the buffer doesn't fit, it will be read from DRAM and the same
  //     coalessing waste factor is applied.
  float rest_bandwidth = gpu_device_info.memory_bandwidth();
  if (n_bytes_net < gpu_device_info.l2_cache_size()) {
    rest_bandwidth *= kL2CacheSpeedup;
    if (n_bytes_net <
        gpu_device_info.l1_cache_size_per_SM() * gpu_device_info.core_count()) {
      rest_bandwidth *= kL1CacheSpeedup;
    }
  } else {
    rest_bandwidth *= hbm_bandwidth_utilization_rate;
  }

  dram_bandwidth = AdjustBandwidth(gpu_device_info, dram_bandwidth, num_blocks);
  rest_bandwidth = AdjustBandwidth(gpu_device_info, rest_bandwidth, num_blocks);

  // n_bytes_net > n_bytes_total can happen when we compute read time of
  // shared operand. This is a flaw in the interface that should be fixed.
  int64_t n_bytes_read_dram = std::min(n_bytes_net, n_bytes_total);

  // Number of bytes that we be re-read, potentially from cache.
  int64_t n_bytes_read_cache = n_bytes_total - n_bytes_read_dram;

  return absl::Seconds(n_bytes_read_dram / dram_bandwidth) +
         absl::Seconds(n_bytes_read_cache / rest_bandwidth);
}

/*static*/
absl::Duration GpuPerformanceModelBase::WriteTime(
    const se::DeviceDescription& gpu_device_info, int64_t bytes_written) {
  return absl::Seconds(1.0f * bytes_written /
                       gpu_device_info.memory_bandwidth());
}

/*static*/
int64_t GpuPerformanceModelBase::CalculateEffectiveFlopsPerNs(
    const se::DeviceDescription& gpu_device_info, int64_t num_blocks,
    int64_t num_threads_per_block) {
  int64_t n_active_fpus_per_core =
      std::min<int64_t>(num_threads_per_block, gpu_device_info.fpus_per_core());

  int64_t n_active_core =
      std::min<int64_t>(num_blocks, gpu_device_info.core_count());
  int64_t fpu_count = n_active_core * n_active_fpus_per_core;

  double flop_per_ns_per_fpu = gpu_device_info.clock_rate_ghz() * /*fma:*/ 2;
  return flop_per_ns_per_fpu * fpu_count;
}

/*static*/
int64_t GpuPerformanceModelBase::CalculatePeakMatrixOpsPerNs(
    const se::DeviceDescription& gpu_device_info, xla::PrimitiveType dtype) {
  const se::ExecutionUnitDescription* caps =
      gpu_device_info.matrix_unit_description();
  std::optional<se::ExecutionUnitDescription::RateInfo> dtype_rates;

  if (caps != nullptr) {
    dtype_rates = caps->GetRateInfo(dtype);
  }

  if (!dtype_rates.has_value()) {
    // Fallback to default flops if matrix unit description is not available
    // or does not support the given dtype.
    return CalculateEffectiveFlopsPerNs(
        gpu_device_info, /*num_blocks=*/gpu_device_info.core_count(),
        /*num_threads_per_block=*/gpu_device_info.fpus_per_core());
  }

  // FMA is counted as 2 ops.
  double flops_per_ns_per_unit =
      dtype_rates->clock_rate_ghz * dtype_rates->ops_per_clock * 2;
  int64_t n_compute_units =
      gpu_device_info.core_count() * dtype_rates->units_per_core;
  return flops_per_ns_per_unit * n_compute_units;
}

/*static*/
absl::Duration GpuPerformanceModelBase::ComputeTime(
    const se::DeviceDescription& gpu_device_info, int64_t flops,
    int64_t num_blocks, int64_t num_threads_per_block) {
  int64_t flop_per_ns_effective = CalculateEffectiveFlopsPerNs(
      gpu_device_info, num_blocks, num_threads_per_block);
  return absl::Nanoseconds(1.0f * flops / flop_per_ns_effective);
}

/*static*/
absl::Duration GpuPerformanceModelBase::CombineComputeAndMemoryAccessTime(
    absl::Duration compute_time, absl::Duration memory_access_time) {
  return compute_time + memory_access_time -
         std::min(compute_time, memory_access_time) * kMemoryComputeParallelism;
}

static int64_t get_safe_rank(const Shape& shape) {
  if (shape.IsArray()) {
    return shape.dimensions().size();
  }
  int64_t max_rank = 0;
  for (const auto& subshape : shape.tuple_shapes()) {
    max_rank = std::max<int64_t>(max_rank, get_safe_rank(subshape));
  }
  return max_rank;
}

namespace {
int64_t GetRegistersPerValue(const HloInstruction* instr) {
  // Input parameters cost 1.
  if (instr->opcode() == HloOpcode::kParameter) {
    return 1;
  }
  if (instr->shape().IsTuple()) {
    return 1;  // Metadata/pointer
  }

  // Broadcast of a scalar just forwards the input.
  if (instr->opcode() == HloOpcode::kBroadcast) {
    if (instr->operand(0)->shape().dimensions().size() == 0) {
      return GetRegistersPerValue(instr->operand(0));
    }
  }

  // Factor in fragmentation and local temporaries missed by HLO-level sort.
  // Using 3x factor as complex fusions tend to have high pressure, from
  // vectorization and unrolling.

  PrimitiveType type = instr->shape().element_type();
  int64_t element_size = ShapeUtil::ByteSizeOfPrimitiveType(type);
  // Assume 32-bit registers. f64 = 2, f32 = 1, pred = 1 (usually).
  size_t registers = std::max<int64_t>(1, element_size / 4);

  if (instr->shape().dimensions().size() == 0) {
    // Literal constants consume no registers.
    if (instr->opcode() == HloOpcode::kConstant) {
      return 0;
    }
    return registers;
  }

  return 3 * registers;
}

int64_t AddressingOverhead(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
      return 16;
    case HloOpcode::kGather:
    case HloOpcode::kScatter:
      return 32;
    case HloOpcode::kTranspose:
    case HloOpcode::kReverse:
      return 8;
    case HloOpcode::kSelect:
      // Select often involves complex condition and multiple values live.
      return 4;
    case HloOpcode::kDivide:
    case HloOpcode::kExp:
    case HloOpcode::kLog:
    case HloOpcode::kPower:
    case HloOpcode::kAtan2:
    case HloOpcode::kRsqrt:
      // Transcendental ops and divisions take extra registers for sequences.
      return 4;
    case HloOpcode::kConstant:
      return 2;
    default:
      return 0;
  }
}

bool ProducesRegister(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kBroadcast:
    case HloOpcode::kReverse:
    case HloOpcode::kReshape:
    case HloOpcode::kBitcast:
    case HloOpcode::kTuple:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kPad:
    case HloOpcode::kConcatenate:
      return false;
    default:
      return true;
  }
}
}  // namespace

size_t current_live_registers(
    const llvm::SmallPtrSet<const HloInstruction*, 16>& active) {
  size_t count = 0;
  for (const HloInstruction* instr : active) {
    count += GetRegistersPerValue(instr);
  }
  return count;
}

/*static*/ size_t simple_register_estimate(const HloInstruction* instr,
                                           const HloFusionAnalysis* analysis) {
  if (instr->opcode() != HloOpcode::kFusion) {
    return 1;
  }

  // A heuristic for register pressure within a fusion.
  // We start with a base count for thread indices, etc.
  int64_t base_registers = 16;

  // Calculate maximum number of live values using a simulated schedule
  // (currently just the topological sort order already present in
  // fused_instructions).

  // 1. Keep track of how many times each instruction is used, to know when its
  // value dies.
  absl::flat_hash_map<const HloInstruction*, int64_t> use_counts;

  // For each value, keep track of the operands which are required to compute
  // it. In the case of non-producing operands, we forward to their inputs.
  absl::flat_hash_map<const HloInstruction*, std::set<const HloInstruction*>>
      indirect_operands;

  // 2. Simulate execution and track live values
  llvm::SmallPtrSet<const HloInstruction*, 16> active;

  for (const HloInstruction* fused_instr : instr->fused_instructions()) {
    for (const HloInstruction* operand : fused_instr->operands()) {
      use_counts[operand]++;
      indirect_operands[fused_instr].insert(operand);
    }
    if (fused_instr->opcode() == HloOpcode::kParameter) {
      active.insert(fused_instr);
    }
  }

  size_t max_live_registers = 0;

  absl::flat_hash_map<const HloInstruction*, int64_t> indirect_users;

  for (const HloInstruction* fused_instr : instr->fused_instructions()) {
    if (fused_instr->opcode() == HloOpcode::kParameter) continue;

    if (!ProducesRegister(fused_instr)) {
      // If the instruction doesn't produce a register, any users of this value
      // actually depend on the current indirect_operands of this instruction.
      for (const auto& operand : indirect_operands[fused_instr]) {
        use_counts[operand] += use_counts[fused_instr];
      }
      use_counts[fused_instr] = 0;
      for (const auto& user : fused_instr->users()) {
        auto& indirect_of_user = indirect_operands[user];
        indirect_of_user.erase(fused_instr);
        for (auto& operand : indirect_operands[fused_instr]) {
          indirect_of_user.insert(operand);
        }
      }
    } else {
      max_live_registers = std::max(
          max_live_registers,
          current_live_registers(active) + AddressingOverhead(fused_instr));
      for (const auto& operand : indirect_operands[fused_instr]) {
        indirect_users[operand]++;
        use_counts[operand]--;
        if (use_counts[operand] == 0) {
          active.erase(operand);
        }
      }
      active.insert(fused_instr);
    }
  }

  int64_t unroll_factor = 1;
  //if (analysis != nullptr) {
  //  unroll_factor = MaxUnrollFactor(analysis);
  //}

  return base_registers + max_live_registers * unroll_factor;
}

/*static*/
absl::StatusOr<RegisterUsage>
GpuPerformanceModelBase::EstimateRegisterUsage(
    const HloInstruction* instr,
    const se::DeviceDescription* device_info) {
  auto mlir_context = CreateMlirContext();

  auto fusion_analysis = HloFusionAnalysis::Create(*instr, *device_info);
  size_t simple_estimate = simple_register_estimate(instr, nullptr);

  if (simple_estimate < 255) {
    /*
    std::ostringstream strm;
    strm << "EstimateRegisterUsage: " << instr->name()
              << " operands: " << instr->operand_count()
               << ", simple_estimate: " << simple_estimate << "\n";
    std::cerr << strm.str();
    */
    RegisterUsage result = {simple_estimate, 0};
    return result;
  }

  auto emitter = GetFusionEmitter(
      PreBufferAssignmentFusionInfo{fusion_analysis}, mlir_context.get());

  const auto* kernel_emitter = dynamic_cast<const EmitterBase*>(emitter.get());
  if (!kernel_emitter) {
    return absl::InternalError("Failed to get kernel emitter.");
  }

  auto entry_function_name = "kernel";
  const BufferAssignment* buffer_assignment = nullptr;
  const auto* fusion_instr = Cast<HloFusionInstruction>(instr);
  llvm::LLVMContext llvm_context;
  /*
  
    TF_ASSIGN_OR_RETURN(
      auto module, kernel_emitter->CreateMLIRModule(*mlir_context, *fusion_instr, entry_function_name,
                                    buffer_assignment));

  mlir::PassManager pm(mlir_context);
  pm.enableVerifier(false);
  emitters::RegisterOptimizationPasses(pm);
  AddLoopTransformationPasses(pm, *device_info, kernel_emitter->unroll_factor());
  if (EnablePDL(*fusion_instr->GetModule(), *device_info)) {
    pm.addPass(CreateInsertPDLPass());
  }
  AddLoweringPasses(pm, *device_info);

  TF_RETURN_IF_ERROR(RunPassPipeline(module.get(), *fusion_instr->GetModule(), pm,
                                     entry_function_name));
                                     */

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<llvm::Module> llvm_module,
      kernel_emitter->CreateLLVMModule(
          *mlir_context, llvm_context, *device_info, *fusion_instr,
          entry_function_name, buffer_assignment,
          /*use_diagnostic_handler=*/false));

  using namespace llvm;


  PassBuilder PB;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PB.registerModuleAnalyses(MAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  auto PM = PB.buildModuleSimplificationPipeline(OptimizationLevel::O1,
                                                 ThinOrFullLTOPhase::None);
  PM.run(*llvm_module.get(), MAM);

  auto result = EstimateRegisterUsageFromModule(llvm_module.get());

  /*
  std::ostringstream strm;
  strm << "EstimateRegisterUsage: " << instr->name()
            << " operands: " << instr->operand_count()
            << ", simple_estimate: " << simple_estimate
            << ", result: " << result.value().registers_per_thread << ", "
            << result.value().spilled_bytes_accessed << "\n";
  std::cerr << strm.str();
  */
  return result;
}

/*static*/
absl::StatusOr<RegisterUsage>
GpuPerformanceModelBase::EstimateRegisterUsageFromModule(
    const llvm::Module* module) {
  llvm::Function* func = module->getFunction("kernel");
  if (!func || func->empty()) {
    return absl::InternalError("Failed to get LLVM function.");
  }

  size_t max_live_values = 0;
  size_t spilling_bytes_accessed = 0;

  llvm::DenseMap<llvm::Value*, int> first_def;
  llvm::DenseMap<llvm::Value*, int> last_use;
  llvm::DenseMap<llvm::Value*, int64_t> bytes_accessed;
  int current_idx = 0;

  // Track interval of all instructions and arguments
  // We must iterate basic blocks in a topological order to assign sequential
  // indices. ReversePostOrderTraversal gives us a topological sort (ignoring
  // backedges).
  llvm::ReversePostOrderTraversal<llvm::Function*> rpo(func);
  for (llvm::BasicBlock* bb : rpo) {
    for (llvm::Instruction& inst : *bb) {
      int idx = current_idx++;
      first_def[&inst] = idx;
      last_use[&inst] = idx;  // Default to def dying immediately

      // Determine byte size of this instruction's type for spills
      int64_t type_bytes = 0;
      llvm::Type* ty = inst.getType();
      if (ty->isSized()) {
        type_bytes = module->getDataLayout().getTypeAllocSize(ty);
      }
      int num_uses = inst.getNumUses();
      // (1 store + N loads) * type_bytes
      if (type_bytes > 0 && num_uses > 0) {
        bytes_accessed[&inst] = (1 + num_uses) * type_bytes;
      }
    }
  }

  for (llvm::Argument& arg : func->args()) {
    first_def[&arg] = 0;
    last_use[&arg] = 0;  // Will be extended by uses

    int64_t type_bytes = 0;
    llvm::Type* ty = arg.getType();
    if (ty->isSized()) {
      type_bytes = module->getDataLayout().getTypeAllocSize(ty);
    }
    int num_uses = arg.getNumUses();
    if (type_bytes > 0 && num_uses > 0) {
      bytes_accessed[&arg] = (1 + num_uses) * type_bytes;
    }
  }

  // phi = phi %a
  // ...
  //. block:
  //.   %a = foo
  //.   ... [no other users of %a]
  //.   br label %phi_block

  for (llvm::BasicBlock* bb : rpo) {
    for (llvm::Instruction& inst : *bb) {
      for (llvm::Use& u : inst.operands()) {
        llvm::Value* operand = u.get();
        if (first_def.count(operand)) {
          last_use[operand] =
              std::max(last_use[operand], first_def[u.getUser()]);
          if (auto PN = dyn_cast<llvm::PHINode>(&inst)) {
            for (auto bb : PN->blocks()) {
              last_use[operand] =
                  std::max(last_use[operand], first_def[bb->getTerminator()]);
            }
          }
        }
      }
    }
  }

  // Greedy register allocator
  std::vector<llvm::Value*> active;
  max_live_values = 0;
  spilling_bytes_accessed = 0;

  // Values sorted by first_def
  std::vector<llvm::Value*> values;
  for (auto& pair : first_def) {
    if (!pair.first->getType()->isVoidTy()) {
      values.push_back(pair.first);
    }
  }
  std::sort(values.begin(), values.end(), [&](llvm::Value* a, llvm::Value* b) {
    return first_def[a] < first_def[b];
  });

  const int kMaxActiveValues = 255;
  int val_idx = 0;
  for (int i = 0; i < current_idx; ++i) {
    // Expire values whose last use < i
    active.erase(
        std::remove_if(active.begin(), active.end(),
                       [&](llvm::Value* v) { return last_use[v] < i; }),
        active.end());

    // Add values defined at i
    while (val_idx < values.size() && first_def[values[val_idx]] == i) {
      llvm::Value* v = values[val_idx++];
      size_t num_registers = 1;

      if (module->getDataLayout().getTypeAllocSize(v->getType()) > 0) {
        num_registers =
            module->getDataLayout().getTypeAllocSize(v->getType()) / 4;
      }

      for (int j = 0; j < num_registers; ++j) {
        if (active.size() < kMaxActiveValues) {
          active.push_back(v);
        } else {
          // Spill the one with the furthest last_use
          auto furthest_it =
              std::max_element(active.begin(), active.end(),
                               [&](llvm::Value* a, llvm::Value* b) {
                                 return last_use[a] < last_use[b];
                               });

          if (last_use[*furthest_it] > last_use[v]) {
            // Spill furthest
            llvm::Value* spilled = *furthest_it;
            if (bytes_accessed.count(spilled)) {
              spilling_bytes_accessed += bytes_accessed[spilled];
            }
            *furthest_it = v;
          } else {
            // Spill current
            if (bytes_accessed.count(v)) {
              spilling_bytes_accessed += bytes_accessed[v];
            }
          }
        }
      }
      max_live_values = std::max<int64_t>(max_live_values, active.size());
    }
  }

  return RegisterUsage{
      static_cast<size_t>(std::min<int64_t>(max_live_values, 255)),
      spilling_bytes_accessed};
}

/*static*/
absl::Duration GpuPerformanceModelBase::CalculateSpillPenalty(
    const se::DeviceDescription& gpu_device_info,
    const RegisterUsage& register_usage, int64_t num_blocks,
    int64_t num_threads_per_block) {
  if (register_usage.spilled_bytes_accessed == 0 &&
      register_usage.registers_per_thread <= 255) {
    // Check occupancy based penalty even if no spills.
    int64_t registers_per_block =
        register_usage.registers_per_thread * num_threads_per_block;
    if (registers_per_block <= gpu_device_info.registers_per_block_limit()) {
      return absl::ZeroDuration();
    }
    return GpuPerformanceModelBase::kSpillBasePenalty;
  }

  int64_t total_threads = num_blocks * num_threads_per_block;
  int64_t total_spill_bytes =
      register_usage.spilled_bytes_accessed * total_threads;

  absl::Duration memory_access_latency = ReadTimeWithDRAMHeuristic(
      gpu_device_info, num_blocks, /*n_bytes_net=*/total_spill_bytes,
      /*n_bytes_total=*/total_spill_bytes, PrimitiveType::F32,
      /*hbm_bandwidth_utilization_rate=*/1.0);

  return GpuPerformanceModelBase::kSpillBasePenalty + memory_access_latency;
}

/*static*/
void GpuPerformanceModelBase::VLogOperandRead(const HloInstruction* operand,
                                              int64_t n_bytes_total,
                                              int64_t n_bytes_net,
                                              bool coalesced) {
  VLOG(8) << "operand " << operand->name()
          << ", n_bytes_total: " << n_bytes_total
          << ", n_bytes_net: " << n_bytes_net << ", coalesced: " << coalesced;
}

double GetCoalescingUtilizationRate(
    PrimitiveType element_type, const se::DeviceDescription& gpu_device_info,
    bool coalesced) {
  int64_t element_size_bytes =
      element_type == PrimitiveType::TUPLE ||
              element_type == PrimitiveType::TOKEN
          ? 4 /* Dummy value. TODO(jreiffers): Model this case. */
          : ShapeUtil::ByteSizeOfPrimitiveType(element_type);
  // Assume we use one element from the cache line and waste the remaining
  // bandwidth. For example, if we're reading f32s, we use 1/16nd of the cache
  // line.
  return coalesced ? 1.0
                   : 1.0 * element_size_bytes /
                         gpu_device_info.dram_to_l2_transaction_size_bytes();
}

}  // namespace gpu
}  // namespace xla
