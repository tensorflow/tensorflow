/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/hlo_fusion_analysis.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/kernel_mapping_scheme.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/union_find.h"
#include "tsl/platform/macros.h"

namespace xla {
namespace gpu {
namespace {

const auto kDimX = TilingScheme::DimX;
const auto kLinearIndexingX = TilingScheme::LinearIndexingX;
const auto kStridedIndexingX = TilingScheme::StridedIndexingX;

// Returns true if `instr` is a non-strided slice.
bool IsSliceWithUnitStrides(const HloInstruction* instr) {
  auto slice = DynCast<HloSliceInstruction>(instr);
  return slice && absl::c_all_of(slice->slice_strides(),
                                 [](int64_t stride) { return stride == 1; });
}

// Returns true if the fusion output contains non-strided slices only.
bool IsInputFusibleNonStridedSlices(
    const std::vector<const HloInstruction*>& fusion_roots) {
  return absl::c_all_of(fusion_roots, IsSliceWithUnitStrides);
}

// Returns true if all slice inputs in a tuple are equal (ignoring type).
bool AllSliceInputsAreCompatible(
    const std::vector<const HloInstruction*>& fusion_roots) {
  const Shape& first_slice_operand_shape = fusion_roots[0]->operand(0)->shape();
  return absl::c_all_of(fusion_roots, [&](const HloInstruction* slice) {
    return ShapeUtil::EqualIgnoringElementType(slice->operand(0)->shape(),
                                               first_slice_operand_shape);
  });
}

bool MayPreventVectorization(
    const std::vector<const HloInstruction*>& fusion_roots,
    const FusionBoundaryFn& fusion_boundary_fn) {
  // An empirically chosen constant: unrolling concat with a large amount of
  // arguments causes excessive register spilling.
  static constexpr int kMaxConcatArgumentsForUnrolling = 10;
  return HloAnyOf(
      fusion_roots, fusion_boundary_fn, [&](const HloInstruction& node) {
        switch (node.opcode()) {
          case HloOpcode::kReduceWindow:
          case HloOpcode::kSort:
          case HloOpcode::kDot:
          case HloOpcode::kSin:
          case HloOpcode::kCos:
          case HloOpcode::kTan:
          case HloOpcode::kPower:
          case HloOpcode::kAtan2:
            return true;
          case HloOpcode::kConcatenate:
            return node.operand_count() > kMaxConcatArgumentsForUnrolling;
          case HloOpcode::kReduce:
            return node.shape().tuple_shapes_size() > 1;
          default:
            return false;
        }
      });
}

// Determines if we enable the row optimized codegen. When we have a fusion with
// only point-wise operations, scalar broadcasting and row broadcasting, we can
// trigger a kernel that vectorizes the row loads. This speeds up the kernel, in
// particular on A100. The int is the number of inputs with rank `out_rank`. Its
// value is only defined if row vectorization is enabled.
std::pair<bool /*enabled*/, int> RowVectorizationEnabled(
    const std::vector<const HloInstruction*>& fusion_roots, int64_t out_rank) {
  const auto is_row_major = [](const HloInstruction* instr) {
    // Only tested when the inputs are row-major. So only enable that case.
    // Maybe it would work if only the inner dimensions is contiguous.
    return LayoutUtil::IsMonotonicWithDim0Major(instr->shape().layout());
  };
  bool row_vectorized = fusion_roots.size() == 1 &&
                        !fusion_roots[0]->shape().IsTuple() &&
                        is_row_major(fusion_roots[0]);
  if (!row_vectorized) {
    return {false, 0};
  }

  // Check that the operations in the fusion are supported.  Each
  // supported operation (or category) must be manually vetted as XLA
  // only unrolls and relies on LLVM to vectorize. But this is brittle.
  // Currently tested and supported operations:
  // Elementwise, scalar and row broadcasting.
  //
  // We also detect at the same time if there is a row broadcasting
  // operation.
  int num_big_inputs = 0;
  bool some_row_broadcasting = false;
  HloBfsConsumersFirstTraversal(
      {fusion_roots.front()},
      [&](const HloInstruction& producer, const HloInstruction& consumer) {
        return consumer.opcode() == HloOpcode::kParameter;
      },
      [&](const HloInstruction& node) -> TraversalResult {
        if (node.IsElementwise()) {
          return TraversalResult::kVisitOperands;
        }

        switch (node.opcode()) {
          case HloOpcode::kConstant:
            return TraversalResult::kDoNotVisitOperands;
          case HloOpcode::kParameter:
            if (node.shape().rank() == out_rank) {
              ++num_big_inputs;
            }
            // TODO(jreiffers): When extending this to work with unfused HLO,
            // move this check to the boundary function.
            if (!is_row_major(&node)) {
              row_vectorized = false;
              return TraversalResult::kAbortTraversal;
            }
            return TraversalResult::kVisitOperands;
          case HloOpcode::kBroadcast:
            if (node.dimensions().empty()) {
              return TraversalResult::kVisitOperands;
            }

            if (node.dimensions().size() == 1 &&
                node.dimensions().front() == node.shape().rank() - 1) {
              some_row_broadcasting = true;
              return TraversalResult::kVisitOperands;
            }
            TF_FALLTHROUGH_INTENDED;
          default:
            VLOG(2) << "Row vectorization not enabled due to: "
                    << node.ToString();
            row_vectorized = false;
            return TraversalResult::kAbortTraversal;
        }
      });
  // Trigger only when there is a row broadcasting.
  return std::make_pair(row_vectorized && some_row_broadcasting,
                        num_big_inputs);
}

// Computes the maximum valid unroll factor for a given instruction.
int ComputeMaxUnrollFactor(int64_t num_elements) {
  constexpr int kMaxUnrollFactor = 4;
  for (int i = kMaxUnrollFactor; i > 1; i /= 2) {
    if (num_elements % i == 0) {
      return i;
    }
  }
  return 1;
}

// For a row reduction, returns the number of rows we can process in parallel
// per warp.
int RowReductionGetRowsPerWarp(int reduced_dimension_size) {
  if (WarpSize() % reduced_dimension_size != 0 ||
      reduced_dimension_size >= WarpSize()) {
    return 1;
  }
  return WarpSize() / reduced_dimension_size;
}

int64_t NearestPowerOfTwo(int64_t v) {
  if (v < 0) {
    return 0;
  }
  int64_t upper = absl::bit_ceil<uint64_t>(v);
  int64_t lower = upper >> 1;
  return upper - v < v - lower ? upper : lower;
}

// Returns a description of a transpose hero, that is compatible with all roots.
//
// A root is compatible with the transpose hero if:
//   * Either the root has a traspose hero with the same normalized dimensions
//   * Or the root output shape is equal to the the transpose input shape
std::optional<TransposeDescription> FindConsistentTransposeHero(
    const std::vector<const HloInstruction*>& hlo_roots,
    const std::vector<const HloInstruction*>& heroes) {
  std::optional<TransposeDescription> tiled_transpose_hero;
  std::vector<const HloInstruction*> non_transpose_roots;

  for (auto [root, hero] : llvm::zip(hlo_roots, heroes)) {
    if (auto tr = GetDescriptionForTiledTransposeEmitter(*root, *hero)) {
      if (!tiled_transpose_hero) {
        // First transpose hero found.
        tiled_transpose_hero = tr;
      } else if (!tiled_transpose_hero->IsEquivalent(*tr)) {
        // Transpose heroes have different shape.
        return std::nullopt;
      }
    } else {
      non_transpose_roots.push_back(root);
    }
  }

  if (!tiled_transpose_hero) return std::nullopt;

  for (auto* root : non_transpose_roots) {
    // Roots that don't have a transpose hero, should have a shape compatible
    // with the transpose input.
    if (!ShapeUtil::IsReshapeOrTransposeBitcast(
            root->shape(), tiled_transpose_hero->input_shape(),
            /*ignore_element_type=*/true)) {
      return std::nullopt;
    }
  }

  return tiled_transpose_hero;
}

}  // namespace

// static
StatusOr<HloFusionAnalysis> HloFusionAnalysis::Create(
    FusionBackendConfig backend_config,
    std::vector<const HloInstruction*> hlo_roots, FusionBoundaryFn boundary_fn,
    const se::DeviceDescription* device_info) {
  std::vector<const HloInstruction*> heroes;
  heroes.reserve(hlo_roots.size());
  for (auto* root : hlo_roots) {
    heroes.push_back(&FindNonTrivialHero(*root, boundary_fn));
  }

  std::vector<const HloInstruction*> fusion_arguments;
  FindFusionArguments(hlo_roots, boundary_fn,
                      [&](const HloInstruction& argument) {
                        fusion_arguments.push_back(&argument);
                      });

  auto is_4bit = [](const HloInstruction* arg) {
    return primitive_util::Is4BitType(arg->shape().element_type());
  };
  bool has_4_bit_input = absl::c_any_of(fusion_arguments, is_4bit);
  bool has_4_bit_output = absl::c_any_of(hlo_roots, is_4bit);

  std::optional<TransposeDescription> tiled_transpose_hero =
      FindConsistentTransposeHero(hlo_roots, heroes);

  return HloFusionAnalysis(std::move(backend_config), std::move(hlo_roots),
                           std::move(boundary_fn), std::move(fusion_arguments),
                           std::move(heroes), device_info, tiled_transpose_hero,
                           has_4_bit_input, has_4_bit_output);
}

// static
StatusOr<HloFusionAnalysis> HloFusionAnalysis::Create(
    const HloFusionInstruction* fusion,
    const se::DeviceDescription* device_info) {
  CHECK(device_info != nullptr);
  TF_ASSIGN_OR_RETURN(auto backend_config,
                      fusion->backend_config<FusionBackendConfig>());

  auto hlo_roots = GetFusionRoots(*fusion->fused_instructions_computation());
  return Create(std::move(backend_config), std::move(hlo_roots),
                DefaultFusionBoundaryFn, device_info);
}

// Returns true if the fusion has consistent transpose heros.
bool HloFusionAnalysis::HasConsistentTransposeHeros() const {
  return tiled_transpose_.has_value();
}

HloFusionAnalysis::EmitterFusionKind HloFusionAnalysis::GetEmitterFusionKind()
    const {
#if GOOGLE_CUDA
  if (fusion_backend_config_.kind() == kTritonGemmFusionKind ||
      fusion_backend_config_.kind() == kTritonSoftmaxFusionKind) {
    return EmitterFusionKind::kTriton;
  }
#endif

  if (has_4_bit_input_ || has_4_bit_output_) {
    // Only loop fusions currently can handle int4 inputs/outputs, due to the
    // special handling with IrArray needed to deal with two values occupying a
    // single byte.
    return EmitterFusionKind::kLoop;
  }

  for (auto [root, hero] : llvm::zip(fusion_roots_, fusion_heroes_)) {
    if (IsRealReductionHero(*root, *hero)) {
      return EmitterFusionKind::kReduction;
    }
  }

  // We expect that the last dimension is swapped with a different dimension.
  if (HasConsistentTransposeHeros() && tiled_transpose_->permutation[2] != 2) {
    return EmitterFusionKind::kTranspose;
  }

  if (fusion_roots_.size() > 1) {
    if (IsInputFusibleNonStridedSlices(fusion_roots_) &&
        AllSliceInputsAreCompatible(fusion_roots_)) {
      return EmitterFusionKind::kInputSlices;
    }
    return EmitterFusionKind::kLoop;
  }

  if (fusion_roots_[0]->opcode() == HloOpcode::kScatter) {
    return EmitterFusionKind::kScatter;
  }

  return EmitterFusionKind::kLoop;
}

StatusOr<LaunchDimensions> HloFusionAnalysis::GetLaunchDimensions() {
  auto emitter_fusion_kind = GetEmitterFusionKind();
  switch (emitter_fusion_kind) {
    case EmitterFusionKind::kLoop: {
      // Disable experimental block size if few_waves or row_vectorized enabled.
      auto loop_fusion_config = GetLoopFusionConfig();
      return CalculateLaunchDimensions(GetElementShape(), *device_info_,
                                       *loop_fusion_config);
    }
    case EmitterFusionKind::kReduction: {
      auto* reduction_codegen_info = GetReductionCodegenInfo();
      const TilingScheme& tiling_scheme =
          reduction_codegen_info->GetTilingScheme();
      size_t blocks_y = reduction_codegen_info->GetIndexGroups().size();
      return LaunchDimensions(
          {/*x=*/tiling_scheme.GetNumberOfBlocksPhysical(),
           /*y=*/static_cast<int64_t>(blocks_y), /*z=*/1},
          {/*x=*/tiling_scheme.GetNumThreadsPerBlockPhysical(),
           /*y=*/1, /*z=*/1});
    }
    case EmitterFusionKind::kTranspose: {
      auto* tiling_scheme = GetTransposeTilingScheme();
      return LaunchDimensions(tiling_scheme->GetNumberOfBlocksPhysical(),
                              tiling_scheme->GetNumThreadsPerBlockPhysical());
    }
    case EmitterFusionKind::kInputSlices: {
      auto* root = fusion_roots().front();
      const auto& shape = root->operands()[0]->shape();
      constexpr int kUnrollFactor = 1;
      return CalculateLaunchDimensions(shape, *device_info_, {kUnrollFactor});
    }
    case EmitterFusionKind::kScatter: {
      const auto& root_shape = fusion_roots().front()->shape();
      int64_t num_elements = ShapeUtil::ElementsIn(root_shape);
      int unroll_factor = num_elements % 4 == 0   ? 4
                          : num_elements % 2 == 0 ? 2
                                                  : 1;
      return CalculateLaunchDimensions(root_shape, *device_info_,
                                       {unroll_factor, /*few_waves=*/false});
    }
    case EmitterFusionKind::kTriton:
      return Unimplemented("GetLaunchDimensions");
  }
}

const HloInstruction* HloFusionAnalysis::FindHeroReduction() const {
  CHECK(GetEmitterFusionKind() == EmitterFusionKind::kReduction);
  auto roots = fusion_roots();
  CHECK(!roots.empty());
  // We always use the first reduce root that triggers unnested reduction
  // emitter as the hero reduction, since all the reductions are required to
  // have the same shape and layout as verified by
  // `IsFusedReductionOutputConsistent()`.
  for (auto [root, hero] : llvm::zip(roots, fusion_heroes_)) {
    if (IsRealReductionHero(*root, *hero)) {
      return hero;
    }
  }
  LOG(FATAL) << "Did not find a hero reduction";
}

const ReductionCodegenInfo* HloFusionAnalysis::GetReductionCodegenInfo() {
  if (reduction_codegen_info_.has_value()) {
    return &reduction_codegen_info_.value();
  }

  const HloInstruction* hero_reduction = FindHeroReduction();

  auto reduction_codegen_info = ComputeReductionCodegenInfo(hero_reduction);
  reduction_codegen_info_.emplace(std::move(reduction_codegen_info));
  return &reduction_codegen_info_.value();
}

const TilingScheme* HloFusionAnalysis::GetTransposeTilingScheme() {
  if (transpose_tiling_scheme_.has_value()) {
    return &transpose_tiling_scheme_.value();
  }

  if (!tiled_transpose_) {
    return nullptr;
  }

  constexpr int kNumRows = 4;
  static_assert(WarpSize() % kNumRows == 0);

  // 3D view over the input shape.
  Vector3 dims = tiled_transpose_->dimensions;
  Vector3 order = tiled_transpose_->permutation;

  Vector3 permuted_dims = {dims[order[0]], dims[order[1]], dims[order[2]]};
  Vector3 tile_sizes{1, 1, 1};
  tile_sizes[order[2]] = WarpSize() / kNumRows;
  Vector3 num_threads{1, 1, WarpSize()};
  num_threads[order[2]] = kNumRows;

  TilingScheme tiling_scheme(
      /*permuted_dims*/ permuted_dims,
      /*tile_sizes=*/tile_sizes,
      /*num_threads=*/num_threads,
      /*indexing_order=*/kLinearIndexingX,
      /*vector_size=*/1,
      /*scaling_factor=*/1,
      /*tiling_dimensions=*/{order[2], 2});
  transpose_tiling_scheme_.emplace(std::move(tiling_scheme));
  return &transpose_tiling_scheme_.value();
}

const LaunchDimensionsConfig* HloFusionAnalysis::GetLoopFusionConfig() {
  if (loop_fusion_config_.has_value()) {
    return &loop_fusion_config_.value();
  }

  int unroll_factor = 1;
  // Unrolling is good to read large inputs with small elements
  // due to vector loads, but increases the register pressure when one
  // thread has to produce multiple output elements.
  // Therefore for fusions with small outputs prefer to use one thread
  // per output element = no unroll.
  // Call 'small' fusions that use less threads than the GPU has.
  int64_t num_elements = ShapeUtil::ElementsIn(GetElementShape());
  int64_t n_threads_max =
      device_info_->threads_per_core_limit() * device_info_->core_count();
  if (num_elements >= n_threads_max &&
      !MayPreventVectorization(fusion_roots_, fusion_boundary_fn_)) {
    unroll_factor = ComputeMaxUnrollFactor(num_elements);
  }
  if (has_4_bit_output_ && unroll_factor == 1) {
    // Ensure a single thread writes to a byte containing two int4 values. The
    // HLO Verifier ensures each int4 array has an even number of elements so
    // it's safe to set the unroll_factor to 2. Setting unroll_factor is safe
    // even if MayPreventVectorization returns false, as the
    // MayPreventVectorization check is an optimization, not a correctness
    // requirement.
    CHECK_EQ(num_elements % 2, 0);
    unroll_factor = 2;
  }
  VLOG(2) << "Unroll factor: " << unroll_factor;

  if (GetEmitterFusionKind() == EmitterFusionKind::kScatter) {
    // Only the unroll factor is used for scatter.
    loop_fusion_config_.emplace(LaunchDimensionsConfig{unroll_factor});
    return &loop_fusion_config_.value();
  }

  bool row_vectorized;
  int num_big_inputs;
  std::tie(row_vectorized, num_big_inputs) =
      RowVectorizationEnabled(fusion_roots(), GetElementShape().rank());
  bool few_waves = !HloAnyOf(
      fusion_roots_, fusion_boundary_fn_, [&](const HloInstruction& instr) {
        if (instr.opcode() == HloOpcode::kParameter ||
            instr.opcode() == HloOpcode::kConstant ||
            HloInstruction::IsOpElementwise(instr.opcode())) {
          return false;
        }
        if (auto broadcast = DynCast<HloBroadcastInstruction>(&instr)) {
          if (broadcast->dimensions().empty() ||
              // More than 3 big inputs cause a speed regression.
              (row_vectorized && num_big_inputs <= 3)) {
            return false;
          }
        }
        VLOG(2) << "few_waves not enabled due to: " << instr.ToString();
        return true;
      });

  LaunchDimensionsConfig launch_config{unroll_factor, few_waves,
                                       row_vectorized};
  // Check that the shapes is supported.
  if (launch_config.row_vectorized &&
      ThreadsPerBlockRowVectorized(GetElementShape(), *device_info_,
                                   launch_config) <= 0) {
    VLOG(2) << "Cancelling row_vectorization as the shape isn't supported.";
    launch_config.row_vectorized = false;
    launch_config.few_waves = false;
  }
  loop_fusion_config_.emplace(std::move(launch_config));
  return &loop_fusion_config_.value();
}

const Shape& HloFusionAnalysis::GetElementShape() const {
  const Shape* shape = &fusion_roots_.front()->shape();
  while (shape->IsTuple()) {
    shape = &shape->tuple_shapes(0);
  }
  return *shape;
}

int HloFusionAnalysis::SmallestInputDtypeBits() const {
  int bits = std::numeric_limits<int>::max();
  for (const HloInstruction* operand : fusion_arguments_) {
    bits = std::min(bits,
                    primitive_util::BitWidth(operand->shape().element_type()));
  }
  return bits;
}

int64_t HloFusionAnalysis::MaxBeneficialColumnReductionUnrollBasedOnBlockSize()
    const {
  int64_t num_reduce_output_elems = 0;
  for (const HloInstruction* root : fusion_roots()) {
    if (!IsReductionFromOrToContiguousDimensions(*root)) {
      continue;
    }
    const Shape* output_shape = &root->shape();
    // Unwrap multi-output reduction.  All outputs should be the same shape.
    if (output_shape->IsTuple()) {
      output_shape = &output_shape->tuple_shapes()[0];
    }
    num_reduce_output_elems =
        std::max(num_reduce_output_elems, ShapeUtil::ElementsIn(*output_shape));
  }

  // A column reduction that's unrolled N times uses one warp to generate N
  // output elements.  The block size is always 32 warps = 1024 threads.
  int64_t num_blocks = CeilOfRatio(num_reduce_output_elems, int64_t{32});
  int64_t num_threads = num_blocks * 1024;
  // Number of SMs we can saturate with this work.
  int num_cores =
      CeilOfRatio<int64_t>(num_threads, device_info_->threads_per_core_limit());
  return static_cast<int>(CeilOfRatio(num_cores, device_info_->core_count()));
}

// Divides `num_reduces` reduces into groups. Different groups will be executed
// in parallel. Generally speaking, we'd like to run the reduce instructions
// in parallel without incurring too much recomputation overhead. The current
// heuristic is to place reduce instructions who share nothing or only
// (broadcasted) scalars/constants into different groups; otherwise, they are
// placed in the same group. Non-reduce instructions always go with the reduce
// instructions into the same group so long as they share any predecessors.
std::vector<std::vector<const HloInstruction*>>
HloFusionAnalysis::GroupDisjointReductions() const {
  const int num_fusion_outputs = fusion_roots().size();

  CHECK_NE(0, num_fusion_outputs);
  if (num_fusion_outputs == 1) {
    return {{fusion_roots()[0]}};
  }

  ConstHloInstructionMap<tensorflow::UnionFind<const HloInstruction*>>
      disjoint_sets;

  // TODO(b/249976438): we currently do not treat properly
  // aliasing between inputs and outputs of the fusion, so for now put all
  // non-reduction roots into one group to avoid read-after-write conflicts.
  const HloInstruction* first_non_reduction_root = nullptr;

  ConstHloInstructionMap<absl::flat_hash_set<const HloInstruction*>>
      reachable_outputs;
  absl::flat_hash_set<const HloInstruction*> roots_with_reduction;
  for (auto [root, hero] : llvm::zip(fusion_roots(), fusion_heroes_)) {
    disjoint_sets[root].Get() = root;
    reachable_outputs[root].insert(root);
    if (IsRealReductionHero(*root, *hero)) {
      roots_with_reduction.insert(root);
    } else if (first_non_reduction_root) {
      disjoint_sets[first_non_reduction_root].Merge(&disjoint_sets[root]);
    } else {
      first_non_reduction_root = root;
    }
  }

  std::vector<const HloInstruction*> instructions;
  HloBfsConsumersFirstTraversal(
      fusion_roots_,
      [&](const HloInstruction& producer, const HloInstruction& consumer) {
        auto& producer_reachable = reachable_outputs[&producer];
        for (auto* instruction : reachable_outputs[&consumer]) {
          producer_reachable.insert(instruction);
        }
        return fusion_boundary_fn_(producer, consumer);
      },
      [&](const HloInstruction& node) {
        instructions.push_back(&node);
        return TraversalResult::kVisitOperands;
      });

  for (const HloInstruction* instr : instructions) {
    const auto& reachable = reachable_outputs[instr];
    std::vector<const HloInstruction*> reached_output_ids;
    bool added_to_reduce = false;
    for (const HloInstruction* output : fusion_roots()) {
      bool has_real_hero = roots_with_reduction.contains(output);
      if (has_real_hero && (hlo_query::IsBroadcastedConstantOrScalar(*instr))) {
        if (added_to_reduce) {
          // Do not group more than one output reduce instructions through
          // broadcasted constants or scalars, as the recomputation should be
          // acceptable.
          VLOG(3) << "Skip broadcasted constant or scalar "
                  << instr->ToString();
          continue;
        }
      }
      // Now group output instructions if they have common predecessors.
      if (reachable.contains(output)) {
        VLOG(3) << "Reaching " << output->ToString() << " from "
                << instr->ToString();
        reached_output_ids.push_back(output);
        if (has_real_hero) {
          added_to_reduce = true;
        }
      }
    }
    for (size_t j = 1; j < reached_output_ids.size(); ++j) {
      disjoint_sets[reached_output_ids[0]].Merge(
          &disjoint_sets[reached_output_ids[j]]);
    }
  }

  // Place output instructions in the same set into the same group.
  ConstHloInstructionMap<std::vector<const HloInstruction*>> groups;
  for (const HloInstruction* root : fusion_roots()) {
    groups[disjoint_sets[root].Get()].push_back(root);
  }

  std::vector<std::vector<const HloInstruction*>> ret;
  ret.reserve(groups.size());
  absl::c_for_each(
      groups, [&](auto& iter) { ret.emplace_back(std::move(iter.second)); });
  return ret;
}

bool HloFusionAnalysis::IsUnrollingColumnReductionBeneficial(
    const Shape& input_shape, int64_t num_kept_minor,
    bool reduction_is_race_free) const {
  if (num_kept_minor % (WarpSize() * 2) != 0) {
    return false;
  }
  if (input_shape.dimensions(input_shape.rank() - 1) < 64) {
    return false;
  }

  int64_t can_be_vectorized = 0;
  int64_t cannot_be_vectorized = 0;
  absl::flat_hash_set<const HloInstruction*> use_chain_endings;

  for (const HloInstruction* fusion_root : fusion_roots()) {
    if (!reduction_is_race_free &&
        IsReductionFromOrToContiguousDimensions(*fusion_root)) {
      // Atomics cannot be vectorized.
      cannot_be_vectorized++;
    } else {
      can_be_vectorized++;
    }
    use_chain_endings.insert(fusion_root);
  }

  // Fusion inputs that have the same dimension as the reduce input and
  // only involve in element-wise operations can be vectorized.
  absl::flat_hash_set<const HloInstruction*> reachable_through_non_elementwise;
  HloBfsConsumersFirstTraversal(
      fusion_roots_,
      [&](const HloInstruction& producer, const HloInstruction& consumer) {
        // We check if the consumer is elementwise, unless this edge is a
        // virtual edge that only exists in partially fused HLO. There are two
        // types of such edges:
        // 1. Edges from producers outside a fusion to a parameter instruction
        //    within a fusion. Here, the producer is a parameter of the fusion
        //    instruction.
        // 2. Edges from fusion roots to fusion nodes.
        if (reachable_through_non_elementwise.contains(&consumer) ||
            (!(consumer.opcode() == HloOpcode::kParameter ||
               consumer.opcode() == HloOpcode::kFusion ||
               consumer.IsElementwise()) &&
             !use_chain_endings.contains(&consumer))) {
          reachable_through_non_elementwise.insert(&producer);
        }

        return fusion_boundary_fn_(producer, consumer);
      },
      [&](const HloInstruction& node) {
        return TraversalResult::kVisitOperands;
      });

  for (auto* argument : fusion_arguments_) {
    if (!reachable_through_non_elementwise.contains(argument) &&
        ShapeUtil::SameDimensions(input_shape, argument->shape())) {
      ++can_be_vectorized;
    }
  }

  // Fusion inputs with more elements than the reduce op input must participate
  // in non-elementwise operations and we assume that they are not vectorizable
  // for the purpose of estimating the benefit of unrolling. If the kernel is
  // unrolled even with such an assumption,  and the accesses to those inputs
  // turn out to be vectorizable, the compiler will still vectorize them.
  int64_t num_elements = ShapeUtil::ElementsIn(input_shape);
  cannot_be_vectorized +=
      absl::c_count_if(fusion_arguments_, [&](const HloInstruction* parameter) {
        return ShapeUtil::ElementsIn(parameter->shape()) > num_elements;
      });
  if (can_be_vectorized < cannot_be_vectorized) {
    return false;
  }

  return MaxBeneficialColumnReductionUnrollBasedOnBlockSize() > 1;
}

bool HloFusionAnalysis::CanVectorizeReduction(
    const ReductionDimensions& reduction_dimensions, int num_threads_x,
    Vector3 reduction_tiling, const Shape& input_shape,
    bool reduction_is_race_free) const {
  if (!reduction_dimensions.is_row_reduction) {
    return IsUnrollingColumnReductionBeneficial(
        input_shape, reduction_dimensions.dimensions[kDimX],
        reduction_is_race_free);
  }

  if (reduction_dimensions.dimensions[kDimX] % 2 != 0 ||
      MayPreventVectorization(fusion_roots_, fusion_boundary_fn_)) {
    return false;
  }

  // Enabling vectorization if number of threads is <= warpsize leads to half or
  // more of the threads not doing any work.
  if (reduction_dimensions.is_row_reduction && num_threads_x <= WarpSize()) {
    return false;
  }

  const auto* cuda_cc = std::get_if<se::CudaComputeCapability>(
      &device_info_->gpu_compute_capability());
  if (cuda_cc == nullptr) return false;
  if (cuda_cc->IsAtLeast(se::CudaComputeCapability::VOLTA)) return true;
  if (cuda_cc->IsAtLeast(se::CudaComputeCapability::PASCAL_)) {
    return SmallestInputDtypeBits() <= 32 &&
           reduction_dimensions.dimensions[kDimX] %
                   (reduction_tiling[2] * num_threads_x) ==
               0;
  }
  return false;
}

int HloFusionAnalysis::CalculateVirtualThreadScalingFactorForReduction(
    const ReductionDimensions& reduction_dimensions) const {
  int64_t dimx = reduction_dimensions.dimensions[kDimX];
  if (reduction_dimensions.is_row_reduction && dimx <= 128) {
    int rows_per_warp = RowReductionGetRowsPerWarp(dimx);
    const auto* cuda_cc = std::get_if<se::CudaComputeCapability>(
        &device_info_->gpu_compute_capability());
    if (cuda_cc != nullptr &&
        cuda_cc->IsAtLeast(se::CudaComputeCapability::AMPERE)) {
      return rows_per_warp * 3;
    }
    return rows_per_warp * 5;
  }
  return 1;
}

ReductionCodegenInfo HloFusionAnalysis::ComputeReductionCodegenInfo(
    const HloInstruction* hero_reduction) const {
  Shape input_shape = hero_reduction->operand(0)->shape();
  ReductionDimensions reduction_dimensions =
      GetReductionKindAndContiguousComponents(*hero_reduction);
  VLOG(10) << "is_row_reduction " << reduction_dimensions.is_row_reduction
           << " " << reduction_dimensions.dimensions[0] << " "
           << reduction_dimensions.dimensions[1] << " "
           << reduction_dimensions.dimensions[2];
  Vector3 reduction_tiling = GetReductionTiling(reduction_dimensions);

  int64_t fan_out = fusion_roots().size();
  int64_t num_threads_y =
      reduction_dimensions.is_row_reduction ? 1 : WarpSize();
  int64_t num_threads_x = [&] {
    if (reduction_dimensions.is_row_reduction) {
      if (RowReductionGetRowsPerWarp(reduction_dimensions.dimensions[2]) > 1) {
        return reduction_dimensions.dimensions[2];
      }
      // Use 512 as default block size (threads per block) for row reductions.
      // For multi-output fusions, reduce the block size further to decrease
      // register pressure when multiple outputs are computed by each thread.
      int64_t max_block_size = std::max(
          MinThreadsXRowReduction(hero_reduction->GetModule()->config()),
          static_cast<int64_t>(512LL / NearestPowerOfTwo(fan_out)));
      return std::min(max_block_size,
                      RoundUpTo(CeilOfRatio(reduction_dimensions.dimensions[2],
                                            reduction_tiling[2]),
                                WarpSize()));
    }
    return WarpSize();
  }();

  TilingScheme::IndexingOrder indexing_order =
      reduction_dimensions.is_row_reduction ? kStridedIndexingX
                                            : kLinearIndexingX;
  auto instr_index_groups = GroupDisjointReductions();
  int64_t shmem_usage = ReductionProjectedShmemUsageBytes(reduction_dimensions,
                                                          instr_index_groups);
  const int64_t shmem_budget = device_info_->shared_memory_per_block();
  bool reduction_is_race_free = ReductionIsRaceFree(
      hero_reduction->GetModule()->config(), reduction_dimensions);
  bool vectorize =
      // Vectorization might cause us to run out of budget.
      (shmem_usage * 2 <= shmem_budget) &&
      CanVectorizeReduction(reduction_dimensions, num_threads_x,
                            reduction_tiling, input_shape,
                            reduction_is_race_free);
  int vector_size = vectorize ? 2 : 1;

  // TODO(b/283542954): Autotune num_partial_results?  This can make a big
  // difference, e.g. by affecting register spilling.
  int num_partial_results = 1;
  if (!reduction_dimensions.is_row_reduction && vectorize) {
    int smallest_input_dtype_bits = SmallestInputDtypeBits();
    if (smallest_input_dtype_bits <= 32) {
      // Make sure to use all the data read at once.
      // Instead of hardcoding the granularity, we can query the granularity we
      // need like this:
      //   size_t granularity = 0;
      //   CUresult res = cuCtxGetLimit(&granularity,
      //   CU_LIMIT_MAX_L2_FETCH_GRANULARITY); // 0x05
      // But we need a context to be active. Which isn't the case here.
      num_partial_results = std::min(64 / smallest_input_dtype_bits, 8);

      // Limit register pressure for MOF, but still use a minimum of 2.
      num_partial_results /= fan_out;
      // We can't go below 2 for the unroll factor -- if we wanted to use 1 as
      // the unroll factor, we should have set this reduction as unvectorized.
      num_partial_results = std::max(num_partial_results, 2);
    } else {
      num_partial_results = 2;
    }

    while (num_partial_results != 1 &&
           shmem_usage * num_partial_results > shmem_budget) {
      num_partial_results /= 2;
    }
    reduction_tiling[kDimX] *= num_partial_results;
  }

  VLOG(3) << "Each thread will produce " << num_partial_results << " output(s)";

  Vector3 num_threads = {1, num_threads_y, num_threads_x};
  int virtual_thread_scaling_factor =
      CalculateVirtualThreadScalingFactorForReduction(reduction_dimensions);
  VLOG(2) << "Using virtual thread scaling: " << virtual_thread_scaling_factor;

  TilingScheme tiling_scheme(reduction_dimensions.dimensions, reduction_tiling,
                             num_threads, indexing_order, vector_size,
                             virtual_thread_scaling_factor);
  return ReductionCodegenInfo(
      tiling_scheme, num_partial_results, reduction_dimensions.is_row_reduction,
      reduction_is_race_free, std::move(instr_index_groups), hero_reduction);
}

}  // namespace gpu
}  // namespace xla
