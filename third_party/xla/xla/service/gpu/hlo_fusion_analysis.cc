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
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
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
#include "tsl/platform/macros.h"

namespace xla {
namespace gpu {
namespace {

const auto kDimX = TilingScheme::DimX;
const auto kLinearIndexingX = TilingScheme::LinearIndexingX;
const auto kStridedIndexingX = TilingScheme::StridedIndexingX;

std::optional<TilingScheme> ComputeTransposeTilingScheme(
    const std::optional<TransposeDescription>& tiled_transpose) {
  if (!tiled_transpose) {
    return std::nullopt;
  }

  constexpr int kNumRows = 4;
  static_assert(WarpSize() % kNumRows == 0);

  // 3D view over the input shape.
  Vector3 dims = tiled_transpose->dimensions;
  Vector3 order = tiled_transpose->permutation;

  Vector3 permuted_dims = {dims[order[0]], dims[order[1]], dims[order[2]]};
  Vector3 tile_sizes{1, 1, 1};
  tile_sizes[order[2]] = WarpSize() / kNumRows;
  Vector3 num_threads{1, 1, WarpSize()};
  num_threads[order[2]] = kNumRows;

  return TilingScheme(
      /*permuted_dims*/ permuted_dims,
      /*tile_sizes=*/tile_sizes,
      /*num_threads=*/num_threads,
      /*indexing_order=*/kLinearIndexingX,
      /*vector_size=*/1,
      /*scaling_factor=*/1,
      /*tiling_dimensions=*/{order[2], 2});
}

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

// Determines if we enable the row optimized codegen. When we have a fusion with
// only point-wise operations, scalar broadcasting and row broadcasting, we can
// trigger a kernel that vectorizes the row loads. This speeds up the kernel, in
// particular on A100. The int is the number of inputs with rank `out_rank`. Its
// value is only defined if row vectorization is enabled.
std::pair<bool /*enabled*/, int> RowVectorizationEnabled(
    const HloFusionAdaptor& fusion, int64_t out_rank) {
  auto roots = fusion.GetRoots();
  const auto is_row_major = [](auto instr) {
    // Only tested when the inputs are row-major. So only enable that case.
    // Maybe it would work if only the inner dimensions is contiguous.
    return LayoutUtil::IsMonotonicWithDim0Major(instr.shape().layout());
  };
  bool row_vectorized = roots.size() == 1 && !roots[0].shape().IsTuple() &&
                        is_row_major(roots[0]);
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
      roots, fusion,
      [&](auto node) -> TraversalResult {
        if (!row_vectorized) {
          return TraversalResult::kAbortTraversal;
        }

        if (node.instruction().IsElementwise()) {
          return TraversalResult::kVisitOperands;
        }

        switch (node.opcode()) {
          case HloOpcode::kConstant:
            return TraversalResult::kDoNotVisitOperands;
          case HloOpcode::kParameter:
            return TraversalResult::kVisitOperands;
          case HloOpcode::kBroadcast: {
            auto dims = node.instruction().dimensions();
            if (dims.empty()) {
              return TraversalResult::kVisitOperands;
            }

            if (dims.size() == 1 && dims.front() == node.shape().rank() - 1) {
              some_row_broadcasting = true;
              return TraversalResult::kVisitOperands;
            }
            TF_FALLTHROUGH_INTENDED;
          }
          default:
            VLOG(2) << "Row vectorization not enabled due to: "
                    << node.ToString();
            row_vectorized = false;
            return TraversalResult::kAbortTraversal;
        }
      },
      [&](auto argument) {
        if (argument.shape().rank() == out_rank) {
          ++num_big_inputs;
        }
        if (!is_row_major(argument)) {
          row_vectorized = false;
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

int SmallestInputDtypeBits(const std::vector<const HloInstruction*>& args) {
  int bits = std::numeric_limits<int>::max();
  for (const HloInstruction* operand : args) {
    if (!operand->shape().IsArray()) continue;
    bits = std::min(bits,
                    primitive_util::BitWidth(operand->shape().element_type()));
  }
  return bits;
}

}  // namespace

HloFusionAnalysis::HloFusionAnalysis(
    FusionBackendConfig fusion_backend_config,
    std::vector<const HloInstruction*> fusion_roots,
    std::unique_ptr<HloFusionAdaptor> fusion,
    std::vector<const HloInstruction*> fusion_heroes,
    const se::DeviceDescription* device_info,
    std::optional<TransposeDescription> tiled_transpose,
    HloFusionAnalysis::InputOutputInfo input_output_info)
    : fusion_backend_config_(std::move(fusion_backend_config)),
      fusion_roots_(std::move(fusion_roots)),
      fusion_(std::move(fusion)),
      fusion_heroes_(std::move(fusion_heroes)),
      device_info_(device_info),
      tiled_transpose_(tiled_transpose),
      input_output_info_(std::move(input_output_info)),
      transpose_tiling_scheme_(ComputeTransposeTilingScheme(tiled_transpose_)),
      loop_fusion_config_(ComputeLoopFusionConfig()) {}

// static
StatusOr<HloFusionAnalysis> HloFusionAnalysis::Create(
    FusionBackendConfig backend_config,
    std::unique_ptr<HloFusionAdaptor> fusion,
    const se::DeviceDescription* device_info) {
  std::vector<const HloInstruction*> roots;
  std::vector<const HloInstruction*> heroes;
  for (auto root : fusion->GetRoots()) {
    roots.push_back(&root.instruction());
    heroes.push_back(&FindNonTrivialHero(*roots.back(), *fusion));
  }

  std::vector<const HloInstruction*> fusion_arguments;
  FindFusionArguments(*fusion, [&](auto argument) {
    fusion_arguments.push_back(&argument.instruction());
  });

  auto is_4bit = [](const HloInstruction* arg) {
    return primitive_util::Is4BitType(arg->shape().element_type());
  };

  InputOutputInfo input_output_info{
      .has_4_bit_input = absl::c_any_of(fusion_arguments, is_4bit),
      .has_4_bit_output = absl::c_any_of(roots, is_4bit),
      .smallest_input_dtype_bits = SmallestInputDtypeBits(fusion_arguments),
  };

  std::optional<TransposeDescription> tiled_transpose_hero =
      FindConsistentTransposeHero(roots, heroes);

  return HloFusionAnalysis(std::move(backend_config), std::move(roots),
                           std::move(fusion), std::move(heroes), device_info,
                           tiled_transpose_hero, std::move(input_output_info));
}

// static
StatusOr<HloFusionAnalysis> HloFusionAnalysis::Create(
    const HloFusionInstruction* fusion,
    const se::DeviceDescription* device_info) {
  CHECK(device_info != nullptr);
  TF_ASSIGN_OR_RETURN(auto backend_config,
                      fusion->backend_config<FusionBackendConfig>());
  return Create(std::move(backend_config),
                HloFusionAdaptor::ForInstruction(fusion), device_info);
}

// Returns true if the fusion has consistent transpose heros.
bool HloFusionAnalysis::HasConsistentTransposeHeros() const {
  return tiled_transpose_.has_value();
}

HloFusionAnalysis::EmitterFusionKind HloFusionAnalysis::GetEmitterFusionKind()
    const {
  if (fusion_backend_config_.kind() == kCustomFusionKind) {
    return EmitterFusionKind::kCustomFusion;
  }

#if GOOGLE_CUDA
  if (fusion_backend_config_.kind() == kTritonGemmFusionKind ||
      fusion_backend_config_.kind() == kTritonSoftmaxFusionKind) {
    return EmitterFusionKind::kTriton;
  }
#endif

  if (input_output_info_.has_4_bit_input ||
      input_output_info_.has_4_bit_output) {
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

StatusOr<LaunchDimensions> HloFusionAnalysis::GetLaunchDimensions() const {
  auto emitter_fusion_kind = GetEmitterFusionKind();
  switch (emitter_fusion_kind) {
    case EmitterFusionKind::kLoop: {
      // Disable experimental block size if few_waves or row_vectorized enabled.
      auto loop_fusion_config = GetLoopFusionConfig();
      return CalculateLaunchDimensions(GetElementShape(), *device_info_,
                                       *loop_fusion_config);
    }
    case EmitterFusionKind::kReduction: {
      return absl::UnimplementedError(
          "GetLaunchDimensions is not implemented for reduction fusions");
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
      const auto& updates_shape = fusion_roots().front()->operand(2)->shape();
      return CalculateLaunchDimensions(updates_shape, *device_info_);
    }
    case EmitterFusionKind::kCustomFusion:
      return absl::UnimplementedError(
          "GetLaunchDimensions is not implemented for custom fusions");
    case EmitterFusionKind::kTriton:
      return absl::UnimplementedError(
          "GetLaunchDimensions is not implemented for Triton fusions");
  }
}

const HloInstruction* HloFusionAnalysis::FindHeroReduction() const {
  if (GetEmitterFusionKind() != EmitterFusionKind::kReduction) {
    return nullptr;
  }
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

std::optional<LaunchDimensionsConfig>
HloFusionAnalysis::ComputeLoopFusionConfig() const {
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
  if (num_elements >= n_threads_max && !MayPreventVectorization(*fusion_)) {
    unroll_factor = ComputeMaxUnrollFactor(num_elements);
  }
  // CHECK that unroll_factor is a power-of-2, as needed by the logic below.
  CHECK(absl::has_single_bit(static_cast<uint64_t>(unroll_factor)));
  if (input_output_info_.has_4_bit_output && unroll_factor == 1) {
    // Ensure a single thread writes to a byte containing two int4 values by
    // setting unroll_factor to 2. unroll_factor is always a power of 2, so
    // setting it to 2 here ensures unroll_factor is even when there are 4-bit
    // outputs. Setting unroll_factor is safe even if there are an odd number of
    // elements, as the parallel loop emitter will insert a bounds check in this
    // case to ensure the out-of-bounds element is not computed and written.
    // Setting unroll_factor is safe even if MayPreventVectorization returns
    // false, as the MayPreventVectorization check is an optimization, not a
    // correctness requirement.
    unroll_factor = 2;
  }
  VLOG(2) << "Unroll factor: " << unroll_factor;

  if (GetEmitterFusionKind() == EmitterFusionKind::kScatter) {
    // Only the unroll factor is used for scatter.
    return LaunchDimensionsConfig{unroll_factor};
  }

  bool row_vectorized;
  int num_big_inputs;
  std::tie(row_vectorized, num_big_inputs) =
      RowVectorizationEnabled(*fusion_, GetElementShape().rank());
  bool few_waves = !HloAnyOf(fusion_->GetRoots(), *fusion_, [&](auto instr) {
    if (instr.opcode() == HloOpcode::kParameter ||
        instr.opcode() == HloOpcode::kConstant ||
        HloInstruction::IsOpElementwise(instr.opcode())) {
      return false;
    }
    if (auto broadcast =
            DynCast<HloBroadcastInstruction>(&instr.instruction())) {
      if (broadcast->dimensions().empty() ||
          // More than 3 big inputs cause a speed regression.
          (row_vectorized && num_big_inputs <= 3)) {
        return false;
      }
    }
    VLOG(2) << "few_waves not enabled due to: "
            << instr.instruction().ToString();
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
  return launch_config;
}

const Shape& HloFusionAnalysis::GetElementShape() const {
  const Shape* shape = &fusion_roots_.front()->shape();
  while (shape->IsTuple()) {
    shape = &shape->tuple_shapes(0);
  }
  return *shape;
}

std::optional<HloFusionAnalysis> AnalyzeProducerConsumerFusion(
    const HloInstruction& producer, const HloInstruction& consumer,
    const se::DeviceDescription& device_info) {
  auto ret = HloFusionAnalysis::Create(
      consumer.has_backend_config()
          ? *consumer.backend_config<FusionBackendConfig>()
          : *producer.backend_config<FusionBackendConfig>(),
      std::make_unique<ProducerConsumerFusion>(
          HloFusionAdaptor::ForInstruction(&producer),
          HloFusionAdaptor::ForInstruction(&consumer)),
      &device_info);
  if (!ret.ok()) return std::nullopt;
  return {std::move(*ret)};
}

std::optional<HloFusionAnalysis> AnalyzeFusion(
    const HloInstruction& consumer, const se::DeviceDescription& device_info) {
  auto ret = HloFusionAnalysis::Create(
      *consumer.backend_config<FusionBackendConfig>(),
      HloFusionAdaptor::ForInstruction(&consumer), &device_info);
  if (!ret.ok()) return std::nullopt;
  return {std::move(*ret)};
}

}  // namespace gpu
}  // namespace xla
