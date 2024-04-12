/* Copyright 2023 The OpenXLA Authors.

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
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {
namespace {

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
      input_output_info_(std::move(input_output_info)) {}

// static
HloFusionAnalysis HloFusionAnalysis::Create(
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
HloFusionAnalysis HloFusionAnalysis::Create(
    const HloFusionInstruction* fusion,
    const se::DeviceDescription* device_info) {
  CHECK(device_info != nullptr);
  FusionBackendConfig backend_config =
      fusion->has_backend_config()
          ? fusion->backend_config<GpuBackendConfig>()->fusion_backend_config()
          : FusionBackendConfig::default_instance();
  return Create(std::move(backend_config),
                HloFusionAdaptor::ForInstruction(fusion), device_info);
}

// Returns true if the fusion has consistent transpose heros.
bool HloFusionAnalysis::HasConsistentTransposeHeros() const {
  return tiled_transpose_.has_value();
}

static bool UseConcatenateFusion(
    const std::vector<const HloInstruction*>& roots,
    const std::vector<const HloInstruction*>& heroes) {
  if (heroes.size() != 1) return false;
  if (heroes.front()->opcode() != HloOpcode::kConcatenate) return false;
  // The concat emitter does not support multiple outputs yet. TODO(csigg): fix.
  if (roots.front()->shape().IsTuple()) return false;
  // Limit the number of operands because the concat emitter produces code for
  // each operand, hurting occupancy.
  if (heroes.front()->operand_count() > 4) return false;
  // The loop emitter is faster when warp divergence and occupancy are both low.
  // TODO(csigg): exclude this case.
  return true;
}

HloFusionAnalysis::EmitterFusionKind HloFusionAnalysis::GetEmitterFusionKind()
    const {
  if (fusion_backend_config_.kind() == kCustomFusionKind) {
    return EmitterFusionKind::kCustomFusion;
  }

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  if (fusion_backend_config_.kind() == kTritonGemmFusionKind ||
      fusion_backend_config_.kind() == kTritonSoftmaxFusionKind) {
    return EmitterFusionKind::kTriton;
  }

  if (fusion_backend_config_.kind() == kCuDnnFusionKind) {
    return EmitterFusionKind::kCuDnn;
  }
#endif

  if (input_output_info_.has_4_bit_input ||
      input_output_info_.has_4_bit_output) {
    // Only loop and input slice fusions currently can handle int4
    // inputs/outputs, due to the special handling with IrArray needed to deal
    // with two values occupying a single byte.
    if (fusion_roots_.size() > 1 &&
        IsInputFusibleNonStridedSlices(fusion_roots_) &&
        AllSliceInputsAreCompatible(fusion_roots_)) {
      return EmitterFusionKind::kInputSlices;
    }
    return EmitterFusionKind::kLoop;
  }

  const HloInstruction* first_reduce_hero = nullptr;
  for (auto [root, hero] : llvm::zip(fusion_roots_, fusion_heroes_)) {
    if (IsRealReductionHero(*root, *hero)) {
      first_reduce_hero = hero;
      break;
    }
  }
  if (first_reduce_hero != nullptr) {
    bool valid_shapes = true;
    Shape hero_operand_shape = first_reduce_hero->operand(0)->shape();
    for (auto [root, hero] : llvm::zip(fusion_roots_, fusion_heroes_)) {
      if (root == first_reduce_hero) {
        continue;
      }
      if (!IsRealReductionHero(*root, *hero)) {
        // Needs to have a compatible shape to the reduce operand (compatible
        // meaning same number of elements).
        if (ShapeUtil::ElementsIn(root->shape()) !=
            ShapeUtil::ElementsIn(hero_operand_shape)) {
          valid_shapes = false;
          break;
        }
      } else if (!AreReductionsMultiOutputFusionCompatible(hero,
                                                           first_reduce_hero)) {
        valid_shapes = false;
        break;
      }
    }
    if (valid_shapes) {
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

  if (UseConcatenateFusion(fusion_roots_, fusion_heroes_)) {
    return EmitterFusionKind::kConcatenate;
  }

  return EmitterFusionKind::kLoop;
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

HloFusionAnalysis AnalyzeProducerConsumerFusion(
    const HloInstruction& producer, const HloInstruction& consumer,
    const se::DeviceDescription& device_info) {
  return HloFusionAnalysis::Create(
      consumer.has_backend_config()
          ? consumer.backend_config<GpuBackendConfig>()->fusion_backend_config()
          : producer.backend_config<GpuBackendConfig>()
                ->fusion_backend_config(),
      std::make_unique<ProducerConsumerFusion>(
          HloFusionAdaptor::ForInstruction(&producer),
          HloFusionAdaptor::ForInstruction(&consumer)),
      &device_info);
}

HloFusionAnalysis AnalyzeFusion(const HloInstruction& consumer,
                                const se::DeviceDescription& device_info) {
  return HloFusionAnalysis::Create(
      consumer.backend_config<GpuBackendConfig>()->fusion_backend_config(),
      HloFusionAdaptor::ForInstruction(&consumer), &device_info);
}

}  // namespace gpu
}  // namespace xla
