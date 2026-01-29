/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_TRANSFORMS_YNN_MATCHER_H_
#define XLA_BACKENDS_CPU_TRANSFORMS_YNN_MATCHER_H_

#include <string>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/transforms/library_matcher.h"
#include "xla/backends/cpu/ynn_support.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "tsl/platform/protobuf.h"

namespace xla::cpu {

class YnnMatcher : public LibraryMatcher {
 public:
  explicit YnnMatcher(const TargetMachineFeatures* target_machine_features,
                      const tsl::protobuf::RepeatedField<int>* fusion_types)
      : LibraryMatcher(target_machine_features, fusion_types) {}
  ~YnnMatcher() override = default;

  // Returns the set of supported HLO instructions.
  absl::flat_hash_set<HloOpcode> SupportedOps() const override {
    static const absl::NoDestructor<absl::flat_hash_set<HloOpcode>>
        kSupportedOps{[]() {
          absl::flat_hash_set<HloOpcode> supported_ops{
              HloOpcode::kDot, HloOpcode::kReduce, HloOpcode::kConstant,
              HloOpcode::kConvolution};
          for (const auto& [op, _] : GetYnnUnaryOpMap()) {
            supported_ops.insert(op);
          }
          for (const auto& [op, _] : GetYnnBinaryOpMap()) {
            supported_ops.insert(op);
          }
          return supported_ops;
        }()};
    return *kSupportedOps;
  }

  // Returns true if the HLO instruction is supported by the library.
  absl::StatusOr<bool> IsOpSupported(const HloInstruction* instr) override {
    if (instr->opcode() == HloOpcode::kReduce) {
      return IsReduceOpOffloadedToYnn(instr);
    }
    if (instr->IsConstant()) {
      return IsConstantSupportedByYnn(instr);
    }

    // TODO(b/441837668): Need to get the reduction performance right before
    // enabling fusions. Fusions make performance analysis quite challenging.
    if (fuse_reduce_) {
      return false;
    }
    if (instr->opcode() == HloOpcode::kDot) {
      return IsDotSupportedByYnn(instr->dot_dimension_numbers(),
                                 instr->operand(0)->shape(),
                                 instr->operand(1)->shape(), instr->shape());
    }

    // We do not want to fuse reductions, with the following exception:
    // init instruction (constant). So the checks below should go after the
    // reduce check.
    if (instr->opcode() == HloOpcode::kConvolution) {
      return IsConvolutionOpSupportedByYnn(instr);
    }
    if (instr->IsElementwise()) {
      return IsElementwiseOpSupportedByYnn(instr);
    }
    return false;
  }

  // Returns true if we should start a new fusion containing just the given HLO
  // instruction. We control the instructions that can start a fusion with the
  // `--xla_cpu_experimental_ynn_fusion_type` flag.
  bool ShouldCreateFusion(const HloInstruction* instr) override {
    if (fuse_dot_ && instr->opcode() == HloOpcode::kDot) {
      return true;
    }
    if (fuse_reduce_ && instr->opcode() == HloOpcode::kReduce) {
      return true;
    }
    return fuse_eltwise_ && instr->IsElementwise();
  }

  PrimitiveType LibraryOpOutputType(const HloInstruction* instr) override {
    auto out_type = instr->shape().element_type();
    if (instr->opcode() != HloOpcode::kDot) {
      return out_type;
    }
    return out_type == BF16 ? F32 : out_type;
  }

  // Returns a prefix string for the fusion op's name.
  std::string fusion_prefix() const override { return "ynn_"; }

  // Returns a string for FusionBackendConfig's fusion kind.
  absl::string_view fusion_kind() const override { return kYnnFusionKind; }

 private:
  absl::flat_hash_set<DebugOptions::LibraryFusionType> fusion_types_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_TRANSFORMS_YNN_MATCHER_H_
