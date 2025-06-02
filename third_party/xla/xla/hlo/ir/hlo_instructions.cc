/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/ir/hlo_instructions.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_domain_metadata.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/hlo_sharding_metadata.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/map_util.h"
#include "xla/printer.h"
#include "xla/protobuf_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/gtl/iterator_range.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/tsl/platform/status.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace {

using absl::CEscape;
using absl::StrCat;

bool IsInstructionElementwiseOnOperand(const HloInstruction* instruction,
                                       const HloInstruction* operand) {
  const auto operand_indices = instruction->OperandIndices(operand);
  return absl::c_all_of(operand_indices, [instruction](int64_t operand_index) {
    return instruction->IsElementwiseOnOperand(operand_index);
  });
}

void PrintPrecisionConfig(HloInstruction::AttributePrinter& printer,
                          const PrecisionConfig& precision_config) {
  if (absl::c_any_of(
          precision_config.operand_precision(), [](int32_t precision) {
            return static_cast<PrecisionConfig::Precision>(precision) !=
                   PrecisionConfig::DEFAULT;
          })) {
    printer.Next([&precision_config](Printer* printer) {
      printer->Append("operand_precision={");
      AppendJoin(printer, precision_config.operand_precision(), ",",
                 [](Printer* printer, int32_t precision) {
                   CHECK(PrecisionConfig::Precision_IsValid(precision))
                       << precision;
                   printer->Append(PrecisionToString(
                       static_cast<PrecisionConfig::Precision>(precision)));
                 });
      printer->Append("}");
    });
  }

  if (precision_config.algorithm() != PrecisionConfig::ALG_UNSET) {
    printer.Next([&precision_config](Printer* printer) {
      printer->Append("algorithm=");
      printer->Append(AlgorithmToString(precision_config.algorithm()));
    });
  }
}

void PrintSparsityDescriptor(HloInstruction::AttributePrinter& printer,
                             absl::Span<const SparsityDescriptor> sparsity) {
  printer.Next([&sparsity](Printer* printer) {
    printer->Append("sparsity=");
    for (int i = 0; i < sparsity.size(); ++i) {
      if (i != 0) {
        printer->Append("_");
      }
      const SparsityDescriptor& cur = sparsity[i];
      printer->Append(cur.index() == 0 ? "L." : "R.");
      printer->Append(cur.dimension());
      printer->Append("@");
      switch (cur.type()) {
        case SPARSITY_STRUCTURED_N_M:
          printer->Append(cur.n());
          printer->Append(":");
          printer->Append(cur.m());
          break;
        default:
          LOG(FATAL) << "Unknown sparsity type: " << cur.type();
      }
    }
  });
}

void SetThreadName(HloComputation* called_computation,
                   absl::string_view execution_thread,
                   bool skip_async_execution_thread_overwrite) {
  called_computation->SetExecutionThread(execution_thread);
  for (HloInstruction* instr : called_computation->instructions()) {
    if (instr->IsAsynchronous()) {
      if (!skip_async_execution_thread_overwrite) {
        // Set async instruction thread name and also recursively set async
        // computations.
        instr->set_async_execution_thread(execution_thread);
      }
      continue;
    }
    for (HloComputation* nested_called_computation :
         instr->called_computations()) {
      SetThreadName(nested_called_computation, execution_thread,
                    skip_async_execution_thread_overwrite);
    }
  }
}

}  // namespace

HloBatchNormInstruction::HloBatchNormInstruction(
    HloOpcode opcode, const Shape& shape, HloInstruction* operand,
    HloInstruction* scale, float epsilon, int64_t feature_index)
    : HloInstruction(opcode, shape),
      epsilon_(epsilon),
      feature_index_(feature_index) {
  AppendOperand(operand);
  AppendOperand(scale);
}

bool HloBatchNormInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloBatchNormInstruction&>(other);
  return feature_index() == casted_other.feature_index() &&
         epsilon() == casted_other.epsilon();
}

HloInstructionProto HloBatchNormInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_epsilon(epsilon_);
  proto.set_feature_index(feature_index_);
  return proto;
}

void HloBatchNormInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  printer.Next(
      [this](Printer* printer) { AppendCat(printer, "epsilon=", epsilon()); });
  printer.Next([this](Printer* printer) {
    AppendCat(printer, "feature_index=", feature_index());
  });
}

HloBatchNormTrainingInstruction::HloBatchNormTrainingInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* scale,
    HloInstruction* offset, float epsilon, int64_t feature_index)
    : HloBatchNormInstruction(HloOpcode::kBatchNormTraining, shape, operand,
                              scale, epsilon, feature_index) {
  AppendOperand(offset);
}

std::unique_ptr<HloInstruction>
HloBatchNormTrainingInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 3);
  return std::make_unique<HloBatchNormTrainingInstruction>(
      shape, new_operands[0], new_operands[1], new_operands[2], epsilon(),
      feature_index());
}

HloBatchNormInferenceInstruction::HloBatchNormInferenceInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* scale,
    HloInstruction* offset, HloInstruction* mean, HloInstruction* variance,
    float epsilon, int64_t feature_index)
    : HloBatchNormInstruction(HloOpcode::kBatchNormInference, shape, operand,
                              scale, epsilon, feature_index) {
  AppendOperand(offset);
  AppendOperand(mean);
  AppendOperand(variance);
}

std::unique_ptr<HloInstruction>
HloBatchNormInferenceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 5);
  return std::make_unique<HloBatchNormInferenceInstruction>(
      shape, new_operands[0], new_operands[1], new_operands[2], new_operands[3],
      new_operands[4], epsilon(), feature_index());
}

HloBatchNormGradInstruction::HloBatchNormGradInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* scale,
    HloInstruction* mean, HloInstruction* variance, HloInstruction* grad_output,
    float epsilon, int64_t feature_index)
    : HloBatchNormInstruction(HloOpcode::kBatchNormGrad, shape, operand, scale,
                              epsilon, feature_index) {
  AppendOperand(mean);
  AppendOperand(variance);
  AppendOperand(grad_output);
}

std::unique_ptr<HloInstruction>
HloBatchNormGradInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 5);
  return std::make_unique<HloBatchNormGradInstruction>(
      shape, new_operands[0], new_operands[1], new_operands[2], new_operands[3],
      new_operands[4], epsilon(), feature_index());
}

HloFftInstruction::HloFftInstruction(const Shape& shape,
                                     HloInstruction* operand, FftType fft_type,
                                     absl::Span<const int64_t> fft_length)
    : HloInstruction(HloOpcode::kFft, shape), fft_type_(fft_type) {
  fft_length_.assign(fft_length.begin(), fft_length.end());
  AppendOperand(operand);
}

HloInstructionProto HloFftInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_fft_type(fft_type_);
  for (int64_t fft_len : fft_length_) {
    proto.add_fft_length(fft_len);
  }
  return proto;
}

void HloFftInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  printer.Next([this](Printer* printer) {
    AppendCat(printer, "fft_type=", FftType_Name(fft_type()));
  });
  printer.Next([this](Printer* printer) {
    printer->Append("fft_length={");
    AppendJoin(printer, fft_length(), ",");
    printer->Append("}");
  });
}

bool HloFftInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloFftInstruction&>(other);
  return fft_type() == casted_other.fft_type() &&
         fft_length() == casted_other.fft_length();
}

std::unique_ptr<HloInstruction> HloFftInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloFftInstruction>(shape, new_operands[0], fft_type_,
                                             fft_length_);
}

HloAsyncInstruction::HloAsyncInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands, HloOpcode async_wrapped_opcode)
    : HloInstruction(opcode, shape) {
  CHECK(opcode == HloOpcode::kAsyncStart || operands.size() == 1);
  for (auto operand : operands) {
    AppendOperand(operand);
  }

  // Drop 'async' from async-{start/update/done} to get the suffix.
  absl::string_view suffix = HloOpcodeString(opcode).substr(5);
  absl::string_view wrapped_name = HloOpcodeString(async_wrapped_opcode);
  SetAndSanitizeName(absl::StrCat(wrapped_name, suffix));
}

HloAsyncInstruction::HloAsyncInstruction(HloOpcode opcode, const Shape& shape,
                                         HloInstruction* operand)
    : HloAsyncInstruction(opcode, shape, absl::MakeConstSpan(&operand, 1),
                          operand->async_wrapped_opcode()) {
  CHECK(operand->opcode() == HloOpcode::kAsyncStart ||
        operand->opcode() == HloOpcode::kAsyncUpdate);
  HloAsyncInstruction* prev = Cast<HloAsyncInstruction>(operand);
  prev->async_chain_next_ = this;
}

HloComputation* HloAsyncInstruction::async_wrapped_computation() const {
  return async_chain_start()->called_computations().front();
}

HloInstruction* HloAsyncInstruction::async_wrapped_instruction() const {
  return async_chain_start()->async_wrapped_computation()->root_instruction();
}

HloOpcode HloAsyncInstruction::async_wrapped_opcode() const {
  return async_chain_start()->async_wrapped_instruction()->opcode();
}

absl::string_view HloAsyncInstruction::async_execution_thread() const {
  return async_chain_start()->async_execution_thread();
}

HloAsyncInstruction* HloAsyncInstruction::async_chain_start() const {
  if (opcode() == HloOpcode::kAsyncStart) {
    return const_cast<HloAsyncInstruction*>(this);
  }

  HloInstruction* prev = operands()[0];
  while (prev->opcode() != HloOpcode::kAsyncStart) {
    // If the prev op in the chain isn't async-start, it must be async-update.
    CHECK(prev->opcode() == HloOpcode::kAsyncUpdate);
    prev = prev->operands()[0];
  }
  return Cast<HloAsyncInstruction>(prev);
}

HloAsyncInstruction* HloAsyncInstruction::async_chain_done() const {
  if (opcode() == HloOpcode::kAsyncDone) {
    return const_cast<HloAsyncInstruction*>(this);
  }

  HloAsyncInstruction* next = async_chain_next_;
  while (next->opcode() != HloOpcode::kAsyncDone) {
    // If the next op in the chain isn't async-done, it must be async-update.
    CHECK(next->opcode() == HloOpcode::kAsyncUpdate);
    next = next->async_chain_next_;
  }
  return next;
}

std::vector<HloAsyncInstruction*> HloAsyncInstruction::GetAsyncChain() const {
  std::vector<HloAsyncInstruction*> chain;
  HloAsyncInstruction* current = async_chain_start();
  do {
    chain.push_back(current);
    current = current->async_chain_next_;
  } while (current != nullptr);
  return chain;
}

bool HloAsyncInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  return opcode() == other.opcode() &&
         eq_computations(async_wrapped_computation(),
                         other.async_wrapped_computation());
}

std::unique_ptr<HloInstruction> HloAsyncInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  return std::make_unique<HloAsyncInstruction>(opcode(), shape,
                                               new_operands[0]);
}

HloAsyncStartInstruction::HloAsyncStartInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    HloComputation* async_computation, absl::string_view async_execution_thread)
    : HloAsyncInstruction(opcode, shape, operands,
                          async_computation->root_instruction()->opcode()) {
  CHECK(async_computation->caller_instructions(HloOpcode::kCustomCall).empty());
  CHECK(!async_computation->IsFusionComputation());
  AppendComputation(async_computation);
  HloAsyncStartInstruction::set_async_execution_thread(async_execution_thread);
}

void HloAsyncStartInstruction::set_async_execution_thread(
    absl::string_view async_execution_thread) {
  async_execution_thread_ = std::string(async_execution_thread);
  SetThreadName(async_wrapped_computation(), async_execution_thread,
                /*skip_async_execution_thread_overwrite=*/false);
}

HloInstructionProto HloAsyncStartInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_async_execution_thread(async_execution_thread_ ==
                                           HloInstruction::kMainExecutionThread
                                       ? ""
                                       : async_execution_thread_);
  return proto;
}

void HloAsyncStartInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  if (async_execution_thread_ != kMainExecutionThread) {
    printer.Next([this](Printer* printer) {
      AppendCat(printer, "async_execution_thread=\"", async_execution_thread_,
                "\"");
    });
  }
  if (options.syntax_sugar_async_ops() &&
      async_wrapped_computation()->CanExpandIntoSingleInstruction()) {
    async_wrapped_instruction()->PrintExtraAttributes(printer, options);
  }
}

std::unique_ptr<HloInstruction>
HloAsyncStartInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  HloModule* module = context != nullptr ? context->module() : GetModule();
  HloComputation* new_wrapped_computation = nullptr;
  if (context != nullptr) {
    new_wrapped_computation =
        context->FindComputation(async_wrapped_computation());
  }
  if (new_wrapped_computation == nullptr) {
    new_wrapped_computation = module->AddEmbeddedComputation(
        async_wrapped_computation()->Clone("clone", context));
    // Give the trampoline a trivial schedule if it already had one.
    if (module->has_schedule() && module->schedule().is_computation_scheduled(
                                      async_wrapped_computation())) {
      module->schedule().set_sequence(
          new_wrapped_computation,
          new_wrapped_computation->MakeInstructionPostOrder());
    }
  }

  return std::make_unique<HloAsyncStartInstruction>(
      opcode(), shape, new_operands, new_wrapped_computation,
      async_execution_thread_);
}

HloCopyStartInstruction::HloCopyStartInstruction(
    const Shape& shape, HloInstruction* operand,
    std::optional<int> cross_program_prefetch_index)
    : HloInstruction(HloOpcode::kCopyStart, shape),
      cross_program_prefetch_index_(cross_program_prefetch_index) {
  AppendOperand(operand);
}

HloInstructionProto HloCopyStartInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  if (cross_program_prefetch_index_.has_value()) {
    proto.set_cross_program_prefetch_index(*cross_program_prefetch_index_);
  }
  return proto;
}

void HloCopyStartInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  if (cross_program_prefetch_index_.has_value()) {
    printer.Next([this](Printer* printer) {
      AppendCat(printer, "cross_program_prefetch_index=",
                *cross_program_prefetch_index_);
    });
  }
}

bool HloCopyStartInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloCopyStartInstruction&>(other);
  return cross_program_prefetch_index() ==
         casted_other.cross_program_prefetch_index();
}

std::unique_ptr<HloInstruction>
HloCopyStartInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloCopyStartInstruction>(
      shape, new_operands[0], cross_program_prefetch_index());
}

HloCompareInstruction::HloCompareInstruction(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    ComparisonDirection direction, std::optional<Comparison::Type> type)
    : HloInstruction(HloOpcode::kCompare, shape),
      compare_(type.has_value()
                   ? Comparison(direction, *type)
                   : Comparison(direction, lhs->shape().element_type())) {
  AppendOperand(lhs);
  AppendOperand(rhs);
}

HloInstructionProto HloCompareInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_comparison_direction(
      ComparisonDirectionToString(compare_.GetDirection()));
  proto.set_comparison_type(ComparisonTypeToString(compare_.GetType()));
  return proto;
}

void HloCompareInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  printer.Next([this](Printer* printer) {
    AppendCat(printer, "direction=", ComparisonDirectionToString(direction()));
  });
  // We might want to print a HloInstruction which has been cleand up and has no
  // operands anymore. This should not result in a crash.
  if (operand_count() == 0 || operand(0) == nullptr ||
      compare_.GetType() != Comparison::DefaultComparisonType(
                                operand(0)->shape().element_type())) {
    printer.Next([this](Printer* printer) {
      AppendCat(printer, "type=", ComparisonTypeToString(compare_.GetType()));
    });
  }
}

bool HloCompareInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloCompareInstruction&>(other);
  return direction() == casted_other.direction();
}

std::unique_ptr<HloInstruction> HloCompareInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return std::make_unique<HloCompareInstruction>(
      shape, new_operands[0], new_operands[1], direction(), type());
}

namespace {

// Converts a protocol buffer message (e.g., TriangularSolveOptions) to a
// vector of "key=value" attribute strings generically, using protocol buffer
// reflection.
//
// Currently implements a small subset of cases; feel free to add more as
// needed.
void PrintAttributeProto(HloInstruction::AttributePrinter& printer,
                         const tsl::protobuf::Message& message) {
  const tsl::protobuf::Reflection* reflection = message.GetReflection();
  std::vector<const tsl::protobuf::FieldDescriptor*> fields;
  reflection->ListFields(message, &fields);

  for (const tsl::protobuf::FieldDescriptor* field : fields) {
    CHECK(!field->is_repeated()) << "Repeated fields aren't implemented";
    printer.Next([&](Printer* printer) {
      printer->Append(field->name());
      printer->Append("=");
      switch (field->type()) {
        case tsl::protobuf::FieldDescriptor::TYPE_BOOL: {
          bool val = reflection->GetBool(message, field);
          printer->Append(val ? "true" : "false");
          break;
        }
        case tsl::protobuf::FieldDescriptor::TYPE_ENUM: {
          const tsl::protobuf::EnumValueDescriptor* evd =
              reflection->GetEnum(message, field);
          printer->Append(evd->name());
          break;
        }
        default:
          LOG(FATAL) << "Unimplemented field type: " << field->DebugString();
      }
    });
  }
}

}  // namespace

HloTriangularSolveInstruction::HloTriangularSolveInstruction(
    const Shape& shape, HloInstruction* a, HloInstruction* b,
    const TriangularSolveOptions& options)
    : HloInstruction(HloOpcode::kTriangularSolve, shape),
      triangular_solve_options_(options) {
  AppendOperand(a);
  AppendOperand(b);
}

HloInstructionProto HloTriangularSolveInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_triangular_solve_options() = triangular_solve_options_;
  return proto;
}

void HloTriangularSolveInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  PrintAttributeProto(printer, triangular_solve_options_);
}

bool HloTriangularSolveInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloTriangularSolveInstruction&>(other);
  const auto& options = triangular_solve_options();
  const auto& other_options = casted_other.triangular_solve_options();

  return options.left_side() == other_options.left_side() &&
         options.lower() == other_options.lower() &&
         options.unit_diagonal() == other_options.unit_diagonal() &&
         options.transpose_a() == other_options.transpose_a();
}

std::unique_ptr<HloInstruction>
HloTriangularSolveInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return std::make_unique<HloTriangularSolveInstruction>(
      shape, new_operands[0], new_operands[1], triangular_solve_options());
}

HloCholeskyInstruction::HloCholeskyInstruction(const Shape& shape,
                                               HloInstruction* a,
                                               const CholeskyOptions& options)
    : HloInstruction(HloOpcode::kCholesky, shape), cholesky_options_(options) {
  AppendOperand(a);
}

HloInstructionProto HloCholeskyInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_cholesky_options() = cholesky_options_;
  return proto;
}

void HloCholeskyInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  PrintAttributeProto(printer, cholesky_options_);
}

bool HloCholeskyInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloCholeskyInstruction&>(other);
  const auto& options = cholesky_options();
  const auto& other_options = casted_other.cholesky_options();

  return options.lower() == other_options.lower();
}

std::unique_ptr<HloInstruction>
HloCholeskyInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloCholeskyInstruction>(shape, new_operands[0],
                                                  cholesky_options());
}

HloChannelInstruction::HloChannelInstruction(
    HloOpcode opcode, const Shape& shape,
    const std::optional<int64_t>& channel_id)
    : HloInstruction(opcode, shape), channel_id_(channel_id) {}

void HloChannelInstruction::set_channel_id(
    const std::optional<int64_t>& channel_id) {
  channel_id_ = channel_id;
}

HloInstructionProto HloChannelInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  if (channel_id_) {
    proto.set_channel_id(*channel_id_);
  }
  return proto;
}

void HloChannelInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& /*options*/) const {
  if (!channel_id_) {
    return;
  }
  printer.Next([this](Printer* printer) {
    AppendCat(printer, "channel_id=", *channel_id_);
  });
}

bool HloChannelInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  if (!IdenticalSlowPathIgnoringChannelIdValues(other, eq_computations)) {
    return false;
  }
  const auto& casted_other = static_cast<const HloChannelInstruction&>(other);
  return channel_id() == casted_other.channel_id();
}

HloTopKInstruction::HloTopKInstruction(const Shape& shape,
                                       HloInstruction* input, int64_t k,
                                       bool largest)
    : HloInstruction(HloOpcode::kTopK, shape), k_(k), largest_(largest) {
  AppendOperand(input);
}

HloInstructionProto HloTopKInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_k(k_);
  proto.set_largest(largest_);
  return proto;
}

void HloTopKInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  printer.Next([this](Printer* p) { AppendCat(p, "k=", k_); });
  printer.Next([this](Printer* p) {
    AppendCat(p, "largest=", (largest_ ? "true" : "false"));
  });
}

std::unique_ptr<HloInstruction> HloTopKInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  return std::make_unique<HloTopKInstruction>(shape, new_operands[0], k(),
                                              largest());
}

bool HloTopKInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloTopKInstruction&>(other);
  return k() == casted_other.k() && largest() == casted_other.largest();
}

HloSendRecvInstruction::HloSendRecvInstruction(
    HloOpcode opcode, const Shape& shape, std::optional<int64_t> channel_id,
    bool is_host_transfer)
    : HloChannelInstruction(opcode, shape, channel_id),
      is_host_transfer_(is_host_transfer) {}

HloInstructionProto HloSendRecvInstruction::ToProto() const {
  HloInstructionProto proto = HloChannelInstruction::ToProto();
  proto.set_is_host_transfer(is_host_transfer_);
  return proto;
}

void HloSendRecvInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  HloChannelInstruction::PrintExtraAttributesImpl(printer, options);
  if (is_host_transfer()) {
    printer.Next(
        [](Printer* printer) { printer->Append("is_host_transfer=true"); });
  }
}

bool HloSendRecvInstruction::IdenticalSlowPathIgnoringChannelIdValues(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  // Not yet supported.
  return false;
}

// Send instruction produces a tuple of {aliased operand, U32 context}.
HloSendInstruction::HloSendInstruction(HloInstruction* operand,
                                       HloInstruction* token,
                                       std::optional<int64_t> channel_id,
                                       bool is_host_transfer)
    : HloSendRecvInstruction(
          HloOpcode::kSend,
          ShapeUtil::MakeTupleShape({CHECK_NOTNULL(operand)->shape(),
                                     ShapeUtil::MakeShape(U32, {}),
                                     ShapeUtil::MakeTokenShape()}),
          channel_id, is_host_transfer) {
  AppendOperand(operand);
  AppendOperand(token);
}

std::unique_ptr<HloInstruction> HloSendInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return std::make_unique<HloSendInstruction>(new_operands[0], new_operands[1],
                                              channel_id(), is_host_transfer());
}

HloSendDoneInstruction::HloSendDoneInstruction(HloSendInstruction* operand,
                                               bool is_host_transfer)
    : HloSendRecvInstruction(HloOpcode::kSendDone, ShapeUtil::MakeTokenShape(),
                             operand->channel_id(), is_host_transfer) {
  AppendOperand(operand);
}

HloSendDoneInstruction::HloSendDoneInstruction(
    HloInstruction* operand, std::optional<int64_t> channel_id,
    bool is_host_transfer)
    : HloSendRecvInstruction(HloOpcode::kSendDone, ShapeUtil::MakeTokenShape(),
                             channel_id, is_host_transfer) {
  AppendOperand(operand);
}

std::unique_ptr<HloInstruction>
HloSendDoneInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  HloSendInstruction* send = dynamic_cast<HloSendInstruction*>(new_operands[0]);
  if (send != nullptr) {
    return std::make_unique<HloSendDoneInstruction>(send, is_host_transfer());
  }

  return std::make_unique<HloSendDoneInstruction>(new_operands[0], channel_id(),
                                                  is_host_transfer());
}

// Recv instruction produces a tuple of {receive buffer, U32 context}.
HloRecvInstruction::HloRecvInstruction(const Shape& shape,
                                       HloInstruction* token,
                                       std::optional<int64_t> channel_id,
                                       bool is_host_transfer)
    : HloSendRecvInstruction(
          HloOpcode::kRecv,
          ShapeUtil::MakeTupleShape({shape, ShapeUtil::MakeShape(U32, {}),
                                     ShapeUtil::MakeTokenShape()}),
          channel_id, is_host_transfer) {
  AppendOperand(token);
}

std::unique_ptr<HloInstruction> HloRecvInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloRecvInstruction>(
      ShapeUtil::GetTupleElementShape(shape, 0), new_operands[0], channel_id(),
      is_host_transfer());
}

HloRecvDoneInstruction::HloRecvDoneInstruction(HloRecvInstruction* operand,
                                               bool is_host_transfer)
    : HloSendRecvInstruction(
          HloOpcode::kRecvDone,
          ShapeUtil::MakeTupleShape(
              {ShapeUtil::GetTupleElementShape(operand->shape(), 0),
               ShapeUtil::MakeTokenShape()}),
          operand->channel_id(), is_host_transfer) {
  AppendOperand(operand);
}

HloRecvDoneInstruction::HloRecvDoneInstruction(
    HloInstruction* operand, std::optional<int64_t> channel_id,
    bool is_host_transfer)
    : HloSendRecvInstruction(
          HloOpcode::kRecvDone,
          ShapeUtil::MakeTupleShape(
              {ShapeUtil::GetTupleElementShape(operand->shape(), 0),
               ShapeUtil::MakeTokenShape()}),
          channel_id, is_host_transfer) {
  AppendOperand(operand);
}

std::unique_ptr<HloInstruction>
HloRecvDoneInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  HloRecvInstruction* recv = dynamic_cast<HloRecvInstruction*>(new_operands[0]);
  if (recv != nullptr) {
    return std::make_unique<HloRecvDoneInstruction>(recv, is_host_transfer());
  }

  return std::make_unique<HloRecvDoneInstruction>(new_operands[0], channel_id(),
                                                  is_host_transfer());
}

HloCollectiveInstruction::HloCollectiveInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    const CollectiveDeviceList& device_list, bool constrain_layout,
    const std::optional<int64_t>& channel_id)
    : HloChannelInstruction(opcode, shape, channel_id),
      device_list_(device_list),
      constrain_layout_(constrain_layout) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
}

HloInstructionProto HloCollectiveInstruction::ToProto() const {
  HloInstructionProto proto = HloChannelInstruction::ToProto();
  *proto.mutable_collective_device_list() = device_list_.ToProto();
  proto.set_constrain_layout(constrain_layout_);
  return proto;
}

void HloCollectiveInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  HloChannelInstruction::PrintExtraAttributesImpl(printer, options);
  printer.Next([this, &options](Printer* printer) {
    VLOG(4) << name() << " replica_groups="
            << device_list_.ToString(options.print_full_replica_group_list());

    printer->Append("replica_groups=");
    device_list_.Print(printer, options.print_full_replica_group_list());
  });
  if (constrain_layout_) {
    printer.Next(
        [](Printer* printer) { printer->Append("constrain_layout=true"); });
  }
}

bool HloCollectiveInstruction::IdenticalSlowPathIgnoringChannelIdValues(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloCollectiveInstruction&>(other);
  return HloChannelInstruction::IdenticalSlowPathIgnoringChannelIdValues(
             other, eq_computations) &&
         constrain_layout() == casted_other.constrain_layout() &&
         absl::c_equal(replica_groups(), casted_other.replica_groups(),
                       [](const ReplicaGroup& a, const ReplicaGroup& b) {
                         return absl::c_equal(a.replica_ids(), b.replica_ids());
                       });
}

HloAllGatherInstruction::HloAllGatherInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands, int64_t all_gather_dimension,
    const CollectiveDeviceList& device_list, bool constrain_layout,
    const std::optional<int64_t>& channel_id, bool use_global_device_ids)
    : HloCollectiveInstruction(opcode, shape, operands, device_list,
                               constrain_layout, channel_id),
      all_gather_dimension_(all_gather_dimension),
      use_global_device_ids_(use_global_device_ids) {}

HloAllGatherInstruction::HloAllGatherInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands, int64_t all_gather_dimension,
    absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
    const std::optional<int64_t>& channel_id, bool use_global_device_ids)
    : HloAllGatherInstruction(opcode, shape, operands, all_gather_dimension,
                              CollectiveDeviceList(replica_groups),
                              constrain_layout, channel_id,
                              use_global_device_ids) {}

void HloAllGatherInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  HloCollectiveInstruction::PrintExtraAttributesImpl(printer, options);
  printer.Next([this](Printer* printer) {
    AppendCat(printer, "dimensions={", all_gather_dimension_, "}");
  });
  if (use_global_device_ids_) {
    printer.Next([](Printer* printer) {
      printer->Append("use_global_device_ids=true");
    });
  }
}

std::unique_ptr<HloInstruction>
HloAllGatherInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  return std::make_unique<HloAllGatherInstruction>(
      opcode(), shape, new_operands, all_gather_dimension(), device_list(),
      constrain_layout(), channel_id(), use_global_device_ids());
}

HloInstructionProto HloAllGatherInstruction::ToProto() const {
  HloInstructionProto proto = HloCollectiveInstruction::ToProto();
  proto.add_dimensions(all_gather_dimension_);
  proto.set_use_global_device_ids(use_global_device_ids_);
  return proto;
}

bool HloAllGatherInstruction::IdenticalSlowPathIgnoringChannelIdValues(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloAllGatherInstruction&>(other);
  return HloCollectiveInstruction::IdenticalSlowPathIgnoringChannelIdValues(
             other, eq_computations) &&
         all_gather_dimension_ == casted_other.all_gather_dimension() &&
         use_global_device_ids() == casted_other.use_global_device_ids();
}

HloAllReduceInstructionBase::HloAllReduceInstructionBase(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    HloComputation* reduce_computation, const CollectiveDeviceList& device_list,
    bool constrain_layout, const std::optional<int64_t>& channel_id,
    bool use_global_device_ids)
    : HloCollectiveInstruction(opcode, shape, operands, device_list,
                               constrain_layout, channel_id),
      use_global_device_ids_(use_global_device_ids) {
  AppendComputation(reduce_computation);
}

HloInstructionProto HloAllReduceInstructionBase::ToProto() const {
  HloInstructionProto proto = HloCollectiveInstruction::ToProto();
  proto.set_use_global_device_ids(use_global_device_ids_);
  return proto;
}

void HloAllReduceInstructionBase::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  HloCollectiveInstruction::PrintExtraAttributesImpl(printer, options);
  if (use_global_device_ids_) {
    printer.Next([](Printer* printer) {
      printer->Append("use_global_device_ids=true");
    });
  }
}

bool HloAllReduceInstructionBase::IdenticalSlowPathIgnoringChannelIdValues(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  if (opcode() != other.opcode()) {
    return false;
  }
  const auto& casted_other =
      static_cast<const HloAllReduceInstructionBase&>(other);
  return HloCollectiveInstruction::IdenticalSlowPathIgnoringChannelIdValues(
             other, eq_computations) &&
         constrain_layout() == casted_other.constrain_layout() &&
         use_global_device_ids() == casted_other.use_global_device_ids() &&
         eq_computations(to_apply(), casted_other.to_apply());
}

bool HloAllReduceInstruction::IsNoop() const {
  for (const auto& replica_group : replica_groups()) {
    if (replica_group.replica_ids().size() != 1) {
      return false;
    }
  }
  return !channel_id();
}

std::unique_ptr<HloInstruction>
HloAllReduceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  return std::make_unique<HloAllReduceInstruction>(
      opcode(), shape, new_operands, to_apply(), device_list(),
      constrain_layout(), channel_id(), use_global_device_ids());
}

HloReduceScatterInstruction::HloReduceScatterInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* reduce_computation, const CollectiveDeviceList& device_list,
    bool constrain_layout, const std::optional<int64_t>& channel_id,
    bool use_global_device_ids, int64_t scatter_dimension)
    : HloAllReduceInstructionBase(
          HloOpcode::kReduceScatter, shape, operands, reduce_computation,
          device_list, constrain_layout, channel_id, use_global_device_ids),
      scatter_dimension_(scatter_dimension) {}

HloReduceScatterInstruction::HloReduceScatterInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* reduce_computation,
    absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
    const std::optional<int64_t>& channel_id, bool use_global_device_ids,
    int64_t scatter_dimension)
    : HloReduceScatterInstruction(shape, operands, reduce_computation,
                                  CollectiveDeviceList(replica_groups),
                                  constrain_layout, channel_id,
                                  use_global_device_ids, scatter_dimension) {}

void HloReduceScatterInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  HloAllReduceInstructionBase::PrintExtraAttributesImpl(printer, options);
  printer.Next([this](Printer* printer) {
    AppendCat(printer, "dimensions={", scatter_dimension_, "}");
  });
}

HloInstructionProto HloReduceScatterInstruction::ToProto() const {
  HloInstructionProto proto = HloAllReduceInstructionBase::ToProto();
  proto.add_dimensions(scatter_dimension_);
  return proto;
}

bool HloReduceScatterInstruction::IdenticalSlowPathIgnoringChannelIdValues(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloReduceScatterInstruction&>(other);
  return HloAllReduceInstructionBase::IdenticalSlowPathIgnoringChannelIdValues(
             other, eq_computations) &&
         scatter_dimension_ == casted_other.scatter_dimension();
}

std::unique_ptr<HloInstruction>
HloReduceScatterInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  return std::make_unique<HloReduceScatterInstruction>(
      shape, new_operands, to_apply(), device_list(), constrain_layout(),
      channel_id(), use_global_device_ids(), scatter_dimension());
}

HloAllToAllInstruction::HloAllToAllInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    const CollectiveDeviceList& device_list, bool constrain_layout,
    const std::optional<int64_t>& channel_id,
    const std::optional<int64_t>& split_dimension)
    : HloCollectiveInstruction(HloOpcode::kAllToAll, shape, operands,
                               device_list, constrain_layout, channel_id),
      split_dimension_(split_dimension) {}

HloAllToAllInstruction::HloAllToAllInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
    const std::optional<int64_t>& channel_id,
    const std::optional<int64_t>& split_dimension)
    : HloAllToAllInstruction(shape, operands,
                             CollectiveDeviceList(replica_groups),
                             constrain_layout, channel_id, split_dimension) {}

std::unique_ptr<HloInstruction>
HloAllToAllInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  return std::make_unique<HloAllToAllInstruction>(
      shape, new_operands, device_list(), constrain_layout(), channel_id(),
      split_dimension());
}

HloInstructionProto HloAllToAllInstruction::ToProto() const {
  HloInstructionProto proto = HloCollectiveInstruction::ToProto();
  if (split_dimension_) {
    proto.add_dimensions(*split_dimension_);
  }
  return proto;
}

void HloAllToAllInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  HloCollectiveInstruction::PrintExtraAttributesImpl(printer, options);
  if (split_dimension_) {
    printer.Next([this](Printer* printer) {
      AppendCat(printer, "dimensions={", *split_dimension_, "}");
    });
  }
}

bool HloAllToAllInstruction::IdenticalSlowPathIgnoringChannelIdValues(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloAllToAllInstruction&>(other);
  return HloCollectiveInstruction::IdenticalSlowPathIgnoringChannelIdValues(
             other, eq_computations) &&
         split_dimension_ == casted_other.split_dimension();
}

HloRaggedAllToAllInstruction::HloRaggedAllToAllInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    const CollectiveDeviceList& device_list,
    const std::optional<int64_t>& channel_id)
    : HloCollectiveInstruction(HloOpcode::kRaggedAllToAll, shape, operands,
                               device_list,
                               /*constrain_layout=*/false, channel_id) {}

HloRaggedAllToAllInstruction::HloRaggedAllToAllInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    absl::Span<const ReplicaGroup> replica_groups,
    const std::optional<int64_t>& channel_id)
    : HloRaggedAllToAllInstruction(
          shape, operands, CollectiveDeviceList(replica_groups), channel_id) {}

std::unique_ptr<HloInstruction>
HloRaggedAllToAllInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  return std::make_unique<HloRaggedAllToAllInstruction>(
      shape, new_operands, device_list(), channel_id());
}

HloInstructionProto HloRaggedAllToAllInstruction::ToProto() const {
  HloInstructionProto proto = HloCollectiveInstruction::ToProto();
  return proto;
}

void HloRaggedAllToAllInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  HloCollectiveInstruction::PrintExtraAttributesImpl(printer, options);
}

HloCollectiveBroadcastInstruction::HloCollectiveBroadcastInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    const CollectiveDeviceList& device_list, bool constrain_layout,
    const std::optional<int64_t>& channel_id)
    : HloCollectiveInstruction(opcode, shape, operands, device_list,
                               constrain_layout, channel_id) {}

HloCollectiveBroadcastInstruction::HloCollectiveBroadcastInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
    const std::optional<int64_t>& channel_id)
    : HloCollectiveBroadcastInstruction(opcode, shape, operands,
                                        CollectiveDeviceList(replica_groups),
                                        constrain_layout, channel_id) {}

HloInstructionProto HloCollectiveBroadcastInstruction::ToProto() const {
  return HloCollectiveInstruction::ToProto();
}

std::unique_ptr<HloInstruction>
HloCollectiveBroadcastInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  return std::make_unique<HloCollectiveBroadcastInstruction>(
      opcode(), shape, new_operands, device_list(), constrain_layout(),
      channel_id());
}

HloCollectivePermuteInstruction::HloCollectivePermuteInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
    const std::optional<int64_t>& channel_id)
    : HloChannelInstruction(opcode, shape, channel_id),
      source_target_pairs_(source_target_pairs) {
  AppendOperands(operands);
  inplace_ = false;
}

HloCollectivePermuteInstruction::HloCollectivePermuteInstruction(
    HloOpcode opcode, const Shape& shape, HloInstruction* input,
    HloInstruction* output, HloInstruction* input_start_indices,
    HloInstruction* output_start_indices,
    absl::Span<const std::pair<int64_t, int64_t>> source_target_pairs,
    absl::Span<const std::vector<int64_t>> slice_sizes,
    const std::optional<int64_t>& channel_id)
    : HloChannelInstruction(opcode, shape, channel_id),
      source_target_pairs_(source_target_pairs.begin(),
                           source_target_pairs.end()),
      slice_sizes_(slice_sizes.begin(), slice_sizes.end()) {
  AppendOperand(input);
  AppendOperand(output);
  AppendOperand(input_start_indices);
  AppendOperand(output_start_indices);
  inplace_ = true;
}

HloInstructionProto HloCollectivePermuteInstruction::ToProto() const {
  HloInstructionProto proto = HloChannelInstruction::ToProto();
  for (const auto& pair : source_target_pairs()) {
    auto* proto_pair = proto.add_source_target_pairs();
    proto_pair->set_source(pair.first);
    proto_pair->set_target(pair.second);
  }
  for (const auto& slice_size : dynamic_slice_sizes_list()) {
    for (const auto& dimension_slice_size : slice_size) {
      proto.add_dynamic_slice_sizes(dimension_slice_size);
    }
  }
  return proto;
}

void HloCollectivePermuteInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  HloChannelInstruction::PrintExtraAttributesImpl(printer, options);
  printer.Next([this](Printer* printer) {
    printer->Append("source_target_pairs={");
    AppendJoin(printer, source_target_pairs(), ",",
               [](Printer* printer, const std::pair<int64_t, int64_t>& pair) {
                 AppendCat(printer, "{", pair.first, ",", pair.second);
                 printer->Append("}");
               });
    printer->Append("}");
  });
  if (!dynamic_slice_sizes_list().empty()) {
    printer.Next([this](Printer* printer) {
      printer->Append("slice_sizes={");
      AppendJoin(printer, dynamic_slice_sizes_list(), ",",
                 [](Printer* printer, const std::vector<int64_t>& slice_sizes) {
                   printer->Append("{");
                   AppendJoin(printer, slice_sizes, ",");
                   printer->Append("}");
                 });
      printer->Append("}");
    });
  }
}

bool HloCollectivePermuteInstruction::IdenticalSlowPathIgnoringChannelIdValues(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  if (opcode() != other.opcode()) {
    return false;
  }
  const auto& casted_other =
      static_cast<const HloCollectivePermuteInstruction&>(other);
  return HloChannelInstruction::IdenticalSlowPathIgnoringChannelIdValues(
             other, eq_computations) &&
         absl::c_equal(
             source_target_pairs(), casted_other.source_target_pairs(),
             [](const std::pair<int64_t, int64_t>& a,
                const std::pair<int64_t, int64_t>& b) { return a == b; }) &&
         absl::c_equal(
             dynamic_slice_sizes_list(),
             casted_other.dynamic_slice_sizes_list(),
             [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
               return absl::c_equal(a, b);
             });
}

std::unique_ptr<HloInstruction>
HloCollectivePermuteInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  if (dynamic_slice_sizes_list().empty()) {
    return std::make_unique<HloCollectivePermuteInstruction>(
        opcode(), shape,
        absl::Span<HloInstruction* const>(new_operands.subspan(0, 1)),
        source_target_pairs(), channel_id());
  }
  return std::make_unique<HloCollectivePermuteInstruction>(
      opcode(), shape, new_operands[0], new_operands[1], new_operands[2],
      new_operands[3], source_target_pairs(), dynamic_slice_sizes_list(),
      channel_id());
}

HloReverseInstruction::HloReverseInstruction(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64_t> dimensions)
    : HloDimensionsInstruction(HloOpcode::kReverse, shape, dimensions) {
  AppendOperand(operand);
}

HloInstructionProto HloDimensionsInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (int64_t dimension : dimensions_) {
    proto.add_dimensions(dimension);
  }
  return proto;
}

void HloDimensionsInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  printer.Next([this](Printer* printer) {
    printer->Append("dimensions={");
    AppendJoin(printer, dimensions(), ",");
    printer->Append("}");
  });
}

bool HloDimensionsInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloDimensionsInstruction&>(other);
  return dimensions() == casted_other.dimensions();
}

std::unique_ptr<HloInstruction> HloReverseInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloReverseInstruction>(shape, new_operands[0],
                                                 dimensions());
}

HloConcatenateInstruction::HloConcatenateInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    int64_t dimension)
    : HloDimensionsInstruction(HloOpcode::kConcatenate, shape, {dimension}) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
}

std::unique_ptr<HloInstruction>
HloConcatenateInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  return std::make_unique<HloConcatenateInstruction>(shape, new_operands,
                                                     concatenate_dimension());
}

HloReduceInstruction::HloReduceInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> args,
    absl::Span<const int64_t> dimensions_to_reduce,
    HloComputation* reduce_computation)
    : HloDimensionsInstruction(HloOpcode::kReduce, shape,
                               dimensions_to_reduce) {
  for (HloInstruction* arg : args) {
    AppendOperand(arg);
  }
  AppendComputation(reduce_computation);
}

bool HloReduceInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloReduceInstruction&>(other);
  // Reduction results are determined by the reduction dimension and the
  // reduction computation.
  return dimensions() == casted_other.dimensions() &&
         eq_computations(to_apply(), casted_other.to_apply());
}

std::unique_ptr<HloInstruction> HloReduceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size() % 2, 0);
  return std::make_unique<HloReduceInstruction>(shape, new_operands,
                                                dimensions(), to_apply());
}

HloSortInstruction::HloSortInstruction(
    const Shape& shape, int64_t dimension,
    absl::Span<HloInstruction* const> operands, HloComputation* compare,
    bool is_stable)
    : HloDimensionsInstruction(HloOpcode::kSort, shape, {dimension}),
      is_stable_(is_stable) {
  for (auto* value : operands) {
    AppendOperand(value);
  }
  AppendComputation(compare);
}

HloInstructionProto HloSortInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (int64_t dimension : dimensions_) {
    proto.add_dimensions(dimension);
  }
  proto.set_is_stable(is_stable());
  return proto;
}

void HloSortInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  printer.Next([this](Printer* printer) {
    printer->Append("dimensions={");
    AppendJoin(printer, dimensions(), ",");
    printer->Append("}");
  });
  if (is_stable()) {
    printer.Next([](Printer* printer) { printer->Append("is_stable=true"); });
  }
}

bool HloSortInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloSortInstruction&>(other);
  if (dimensions() != casted_other.dimensions()) {
    return false;
  }
  if (is_stable() != casted_other.is_stable()) {
    return false;
  }
  return eq_computations(to_apply(), other.to_apply());
}

std::unique_ptr<HloInstruction> HloSortInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  return std::make_unique<HloSortInstruction>(
      shape, dimensions_[0], new_operands, to_apply(), is_stable());
}

HloTransposeInstruction::HloTransposeInstruction(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64_t> dimensions)
    : HloDimensionsInstruction(HloOpcode::kTranspose, shape, dimensions) {
  AppendOperand(operand);
}

bool HloTransposeInstruction::IsRank2Transpose() const {
  return dimensions() == std::vector<int64_t>({1, 0}) &&
         shape().dimensions().size() == 2 &&
         std::equal(shape().dimensions().begin(), shape().dimensions().end(),
                    operand(0)->shape().dimensions().rbegin());
}

std::unique_ptr<HloInstruction>
HloTransposeInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloTransposeInstruction>(shape, new_operands[0],
                                                   dimensions());
}

HloBroadcastInstruction::HloBroadcastInstruction(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64_t> broadcast_dimension)
    : HloDimensionsInstruction(HloOpcode::kBroadcast, shape,
                               broadcast_dimension) {
  AppendOperand(operand);
}

std::unique_ptr<HloInstruction>
HloBroadcastInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloBroadcastInstruction>(shape, new_operands[0],
                                                   dimensions());
}

HloDynamicReshapeInstruction::HloDynamicReshapeInstruction(
    const Shape& shape, HloInstruction* data_operand,
    absl::Span<HloInstruction* const> dim_sizes)
    : HloInstruction(HloOpcode::kDynamicReshape, shape) {
  AppendOperand(data_operand);
  for (auto operand : dim_sizes) {
    AppendOperand(operand);
  }
}

std::unique_ptr<HloInstruction>
HloDynamicReshapeInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_GE(new_operands.size(), 1);
  return std::make_unique<HloDynamicReshapeInstruction>(
      shape, new_operands[0], new_operands.subspan(1));
}

HloReshapeInstruction::HloReshapeInstruction(const Shape& shape,
                                             HloInstruction* operand,
                                             int64_t inferred_dimension)
    : HloInstruction(HloOpcode::kReshape, shape),
      inferred_dimension_(inferred_dimension) {
  AppendOperand(operand);
}

HloInstructionProto HloReshapeInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  if (inferred_dimension_ != -1) {
    proto.add_dimensions(inferred_dimension_);
  }
  return proto;
}

void HloReshapeInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  if (inferred_dimension() == -1) {
    return;
  }
  printer.Next([this](Printer* printer) {
    AppendCat(printer, "inferred_dimension=", inferred_dimension());
  });
}

bool HloReshapeInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloReshapeInstruction&>(other);
  return inferred_dimension() == casted_other.inferred_dimension();
}

std::unique_ptr<HloInstruction> HloReshapeInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloReshapeInstruction>(shape, new_operands[0],
                                                 inferred_dimension());
}

HloMapInstruction::HloMapInstruction(const Shape& shape,
                                     absl::Span<HloInstruction* const> operands,
                                     HloComputation* map_computation)
    : HloInstruction(HloOpcode::kMap, shape) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
  AppendComputation(map_computation);
  // TODO(b/65689298) Remove code below once Map is generalized to accept
  // arbitrary map dimensions.
  dimensions_.resize(shape.dimensions().size());
  std::iota(dimensions_.begin(), dimensions_.end(), 0);
}

HloInstructionProto HloMapInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (int64_t dimension : dimensions_) {
    proto.add_dimensions(dimension);
  }
  return proto;
}

bool HloMapInstruction::IsElementwiseImpl(
    const std::optional<int64_t>& operand_idx) const {
  if (!dimensions().empty()) {
    // Check that the map is executed in elementwise compatible dimensions.
    if (dimensions().size() != shape().dimensions().size()) {
      return false;
    }
    for (int i = 0; i < dimensions().size(); ++i) {
      if (dimensions()[i] != i) {
        return false;
      }
    }
  }
  return true;
}

void HloMapInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  printer.Next([this](Printer* printer) {
    printer->Append("dimensions={");
    AppendJoin(printer, dimensions(), ",");
    printer->Append("}");
  });
}

bool HloMapInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloMapInstruction&>(other);
  return eq_computations(to_apply(), casted_other.to_apply()) &&
         dimensions() == casted_other.dimensions();
}

std::unique_ptr<HloInstruction> HloMapInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  return std::make_unique<HloMapInstruction>(shape, new_operands, to_apply());
}

HloSliceInstruction::HloSliceInstruction(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64_t> start_indices,
    absl::Span<const int64_t> limit_indices, absl::Span<const int64_t> strides)
    : HloInstruction(HloOpcode::kSlice, shape),
      slice_starts_(start_indices.begin(), start_indices.end()),
      slice_limits_(limit_indices.begin(), limit_indices.end()),
      slice_strides_(strides.begin(), strides.end()) {
  AppendOperand(operand);
  // For backward compatibility with old serialized computations: if there are
  // no strides, assume all strides are 1.
  // TODO(b/63317920): remove this code.
  if (slice_strides_.empty()) {
    slice_strides_ = std::vector<int64_t>(start_indices.size(), 1LL);
  }
}

HloInstructionProto HloSliceInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (int i = 0; i < slice_starts_.size(); ++i) {
    auto* slice_dimension = proto.add_slice_dimensions();
    slice_dimension->set_start(slice_starts_[i]);
    slice_dimension->set_limit(slice_limits_[i]);
    slice_dimension->set_stride(slice_strides_[i]);
  }
  return proto;
}

void HloSliceInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  printer.Next([this](Printer* printer) {
    const bool omit_stride = absl::c_all_of(
        slice_strides_, [](int64_t stride) { return stride == 1; });
    printer->Append("slice={");
    AppendJoin(printer, slice_starts_, ", ",
               [&](Printer* printer, auto& slice_start) {
                 const auto i = &slice_start - slice_starts_.data();
                 AppendCat(printer, "[", slice_start, ":", slice_limits_[i]);
                 if (!omit_stride) {
                   AppendCat(printer, ":", slice_strides_[i]);
                 }
                 printer->Append("]");
               });
    printer->Append("}");
  });
}

bool HloSliceInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& other_slice = static_cast<const HloSliceInstruction&>(other);
  return slice_starts_ == other_slice.slice_starts_ &&
         slice_limits_ == other_slice.slice_limits_ &&
         slice_strides_ == other_slice.slice_strides_;
}

std::unique_ptr<HloInstruction> HloSliceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloSliceInstruction>(
      shape, new_operands[0], slice_starts_, slice_limits_, slice_strides_);
}

HloConstantInstruction::HloConstantInstruction(Literal literal)
    : HloInstruction(HloOpcode::kConstant, literal.shape()),
      literal_(new Literal(std::move(literal))) {}

HloConstantInstruction::HloConstantInstruction(Literal literal,
                                               const Shape& shape)
    : HloInstruction(HloOpcode::kConstant, shape),
      literal_(new Literal(std::move(literal))) {}

HloConstantInstruction::HloConstantInstruction(std::shared_ptr<Literal> literal,
                                               const Shape& shape)
    : HloInstruction(HloOpcode::kConstant, shape), literal_(literal) {}

HloConstantInstruction::HloConstantInstruction(const Shape& shape)
    : HloInstruction(HloOpcode::kConstant, shape) {}

HloInstructionProto HloConstantInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  if (literal_) {
    *proto.mutable_literal() = literal_->ToProto();
  }
  return proto;
}

bool HloConstantInstruction::IsElementwiseImpl(
    const std::optional<int64_t>& operand_idx) const {
  return true;
}

void HloConstantInstruction::RelayoutConstant(const Layout& new_layout,
                                              const ShapeIndex& shape_index) {
  Shape* mutable_array_subshape =
      ShapeUtil::GetMutableSubshape(mutable_shape(), shape_index);
  CHECK(mutable_array_subshape->IsArray());

  // Normally array_subshape will always have a layout, but this invariant is
  // temporarily broken in LayoutAssignment::AssignLayouts where all shape
  // layouts are cleared. The inner condition below ensures that we don't
  // unnecessarily relayout literals in that case.

  if (!mutable_array_subshape->has_layout() ||
      !LayoutUtil::Equal(mutable_array_subshape->layout(), new_layout)) {
    if (!LayoutUtil::Equal(
            new_layout,
            ShapeUtil::GetSubshape(literal().shape(), shape_index).layout())) {
      // Only relayout literals if that's really necessary.
      Literal new_literal = literal_->Relayout(new_layout, shape_index);
      *mutable_literal() = std::move(new_literal);
    }
    *mutable_array_subshape->mutable_layout() = new_layout;
  }
}

bool HloConstantInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& other_slice = static_cast<const HloSliceInstruction&>(other);
  return literal() == other_slice.literal();
}

std::unique_ptr<HloInstruction>
HloConstantInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  if (!literal_) {
    return std::make_unique<HloConstantInstruction>(this->shape());
  }
  // Literal's shape may have no/different tiling info. Use this instruction's
  // shape instead.
  CHECK(Shape::Equal().MinorToMajorOnlyInLayout()(literal_->shape(),
                                                  this->shape()));
  return std::make_unique<HloConstantInstruction>(literal_, this->shape());
}

void HloConstantInstruction::PrintOperandsWithCanonicalNameMap(
    Printer* printer, const HloPrintOptions& options,
    CanonicalNameMap* canonical_name_map) const {
  if (options.print_only_essential_constants()) {
    if (!literal_) {
      printer->Append("{...}");
      return;
    }
    if (literal().IsAll(0)) {
      printer->Append("0");
      return;
    }
    if (literal().IsAll(1)) {
      printer->Append("1");
      return;
    }
    if (shape().AreAllLeavesIntegers()) {
      // The following prevents high compilation latencies caused by serializing
      // large constant tensors; for example: b/265669625. The limit of 500k was
      // chosen empirically to make sure that serialization of the `literal_` is
      // less than a second.
      const auto num_constants =
          shape().IsArray()
              ? absl::c_accumulate(shape().dimensions(), 1, std::multiplies<>())
              : 1;
      if (num_constants <= 500'000) {
        literal_->PrintWithoutShapeOneline(printer);
        return;
      }
    }
    printer->Append("{...}");
    return;
  }

  // For constants, show the actual value in place of an empty operand list.
  if (literal_ &&
      ((shape().IsArray() && ShapeUtil::ElementsIn(shape()) <= 10) ||
       options.print_large_constants())) {
    // Literal::ToString emits multidimensional arrays over multiple
    // lines. Compact this into one line by stripping out white space.
    literal_->PrintWithoutShapeOneline(printer);
  } else {
    // Do not show large constants or tuples.
    printer->Append("{...}");
  }
}

HloCallableInstruction::HloCallableInstruction(HloOpcode opcode,
                                               const Shape& shape)
    : HloInstruction(opcode, shape) {}

HloCallableInstruction::HloCallableInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands)
    : HloInstruction(opcode, shape) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
  SetAndSanitizeName(HloOpcodeString(opcode));
}

HloCallableInstruction::HloCallableInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    HloComputation* called_computation, absl::string_view prefix)
    : HloInstruction(opcode, shape) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
  SetAndSanitizeName(absl::StrCat(prefix, HloOpcodeString(opcode)));
  AppendComputation(called_computation);
}

HloCallableInstruction::HloCallableInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    absl::Span<HloComputation* const> called_computations)
    : HloInstruction(opcode, shape) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
  SetAndSanitizeName(HloOpcodeString(opcode));
  for (auto called_computation : called_computations) {
    AppendComputation(called_computation);
  }
}

HloCallableInstruction::HloCallableInstruction(HloOpcode opcode,
                                               const Shape& shape,
                                               const std::string& name,
                                               const std::string& attributes,
                                               int64_t version)
    : HloInstruction(opcode, shape) {
  auto frontend_attributes =
      BuildFrontendAttributesForComposite(name, attributes, version);
  add_frontend_attributes(frontend_attributes);
  set_is_composite(true);
}

HloCallableInstruction::HloCallableInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands, HloComputation* decomposition,
    const std::string& name, const std::string& attributes, int64_t version)
    : HloInstruction(opcode, shape) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
  SetAndSanitizeName(HloOpcodeString(opcode));
  AppendComputation(decomposition);

  auto frontend_attributes =
      BuildFrontendAttributesForComposite(name, attributes, version);
  add_frontend_attributes(frontend_attributes);
  set_is_composite(true);
}

HloCallableInstruction::~HloCallableInstruction() { ClearCalledComputations(); }

HloComputation* HloCallableInstruction::called_computation() const {
  CHECK(!called_computations().empty());
  return called_computations().front();
}

HloInstruction* HloCallableInstruction::called_computation_root() const {
  return called_computation()->root_instruction();
}

HloInstruction* HloCallableInstruction::AddCallOperand(
    HloInstruction* new_operand) {
  CHECK_EQ(operand_count(),
           called_computation()->parameter_instructions().size());
  const int64_t param_no = operand_count();
  std::string param_name = StrCat("param_", param_no);
  HloInstruction* called_computation_parameter =
      called_computation()->AddParameter(HloInstruction::CreateParameter(
          param_no, new_operand->shape(), param_name));
  AppendOperand(new_operand);
  return called_computation_parameter;
}

HloInstruction* HloCallableInstruction::AppendInstructionIntoCalledComputation(
    HloInstruction* instruction_to_append, bool add_output) {
  // When add_output is false, this callable instruction must be a user of
  // instruction_to_append.
  if (!add_output) {
    CHECK(IsUserOf(instruction_to_append));
  }
  return CloneAndAppendInstructionIntoCalledComputation(instruction_to_append,
                                                        add_output);
}

HloInstruction*
HloCallableInstruction::CloneAndAppendInstructionIntoCalledComputation(
    HloInstruction* instruction_to_append, bool add_output) {
  VLOG(3) << "CloneAndAppendInstructionIntoCalledComputation:\n"
          << instruction_to_append->ToString();
  HloInstruction* clone = nullptr;
  bool do_not_clone =
      instruction_to_append->opcode() == HloOpcode::kTuple &&
      absl::c_all_of(instruction_to_append->users(), [](HloInstruction* u) {
        return u->opcode() == HloOpcode::kGetTupleElement;
      });
  if (called_computations().empty()) {
    // New fusion instruction. It should not be a multi-output instruction.
    CHECK(!add_output);
    auto builder = HloComputation::Builder(default_called_computation_name());
    builder.AddInstruction(instruction_to_append->Clone(/*suffix=*/""));
    auto* new_computation = CHECK_NOTNULL(instruction_to_append->GetModule())
                                ->AddEmbeddedComputation(builder.Build());
    AppendComputation(new_computation);
    if (opcode() == HloOpcode::kFusion) {
      new_computation->SetFusionInstruction(this);
    }

    clone = called_computation_root();
  } else {
    // When add_output is false, instruction_to_append is necessarily an
    // operand of the callable instruction. After appending this will no
    // longer be the case. Remove the operand from the operand list and remove
    // its corresponding called computation parameter instruction.
    bool in_operand_list =
        absl::c_linear_search(operands(), instruction_to_append);
    CHECK(add_output || in_operand_list);
    if (do_not_clone) {
      // We assume all uses of a kTuple operation are GTE ops. In this case,
      // we don't need to clone 'instruction_to_append'.
      CHECK(!in_operand_list);
      clone = instruction_to_append;
    } else {
      clone = called_computation()->AddInstruction(
          instruction_to_append->Clone(/*suffix=*/""));
    }
    const auto& called_computation_parameters =
        called_computation()->parameter_instructions();
    for (int64_t operand_num = 0; operand_num < operand_count();
         ++operand_num) {
      if (instruction_to_append == operand(operand_num)) {
        // Replace the called computation parameter instruction's uses with
        // the clone.
        HloInstruction* called_computation_parameter =
            called_computation_parameters[operand_num];
        TF_CHECK_OK(called_computation_parameter->ReplaceAllUsesWith(clone));

        // Remove the corresponding called computation parameter and operand
        // from their respective vectors.
        TF_CHECK_OK(called_computation()->RemoveParameter(operand_num));
        RemoveOperandAt(operand_num);
        break;
      }
    }
    // We've cloned instruction_to_append into this callable instruction, so
    // this callable instruction is no longer a use of instruction_to_append.
    if (in_operand_list) {
      DetachFrom(instruction_to_append);
      // When the instruction_to_append does not have other users, we don't
      // need to generate a multi-output instruction.
      if (instruction_to_append->user_count() == 0) {
        add_output = false;
      }
    }
  }

  // Reread the parameters in the computation.
  const auto& called_computation_parameters =
      called_computation()->parameter_instructions();

  // Add each operand of the clone as an operand of the callable instruction.
  // A complication is that some clone operands may already be operands of the
  // callable instruction.
  for (int64_t operand_num = 0; operand_num < clone->operand_count();
       ++operand_num) {
    HloInstruction* operand = clone->mutable_operand(operand_num);

    // See if this operand is already an operand of the callable instruction.
    CHECK_EQ(operands().size(), called_computation_parameters.size());
    HloInstruction* called_computation_parameter = nullptr;
    for (int64_t i = 0; i < operands().size(); ++i) {
      if (this->operand(i) == operand) {
        called_computation_parameter = called_computation_parameters[i];
        break;
      }
    }

    if (called_computation_parameter == nullptr) {
      // Clone's operand was not already an operand of the callable
      // instruction. Add it as an operand and add a corresponding called
      // computation parameter instruction.

      // No need to create an original value for an added parameter as the
      // original value is saved in the corresponding argument.
      called_computation_parameter = AddCallOperand(operand);
    }
    TF_CHECK_OK(
        clone->ReplaceOperandWith(operand_num, called_computation_parameter));
  }

  if (clone != instruction_to_append) {
    // Copy over the original value to the clone of a fused instruction.
    clone->CopyOriginalValue(instruction_to_append,
                             /*clone=*/false);
    VLOG(2) << "New clone:\n" << clone->ToString();
  }

  if (add_output) {
    int64_t user_count = instruction_to_append->user_count();
    CHECK(user_count > 0 || instruction_to_append->IsRoot())
        << "Unable to append instruction: " << instruction_to_append->ToString()
        << ", which has " << user_count << " users.";
    HloInstruction* root = called_computation_root();
    // Check whether we have replaced an existing fusion root with 'clone'. If
    // yes, no need to add a duplicate root.
    if (root->opcode() == HloOpcode::kTuple) {
      for (int64_t i = 0; i < root->operand_count(); ++i) {
        if (root->operand(i) == clone) {
          HloInstruction* new_gte = AddInstruction(
              HloInstruction::CreateGetTupleElement(clone->shape(), this, i));
          TF_CHECK_OK(instruction_to_append->ReplaceAllUsesWith(new_gte));
          return clone;
        }
      }
    }
    // If this is already a multioutput instruction, expand the root tuple
    // by 1.
    HloInstruction::InstructionVector tuple_elements;
    bool newly_created_tuple_instr = false;
    if (root->opcode() == HloOpcode::kTuple) {
      tuple_elements = root->operands();
    } else {
      tuple_elements.push_back(root);
      newly_created_tuple_instr = true;
    }
    if (clone->opcode() == HloOpcode::kTuple) {
      for (auto inst : clone->operands()) {
        tuple_elements.push_back(inst);
      }
    } else {
      tuple_elements.push_back(clone);
    }
    HloInstruction* new_root = called_computation()->AddInstruction(
        HloInstruction::CreateTuple(tuple_elements));

    // No need to create an original value for a new root with added outputs
    // as the original value is saved in the get-tuple-element instructions
    // that use it.
    called_computation()->set_root_instruction(new_root,
                                               /*accept_different_shape=*/true);
    *mutable_shape() = new_root->shape();
    // The instruction might have an existing sharding, which will no longer
    // be valid after we change the shape. So clear the sharding.
    clear_sharding();
    if (root->opcode() == HloOpcode::kTuple) {
      TF_CHECK_OK(called_computation()->RemoveInstruction(root));
    }

    // If this is a newly created multioutput instruction, we need to update
    // the use of the original callable instruction.
    if (newly_created_tuple_instr) {
      HloInstruction* new_instr = AddInstruction(
          HloInstruction::CreateGetTupleElement(root->shape(), this, 0));
      TF_CHECK_OK(ReplaceAllUsesWithDifferentShape(new_instr));
    }
    int64_t index = tuple_elements.size();
    if (do_not_clone) {
      CHECK_EQ(clone, instruction_to_append);
      index -= instruction_to_append->operand_count();
      std::vector<HloInstruction*> to_be_removed;
      const auto& users = instruction_to_append->users();
      to_be_removed.reserve(users.size());
      for (auto old_gte : users) {
        CHECK_EQ(old_gte->opcode(), HloOpcode::kGetTupleElement);
        int64_t old_tuple_index = old_gte->tuple_index();
        HloInstruction* new_gte =
            AddInstruction(HloInstruction::CreateGetTupleElement(
                old_gte->shape(), this, index + old_tuple_index));
        TF_CHECK_OK(old_gte->ReplaceAllUsesWith(new_gte));
        to_be_removed.push_back(old_gte);
      }
      for (auto old_gte : to_be_removed) {
        TF_CHECK_OK(parent()->RemoveInstruction(old_gte));
      }
    } else {
      HloInstruction* new_gte =
          AddInstruction(HloInstruction::CreateGetTupleElement(
              clone->shape(), this, index - 1));
      TF_CHECK_OK(instruction_to_append->ReplaceAllUsesWith(new_gte));
    }
  }

  return clone;
}

absl::InlinedVector<HloComputation*, 1>
HloCallableInstruction::GetOrCloneCalledComputations(
    HloCloneContext* context) const {
  HloModule* module = context != nullptr ? context->module() : GetModule();
  absl::InlinedVector<HloComputation*, 1> new_called_computations;
  for (auto* comp : called_computations()) {
    HloComputation* new_custom_call_computation = nullptr;
    if (context != nullptr) {
      new_custom_call_computation = context->FindComputation(comp);
    }
    if (new_custom_call_computation == nullptr) {
      new_custom_call_computation =
          module->AddEmbeddedComputation(comp->Clone("clone", context));
    }
    new_called_computations.push_back(new_custom_call_computation);
  }
  return new_called_computations;
}

void HloCallableInstruction::RecursivelySetComputationsThreadName(
    absl::string_view execution_thread,
    bool skip_async_execution_thread_overwrite) {
  for (HloComputation* comp : called_computations()) {
    SetThreadName(comp, execution_thread,
                  skip_async_execution_thread_overwrite);
  }
}

HloFusionInstruction::HloFusionInstruction(const Shape& shape,
                                           FusionKind fusion_kind,
                                           HloInstruction* fused_root,
                                           absl::string_view prefix)
    : HloCallableInstruction(HloOpcode::kFusion, shape),
      fusion_kind_(fusion_kind) {
  CHECK(fused_root != nullptr);
  SetAndSanitizeName(absl::StrCat(prefix, HloOpcodeString(opcode())));

  set_metadata(fused_root->metadata());
  set_frontend_attributes(fused_root->frontend_attributes());
  CHECK(fused_root->IsFusible()) << fused_root->ToString();
  CloneAndAppendInstructionIntoCalledComputation(fused_root);
}

HloFusionInstruction::HloFusionInstruction(
    const Shape& shape, FusionKind fusion_kind,
    absl::Span<HloInstruction* const> operands,
    HloComputation* fusion_computation, absl::string_view prefix)
    : HloCallableInstruction(HloOpcode::kFusion, shape, operands,
                             fusion_computation, prefix),
      fusion_kind_(fusion_kind) {
  fusion_computation->SetFusionInstruction(this);
}

HloFusionInstruction::~HloFusionInstruction() {
  ClearFusionComputationInstruction();
}

void HloFusionInstruction::ClearFusionComputationInstruction() {
  // Each fusion calls a single computation, but we use called_computations()
  // instead of fused_instructions_computation(), because the order in which
  // things get destructed can vary; the fusion computation's back-pointer may
  // already be null, which violates a check in
  // fused_instructions_computation.
  for (HloComputation* computation : called_computations()) {
    // Some passes that rewrite fusions may reassign a fusion computation to a
    // different fusion instruction as this instruction gets destructed.
    if (computation->FusionInstruction() == this) {
      computation->SetFusionInstruction(nullptr);
    }
  }
}

void HloFusionInstruction::ClearCalledComputations() {
  ClearFusionComputationInstruction();
  HloInstruction::ClearCalledComputations();
}

HloInstruction*
HloFusionInstruction::CloneAndAppendInstructionIntoCalledComputation(
    HloInstruction* instruction_to_append, bool add_output) {
  CHECK(instruction_to_append->IsFusible())
      << instruction_to_append->ToString();
  return HloCallableInstruction::CloneAndAppendInstructionIntoCalledComputation(
      instruction_to_append, add_output);
}

std::string HloFusionInstruction::ToCategory() const {
  switch (fusion_kind()) {
    case FusionKind::kLoop:
      return "loop fusion";
    case FusionKind::kInput:
      return "input fusion";
    case FusionKind::kOutput:
      return "output fusion";
    case FusionKind::kCustom:
      return "custom fusion";
  }
}

HloInstructionProto HloFusionInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_fusion_kind() = std::string(xla::ToString(fusion_kind()));
  for (const auto& pair : output_to_operand_aliasing()) {
    auto aliasing = proto.add_output_operand_aliasing();
    aliasing->set_operand_index(pair.second.first);
    for (int64_t index : pair.first) {
      aliasing->add_output_shape_index(index);
    }
    for (int64_t index : pair.second.second) {
      aliasing->add_operand_shape_index(index);
    }
  }
  proto.add_called_computation_ids(
      fused_instructions_computation()->unique_id());
  return proto;
}

bool HloFusionInstruction::IsElementwiseImpl(
    const std::optional<int64_t>& operand_idx) const {
  if (!operand_idx.has_value()) {
    for (auto* fused : fused_instructions()) {
      if (fused->opcode() != HloOpcode::kParameter && !fused->IsElementwise()) {
        return false;
      }
    }
    return true;
  }
  // A loop-fusion is elementwise on an operand if all operations (computed
  // using BFS) between the operand and the fused root are elementwise.
  std::deque<HloInstruction*> worklist;
  absl::flat_hash_set<const HloInstruction*> visited;
  worklist.push_back(fused_parameter(operand_idx.value()));
  visited.insert(fused_parameter(operand_idx.value()));
  while (!worklist.empty()) {
    HloInstruction* operand = worklist.front();
    worklist.pop_front();
    for (HloInstruction* user : operand->users()) {
      CHECK_GE(user->unique_id(), 0);
      if (ContainsKey(visited, user)) {
        continue;
      }
      if (user->IsElementwise() ||
          IsInstructionElementwiseOnOperand(user, operand)) {
        worklist.push_back(user);
        visited.insert(user);
      } else {
        return false;
      }
    }
  }
  return true;
}

HloInstruction* HloFusionInstruction::AddFusionOperand(
    HloInstruction* new_operand) {
  return AddCallOperand(new_operand);
}

void HloFusionInstruction::MergeFusionInstruction(
    HloFusionInstruction* instruction_to_merge) {
  CHECK(absl::c_linear_search(operands(), instruction_to_merge));
  // Clone the instruction from which to merge fused instructions.
  std::unique_ptr<HloInstruction> cloned = instruction_to_merge->Clone();
  HloFusionInstruction* cloned_fusion =
      static_cast<HloFusionInstruction*>(cloned.get());
  // Replace uses of fused parameters with the corresponding operand of the
  // fusion.  Add all non-parameter fused instructions to
  // 'unfused_instructions' to be merged into 'this'.  This is done in reverse
  // post order.
  std::vector<HloInstruction*> unfused_instructions;
  auto fused_instructions = cloned_fusion->fused_instructions_computation()
                                ->MakeInstructionPostOrder();
  for (auto fused_it = fused_instructions.rbegin();
       fused_it != fused_instructions.rend(); ++fused_it) {
    auto fused_instruction = *fused_it;
    if (fused_instruction->opcode() == HloOpcode::kParameter) {
      TF_CHECK_OK(
          fused_instruction->ReplaceAllUsesWith(cloned_fusion->mutable_operand(
              fused_instruction->parameter_number())));
    } else {
      unfused_instructions.push_back(fused_instruction);
    }
  }

  // If there are no unfused instructions, the fused computation must consist
  // only of kParameter instructions. Make the operand of the corresponding
  // parameter number the new root.
  HloInstruction* unfused_root =
      unfused_instructions.empty()
          ? instruction_to_merge->mutable_operand(
                instruction_to_merge->fused_instructions_computation()
                    ->root_instruction()
                    ->parameter_number())
          : unfused_instructions.front();
  CHECK(unfused_root == cloned_fusion->fused_expression_root() ||
        unfused_instructions.empty());
  // Replace instruction_to_merge use of 'this' with unfused_root.
  TF_CHECK_OK(instruction_to_merge->ReplaceUseWith(this, unfused_root));

  // Build a dummy root for the cloned fusion as we may remove the original
  // root in the fusion process.
  if (!unfused_instructions.empty()) {
    HloComputation* computation = unfused_root->parent();
    auto* dummy_root = computation->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(U32)));
    computation->set_root_instruction(dummy_root,
                                      /*accept_different_shape=*/true);
  }

  // Fuse 'unfused_instructions' into 'this'. Everytime we fuse an instruction
  // we remove it from the closed fusion node. This is so that we don't add
  // extra users to the producer of that instruction (we use user count to
  // decide if a side-effectful instruction is fusible).
  for (auto& instruction : unfused_instructions) {
    auto* fused = FuseInstruction(instruction);
    TF_CHECK_OK(instruction->ReplaceAllUsesWith(fused));
    TF_CHECK_OK(instruction->parent()->RemoveInstruction(instruction));
  }
  CHECK_EQ(0, cloned_fusion->user_count());
  TF_CHECK_OK(GetModule()->RemoveEmbeddedComputation(
      cloned_fusion->fused_instructions_computation()));
}

void HloFusionInstruction::MergeFusionInstructionIntoMultiOutput(
    HloFusionInstruction* instruction_to_merge) {
  // Add all non-parameter fused instructions to 'unfused_instructions' to be
  // merged into 'this'. `old_to_new' maps the instructions in the fused node
  // to the disassembled fusion instructions.
  // Note that we add the unfused instructions to this->parent_ computation.
  // This is necessary because the unique_id needs for an instruction and
  // it's only added when inserting to the computation.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> old_to_new;
  std::vector<HloInstruction*> unfused_instructions;
  absl::flat_hash_set<const HloInstruction*> new_roots;
  std::vector<std::pair<HloInstruction*, int64_t>> old_fusion_outputs;
  auto computation_to_merge =
      instruction_to_merge->fused_instructions_computation();
  for (auto fused_instruction :
       computation_to_merge->MakeInstructionPostOrder()) {
    if (fused_instruction->opcode() == HloOpcode::kParameter) {
      InsertOrDie(&old_to_new, fused_instruction,
                  instruction_to_merge->mutable_operand(
                      fused_instruction->parameter_number()));
      continue;
    }
    // If 'instruction_to_merge' is a multi-output fusion, we need to skip the
    // root tuple, but remember which of the fusion outputs need to become
    // fusion outputs of the merged fusion.
    if (fused_instruction->opcode() == HloOpcode::kTuple &&
        fused_instruction == instruction_to_merge->fused_expression_root()) {
      for (const HloInstruction* user : instruction_to_merge->users()) {
        CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
        old_fusion_outputs.emplace_back(
            fused_instruction->mutable_operand(user->tuple_index()),
            user->tuple_index());
        bool has_outside_user = false;
        for (HloInstruction* gte_user : user->users()) {
          if (gte_user != this) {
            has_outside_user = true;
            break;
          }
        }
        if (!has_outside_user && !user->IsRoot()) {
          continue;
        }

        new_roots.insert(
            FindOrDie(old_to_new, old_fusion_outputs.back().first));
      }
      continue;
    }

    // Here we clone the insertion and call FuseInstructionIntoMultiOutput()
    // which clones again. This can be improved.
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(fused_instruction->operand_count());
    for (HloInstruction* new_operand : fused_instruction->mutable_operands()) {
      new_operands.push_back(FindOrDie(old_to_new, new_operand));
    }
    auto cloned_instruction =
        parent()->AddInstruction(fused_instruction->CloneWithNewOperands(
            fused_instruction->shape(), new_operands, /*suffix=*/"clone"));
    // Copy over the original value to the clone of a fused instruction.
    // This is necessary as the clone will be cloned again when the clone is
    // fused in FuseInstructionIntoMultiOutput(). This can be skipped if we
    // improve the code to only clone once as stated in the preceding comment.
    cloned_instruction->CopyOriginalValue(fused_instruction, /*clone=*/true);
    unfused_instructions.push_back(cloned_instruction);
    InsertOrDie(&old_to_new, fused_instruction, cloned_instruction);
  }
  if (instruction_to_merge->IsMultiOutputFusion()) {
    for (auto [old_root, tuple_index] : old_fusion_outputs) {
      auto new_root = FindOrDie(old_to_new, old_root);
      // Replace the get-tuple-element op on 'instruction_to_merge' referencing
      // the same tuple index as 'old_root' with 'new_root'.
      for (HloInstruction* gte : instruction_to_merge->users()) {
        if (gte->opcode() == HloOpcode::kGetTupleElement &&
            gte->tuple_index() == tuple_index) {
          TF_CHECK_OK(gte->ReplaceAllUsesWith(new_root));
          TF_CHECK_OK(gte->parent()->RemoveInstruction(gte));
        }
      }
    }
  } else {
    // If there are no unfused instructions, the fused computation must consist
    // only of kParameter instructions. Make the operand of the corresponding
    // parameter number the new root.
    HloInstruction* unfused_root =
        unfused_instructions.empty()
            ? instruction_to_merge->mutable_operand(
                  instruction_to_merge->fused_instructions_computation()
                      ->root_instruction()
                      ->parameter_number())
            : unfused_instructions.back();
    new_roots.insert(unfused_root);
    TF_CHECK_OK(instruction_to_merge->ReplaceAllUsesWith(unfused_root));
  }
  TF_CHECK_OK(
      instruction_to_merge->parent()->RemoveInstruction(instruction_to_merge));
  if (GetModule()) {
    TF_CHECK_OK(GetModule()->RemoveEmbeddedComputation(computation_to_merge));
  }
  for (int64_t i = unfused_instructions.size() - 1; i >= 0; --i) {
    HloInstruction* instruction = unfused_instructions[i];
    if (new_roots.contains(instruction)) {
      FuseInstructionIntoMultiOutput(instruction);
    } else {
      FuseInstruction(instruction);
    }
    TF_CHECK_OK(instruction->parent()->RemoveInstruction(instruction));
  }
}

HloComputation* HloFusionInstruction::fused_instructions_computation() const {
  CHECK_EQ(called_computations().size(), 1);
  auto* fused_instructions_computation = called_computations().front();
  CHECK(fused_instructions_computation->IsFusionComputation())
      << "Computation " << fused_instructions_computation->name()
      << " is not a fusion kind";
  return fused_instructions_computation;
}

HloInstruction* HloFusionInstruction::fused_expression_root() const {
  return fused_instructions_computation()->root_instruction();
}

HloInstruction* HloFusionInstruction::fused_parameter(
    int64_t parameter_number) const {
  return fused_instructions_computation()->parameter_instruction(
      parameter_number);
}

const HloInstruction::InstructionVector&
HloFusionInstruction::fused_parameters() const {
  return fused_instructions_computation()->parameter_instructions();
}

tsl::gtl::iterator_range<HloInstructionUnwrappingConstIterator>
HloFusionInstruction::fused_instructions() const {
  const HloComputation* subcomp = fused_instructions_computation();
  return subcomp->instructions();
}

tsl::gtl::iterator_range<HloInstructionUnwrappingIterator>
HloFusionInstruction::fused_instructions() {
  return fused_instructions_computation()->instructions();
}

int64_t HloFusionInstruction::fused_instruction_count() const {
  return fused_instructions_computation()->instruction_count();
}

void HloFusionInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  printer.Next([this](Printer* printer) {
    AppendCat(printer, "kind=", xla::ToString(fusion_kind()));
  });
  if (!output_to_operand_aliasing().empty()) {
    printer.Next([this](Printer* printer) {
      printer->Append("output_to_operand_aliasing={");
      AppendJoin(printer, output_to_operand_aliasing(), ", ",
                 [](Printer* printer, auto& pair) {
                   AppendCat(printer, pair.first.ToString(), ": (",
                             pair.second.first, ", ");
                   AppendCat(printer, pair.second.second.ToString(), ")");
                 });
      printer->Append("}");
    });
  }
}

bool HloFusionInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  return fusion_kind() == other.fusion_kind() &&
         output_to_operand_aliasing() ==
             static_cast<const HloFusionInstruction&>(other)
                 .output_to_operand_aliasing() &&
         eq_computations(fused_instructions_computation(),
                         other.fused_instructions_computation());
}

std::unique_ptr<HloInstruction> HloFusionInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  auto new_fused_computation = GetOrCloneCalledComputations(context);
  CHECK_EQ(new_fused_computation.size(), 1);
  auto new_fusion_instruction = std::make_unique<HloFusionInstruction>(
      shape, fusion_kind(), new_operands, new_fused_computation.front());
  new_fusion_instruction->set_output_to_operand_aliasing(
      output_to_operand_aliasing());
  return new_fusion_instruction;
}

absl::Status HloFusionInstruction::DeduplicateFusionOperands() {
  if (IsCustomFusion()) {
    return absl::OkStatus();
  }
  absl::flat_hash_map<const HloInstruction*, int> operand_indices;
  std::vector<int> operands_to_remove;
  const int count = operand_count();
  operands_to_remove.reserve(count);
  for (int i = 0; i < count; ++i) {
    auto emplace_result = operand_indices.emplace(operand(i), i);
    if (!emplace_result.second) {
      TF_RETURN_IF_ERROR(fused_parameter(i)->ReplaceAllUsesWith(
          fused_parameter(emplace_result.first->second)));
      operands_to_remove.push_back(i);
    }
  }
  if (operands_to_remove.empty()) {
    return absl::OkStatus();
  }
  TF_RETURN_IF_ERROR(fused_instructions_computation()
                         ->RemoveUnusedParametersFromFusedComputation());
  RemoveOperandsAtAscendingIndices(operands_to_remove);
  return absl::OkStatus();
}

HloCallInstruction::HloCallInstruction(const Shape& shape,
                                       HloInstruction* called_computation_root)
    : HloCallableInstruction(HloOpcode::kCall, shape) {
  CHECK(called_computation_root != nullptr);
  SetAndSanitizeName(HloOpcodeString(opcode()));
  set_metadata(called_computation_root->metadata());
  CloneAndAppendInstructionIntoCalledComputation(called_computation_root);
}

HloCallInstruction::HloCallInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* called_computation)
    : HloCallableInstruction(HloOpcode::kCall, shape, operands,
                             called_computation) {}

HloCallInstruction::HloCallInstruction(const Shape& shape,
                                       HloInstruction* decomposition_root,
                                       const std::string& name,
                                       const std::string& attributes,
                                       int64_t version)
    : HloCallableInstruction(HloOpcode::kCall, shape, name, attributes,
                             version) {
  CHECK(decomposition_root != nullptr);
  SetAndSanitizeName(HloOpcodeString(opcode()));

  FrontendAttributes frontend_attributes;
  frontend_attributes.mutable_map()->insert({"composite.name", name});
  frontend_attributes.mutable_map()->insert(
      {"composite.attributes", attributes});
  frontend_attributes.mutable_map()->insert(
      {"composite.version", std::to_string(version)});

  add_frontend_attributes(frontend_attributes);
  set_is_composite(true);
  set_metadata(decomposition_root->metadata());
  CloneAndAppendInstructionIntoCalledComputation(decomposition_root);
}

HloCallInstruction::HloCallInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* decomposition, const std::string& name,
    const std::string& attributes, int64_t version)
    : HloCallableInstruction(HloOpcode::kCall, shape, operands, decomposition,
                             name, attributes, version) {
  FrontendAttributes frontend_attributes;
  frontend_attributes.mutable_map()->insert({"composite.name", name});
  frontend_attributes.mutable_map()->insert(
      {"composite.attributes", attributes});
  frontend_attributes.mutable_map()->insert(
      {"composite.version", std::to_string(version)});

  add_frontend_attributes(frontend_attributes);
  set_is_composite(true);
}

HloRngInstruction::HloRngInstruction(
    const Shape& shape, RandomDistribution distribution,
    absl::Span<HloInstruction* const> parameters)
    : HloInstruction(HloOpcode::kRng, shape), distribution_(distribution) {
  for (HloInstruction* param : parameters) {
    AppendOperand(param);
  }
}

HloInstructionProto HloRngInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_distribution(distribution_);
  return proto;
}

void HloRngInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  printer.Next([this](Printer* printer) {
    AppendCat(printer,
              "distribution=", RandomDistributionToString(distribution_));
  });
}

bool HloRngInstruction::IsElementwiseImpl(
    const std::optional<int64_t>& operand_idx) const {
  return true;
}

bool HloRngInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloRngInstruction&>(other);
  return distribution_ == casted_other.distribution_;
}

std::unique_ptr<HloInstruction> HloRngInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  return std::make_unique<HloRngInstruction>(shape, distribution_,
                                             new_operands);
}

HloParameterInstruction::HloParameterInstruction(int64_t parameter_number,
                                                 const Shape& shape,
                                                 absl::string_view name)
    : HloInstruction(HloOpcode::kParameter, shape),
      parameter_number_(parameter_number) {
  SetAndSanitizeName(name);
}

HloInstructionProto HloParameterInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_parameter_number(parameter_number_);
  if (parameter_replicated_at_leaf_buffers_) {
    for (bool replicated : *parameter_replicated_at_leaf_buffers_) {
      proto.mutable_parameter_replication()->add_replicated_at_leaf_buffers(
          replicated);
    }
  }
  return proto;
}

void HloParameterInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  if (!parameter_replicated_at_leaf_buffers_ || !options.print_ids()) {
    return;
  }
  printer.Next([this](Printer* printer) {
    printer->Append("parameter_replication={");
    AppendJoin(printer, *parameter_replicated_at_leaf_buffers_, ",",
               [](Printer* printer, bool replicated) {
                 printer->Append(replicated ? "true" : "false");
               });
    printer->Append("}");
  });
}

void HloParameterInstruction::PrintOperandsWithCanonicalNameMap(
    Printer* printer, const HloPrintOptions& options,
    CanonicalNameMap* canonical_name_map) const {
  if (options.print_parameter_number()) {
    printer->Append(parameter_number_);
  }
}

bool HloParameterInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloParameterInstruction&>(other);
  return parameter_number() == casted_other.parameter_number();
}

std::unique_ptr<HloInstruction>
HloParameterInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  auto clone = std::make_unique<HloParameterInstruction>(parameter_number_,
                                                         shape, name());
  if (parameter_replicated_at_leaf_buffers_ &&
      ShapeUtil::Equal(shape, this->shape())) {
    clone->set_parameter_replicated_at_leaf_buffers(
        *parameter_replicated_at_leaf_buffers_);
  }
  return clone;
}

HloGetTupleElementInstruction::HloGetTupleElementInstruction(
    const Shape& shape, HloInstruction* operand, int64_t index)
    : HloInstruction(HloOpcode::kGetTupleElement, shape), tuple_index_(index) {
  AppendOperand(operand);
}

HloInstructionProto HloGetTupleElementInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_tuple_index(tuple_index_);
  return proto;
}

void HloGetTupleElementInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  printer.Next([this](Printer* printer) {
    AppendCat(printer, "index=", tuple_index());
  });
}

bool HloGetTupleElementInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloGetTupleElementInstruction&>(other);
  return tuple_index() == casted_other.tuple_index();
}

std::unique_ptr<HloInstruction>
HloGetTupleElementInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloGetTupleElementInstruction>(shape, new_operands[0],
                                                         tuple_index());
}

HloReducePrecisionInstruction::HloReducePrecisionInstruction(
    const Shape& shape, HloInstruction* operand, const int exponent_bits,
    const int mantissa_bits)
    : HloInstruction(HloOpcode::kReducePrecision, shape),
      exponent_bits_(exponent_bits),
      mantissa_bits_(mantissa_bits) {
  AppendOperand(operand);
}

HloInstructionProto HloReducePrecisionInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_exponent_bits(exponent_bits_);
  proto.set_mantissa_bits(mantissa_bits_);
  return proto;
}

void HloReducePrecisionInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  printer.Next([this](Printer* printer) {
    AppendCat(printer, "exponent_bits=", exponent_bits_);
  });
  printer.Next([this](Printer* printer) {
    AppendCat(printer, "mantissa_bits=", mantissa_bits_);
  });
}

bool HloReducePrecisionInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloReducePrecisionInstruction&>(other);
  // A reduce-precision operation is determined by the bit sizes.
  return exponent_bits() == casted_other.exponent_bits() &&
         mantissa_bits() == casted_other.mantissa_bits();
}

std::unique_ptr<HloInstruction>
HloReducePrecisionInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloReducePrecisionInstruction>(
      shape, new_operands[0], exponent_bits(), mantissa_bits());
}

HloInfeedInstruction::HloInfeedInstruction(const Shape& infeed_shape,
                                           HloInstruction* token_operand,
                                           const std::string& config)
    : HloInstruction(HloOpcode::kInfeed,
                     ShapeUtil::MakeTupleShape(
                         {infeed_shape, ShapeUtil::MakeTokenShape()})),
      infeed_config_(config) {
  AppendOperand(token_operand);
}

HloInstructionProto HloInfeedInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_infeed_config(infeed_config_);
  return proto;
}

void HloInfeedInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  if (!options.print_infeed_outfeed_config() || infeed_config_.empty()) {
    return;
  }
  printer.Next([this](Printer* printer) {
    AppendCat(printer, "infeed_config=\"", CEscape(infeed_config_), "\"");
  });
}

bool HloInfeedInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  // Not yet supported.
  return false;
}

std::unique_ptr<HloInstruction> HloInfeedInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloInfeedInstruction>(infeed_shape(), new_operands[0],
                                                infeed_config());
}

HloOutfeedInstruction::HloOutfeedInstruction(const Shape& outfeed_shape,
                                             HloInstruction* operand,
                                             HloInstruction* token_operand,
                                             absl::string_view outfeed_config)
    : HloInstruction(HloOpcode::kOutfeed, ShapeUtil::MakeTokenShape()),
      outfeed_shape_(outfeed_shape),
      outfeed_config_(outfeed_config) {
  AppendOperand(operand);
  AppendOperand(token_operand);
}

HloInstructionProto HloOutfeedInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_outfeed_config(outfeed_config());
  *proto.mutable_outfeed_shape() = outfeed_shape().ToProto();
  return proto;
}

void HloOutfeedInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  printer.Next([this](Printer* printer) {
    printer->Append("outfeed_shape=");
    ShapeUtil::PrintHumanStringWithLayout(printer, outfeed_shape_);
  });
  if (options.print_infeed_outfeed_config() && !outfeed_config_.empty()) {
    printer.Next([this](Printer* printer) {
      AppendCat(printer, "outfeed_config=\"", CEscape(outfeed_config_), "\"");
    });
  }
}

bool HloOutfeedInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  // Not yet supported.
  return false;
}

std::unique_ptr<HloInstruction> HloOutfeedInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return std::make_unique<HloOutfeedInstruction>(
      outfeed_shape(), new_operands[0], new_operands[1], outfeed_config());
}

HloConvolutionInstruction::HloConvolutionInstruction(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    int64_t feature_group_count, int64_t batch_group_count,
    const Window& window, const ConvolutionDimensionNumbers& dimension_numbers,
    const PrecisionConfig& precision_config)
    : HloInstruction(HloOpcode::kConvolution, shape),
      feature_group_count_(feature_group_count),
      batch_group_count_(batch_group_count),
      window_(window),
      convolution_dimension_numbers_(dimension_numbers),
      precision_config_(precision_config) {
  if (window_util::HasBaseDilation(window)) {
    SetAndSanitizeName(StrCat(name(), "-base-dilated"));
  }
  if (window_util::HasWindowDilation(window)) {
    SetAndSanitizeName(StrCat(name(), "-window-dilated"));
  }
  AppendOperand(lhs);
  AppendOperand(rhs);
}

std::string HloConvolutionInstruction::ToCategory() const {
  std::string category = "convolution";
  if (window_util::HasBaseDilation(window())) {
    absl::StrAppend(&category, " base-dilated");
  }
  if (window_util::HasWindowDilation(window())) {
    absl::StrAppend(&category, " window-dilated");
  }
  return category;
}

HloInstructionProto HloConvolutionInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_window() = window_;
  *proto.mutable_convolution_dimension_numbers() =
      convolution_dimension_numbers_;
  proto.set_feature_group_count(feature_group_count_);
  proto.set_batch_group_count(batch_group_count_);
  *proto.mutable_precision_config() = precision_config_;
  return proto;
}

void HloConvolutionInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  if (!window_.dimensions().empty()) {
    printer.Next([this](Printer* printer) {
      AppendCat(printer, "window={", window_util::ToString(window()), "}");
    });
  }
  printer.Next([this](Printer* printer) {
    AppendCat(
        printer, "dim_labels=",
        ConvolutionDimensionNumbersToString(convolution_dimension_numbers_));
  });
  if (feature_group_count_ != 1) {
    printer.Next([this](Printer* printer) {
      AppendCat(printer, "feature_group_count=", feature_group_count_);
    });
  }

  if (batch_group_count_ != 1) {
    printer.Next([this](Printer* printer) {
      AppendCat(printer, "batch_group_count=", batch_group_count_);
    });
  }
  PrintPrecisionConfig(printer, precision_config_);
}

bool HloConvolutionInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloConvolutionInstruction&>(other);
  if (feature_group_count_ != other.feature_group_count()) {
    return false;
  }
  if (batch_group_count_ != other.batch_group_count()) {
    return false;
  }
  return protobuf_util::HaveSameSerialization(window(),
                                              casted_other.window()) &&
         protobuf_util::HaveSameSerialization(
             convolution_dimension_numbers(),
             casted_other.convolution_dimension_numbers()) &&
         protobuf_util::HaveSameSerialization(precision_config(),
                                              casted_other.precision_config());
}

std::unique_ptr<HloInstruction>
HloConvolutionInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return std::make_unique<HloConvolutionInstruction>(
      shape, new_operands[0], new_operands[1], feature_group_count_,
      batch_group_count_, window(), convolution_dimension_numbers_,
      precision_config_);
}

HloReduceWindowInstruction::HloReduceWindowInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* init_value,
    const Window& window, HloComputation* reduce_computation)
    : HloReduceWindowInstruction(shape, absl::MakeSpan(&operand, 1),
                                 absl::MakeSpan(&init_value, 1), window,
                                 reduce_computation) {}

HloReduceWindowInstruction::HloReduceWindowInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::Span<HloInstruction* const> init_values, const Window& window,
    HloComputation* reduce_computation)
    : HloInstruction(HloOpcode::kReduceWindow, shape), window_(window) {
  for (auto* operand : operands) {
    AppendOperand(operand);
  }
  for (auto* init_value : init_values) {
    AppendOperand(init_value);
  }
  AppendComputation(reduce_computation);
}

HloInstructionProto HloReduceWindowInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_window() = window_;
  return proto;
}

void HloReduceWindowInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  if (!window_.dimensions().empty()) {
    printer.Next([this](Printer* printer) {
      AppendCat(printer, "window={", window_util::ToString(window()), "}");
    });
  }
}

bool HloReduceWindowInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloReduceWindowInstruction&>(other);
  return eq_computations(to_apply(), casted_other.to_apply()) &&
         protobuf_util::HaveSameSerialization(window(), casted_other.window());
}

std::unique_ptr<HloInstruction>
HloReduceWindowInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size() % 2, 0);
  int64_t num_operands = new_operands.size() / 2;
  return std::make_unique<HloReduceWindowInstruction>(
      shape, absl::MakeSpan(new_operands).subspan(0, num_operands),
      absl::MakeSpan(new_operands)
          .subspan(num_operands, new_operands.size() / 2),
      window(), to_apply());
}

HloSelectAndScatterInstruction::HloSelectAndScatterInstruction(
    const Shape& shape, HloInstruction* operand, HloComputation* select,
    const Window& window, HloInstruction* source, HloInstruction* init_value,
    HloComputation* scatter)
    : HloInstruction(HloOpcode::kSelectAndScatter, shape), window_(window) {
  AppendOperand(operand);
  AppendOperand(source);
  AppendOperand(init_value);
  // Select comes before scatter in the vector.
  AppendComputation(select);
  AppendComputation(scatter);
}

HloInstructionProto HloSelectAndScatterInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_window() = window_;
  return proto;
}

void HloSelectAndScatterInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  if (!window_.dimensions().empty()) {
    printer.Next([this](Printer* printer) {
      AppendCat(printer, "window={", window_util::ToString(window()), "}");
    });
  }
}

bool HloSelectAndScatterInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloSelectAndScatterInstruction&>(other);
  return eq_computations(select(), casted_other.select()) &&
         eq_computations(scatter(), casted_other.scatter()) &&
         protobuf_util::HaveSameSerialization(window(), casted_other.window());
}

std::unique_ptr<HloInstruction>
HloSelectAndScatterInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 3);
  return std::make_unique<HloSelectAndScatterInstruction>(
      shape, new_operands[0], select(), window(), new_operands[1],
      new_operands[2], scatter());
}

HloCustomCallInstruction::HloCustomCallInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::string_view custom_call_target, std::string opaque,
    CustomCallApiVersion api_version)
    : HloCallableInstruction(HloOpcode::kCustomCall, shape, operands),
      custom_call_target_(custom_call_target),
      feature_group_count_(1),
      batch_group_count_(1),
      layout_constrained_(false),
      padding_type_(PaddingType::PADDING_INVALID),
      custom_call_has_side_effect_(false),
      custom_call_schedule_(CustomCallSchedule::SCHEDULE_NONE),
      api_version_(api_version) {
  set_raw_backend_config_string(std::move(opaque));
}

HloCustomCallInstruction::HloCustomCallInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* to_apply, absl::string_view custom_call_target,
    std::string opaque, CustomCallApiVersion api_version)
    : HloCallableInstruction(HloOpcode::kCustomCall, shape, operands, to_apply),
      custom_call_target_(custom_call_target),
      feature_group_count_(1),
      batch_group_count_(1),
      layout_constrained_(false),
      padding_type_(PaddingType::PADDING_INVALID),
      custom_call_has_side_effect_(false),
      custom_call_schedule_(CustomCallSchedule::SCHEDULE_NONE),
      api_version_(api_version) {
  set_raw_backend_config_string(std::move(opaque));
}

HloCustomCallInstruction::HloCustomCallInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::Span<HloComputation* const> called_computations,
    absl::string_view custom_call_target, std::string opaque,
    CustomCallApiVersion api_version)
    : HloCallableInstruction(HloOpcode::kCustomCall, shape, operands,
                             called_computations),
      custom_call_target_(custom_call_target),
      feature_group_count_(1),
      batch_group_count_(1),
      layout_constrained_(false),
      padding_type_(PaddingType::PADDING_INVALID),
      custom_call_has_side_effect_(false),
      custom_call_schedule_(CustomCallSchedule::SCHEDULE_NONE),
      api_version_(api_version) {
  set_raw_backend_config_string(std::move(opaque));
}

HloCustomCallInstruction::HloCustomCallInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::string_view custom_call_target, std::string opaque,
    absl::Span<const Shape> operand_shapes_with_layout,
    CustomCallApiVersion api_version)
    : HloCallableInstruction(HloOpcode::kCustomCall, shape, operands),
      custom_call_target_(custom_call_target),
      feature_group_count_(1),
      batch_group_count_(1),
      layout_constrained_(true),
      padding_type_(PaddingType::PADDING_INVALID),
      operand_shapes_with_layout_(operand_shapes_with_layout.begin(),
                                  operand_shapes_with_layout.end()),
      custom_call_has_side_effect_(false),
      custom_call_schedule_(CustomCallSchedule::SCHEDULE_NONE),
      api_version_(api_version) {
  set_raw_backend_config_string(std::move(opaque));
}

HloInstructionProto HloCustomCallInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  if (window_ != nullptr) {
    *proto.mutable_window() = *window_;
  }
  if (convolution_dimension_numbers_ != nullptr) {
    *proto.mutable_convolution_dimension_numbers() =
        *convolution_dimension_numbers_;
  }
  proto.set_custom_call_target(custom_call_target_);
  proto.set_feature_group_count(feature_group_count_);
  proto.set_batch_group_count(batch_group_count_);
  *proto.mutable_precision_config() = precision_config_;
  proto.set_padding_type(padding_type_);
  if (layout_constrained()) {
    proto.set_constrain_layout(true);
    for (const Shape& shape : operand_shapes_with_layout_) {
      *proto.add_operand_shapes_with_layout() = shape.ToProto();
    }
  }
  proto.set_custom_call_has_side_effect(custom_call_has_side_effect_);
  if (literal_) {
    *proto.mutable_literal() = literal_->ToProto();
  }
  for (const auto& pair : output_to_operand_aliasing()) {
    auto aliasing = proto.add_output_operand_aliasing();
    aliasing->set_operand_index(pair.second.first);
    for (int64_t index : pair.first) {
      aliasing->add_output_shape_index(index);
    }
    for (int64_t index : pair.second.second) {
      aliasing->add_operand_shape_index(index);
    }
  }
  proto.set_custom_call_schedule(custom_call_schedule_);
  proto.set_custom_call_api_version(api_version_);
  return proto;
}

void HloCustomCallInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  if (window_ != nullptr) {
    printer.Next([this](Printer* printer) {
      AppendCat(printer, "window={", window_util::ToString(*window_), "}");
    });
  }
  if (convolution_dimension_numbers_ != nullptr) {
    printer.Next([this](Printer* printer) {
      AppendCat(
          printer, "dim_labels=",
          ConvolutionDimensionNumbersToString(*convolution_dimension_numbers_));
    });
  }
  if (feature_group_count_ != 1) {
    printer.Next([this](Printer* printer) {
      AppendCat(printer, "feature_group_count=", feature_group_count_);
    });
  }
  if (batch_group_count_ != 1) {
    printer.Next([this](Printer* printer) {
      AppendCat(printer, "batch_group_count=", batch_group_count_);
    });
  }
  PrintPrecisionConfig(printer, precision_config_);
  if (padding_type_ != PaddingType::PADDING_INVALID) {
    printer.Next([this](Printer* printer) {
      AppendCat(printer, "padding_type=", PaddingType_Name(padding_type()));
    });
  }
  // By contract, we print the custom call target even if
  // options.print_subcomputation_mode() == kOff, because the call target is
  // not an HloComputation.
  printer.Next([this](Printer* printer) {
    AppendCat(printer, "custom_call_target=\"", CEscape(custom_call_target_),
              "\"");
  });

  if (layout_constrained()) {
    printer.Next([this](Printer* printer) {
      printer->Append("operand_layout_constraints={");
      if (!operand_shapes_with_layout_.empty()) {
        ShapeUtil::PrintHumanStringWithLayout(printer,
                                              operand_shapes_with_layout_[0]);
        for (const Shape& shape :
             absl::MakeSpan(operand_shapes_with_layout_).subspan(1)) {
          printer->Append(", ");
          ShapeUtil::PrintHumanStringWithLayout(printer, shape);
        }
      }
      printer->Append("}");
    });
  }
  if (custom_call_has_side_effect_) {
    printer.Next([](Printer* printer) {
      printer->Append("custom_call_has_side_effect=true");
    });
  }
  if (literal_) {
    printer.Next([this](Printer* printer) {
      printer->Append("literal=");
      literal_->PrintWithLayoutOneline(printer);
    });
  }
  if (!output_to_operand_aliasing().empty()) {
    printer.Next([this](Printer* printer) {
      printer->Append("output_to_operand_aliasing={");
      AppendJoin(printer, output_to_operand_aliasing(), ", ",
                 [](Printer* printer, auto& pair) {
                   AppendCat(printer, pair.first.ToString(), ": (",
                             pair.second.first, ", ");
                   AppendCat(printer, pair.second.second.ToString(), ")");
                 });
      printer->Append("}");
    });
  }
  if (custom_call_schedule_ != CustomCallSchedule::SCHEDULE_NONE) {
    printer.Next([this](Printer* printer) {
      AppendCat(printer,
                "schedule=", CustomCallSchedule_Name(custom_call_schedule_));
    });
  }
  if (api_version_ != CustomCallApiVersion::API_VERSION_ORIGINAL) {
    printer.Next([this](Printer* printer) {
      AppendCat(printer,
                "api_version=", CustomCallApiVersion_Name(api_version_));
    });
  }
}

bool HloCustomCallInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloCustomCallInstruction&>(other);
  if ((window_ == nullptr) != (casted_other.window_ == nullptr) ||
      (window_ != nullptr && !protobuf_util::HaveSameSerialization(
                                 *window_, *casted_other.window_))) {
    return false;
  }
  if ((convolution_dimension_numbers_ == nullptr) !=
          (casted_other.convolution_dimension_numbers_ == nullptr) ||
      (convolution_dimension_numbers_ != nullptr &&
       !protobuf_util::HaveSameSerialization(
           convolution_dimension_numbers(),
           casted_other.convolution_dimension_numbers()))) {
    return false;
  }
  if (feature_group_count_ != casted_other.feature_group_count_) {
    return false;
  }
  if (batch_group_count_ != casted_other.batch_group_count_) {
    return false;
  }

  if (padding_type_ != casted_other.padding_type()) {
    return false;
  }

  if (layout_constrained() != casted_other.layout_constrained()) {
    return false;
  }
  if (layout_constrained()) {
    for (int64_t i = 0; i < operand_shapes_with_layout_.size(); ++i) {
      if (!ShapeUtil::Equal(operand_shapes_with_layout_[i],
                            casted_other.operand_shapes_with_layout_[i])) {
        return false;
      }
    }
  }
  if (custom_call_has_side_effect_ !=
      casted_other.custom_call_has_side_effect()) {
    return false;
  }
  if (output_to_operand_aliasing() !=
      casted_other.output_to_operand_aliasing()) {
    return false;
  }
  if (!protobuf_util::HaveSameSerialization(precision_config(),
                                            casted_other.precision_config())) {
    return false;
  }

  if (called_computations().size() != other.called_computations().size()) {
    return false;
  }
  for (int64_t i = 0; i < called_computations().size(); ++i) {
    if (!eq_computations(called_computations()[i],
                         other.called_computations()[i])) {
      return false;
    }
  }
  if (custom_call_schedule_ != casted_other.custom_call_schedule()) {
    return false;
  }
  if (HasLiteral() != casted_other.HasLiteral()) {
    return false;
  }
  if (HasLiteral() && literal() != casted_other.literal()) {
    return false;
  }
  if (api_version_ != casted_other.api_version_) {
    return false;
  }
  // Note: backend_config comparison is done in Identical, which is the
  // intended/exposed way to compare computations, and so not repeated here.
  return custom_call_target_ == casted_other.custom_call_target_;
}

std::unique_ptr<HloInstruction>
HloCustomCallInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  absl::InlinedVector<HloComputation*, 1> new_called_computations =
      GetOrCloneCalledComputations(context);

  auto cloned = std::make_unique<HloCustomCallInstruction>(
      shape, new_operands, new_called_computations, custom_call_target(),
      opaque(), api_version_);
  if (layout_constrained()) {
    cloned->layout_constrained_ = true;
    cloned->operand_shapes_with_layout_ = operand_shapes_with_layout();
  }
  if (window_ != nullptr) {
    cloned->set_window(*window_);
  }
  if (convolution_dimension_numbers_ != nullptr) {
    cloned->set_convolution_dimension_numbers(*convolution_dimension_numbers_);
  }
  if (HasLiteral()) {
    cloned->set_literal(literal().Clone());
  }
  cloned->set_feature_group_count(feature_group_count_);
  cloned->set_batch_group_count(batch_group_count_);
  cloned->set_custom_call_has_side_effect(custom_call_has_side_effect_);
  cloned->set_output_to_operand_aliasing(output_to_operand_aliasing());
  cloned->set_padding_type(padding_type_);
  *cloned->mutable_precision_config() = precision_config();
  cloned->set_custom_call_schedule(custom_call_schedule_);
  return cloned;
}

HloPadInstruction::HloPadInstruction(const Shape& shape,
                                     HloInstruction* operand,
                                     HloInstruction* padding_value,
                                     const PaddingConfig& padding_config)
    : HloInstruction(HloOpcode::kPad, shape), padding_config_(padding_config) {
  AppendOperand(operand);
  AppendOperand(padding_value);
}

HloInstructionProto HloPadInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_padding_config() = padding_config_;
  return proto;
}

void HloPadInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  printer.Next([this](Printer* printer) {
    AppendCat(printer, "padding=", xla::PaddingConfigToString(padding_config_));
  });
}

bool HloPadInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloPadInstruction&>(other);
  return protobuf_util::HaveSameSerialization(padding_config(),
                                              casted_other.padding_config());
}

std::unique_ptr<HloInstruction> HloPadInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return std::make_unique<HloPadInstruction>(shape, new_operands[0],
                                             new_operands[1], padding_config_);
}

HloDynamicSliceInstruction::HloDynamicSliceInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* start_indices,
    absl::Span<const int64_t> slice_sizes)
    : HloDynamicIndexInstruction(HloOpcode::kDynamicSlice, shape),
      dynamic_slice_sizes_(slice_sizes.begin(), slice_sizes.end()) {
  AppendOperand(operand);
  AppendOperand(start_indices);
}

HloDynamicSliceInstruction::HloDynamicSliceInstruction(
    const Shape& shape, HloInstruction* operand,
    absl::Span<HloInstruction* const> start_indices,
    absl::Span<const int64_t> slice_sizes)
    : HloDynamicIndexInstruction(HloOpcode::kDynamicSlice, shape),
      dynamic_slice_sizes_(slice_sizes.begin(), slice_sizes.end()) {
  AppendOperand(operand);
  for (HloInstruction* index : start_indices) {
    AppendOperand(index);
  }
}

HloDynamicUpdateSliceInstruction::HloDynamicUpdateSliceInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* update,
    HloInstruction* start_indices)
    : HloDynamicIndexInstruction(HloOpcode::kDynamicUpdateSlice, shape) {
  AppendOperand(operand);
  AppendOperand(update);
  AppendOperand(start_indices);
}

HloDynamicUpdateSliceInstruction::HloDynamicUpdateSliceInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* update,
    absl::Span<HloInstruction* const> start_indices)
    : HloDynamicIndexInstruction(HloOpcode::kDynamicUpdateSlice, shape) {
  AppendOperand(operand);
  AppendOperand(update);
  for (HloInstruction* index : start_indices) {
    AppendOperand(index);
  }
}

HloInstructionProto HloDynamicSliceInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (int64_t slice_size : dynamic_slice_sizes_) {
    proto.add_dynamic_slice_sizes(slice_size);
  }
  return proto;
}

void HloDynamicSliceInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  printer.Next([this](Printer* printer) {
    printer->Append("dynamic_slice_sizes={");
    AppendJoin(printer, dynamic_slice_sizes(), ",");
    printer->Append("}");
  });
}

bool HloDynamicSliceInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloMapInstruction&>(other);
  return dynamic_slice_sizes() == casted_other.dynamic_slice_sizes();
}

std::unique_ptr<HloInstruction>
HloDynamicSliceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  if (new_operands.size() == 2 &&
      new_operands[1]->shape().dimensions().size() == 1) {
    // TODO(b/118437727): Old form, remove this path.
    return std::make_unique<HloDynamicSliceInstruction>(
        shape, new_operands[0], new_operands[1], dynamic_slice_sizes_);
  }
  return std::make_unique<HloDynamicSliceInstruction>(
      shape, new_operands[0], new_operands.subspan(1), dynamic_slice_sizes_);
}

HloGatherInstruction::HloGatherInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* start_indices,
    const GatherDimensionNumbers& gather_dim_numbers,
    absl::Span<const int64_t> slice_sizes, bool indices_are_sorted)
    : HloInstruction(HloOpcode::kGather, shape),
      indices_are_sorted_(indices_are_sorted) {
  AppendOperand(operand);
  AppendOperand(start_indices);
  gather_dimension_numbers_ =
      std::make_unique<GatherDimensionNumbers>(gather_dim_numbers);
  absl::c_copy(slice_sizes, std::back_inserter(gather_slice_sizes_));
}

/*static*/ std::string HloGatherInstruction::GatherDimensionNumbersToString(
    const GatherDimensionNumbers& dim_numbers) {
  StringPrinter printer;
  PrintGatherDimensionNumbers(&printer, dim_numbers);
  return std::move(printer).ToString();
}

/*static*/ void HloGatherInstruction::PrintGatherDimensionNumbers(
    Printer* printer, const GatherDimensionNumbers& dim_numbers) {
  printer->Append("offset_dims={");
  AppendJoin(printer, dim_numbers.offset_dims(), ",");
  printer->Append("}, collapsed_slice_dims={");
  AppendJoin(printer, dim_numbers.collapsed_slice_dims(), ",");
  printer->Append("}, start_index_map={");
  AppendJoin(printer, dim_numbers.start_index_map(), ",");
  if (dim_numbers.operand_batching_dims_size()) {
    printer->Append("}, operand_batching_dims={");
    AppendJoin(printer, dim_numbers.operand_batching_dims(), ",");
  }
  if (dim_numbers.start_indices_batching_dims_size()) {
    printer->Append("}, start_indices_batching_dims={");
    AppendJoin(printer, dim_numbers.start_indices_batching_dims(), ",");
  }
  AppendCat(printer, "}, index_vector_dim=", dim_numbers.index_vector_dim());
}

/* static */ GatherDimensionNumbers HloGatherInstruction::MakeGatherDimNumbers(
    absl::Span<const int64_t> offset_dims,
    absl::Span<const int64_t> collapsed_slice_dims,
    absl::Span<const int64_t> start_index_map, int64_t index_vector_dim,
    absl::Span<const int64_t> operand_batching_dims,
    absl::Span<const int64_t> start_indices_batching_dims) {
  GatherDimensionNumbers gather_dim_numbers;
  for (int64_t output_window_dim : offset_dims) {
    gather_dim_numbers.add_offset_dims(output_window_dim);
  }
  for (int64_t elided_window_dim : collapsed_slice_dims) {
    gather_dim_numbers.add_collapsed_slice_dims(elided_window_dim);
  }
  for (int64_t gather_dim_to_input_dim : start_index_map) {
    gather_dim_numbers.add_start_index_map(gather_dim_to_input_dim);
  }
  for (int64_t operand_batching_dim : operand_batching_dims) {
    gather_dim_numbers.add_operand_batching_dims(operand_batching_dim);
  }
  for (int64_t start_indices_batching_dim : start_indices_batching_dims) {
    gather_dim_numbers.add_start_indices_batching_dims(
        start_indices_batching_dim);
  }

  gather_dim_numbers.set_index_vector_dim(index_vector_dim);
  return gather_dim_numbers;
}

HloInstructionProto HloGatherInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_gather_dimension_numbers() = gather_dimension_numbers();
  for (int64_t bound : gather_slice_sizes()) {
    proto.add_gather_slice_sizes(bound);
  }
  proto.set_indices_are_sorted(indices_are_sorted());
  return proto;
}

void HloGatherInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  printer.Next([this](Printer* printer) {
    PrintGatherDimensionNumbers(printer, gather_dimension_numbers());
  });
  printer.Next([this](Printer* printer) {
    printer->Append("slice_sizes={");
    AppendJoin(printer, gather_slice_sizes(), ",");
    printer->Append("}");
  });
  if (indices_are_sorted()) {
    printer.Next(
        [](Printer* printer) { printer->Append("indices_are_sorted=true"); });
  }
}

bool HloGatherInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloGatherInstruction&>(other);
  return protobuf_util::HaveSameSerialization(
             gather_dimension_numbers(),
             casted_other.gather_dimension_numbers()) &&
         gather_slice_sizes() == casted_other.gather_slice_sizes() &&
         indices_are_sorted() == casted_other.indices_are_sorted();
}

std::unique_ptr<HloInstruction> HloGatherInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return std::make_unique<HloGatherInstruction>(
      shape, new_operands[0], new_operands[1], gather_dimension_numbers(),
      gather_slice_sizes(), indices_are_sorted());
}

HloScatterInstruction::HloScatterInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> args,
    HloComputation* update_computation,
    const ScatterDimensionNumbers& scatter_dim_numbers, bool indices_are_sorted,
    bool unique_indices)
    : HloInstruction(HloOpcode::kScatter, shape),
      indices_are_sorted_(indices_are_sorted),
      unique_indices_(unique_indices) {
  mutable_operands().reserve(args.size());
  for (HloInstruction* arg : args) {
    AppendOperand(arg);
  }
  AppendComputation(update_computation);
  scatter_dimension_numbers_ =
      std::make_unique<ScatterDimensionNumbers>(scatter_dim_numbers);
}

/*static*/ std::string HloScatterInstruction::ScatterDimensionNumbersToString(
    const ScatterDimensionNumbers& dim_numbers) {
  StringPrinter printer;
  PrintScatterDimensionNumbers(&printer, dim_numbers);
  return std::move(printer).ToString();
}

/*static*/ void HloScatterInstruction::PrintScatterDimensionNumbers(
    Printer* printer, const ScatterDimensionNumbers& dim_numbers) {
  printer->Append("update_window_dims={");
  AppendJoin(printer, dim_numbers.update_window_dims(), ",");
  printer->Append("}, inserted_window_dims={");
  AppendJoin(printer, dim_numbers.inserted_window_dims(), ",");
  printer->Append("}, scatter_dims_to_operand_dims={");
  AppendJoin(printer, dim_numbers.scatter_dims_to_operand_dims(), ",");
  if (dim_numbers.input_batching_dims_size()) {
    printer->Append("}, input_batching_dims={");
    AppendJoin(printer, dim_numbers.input_batching_dims(), ",");
  }
  if (dim_numbers.scatter_indices_batching_dims_size()) {
    printer->Append("}, scatter_indices_batching_dims={");
    AppendJoin(printer, dim_numbers.scatter_indices_batching_dims(), ",");
  }
  AppendCat(printer, "}, index_vector_dim=", dim_numbers.index_vector_dim());
}

/* static */ ScatterDimensionNumbers
HloScatterInstruction::MakeScatterDimNumbers(
    absl::Span<const int64_t> update_window_dims,
    absl::Span<const int64_t> inserted_window_dims,
    absl::Span<const int64_t> scatter_dims_to_operand_dims,
    int64_t index_vector_dim, absl::Span<const int64_t> input_batching_dims,
    absl::Span<const int64_t> scatter_indices_batching_dims) {
  ScatterDimensionNumbers scatter_dim_numbers;
  for (int64_t update_window_dim : update_window_dims) {
    scatter_dim_numbers.add_update_window_dims(update_window_dim);
  }
  for (int64_t inserted_window_dim : inserted_window_dims) {
    scatter_dim_numbers.add_inserted_window_dims(inserted_window_dim);
  }
  for (int64_t scatter_dim_to_operand_dim : scatter_dims_to_operand_dims) {
    scatter_dim_numbers.add_scatter_dims_to_operand_dims(
        scatter_dim_to_operand_dim);
  }
  for (int64_t input_batching_dim : input_batching_dims) {
    scatter_dim_numbers.add_input_batching_dims(input_batching_dim);
  }
  for (int64_t scatter_indices_batching_dim : scatter_indices_batching_dims) {
    scatter_dim_numbers.add_scatter_indices_batching_dims(
        scatter_indices_batching_dim);
  }
  scatter_dim_numbers.set_index_vector_dim(index_vector_dim);
  return scatter_dim_numbers;
}

HloInstructionProto HloScatterInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_scatter_dimension_numbers() = scatter_dimension_numbers();
  proto.set_indices_are_sorted(indices_are_sorted());
  proto.set_unique_indices(unique_indices());
  return proto;
}

void HloScatterInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  printer.Next([this](Printer* printer) {
    printer->Append(
        ScatterDimensionNumbersToString(scatter_dimension_numbers()));
  });
  if (indices_are_sorted()) {
    printer.Next(
        [](Printer* printer) { printer->Append("indices_are_sorted=true"); });
  }
  if (unique_indices()) {
    printer.Next(
        [](Printer* printer) { printer->Append("unique_indices=true"); });
  }
}

bool HloScatterInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloScatterInstruction&>(other);
  return protobuf_util::HaveSameSerialization(
             scatter_dimension_numbers(),
             casted_other.scatter_dimension_numbers()) &&
         eq_computations(to_apply(), casted_other.to_apply()) &&
         indices_are_sorted() == casted_other.indices_are_sorted() &&
         unique_indices() == casted_other.unique_indices();
}

std::unique_ptr<HloInstruction> HloScatterInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  return std::make_unique<HloScatterInstruction>(
      shape, new_operands, to_apply(), scatter_dimension_numbers(),
      indices_are_sorted(), unique_indices());
}

HloIotaInstruction::HloIotaInstruction(const Shape& shape,
                                       int64_t iota_dimension)
    : HloInstruction(HloOpcode::kIota, shape),
      iota_dimension_(iota_dimension) {}

HloInstructionProto HloIotaInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.add_dimensions(iota_dimension());
  return proto;
}

void HloIotaInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  printer.Next([this](Printer* printer) {
    AppendCat(printer, "iota_dimension=", iota_dimension());
  });
}

bool HloIotaInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloIotaInstruction&>(other);
  return iota_dimension() == casted_other.iota_dimension();
}

std::unique_ptr<HloInstruction> HloIotaInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  return std::make_unique<HloIotaInstruction>(shape, iota_dimension());
}

HloDotInstruction::HloDotInstruction(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    const DotDimensionNumbers& dimension_numbers,
    const PrecisionConfig& precision_config,
    std::vector<SparsityDescriptor> sparsity,
    absl::Span<HloInstruction* const> sparse_meta)
    : HloInstruction(HloOpcode::kDot, shape),
      dot_dimension_numbers_(dimension_numbers),
      precision_config_(precision_config),
      sparsity_(std::move(sparsity)) {
  AppendOperand(lhs);
  AppendOperand(rhs);
  CHECK_LE(sparsity_.size(), kOperands);
  CHECK_EQ(sparsity_.size(), sparse_meta.size());
  for (HloInstruction* meta : sparse_meta) {
    AppendOperand(meta);
  }
  if (sparsity_.size() == kOperands &&
      sparsity_[0].index() > sparsity_[1].index()) {
    std::swap(sparsity_[0], sparsity_[1]);  // Keep descriptors ordered.
    std::swap(mutable_operands()[2], mutable_operands()[3]);
  }
}

HloInstructionProto HloDotInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_dot_dimension_numbers() = dot_dimension_numbers_;
  *proto.mutable_precision_config() = precision_config_;
  for (const SparsityDescriptor& descriptor : sparsity_) {
    *proto.add_dot_sparsity() = descriptor;
  }
  return proto;
}

void HloDotInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  printer.Next([this](Printer* printer) {
    printer->Append(DotDimensionNumbersToString(dot_dimension_numbers_));
  });
  PrintPrecisionConfig(printer, precision_config_);
  if (!sparsity_.empty()) {
    PrintSparsityDescriptor(printer, absl::MakeSpan(sparsity_));
  }
}

bool HloDotInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloDotInstruction&>(other);
  return protobuf_util::HaveSameSerialization(
             dot_dimension_numbers(), casted_other.dot_dimension_numbers()) &&
         protobuf_util::HaveSameSerialization(
             precision_config(), casted_other.precision_config()) &&
         absl::c_equal(sparsity_, casted_other.sparsity_,
                       protobuf_util::HaveSameSerialization);
}

std::unique_ptr<HloInstruction> HloDotInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), kOperands + sparse_operands());
  return std::make_unique<HloDotInstruction>(
      shape, new_operands[0], new_operands[1], dot_dimension_numbers_,
      precision_config_, sparsity_, new_operands.subspan(kOperands));
}

HloRaggedDotInstruction::HloRaggedDotInstruction(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    HloInstruction* group_sizes,
    const RaggedDotDimensionNumbers& dimension_numbers,
    const PrecisionConfig& precision_config)
    : HloInstruction(HloOpcode::kRaggedDot, shape),
      ragged_dot_dimension_numbers_(dimension_numbers),
      precision_config_(precision_config) {
  AppendOperand(lhs);
  AppendOperand(rhs);
  AppendOperand(group_sizes);
}

HloInstructionProto HloRaggedDotInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_ragged_dot_dimension_numbers() = ragged_dot_dimension_numbers_;
  *proto.mutable_precision_config() = precision_config_;
  return proto;
}

void HloRaggedDotInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  printer.Next([this](Printer* printer) {
    printer->Append(
        RaggedDotDimensionNumbersToString(ragged_dot_dimension_numbers_));
  });
  PrintPrecisionConfig(printer, precision_config_);
}

bool HloRaggedDotInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloRaggedDotInstruction&>(other);
  return protobuf_util::HaveSameSerialization(
             ragged_dot_dimension_numbers(),
             casted_other.ragged_dot_dimension_numbers()) &&
         protobuf_util::HaveSameSerialization(precision_config(),
                                              casted_other.precision_config());
}

std::unique_ptr<HloInstruction>
HloRaggedDotInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), kOperands);
  return std::make_unique<HloRaggedDotInstruction>(
      shape, new_operands[0], new_operands[1], new_operands[2],
      ragged_dot_dimension_numbers_, precision_config_);
}

HloDomainInstruction::HloDomainInstruction(
    const Shape& shape, HloInstruction* operand,
    std::unique_ptr<DomainMetadata> operand_side_metadata,
    std::unique_ptr<DomainMetadata> user_side_metadata)
    : HloInstruction(HloOpcode::kDomain, shape),
      operand_side_metadata_(std::move(operand_side_metadata)),
      user_side_metadata_(std::move(user_side_metadata)) {
  AppendOperand(operand);
}

void HloDomainInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  if (operand_side_metadata_ != nullptr && user_side_metadata_ != nullptr) {
    printer.Next([this](Printer* printer) {
      AppendCat(printer, "domain={kind=\"", operand_side_metadata_->Kind(),
                "\", entry=");
      AppendCat(printer, user_side_metadata_->ToString(),
                ", exit=", operand_side_metadata_->ToString(), "}");
    });
  }
}

bool HloDomainInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloDomainInstruction&>(other);
  return operand_side_metadata().Matches(
             casted_other.operand_side_metadata()) &&
         user_side_metadata().Matches(casted_other.user_side_metadata());
}

std::unique_ptr<HloInstruction> HloDomainInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloDomainInstruction>(shape, new_operands[0],
                                                operand_side_metadata_->Clone(),
                                                user_side_metadata_->Clone());
}

HloInstructionProto HloDomainInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  auto operand_side_sharding =
      dynamic_cast<const ShardingMetadata*>(operand_side_metadata_.get());
  if (operand_side_sharding && operand_side_sharding->sharding() != nullptr) {
    *proto.mutable_domain_entry_sharding() =
        operand_side_sharding->sharding()->ToProto();
  }

  auto user_side_sharding =
      dynamic_cast<const ShardingMetadata*>(user_side_metadata_.get());
  if (user_side_sharding && user_side_sharding->sharding() != nullptr) {
    *proto.mutable_domain_exit_sharding() =
        user_side_sharding->sharding()->ToProto();
  }

  return proto;
}

HloGetDimensionSizeInstruction::HloGetDimensionSizeInstruction(
    const Shape& shape, HloInstruction* operand, int64_t dimension)
    : HloInstruction(HloOpcode::kGetDimensionSize, shape),
      dimension_(dimension) {
  AppendOperand(operand);
}

HloInstructionProto HloGetDimensionSizeInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.add_dimensions(dimension());
  return proto;
}

void HloGetDimensionSizeInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& /*options*/) const {
  printer.Next([this](Printer* printer) {
    AppendCat(printer, "dimensions={", dimension(), "}");
  });
}

bool HloGetDimensionSizeInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
    /*eq_computations*/) const {
  const auto& casted_other =
      static_cast<const HloGetDimensionSizeInstruction&>(other);
  return dimension() == casted_other.dimension();
}

std::unique_ptr<HloInstruction>
HloGetDimensionSizeInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  if (new_operands.size() != 1) {
    LOG(FATAL) << "expects 1 operand";
  }
  return std::make_unique<HloGetDimensionSizeInstruction>(
      shape, new_operands[0], dimension());
}

HloSetDimensionSizeInstruction::HloSetDimensionSizeInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* val,
    int64_t dimension)
    : HloInstruction(HloOpcode::kSetDimensionSize, shape),
      dimension_(dimension) {
  AppendOperand(operand);
  AppendOperand(val);
}

void HloSetDimensionSizeInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& /*options*/) const {
  printer.Next([this](Printer* printer) {
    AppendCat(printer, "dimensions={", dimension(), "}");
  });
}

HloInstructionProto HloSetDimensionSizeInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.add_dimensions(dimension());
  return proto;
}

bool HloSetDimensionSizeInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
    /*eq_computations*/) const {
  const auto& casted_other =
      static_cast<const HloSetDimensionSizeInstruction&>(other);
  return dimension() == casted_other.dimension();
}

std::unique_ptr<HloInstruction>
HloSetDimensionSizeInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  if (new_operands.size() != 2) {
    LOG(FATAL) << "expects 2 operand";
  }
  return std::make_unique<HloSetDimensionSizeInstruction>(
      shape, new_operands[0], new_operands[1], dimension());
}

HloRngGetAndUpdateStateInstruction::HloRngGetAndUpdateStateInstruction(
    const Shape& shape, int64_t delta)
    : HloInstruction(HloOpcode::kRngGetAndUpdateState, shape), delta_(delta) {}

HloInstructionProto HloRngGetAndUpdateStateInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_delta(delta_);
  return proto;
}

void HloRngGetAndUpdateStateInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& /*options*/) const {
  printer.Next(
      [this](Printer* printer) { AppendCat(printer, "delta=", delta()); });
}

bool HloRngGetAndUpdateStateInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
    /*eq_computations*/) const {
  const auto& casted_other =
      static_cast<const HloRngGetAndUpdateStateInstruction&>(other);
  return delta() == casted_other.delta();
}

std::unique_ptr<HloInstruction>
HloRngGetAndUpdateStateInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  if (!new_operands.empty()) {
    LOG(FATAL) << "expects 0 operand";
  }
  return std::make_unique<HloRngGetAndUpdateStateInstruction>(shape, delta());
}

HloRngBitGeneratorInstruction::HloRngBitGeneratorInstruction(
    const Shape& shape, HloInstruction* state, RandomAlgorithm algorithm)
    : HloInstruction(HloOpcode::kRngBitGenerator, shape),
      algorithm_(algorithm) {
  AppendOperand(state);
}

HloInstructionProto HloRngBitGeneratorInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_rng_algorithm(algorithm_);
  return proto;
}

void HloRngBitGeneratorInstruction::PrintExtraAttributesImpl(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  printer.Next([this](Printer* printer) {
    AppendCat(printer, "algorithm=", RandomAlgorithmToString(algorithm_));
  });
}

bool HloRngBitGeneratorInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloRngBitGeneratorInstruction&>(other);
  return algorithm() == casted_other.algorithm();
}

std::unique_ptr<HloInstruction>
HloRngBitGeneratorInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloRngBitGeneratorInstruction>(shape, new_operands[0],
                                                         algorithm());
}

}  // namespace xla
