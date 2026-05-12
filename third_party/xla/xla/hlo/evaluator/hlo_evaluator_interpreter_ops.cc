/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/evaluator/hlo_evaluator_interpreter_ops.h"

#include <cstdint>
#include <cstring>

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/comparison_util.h"
#include "xla/hlo/evaluator/hlo_evaluator_interpreter.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/types.h"

namespace xla {

template <typename Op>
absl::Status LinearizedInterpreter::PopulateSimpleBinary(Step& step,
                                                         const char* op_name) {
  step.execute_fn =
      Op::GetExecuteFn(step.type, step.operand_types[0], step.operand_types[1]);
  if (!step.execute_fn) {
    return absl::UnimplementedError(
        absl::StrCat("Unsupported ", op_name, " types"));
  }
  return absl::OkStatus();
}

template <typename Op>
absl::Status LinearizedInterpreter::PopulateLogicalBinary(Step& step,
                                                          const char* op_name) {
  step.execute_fn = Op::GetExecuteFn(step.type);
  if (!step.execute_fn) {
    return absl::UnimplementedError(
        absl::StrCat("Unsupported ", op_name, " types"));
  }
  return absl::OkStatus();
}

const LinearizedInterpreter::OpRegistry&
LinearizedInterpreter::GetDefaultOpRegistry() {
  static absl::NoDestructor<OpRegistry> registry([] {
    OpRegistry r;
    r.Register(HloOpcode::kAdd, [](Step& step, const HloInstruction* instr,
                                   PrimitiveType promoted_type) {
      return PopulateSimpleBinary<Ops::Add>(step, "Add");
    });
    r.Register(HloOpcode::kMaximum, [](Step& step, const HloInstruction* instr,
                                       PrimitiveType promoted_type) {
      return PopulateSimpleBinary<Ops::Maximum>(step, "Maximum");
    });
    r.Register(HloOpcode::kCompare, [](Step& step, const HloInstruction* instr,
                                       PrimitiveType promoted_type) {
      const auto* compare = Cast<HloCompareInstruction>(instr);
      step.execute_fn = Ops::Compare::GetExecuteFn(compare->direction(),
                                                   step.operand_types[0]);
      if (!step.execute_fn) {
        return absl::UnimplementedError("Unsupported compare types");
      }
      return absl::OkStatus();
    });
    r.Register(HloOpcode::kOr, [](Step& step, const HloInstruction* instr,
                                  PrimitiveType promoted_type) {
      return PopulateLogicalBinary<Ops::Or>(step, "Or");
    });
    r.Register(HloOpcode::kAnd, [](Step& step, const HloInstruction* instr,
                                   PrimitiveType promoted_type) {
      return PopulateLogicalBinary<Ops::And>(step, "And");
    });
    r.Register(HloOpcode::kSelect, [](Step& step, const HloInstruction* instr,
                                      PrimitiveType promoted_type) {
      PrimitiveType cond_type = step.operand_types[0];
      if (cond_type != PRED) {
        return absl::UnimplementedError("Select condition must be PRED");
      }
      step.execute_fn = Ops::Select::GetExecuteFn(step.type);
      if (!step.execute_fn) {
        return absl::UnimplementedError("Unsupported select types");
      }
      return absl::OkStatus();
    });
    r.Register(HloOpcode::kConstant, [](Step& step, const HloInstruction* instr,
                                        PrimitiveType promoted_type) {
      step.aux_data = instr->literal().untyped_data();
      step.execute_fn = [](const Step* s, void* base) {
        size_t bytes =
            ShapeUtil::ByteSizeOfPrimitiveType(s->type) * s->element_count;
        if (bytes > 0) {
          char* dest = static_cast<char*>(base) + s->result_offset;
          size_t src_bytes = bytes / s->batch_size;
          for (int i = 0; i < s->batch_size; ++i) {
            std::memcpy(dest + i * src_bytes, s->aux_data, src_bytes);
          }
        }
      };
      return absl::OkStatus();
    });
    return r;
  }());
  return *registry;
}

LinearizedInterpreter::Step::ExecuteFn
LinearizedInterpreter::Ops::Add::GetExecuteFn(PrimitiveType res_type,
                                              PrimitiveType lhs_type,
                                              PrimitiveType rhs_type) {
  if (res_type == F32 && lhs_type == F32 && rhs_type == F32) {
    return &Execute<float, float, float>;
  }
  if (res_type == F64) {
    if (lhs_type == F64 && rhs_type == F32) {
      return &Execute<double, double, float>;
    }
    if (lhs_type == F32 && rhs_type == F64) {
      return &Execute<double, float, double>;
    }
    if (lhs_type == F64 && rhs_type == F64) {
      return &Execute<double, double, double>;
    }
  }
  if (res_type == S32 && lhs_type == S32 && rhs_type == S32) {
    return &Execute<int32_t, int32_t, int32_t>;
  }
  return nullptr;
}

LinearizedInterpreter::Step::ExecuteFn
LinearizedInterpreter::Ops::Maximum::GetExecuteFn(PrimitiveType res_type,
                                                  PrimitiveType lhs_type,
                                                  PrimitiveType rhs_type) {
  if (res_type == lhs_type && res_type == rhs_type) {
    switch (res_type) {
      case F32:
        return &Execute<float>;
      case S32:
        return &Execute<int32_t>;
      default:
        break;
    }
  }
  return nullptr;
}

LinearizedInterpreter::Step::ExecuteFn
LinearizedInterpreter::Ops::Compare::GetExecuteFn(ComparisonDirection direction,
                                                  PrimitiveType operand_type) {
  return primitive_util::PrimitiveTypeSwitch<Step::ExecuteFn>(
      [&](auto type_constant) -> Step::ExecuteFn {
        constexpr PrimitiveType kType = decltype(type_constant)::value;
        if constexpr (kType == BF16 || kType == S32 || kType == S64) {
          using T = primitive_util::NativeTypeOf<kType>;
          switch (direction) {
            case ComparisonDirection::kGt:
              return &Execute<T, ComparisonDirection::kGt>;
            case ComparisonDirection::kNe:
              return &Execute<T, ComparisonDirection::kNe>;
            case ComparisonDirection::kEq:
              return &Execute<T, ComparisonDirection::kEq>;
            case ComparisonDirection::kLt:
              return &Execute<T, ComparisonDirection::kLt>;
            case ComparisonDirection::kGe:
              return &Execute<T, ComparisonDirection::kGe>;
            case ComparisonDirection::kLe:
              return &Execute<T, ComparisonDirection::kLe>;
            default:
              return nullptr;
          }
        }
        return nullptr;
      },
      operand_type);
}

LinearizedInterpreter::Step::ExecuteFn
LinearizedInterpreter::Ops::Select::GetExecuteFn(PrimitiveType type) {
  switch (type) {
    case BF16:
      return &Execute<bfloat16>;
    case S32:
      return &Execute<int32_t>;
    case S64:
      return &Execute<int64_t>;
    default:
      return nullptr;
  }
}

}  // namespace xla
