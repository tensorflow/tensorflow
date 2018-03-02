/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/index_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/bitmap.h"
#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

namespace {

template <typename T>
struct is_complex_t : public std::false_type {};

template <>
struct is_complex_t<complex64> : public std::true_type {};

template <typename T>
struct is_complex64_t : public std::false_type {};

template <>
struct is_complex64_t<complex64> : public std::true_type {};

template <typename OperandT>
StatusOr<std::unique_ptr<Literal>> Compare(const Shape& shape, HloOpcode opcode,
                                           const Literal& lhs_literal,
                                           const Literal& rhs_literal) {
  std::function<bool(OperandT, OperandT)> compare_op;
  switch (opcode) {
    case HloOpcode::kEq:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el == rhs_el;
      };
      break;
    case HloOpcode::kNe:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el != rhs_el;
      };
      break;
    case HloOpcode::kGe:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el >= rhs_el;
      };
      break;
    case HloOpcode::kGt:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el > rhs_el;
      };
      break;
    case HloOpcode::kLe:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el <= rhs_el;
      };
      break;
    case HloOpcode::kLt:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el < rhs_el;
      };
      break;
    default:
      LOG(FATAL) << "unhandled HLO opcode for conversion to Comparison: "
                 << HloOpcodeString(opcode);
  }

  auto result = Literal::CreateFromShape(shape);
  TF_RETURN_IF_ERROR(result->Populate<bool>(
      [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
        return compare_op(lhs_literal.Get<OperandT>(multi_index),
                          rhs_literal.Get<OperandT>(multi_index));
      }));

  return std::move(result);
}

template <>
StatusOr<std::unique_ptr<Literal>> Compare<complex64>(
    const Shape& shape, HloOpcode opcode, const Literal& lhs_literal,
    const Literal& rhs_literal) {
  std::function<bool(complex64, complex64)> compare_op;
  switch (opcode) {
    case HloOpcode::kEq:
      compare_op = [](complex64 lhs_el, complex64 rhs_el) {
        return lhs_el == rhs_el;
      };
      break;
    case HloOpcode::kNe:
      compare_op = [](complex64 lhs_el, complex64 rhs_el) {
        return lhs_el != rhs_el;
      };
      break;
    default:
      LOG(FATAL) << "unhandled HLO opcode for conversion to Comparison: "
                 << HloOpcodeString(opcode);
  }

  auto result = Literal::CreateFromShape(shape);
  TF_RETURN_IF_ERROR(result->Populate<bool>(
      [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
        return compare_op(lhs_literal.Get<complex64>(multi_index),
                          rhs_literal.Get<complex64>(multi_index));
      }));

  return std::move(result);
}

template <typename ReturnT, typename NativeT>
StatusOr<std::unique_ptr<Literal>> ElementWiseUnaryOpImpl(
    HloInstruction* instruction,
    const std::function<ReturnT(NativeT)>& unary_op,
    const Literal& operand_literal) {
  const auto shape = instruction->shape();
  const auto* operand = instruction->operand(0);

  // TODO(b/35950897, b/27796129): add DCHECK back once implicit broadcast is
  // removed.
  if (!ShapeUtil::SameDimensions(shape, operand->shape())) {
    return Unimplemented(
        "Implicit broadcasting is currently unsupported in HLO evaluator "
        "Shape Mismatch: %s vs %s",
        ShapeUtil::HumanString(shape).c_str(),
        ShapeUtil::HumanString(operand->shape()).c_str());
  }

  auto result = Literal::CreateFromShape(shape);

  TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
      [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
        return unary_op(operand_literal.Get<NativeT>(multi_index));
      }));
  return std::move(result);
}

// For one particular placement of a window in a base shape (the placement is
// represented as `window_count_index`), iterates inside the window. Translates
// the window index into base index. If the base index is within bound, call `f`
// with the base index.
void IterateThroughWindow(
    const Shape& window_shape, const Window& window, const Shape& base_shape,
    const tensorflow::gtl::ArraySlice<int64>& window_count_index,
    const std::function<void(const std::vector<int64>&)>& f) {
  const int64 rank = ShapeUtil::Rank(base_shape);
  DimensionVector window_index(rank);
  std::fill(window_index.begin(), window_index.end(), 0);
  do {
    std::vector<int64> base_index(rank);
    bool out_of_bound = false;
    for (int64 i = 0; i < rank; ++i) {
      base_index[i] = window_count_index[i] * window.dimensions(i).stride() +
                      window_index[i] - window.dimensions(i).padding_low();
      if (base_index[i] < 0 || base_index[i] >= base_shape.dimensions(i)) {
        out_of_bound = true;
        break;
      }
    }
    if (!out_of_bound) {
      f(base_index);
    }
  } while (IndexUtil::BumpIndices(window_shape, &window_index));
}

}  // namespace

template <typename ReturnT, typename ElementwiseT>
class HloEvaluator::TypedVisitor : public DfsHloVisitorWithDefault {
 public:
  explicit TypedVisitor(HloEvaluator* p) : parent_(p) {}

  // The following higher-order functions convert a function with ElementwiseT
  // to a function with ReturnT.
  std::function<ReturnT(ReturnT)> ConvertUnaryFunction(
      const std::function<ElementwiseT(ElementwiseT)>& unary_op) {
    return [&unary_op](ReturnT arg) {
      return static_cast<ReturnT>(unary_op(static_cast<ElementwiseT>(arg)));
    };
  }
  std::function<ReturnT(ReturnT, ReturnT)> ConvertBinaryFunction(
      const std::function<ElementwiseT(ElementwiseT, ElementwiseT)>&
          binary_op) {
    return [&binary_op](ReturnT arg1, ReturnT arg2) {
      return static_cast<ReturnT>(binary_op(static_cast<ElementwiseT>(arg1),
                                            static_cast<ElementwiseT>(arg2)));
    };
  }
  std::function<ReturnT(ReturnT, ReturnT, ReturnT)> ConvertTernaryFunction(
      const std::function<ElementwiseT(ElementwiseT, ElementwiseT,
                                       ElementwiseT)>& ternary_op) {
    return [&ternary_op](ReturnT arg1, ReturnT arg2, ReturnT arg3) {
      return static_cast<ReturnT>(ternary_op(static_cast<ElementwiseT>(arg1),
                                             static_cast<ElementwiseT>(arg2),
                                             static_cast<ElementwiseT>(arg3)));
    };
  }

  Status DefaultAction(HloInstruction* hlo_instruction) override {
    return Unimplemented("unhandled HLO ops for HloEvaluator: %s.",
                         HloOpcodeString(hlo_instruction->opcode()).c_str());
  }

  // TODO(b/35950897): many of the stl functions used in the handlers are not
  // overloaded for every XLA primitive types.

  template <typename NativeT,
            typename std::enable_if<std::is_unsigned<NativeT>::value>::type* =
                nullptr>
  Status HandleAbs(HloInstruction* abs) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[abs],
                        ElementWiseUnaryOp(abs, [](NativeT elem_operand) {
                          return elem_operand;
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<std::is_signed<NativeT>::value>::type* = nullptr>
  Status HandleAbs(HloInstruction* abs) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[abs],
                        ElementWiseUnaryOp(abs, [](NativeT elem_operand) {
                          return std::abs(elem_operand);
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex64_t<NativeT>::value>::type* = nullptr>
  Status HandleAbs(HloInstruction* abs) {
    const Literal& operand_literal =
        parent_->GetEvaluatedLiteralFor(abs->operand(0));
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[abs],
        (ElementWiseUnaryOpImpl<float, NativeT>(
            abs, [](NativeT elem_operand) { return std::abs(elem_operand); },
            operand_literal)));

    return Status::OK();
  }

  Status HandleAbs(HloInstruction* abs) override {
    // If the operand is of C64 type, the return type of abs will be F32.
    // However, ElementwiseT would still be the return type, F32, and thus
    // specifying the ElementwiseT explicitly as C64 is needed below.
    if (abs->operand(0)->shape().element_type() == C64) {
      return HandleAbs<complex64>(abs);
    }
    return HandleAbs<ElementwiseT>(abs);
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleRound(HloInstruction* round) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[round],
        ElementWiseUnaryOp(round, [](ElementwiseT elem_operand) {
          return std::round(elem_operand);
        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleRound(HloInstruction* round) {
    return InvalidArgument("Unsupported type for Round");
  }

  Status HandleRound(HloInstruction* round) override {
    return HandleRound<ReturnT>(round);
  }

  Status HandleBroadcast(HloInstruction* broadcast) override {
    parent_->evaluated_[broadcast] =
        Literal::CreateFromShape(broadcast->shape());
    auto output = parent_->evaluated_[broadcast].get();
    const Literal& operand_to_broadcast =
        parent_->GetEvaluatedLiteralFor(broadcast->operand(0));
    std::vector<int64> broadcast_indices(
        ShapeUtil::Rank(broadcast->operand(0)->shape()), 0);

    TF_RET_CHECK(broadcast->dimensions().size() ==
                 ShapeUtil::Rank(operand_to_broadcast.shape()))
        << "broadcast dimensions is of size: " << broadcast->dimensions().size()
        << " and rank of operand_to_broadcast is: "
        << ShapeUtil::Rank(operand_to_broadcast.shape());
    // Checks that operand's dimensions are the same as the broadcast's
    // dimensions along the dimensions to be broadcasted.
    for (int64 i = 0; i < broadcast->dimensions().size(); ++i) {
      TF_RET_CHECK(broadcast->shape().dimensions(broadcast->dimensions(i)) ==
                   operand_to_broadcast.shape().dimensions(i));
    }

    return output->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
          for (int64 i = 0; i < broadcast->dimensions().size(); ++i) {
            broadcast_indices[i] = multi_index[broadcast->dimensions(i)];
          }
          return operand_to_broadcast.Get<ReturnT>(broadcast_indices);
        });
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleCeil(HloInstruction* ceil) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[ceil],
                        ElementWiseUnaryOp(ceil, [](ElementwiseT elem_operand) {
                          return std::ceil(elem_operand);
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleCeil(HloInstruction* ceil) {
    return InvalidArgument("Unsupported type for Ceil");
  }

  Status HandleCeil(HloInstruction* ceil) override {
    return HandleCeil<ReturnT>(ceil);
  }

  Status HandleConvert(HloInstruction* convert) override {
    const HloInstruction* operand = convert->operand(0);
    TF_RET_CHECK(ShapeUtil::SameDimensions(operand->shape(), convert->shape()));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<Literal> result,
                        parent_->GetEvaluatedLiteralFor(operand).Convert(
                            convert->shape().element_type()));

    if (LayoutUtil::LayoutsInShapesEqual(result->shape(), convert->shape())) {
      parent_->evaluated_[convert] = std::move(result);
    } else {
      parent_->evaluated_[convert] =
          result->Relayout(convert->shape().layout());
    }
    return Status::OK();
  }

  Status HandleExp(HloInstruction* exp) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[exp],
                        ElementWiseUnaryOp(exp, [](ElementwiseT elem_operand) {
                          return std::exp(elem_operand);
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleFloor(HloInstruction* floor) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[floor],
        ElementWiseUnaryOp(floor, [](ElementwiseT elem_operand) {
          return std::floor(elem_operand);
        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleFloor(HloInstruction* floor) {
    return InvalidArgument("Unsupported type for Floor");
  }

  Status HandleFloor(HloInstruction* floor) override {
    return HandleFloor<ReturnT>(floor);
  }

  Status HandleLog(HloInstruction* log) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[log],
                        ElementWiseUnaryOp(log, [](ElementwiseT elem_operand) {
                          return std::log(elem_operand);
                        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<
                std::is_integral<NativeT>::value &&
                !std::is_same<NativeT, bool>::value>::type* = nullptr>
  Status HandleNot(HloInstruction* not_) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[not_],
                        ElementWiseUnaryOp(not_, [](ElementwiseT elem_operand) {
                          return ~elem_operand;
                        }));
    return Status::OK();
  }

  template <typename NativeT, typename std::enable_if<std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleNot(HloInstruction* not_) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[not_],
                        ElementWiseUnaryOp(not_, [](ElementwiseT elem_operand) {
                          return !elem_operand;
                        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<std::is_same<NativeT, bool>::value>::type* =
                nullptr>
  Status HandleNot(HloInstruction* not_) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[not_],
                        ElementWiseUnaryOp(not_, [](ElementwiseT elem_operand) {
                          return !elem_operand;
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleNot(HloInstruction* not_) {
    return InvalidArgument("Unsupported type for Not");
  }

  Status HandleNot(HloInstruction* not_) override {
    return HandleNot<ElementwiseT>(not_);
  }

  template <typename NativeT,
            typename std::enable_if<
                std::is_signed<NativeT>::value &&
                !std::is_floating_point<NativeT>::value>::type* = nullptr>
  Status HandleNegate(HloInstruction* negate) {
    using type = typename std::make_unsigned<NativeT>::type;
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[negate],
        ElementWiseUnaryOp(negate, [](ElementwiseT elem_operand) {
          return NativeT(-type(elem_operand));
        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<
                !std::is_signed<NativeT>::value ||
                std::is_floating_point<NativeT>::value>::type* = nullptr>
  Status HandleNegate(HloInstruction* negate) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[negate],
        ElementWiseUnaryOp(
            negate, [](ElementwiseT elem_operand) { return -elem_operand; }));
    return Status::OK();
  }

  Status HandleNegate(HloInstruction* negate) override {
    return HandleNegate<ReturnT>(negate);
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleSign(HloInstruction* sign) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[sign],
                        ElementWiseUnaryOp(sign, [](ElementwiseT elem_operand) {
                          return (ElementwiseT(0) < elem_operand) -
                                 (elem_operand < ElementwiseT(0));
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleSign(HloInstruction* sign) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[sign],
                        ElementWiseUnaryOp(sign, [](ElementwiseT elem_operand) {
                          auto abs_val = std::abs(elem_operand);
                          return 0 == abs_val ? ElementwiseT(0)
                                              : elem_operand / abs_val;
                        }));
    return Status::OK();
  }

  Status HandleSign(HloInstruction* sign) override {
    return HandleSign<ReturnT>(sign);
  }

  template <typename NativeT, typename std::enable_if<std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleAtan2(HloInstruction* atan2) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[atan2],
                        ElementWiseBinaryOp(atan2, [](ElementwiseT lhs_elem,
                                                      ElementwiseT rhs_elem) {
                          return std::atan2(lhs_elem, rhs_elem);
                        }));
    return Status::OK();
  }

  template <typename NativeT, typename std::enable_if<!std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleAtan2(HloInstruction* atan2) {
    return InvalidArgument("Unsupported type for Atan2");
  }

  Status HandleAtan2(HloInstruction* atan2) override {
    return HandleAtan2<ElementwiseT>(atan2);
  }

  Status HandleTanh(HloInstruction* tanh) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[tanh],
                        ElementWiseUnaryOp(tanh, [](ElementwiseT elem_operand) {
                          return std::tanh(elem_operand);
                        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<
                std::is_signed<NativeT>::value &&
                !std::is_floating_point<NativeT>::value>::type* = nullptr>
  Status HandleMultiply(HloInstruction* multiply) {
    using type = typename std::make_unsigned<NativeT>::type;
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[multiply],
        ElementWiseBinaryOp(multiply,
                            [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
                              return NativeT(type(lhs_elem) * type(rhs_elem));
                            }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<std::is_unsigned<NativeT>::value ||
                              std::is_floating_point<NativeT>::value ||
                              is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleMultiply(HloInstruction* multiply) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[multiply],
        ElementWiseBinaryOp(multiply,
                            [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
                              return lhs_elem * rhs_elem;
                            }));
    return Status::OK();
  }

  Status HandleMultiply(HloInstruction* multiply) override {
    return HandleMultiply<ElementwiseT>(multiply);
  }

  Status HandleSubtract(HloInstruction* subtract) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[subtract],
        ElementWiseBinaryOp(subtract,
                            [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
                              return lhs_elem - rhs_elem;
                            }));
    return Status::OK();
  }

  Status HandleAdd(HloInstruction* add) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[add],
                        ElementWiseBinaryOp(add, [](ElementwiseT lhs_elem,
                                                    ElementwiseT rhs_elem) {
                          return lhs_elem + rhs_elem;
                        }));
    return Status::OK();
  }

  Status HandleDivide(HloInstruction* divide) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[divide],
                        ElementWiseBinaryOp(divide, [](ElementwiseT lhs_elem,
                                                       ElementwiseT rhs_elem) {
                          return lhs_elem / rhs_elem;
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleMaximum(HloInstruction* maximum) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[maximum],
        ElementWiseBinaryOp(maximum, [](ElementwiseT lhs, ElementwiseT rhs) {
          return std::fmax(lhs, rhs);
        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleMaximum(HloInstruction* maximum) {
    return InvalidArgument("Unsupported type for Maximum");
  }

  Status HandleMaximum(HloInstruction* maximum) override {
    return HandleMaximum<ElementwiseT>(maximum);
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleMinimum(HloInstruction* minimum) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[minimum],
                        ElementWiseBinaryOp(minimum, [](ElementwiseT lhs_el,
                                                        ElementwiseT rhs_el) {
                          return std::fmin(lhs_el, rhs_el);
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleMinimum(HloInstruction* minimum) {
    return InvalidArgument("Unsupported type for Minimum");
  }

  Status HandleMinimum(HloInstruction* minimum) override {
    return HandleMinimum<ElementwiseT>(minimum);
  }

  Status HandlePower(HloInstruction* power) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[power],
                        ElementWiseBinaryOp(power, [](ElementwiseT lhs_el,
                                                      ElementwiseT rhs_el) {
                          return std::pow(lhs_el, rhs_el);
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleRemainder(HloInstruction* remainder) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[remainder],
                        ElementWiseBinaryOp(remainder, [](ElementwiseT lhs_el,
                                                          ElementwiseT rhs_el) {
                          return std::fmod(lhs_el, rhs_el);
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleRemainder(HloInstruction* remainder) {
    return InvalidArgument("Unsupported type for Remainder");
  }

  Status HandleRemainder(HloInstruction* remainder) override {
    return HandleRemainder<ElementwiseT>(remainder);
  }

  template <typename NativeT,
            typename std::enable_if<std::is_integral<NativeT>::value>::type* =
                nullptr>
  Status HandleAnd(HloInstruction* and_) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[and_],
        ElementWiseBinaryOp(and_, [](ElementwiseT lhs_el, ElementwiseT rhs_el) {
          return lhs_el & rhs_el;
        }));
    return Status::OK();
  }

  template <typename NativeT, typename std::enable_if<std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleAnd(HloInstruction* and_) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[and_],
        ElementWiseBinaryOp(and_, [](ElementwiseT lhs_el, ElementwiseT rhs_el) {
          return lhs_el && rhs_el;
        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleAnd(HloInstruction* and_) {
    return InvalidArgument("Unsupported type for And");
  }

  Status HandleAnd(HloInstruction* and_) override {
    return HandleAnd<ElementwiseT>(and_);
  }

  template <typename NativeT,
            typename std::enable_if<std::is_integral<NativeT>::value>::type* =
                nullptr>
  Status HandleOr(HloInstruction* or_) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[or_],
        ElementWiseBinaryOp(or_, [](ElementwiseT lhs_el, ElementwiseT rhs_el) {
          return lhs_el | rhs_el;
        }));
    return Status::OK();
  }

  template <typename NativeT, typename std::enable_if<std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleOr(HloInstruction* or_) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[or_],
        ElementWiseBinaryOp(or_, [](ElementwiseT lhs_el, ElementwiseT rhs_el) {
          return lhs_el || rhs_el;
        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleOr(HloInstruction* or_) {
    return InvalidArgument("Unsupported type for Or");
  }

  Status HandleOr(HloInstruction* or_) override {
    return HandleOr<ElementwiseT>(or_);
  }

  template <typename NativeT,
            typename std::enable_if<
                std::is_integral<NativeT>::value &&
                !std::is_same<NativeT, bool>::value>::type* = nullptr>
  Status HandleShiftLeft(HloInstruction* shl) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[shl],
        ElementWiseBinaryOp(shl, [](NativeT lhs_elem, NativeT rhs_elem) {
          return IsShiftOutOfBounds<NativeT>(rhs_elem) ? 0
                                                       : (lhs_elem << rhs_elem);
        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<!std::is_integral<NativeT>::value ||
                                    std::is_same<NativeT, bool>::value>::type* =
                nullptr>
  Status HandleShiftLeft(HloInstruction*) {
    return InvalidArgument("Unsupported type for ShiftLeft");
  }

  Status HandleShiftLeft(HloInstruction* shl) override {
    return HandleShiftLeft<ElementwiseT>(shl);
  }
  template <typename NativeT,
            typename std::enable_if<
                std::is_integral<NativeT>::value &&
                !std::is_same<NativeT, bool>::value>::type* = nullptr>
  Status HandleShiftRightArithmetic(HloInstruction* shr) {
    typedef typename std::make_signed<NativeT>::type SignedT;
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[shr],
        ElementWiseBinaryOp(shr, [](NativeT lhs_elem, NativeT rhs_elem) {
          SignedT lhs_signed = static_cast<SignedT>(lhs_elem);
          if (IsShiftOutOfBounds<NativeT>(rhs_elem)) {
            return lhs_signed < 0 ? static_cast<SignedT>(-1) : 0;
          } else {
            return lhs_signed >> rhs_elem;
          }
        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<!std::is_integral<NativeT>::value ||
                                    std::is_same<NativeT, bool>::value>::type* =
                nullptr>
  Status HandleShiftRightArithmetic(HloInstruction*) {
    return InvalidArgument("Unsupported type for ShiftRightArithmetic");
  }

  Status HandleShiftRightArithmetic(HloInstruction* shra) override {
    return HandleShiftRightArithmetic<ElementwiseT>(shra);
  }

  template <typename NativeT,
            typename std::enable_if<
                std::is_integral<NativeT>::value &&
                !std::is_same<NativeT, bool>::value>::type* = nullptr>
  Status HandleShiftRightLogical(HloInstruction* shr) {
    typedef typename std::make_unsigned<NativeT>::type UnsignedT;
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[shr],
        ElementWiseBinaryOp(shr, [](NativeT lhs_elem, NativeT rhs_elem) {
          // If shift amount is greater than the number of bits, then return 0.
          if (IsShiftOutOfBounds<NativeT>(rhs_elem)) {
            return static_cast<NativeT>(0);
          }
          return static_cast<NativeT>(static_cast<UnsignedT>(lhs_elem) >>
                                      rhs_elem);
        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<!std::is_integral<NativeT>::value ||
                                    std::is_same<NativeT, bool>::value>::type* =
                nullptr>
  Status HandleShiftRightLogical(HloInstruction*) {
    return InvalidArgument("Unsupported type for ShiftRightLogical");
  }

  Status HandleShiftRightLogical(HloInstruction* shrl) override {
    return HandleShiftRightLogical<ElementwiseT>(shrl);
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleClamp(HloInstruction* clamp) {
    std::function<ElementwiseT(ElementwiseT, ElementwiseT, ElementwiseT)>
        clamp_op = [](ElementwiseT low, ElementwiseT value, ElementwiseT high) {
          return std::fmax(low, std::fmin(value, high));
        };
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[clamp],
        ElementwiseTernaryOp(clamp,
                             std::move(ConvertTernaryFunction(clamp_op))));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleClamp(HloInstruction*) {
    return InvalidArgument("Unsupported type for Clamp");
  }

  Status HandleClamp(HloInstruction* clamp) override {
    return HandleClamp<ElementwiseT>(clamp);
  }

  Status HandleSelect(HloInstruction* select) override {
    CHECK(!ShapeUtil::IsTuple(select->shape()));
    std::function<ReturnT(bool, ReturnT, ReturnT)> select_op =
        [](bool pred, ReturnT on_true, ReturnT on_false) {
          if (pred) {
            return on_true;
          }
          return on_false;
        };
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[select],
                        ElementwiseTernaryOp(select, std::move(select_op)));
    return Status::OK();
  }

  Status HandleReverse(HloInstruction* reverse) override {
    const auto result_shape = reverse->shape();
    const auto reverse_dimensions = reverse->dimensions();

    auto operand = reverse->operand(0);
    TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                        ShapeInference::InferReverseShape(operand->shape(),
                                                          reverse_dimensions));

    TF_RET_CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
        << "return shape set to: " << ShapeUtil::HumanString(result_shape)
        << " but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);

    const Literal& operand_literal = parent_->GetEvaluatedLiteralFor(operand);
    auto result = Literal::CreateFromShape(result_shape);

    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> out_index) {
          std::vector<int64> from_index(out_index.begin(), out_index.end());
          for (const int64 dim : reverse_dimensions) {
            from_index[dim] = result_shape.dimensions(dim) - 1 - out_index[dim];
          }
          return operand_literal.Get<ReturnT>(from_index);
        }));

    parent_->evaluated_[reverse] = std::move(result);
    return Status::OK();
  }

  Status HandleConvolution(HloInstruction* conv) override {
    auto lhs = conv->operand(0);
    auto rhs = conv->operand(1);
    const auto& window = conv->window();
    const Shape& result_shape = conv->shape();
    const Shape& lhs_shape = lhs->shape();
    const Shape& rhs_shape = rhs->shape();

    TF_CHECK_OK(ShapeUtil::ValidateShape(lhs_shape));
    TF_CHECK_OK(ShapeUtil::ValidateShape(rhs_shape));
    CHECK(ShapeUtil::IsArray(lhs_shape));
    CHECK(ShapeUtil::IsArray(rhs_shape));
    CHECK(ShapeUtil::SameElementType(lhs_shape, rhs_shape));
    CHECK(ShapeUtil::SameElementType(lhs_shape, result_shape));

    const auto& dnums = conv->convolution_dimension_numbers();
    const int64 num_spatial_dims = dnums.output_spatial_dimensions_size();
    CHECK_EQ(num_spatial_dims, dnums.input_spatial_dimensions_size());
    CHECK_EQ(num_spatial_dims, dnums.kernel_spatial_dimensions_size());
    CHECK_GE(num_spatial_dims, 0);
    CHECK_EQ(window.dimensions_size(), num_spatial_dims);

    const auto lhs_rank = ShapeUtil::Rank(lhs_shape);
    const auto rhs_rank = ShapeUtil::Rank(rhs_shape);

    CHECK_EQ(num_spatial_dims + 2, lhs_rank);
    CHECK_EQ(num_spatial_dims + 2, rhs_rank);

    TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                        ShapeInference::InferConvolveShape(lhs_shape, rhs_shape,
                                                           window, dnums));
    CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
        << "return shape set to: " << ShapeUtil::HumanString(result_shape)
        << " but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);

    const Literal& lhs_literal = parent_->GetEvaluatedLiteralFor(lhs);
    const Literal& rhs_literal = parent_->GetEvaluatedLiteralFor(rhs);

    // Dimension number applicable for input (lhs).
    const int64 input_batch_dim = dnums.input_batch_dimension();
    const int64 input_z_dim = dnums.input_feature_dimension();
    // Dimension number applicable for kernel (rhs).
    const int64 kernel_input_z_dim = dnums.kernel_input_feature_dimension();
    const int64 kernel_output_z_dim = dnums.kernel_output_feature_dimension();
    // Dimension number applicable for output.
    const int64 output_batch_dim = dnums.output_batch_dimension();
    const int64 output_z_dim = dnums.output_feature_dimension();

    const int64 z_size = ShapeUtil::GetDimension(lhs_shape, input_z_dim);

    std::vector<int64> window_dimension_sizes;
    for (auto i : dnums.kernel_spatial_dimensions()) {
      window_dimension_sizes.push_back(ShapeUtil::GetDimension(rhs_shape, i));
    }

    const Shape& window_shape =
        ShapeUtil::MakeShape(rhs_shape.element_type(), window_dimension_sizes);

    DimensionVector lhs_index(lhs_rank);
    DimensionVector rhs_index(rhs_rank);
    DimensionVector rhs_spatial_index(dnums.kernel_spatial_dimensions_size());

    auto func = [&](tensorflow::gtl::ArraySlice<int64> out_index) {
      ElementwiseT result_val = static_cast<ElementwiseT>(0);

      std::fill(lhs_index.begin(), lhs_index.end(), 0);
      std::fill(rhs_index.begin(), rhs_index.end(), 0);
      std::fill(rhs_spatial_index.begin(), rhs_spatial_index.end(), 0);

      lhs_index[input_batch_dim] = out_index[output_batch_dim];
      rhs_index[kernel_output_z_dim] = out_index[output_z_dim];

      // Convolve input feature with kernel.
      do {
        for (int64 iz = 0; iz < z_size; ++iz) {
          lhs_index[input_z_dim] = iz;
          rhs_index[kernel_input_z_dim] = iz;

          // Find corresponding spatial dimension index for input (lhs).
          for (int64 ki = 0; ki < rhs_spatial_index.size(); ++ki) {
            // Spatial dimension number for input (lhs) and output.
            const int64 input_spatial_dim = dnums.input_spatial_dimensions(ki);
            const int64 output_spatial_dim =
                dnums.output_spatial_dimensions(ki);

            // Calculate lhs (input) index without taking base dilation into
            // account.
            const auto& window_dim = window.dimensions(ki);
            const int64 undilated_index =
                out_index[output_spatial_dim] * window_dim.stride() -
                window_dim.padding_low() +
                rhs_spatial_index[ki] * window_dim.window_dilation();
            // Skip if the lhs (input) index is to be dilated.  As an
            // optimization, skip this mod if there's no dilation.
            if (window_dim.base_dilation() > 1 &&
                undilated_index % window_dim.base_dilation() != 0) {
              goto cnt;
            }

            // Calculate the actual lhs (input) index after dilation.  As an
            // optimization, skip this integer divide if there's no dilation.
            if (window_dim.base_dilation() > 1) {
              lhs_index[input_spatial_dim] =
                  undilated_index / window_dim.base_dilation();
            } else {
              lhs_index[input_spatial_dim] = undilated_index;
            }

            // Skip if input index is not in bound.
            if (!(lhs_index[input_spatial_dim] >= 0 &&
                  lhs_index[input_spatial_dim] <
                      lhs_shape.dimensions(input_spatial_dim))) {
              goto cnt;
            }

            rhs_index[dnums.kernel_spatial_dimensions(ki)] =
                window_dim.window_reversal()
                    ? ((window_dim.size() - 1) - rhs_spatial_index[ki])
                    : rhs_spatial_index[ki];
          }

          result_val +=
              static_cast<ElementwiseT>(lhs_literal.Get<ReturnT>(lhs_index)) *
              static_cast<ElementwiseT>(rhs_literal.Get<ReturnT>(rhs_index));
        }
      cnt : {}
      } while (IndexUtil::BumpIndices(window_shape, &rhs_spatial_index));

      return static_cast<ReturnT>(result_val);
    };

    auto result = Literal::CreateFromShape(result_shape);
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(func));

    parent_->evaluated_[conv] = std::move(result);
    return Status::OK();
  }

  Status HandleDot(HloInstruction* dot) override {
    auto lhs = dot->operand(0);
    auto rhs = dot->operand(1);
    CHECK(ShapeUtil::IsArray(dot->shape()));
    CHECK(ShapeUtil::IsArray(lhs->shape()));
    CHECK(ShapeUtil::IsArray(rhs->shape()));

    const auto& dnums = dot->dot_dimension_numbers();

    const auto lhs_rank = ShapeUtil::Rank(lhs->shape());
    const auto rhs_rank = ShapeUtil::Rank(rhs->shape());

    CHECK(ShapeUtil::SameElementType(lhs->shape(), rhs->shape()));
    CHECK(ShapeUtil::SameElementType(lhs->shape(), dot->shape()));

    // There must be 1 and only 1 Contracting dimension for lhs and rhs.
    CHECK_EQ(dnums.lhs_contracting_dimensions_size(), 1);
    CHECK_EQ(dnums.rhs_contracting_dimensions_size(), 1);
    const int64 lhs_contracting_dimension = dnums.lhs_contracting_dimensions(0);
    const int64 rhs_contracting_dimension = dnums.rhs_contracting_dimensions(0);
    // Contracted dimension sizes must be the same.
    CHECK_EQ(lhs->shape().dimensions(lhs_contracting_dimension),
             rhs->shape().dimensions(rhs_contracting_dimension))
        << "lhs contracted dimension: "
        << lhs->shape().dimensions(lhs_contracting_dimension)
        << " rhs contracted dimension: "
        << rhs->shape().dimensions(rhs_contracting_dimension);
    const int64 contracted_dimension_size =
        lhs->shape().dimensions(lhs_contracting_dimension);

    const Literal& lhs_literal = parent_->GetEvaluatedLiteralFor(lhs);
    const Literal& rhs_literal = parent_->GetEvaluatedLiteralFor(rhs);

    auto result = Literal::CreateFromShape(dot->shape());

    CHECK_EQ(dnums.lhs_batch_dimensions_size(),
             dnums.rhs_batch_dimensions_size());

    std::vector<int64> lhs_non_contracting_dims;
    for (int64 i = 0; i < lhs_rank; i++) {
      if (i != lhs_contracting_dimension) {
        lhs_non_contracting_dims.push_back(i);
      }
    }

    std::vector<int64> rhs_non_batch_non_contracting_dims;
    tensorflow::gtl::FlatSet<int64> batch_dims_set(
        dnums.rhs_batch_dimensions().begin(),
        dnums.rhs_batch_dimensions().end());
    for (int64 i = 0; i < rhs_rank; i++) {
      if (i != rhs_contracting_dimension && batch_dims_set.count(i) == 0) {
        rhs_non_batch_non_contracting_dims.push_back(i);
      }
    }

    const int64 batch_dim_size = dnums.lhs_batch_dimensions_size();
    const int64 lhs_non_contracting_size = lhs_non_contracting_dims.size();

    DimensionVector lhs_index(lhs_rank);
    DimensionVector rhs_index(rhs_rank);
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> result_index) {
          ElementwiseT result_val = static_cast<ElementwiseT>(0);

          // Find the corresponding non-contracting indices for lhs and rhs.
          //
          // For `result_index`, its batch dimension, if exists, will be at the
          // same dimension as the batch dimension of lhs and rhs. More
          // specifically:
          // - For lhs, the non-contracting dimensions, including the batch
          // dimension have the same index as the `result_index`.
          // - For rhs, the batch dimension is set seperately from other
          // non-contracting dimensions, since these other non-contracting
          // dimensions in rhs follow the non-contracting dimensions of lhs in
          // the resulting index.
          //
          // As an example, for a resulting index:
          //  result_index [result_batch, result_x, result_y]
          // the effecting lhs and rhs indices are:
          //  lhs [result_batch, lhs_non_contracting_dim, contracting_dim
          //  rhs [result_batch, contracting_dim, rhs_non_contracting_dim]
          // `result_x` is only affected by the lhs_non_contracting_dim and
          // likewise `result_y` only depends on rhs_non_contracting_dim.
          //
          // so we can look up the lhs and rhs indices by:
          //
          // lhs:
          //  batch index is the same as `result_batch`.
          //    non-contracting dimension is the same as
          //    result_index[lhs_non_contracting_dim]
          // rhs:
          //  batch index: the same as `result_batch`.
          //  non-contracting dimension index: *not* the same as
          //    result_index[rhs_non_contractng_dim], since the
          //    non-contracting dimensions of lhs are included in the
          //    result_index first. Instead, the non_contracting_dim of rhs must
          //    be calculated as following:
          //      lhs_non_contracting_dimensions_size +
          //      (rhs_non_batch_non_contracting_dim - batch_dim_size) - 1
          //
          //    Note that (rhs_non_batch_contracting_dim - batch_dim_size) is
          //    the index offset to the result_index that only depends on
          //    the non_batch and non-contracting dimensions of rhs. -1 at the
          //    end translates size to index.
          for (auto i : lhs_non_contracting_dims) {
            lhs_index[i] = result_index[i];
          }
          for (auto i : dnums.rhs_batch_dimensions()) {
            rhs_index[i] = result_index[i];
          }
          for (auto i : rhs_non_batch_non_contracting_dims) {
            const int64 rhs_non_batch_non_contracting_dim =
                lhs_non_contracting_size + (i - batch_dim_size) - 1;
            rhs_index[i] = result_index[rhs_non_batch_non_contracting_dim];
          }

          // Accumulates resulting product along the contracted dimension.
          for (int64 i = 0; i < contracted_dimension_size; ++i) {
            lhs_index[lhs_contracting_dimension] = i;
            rhs_index[rhs_contracting_dimension] = i;

            result_val +=
                static_cast<ElementwiseT>(lhs_literal.Get<ReturnT>(lhs_index)) *
                static_cast<ElementwiseT>(rhs_literal.Get<ReturnT>(rhs_index));
          }

          return static_cast<ReturnT>(result_val);
        }));

    parent_->evaluated_[dot] = std::move(result);
    return Status::OK();
  }

  Status HandlePad(HloInstruction* pad) override {
    CHECK(!ShapeUtil::IsTuple(pad->operand(0)->shape()));
    // Padding value must be scalar.
    CHECK(ShapeUtil::IsScalar(pad->operand(1)->shape()));
    CHECK_EQ(ShapeUtil::Rank(pad->operand(0)->shape()),
             pad->padding_config().dimensions_size());

    TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                        ShapeInference::InferPadShape(
                            /*operand_shape=*/pad->operand(0)->shape(),
                            /*padding_value_shape=*/pad->operand(1)->shape(),
                            /*padding_config=*/pad->padding_config()));
    CHECK(ShapeUtil::Compatible(pad->shape(), inferred_return_shape))
        << "return shape is set to: " << ShapeUtil::HumanString(pad->shape())
        << "but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);

    // Create new HLO of padded shape with padding value.
    ReturnT scalar =
        parent_->GetEvaluatedLiteralFor(pad->operand(1)).Get<ReturnT>({});
    auto result = Literal::CreateFromShape(pad->shape());
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&scalar](tensorflow::gtl::ArraySlice<int64> multi_index) {
          return scalar;
        }));

    const Literal& evaluated_operand =
        parent_->GetEvaluatedLiteralFor(pad->operand(0));

    std::vector<int64> input_index(ShapeUtil::Rank(evaluated_operand.shape()),
                                   0);
    std::vector<int64> target_index(ShapeUtil::Rank(result->shape()), 0);

    // Loop through each element of the operand, assign them to the
    // corresponding index of the resulting padded literal.
    const PaddingConfig& pad_config = pad->padding_config();

    auto func = [&](const std::vector<int64>& input_index) {
      for (auto i = 0; i < input_index.size(); ++i) {
        // Interior padding occurs logically before edge padding, so in the case
        // of negative edge padding elements are removed from the
        // interior-padded operand.
        target_index[i] =
            pad_config.dimensions(i).edge_padding_low() +
            input_index[i] * (pad_config.dimensions(i).interior_padding() + 1);

        // Account for negative low and high padding: skip assignment if the
        // any target index is out of range.
        if (!(target_index[i] >= 0 &&
              target_index[i] < pad->shape().dimensions(i))) {
          return true;
        }
      }
      result->Set<ReturnT>(target_index,
                           evaluated_operand.Get<ReturnT>(input_index));
      return true;
    };

    std::vector<int64> zero_base(evaluated_operand.shape().dimensions_size(),
                                 0);
    std::vector<int64> step(evaluated_operand.shape().dimensions_size(), 1);

    ShapeUtil::ForEachIndex(
        evaluated_operand.shape(), zero_base,
        AsInt64Slice(evaluated_operand.shape().dimensions()), step, func);

    parent_->evaluated_[pad] = std::move(result);
    return Status::OK();
  }

  Status HandleDynamicSlice(HloInstruction* dynamic_slice) override {
    auto operand = dynamic_slice->operand(0);
    auto start_indices = dynamic_slice->operand(1);
    auto result_shape = dynamic_slice->shape();
    TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                        ShapeInference::InferDynamicSliceShape(
                            operand->shape(), start_indices->shape(),
                            dynamic_slice->dynamic_slice_sizes()));
    TF_RET_CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
        << "return shape is set to: " << ShapeUtil::HumanString(result_shape)
        << "but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);
    TF_RET_CHECK(
        primitive_util::IsIntegralType(start_indices->shape().element_type()));

    const Literal& operand_literal = parent_->GetEvaluatedLiteralFor(operand);
    const Literal& start_indices_literal =
        parent_->GetEvaluatedLiteralFor(start_indices);

    switch (start_indices->shape().element_type()) {
      case S32: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_slice],
            DynamicSlice<int32>(operand_literal, start_indices_literal,
                                result_shape));
      } break;
      case S64: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_slice],
            DynamicSlice<int64>(operand_literal, start_indices_literal,
                                result_shape));
      } break;
      case U32: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_slice],
            DynamicSlice<uint32>(operand_literal, start_indices_literal,
                                 result_shape));
      } break;
      case U64: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_slice],
            DynamicSlice<uint64>(operand_literal, start_indices_literal,
                                 result_shape));
      } break;
      default:
        LOG(FATAL) << "HandleDynamicSlice: unhandled primitive type for "
                      "start_indices: "
                   << PrimitiveType_Name(start_indices->shape().element_type());
    }

    return Status::OK();
  }

  Status HandleDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice) override {
    auto operand = dynamic_update_slice->operand(0);
    auto update = dynamic_update_slice->operand(1);
    auto start_indices = dynamic_update_slice->operand(2);
    auto result_shape = dynamic_update_slice->shape();
    TF_ASSIGN_OR_RETURN(
        auto inferred_return_shape,
        ShapeInference::InferDynamicUpdateSliceShape(
            operand->shape(), update->shape(), start_indices->shape()));
    TF_RET_CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
        << "return shape is set to: " << ShapeUtil::HumanString(result_shape)
        << "but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);
    TF_RET_CHECK(
        primitive_util::IsIntegralType(start_indices->shape().element_type()));
    TF_RET_CHECK(ShapeUtil::Compatible(result_shape, operand->shape()));

    const Literal& operand_literal = parent_->GetEvaluatedLiteralFor(operand);
    const Literal& update_literal = parent_->GetEvaluatedLiteralFor(update);
    const Literal& start_indices_literal =
        parent_->GetEvaluatedLiteralFor(start_indices);

    switch (start_indices->shape().element_type()) {
      case S32: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_update_slice],
            DynamicUpdateSlice<int32>(operand_literal, update_literal,
                                      start_indices_literal));
      } break;
      case S64: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_update_slice],
            DynamicUpdateSlice<int64>(operand_literal, update_literal,
                                      start_indices_literal));
      } break;
      case U32: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_update_slice],
            DynamicUpdateSlice<uint32>(operand_literal, update_literal,
                                       start_indices_literal));
      } break;
      case U64: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_update_slice],
            DynamicUpdateSlice<uint64>(operand_literal, update_literal,
                                       start_indices_literal));
      } break;
      default:
        LOG(FATAL) << "HandleDynamicUpdateSlice: unhandled primitive type for "
                      "start_indices: "
                   << PrimitiveType_Name(start_indices->shape().element_type());
    }

    return Status::OK();
  }

  template <typename NativeT>
  StatusOr<std::unique_ptr<Literal>> MapImpl(HloInstruction* map) {
    auto operands = map->operands();
    HloComputation* computation = map->to_apply();

    auto result = Literal::CreateFromShape(map->shape());

    HloEvaluator embedded_evaluator;
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
          std::vector<std::unique_ptr<Literal>> arg_literals;
          arg_literals.reserve(operands.size());

          // Construct scalar literal parameters to be passed to the map
          // computation.
          for (auto operand : operands) {
            const Literal& arg_literal =
                parent_->GetEvaluatedLiteralFor(operand);

            auto curr_val = arg_literal.Get<NativeT>(multi_index);
            auto curr_val_literal = Literal::CreateR0<NativeT>(curr_val);

            arg_literals.push_back(std::move(curr_val_literal));
          }

          std::unique_ptr<Literal> computed_result =
              embedded_evaluator
                  .Evaluate<std::unique_ptr<Literal>>(*computation,
                                                      arg_literals)
                  .ConsumeValueOrDie();
          // Clear visit states so that the we can use the evaluate again on
          // the same computation.
          embedded_evaluator.ResetVisitStates();

          return computed_result->Get<ReturnT>({});
        }));
    return std::move(result);
  }

  Status HandleMap(HloInstruction* map) override {
    switch (map->operand(0)->shape().element_type()) {
      case PRED: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<bool>(map));
        break;
      }
      case U8: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<uint8>(map));
        break;
      }
      case U32: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<uint32>(map));
        break;
      }
      case U64: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<uint64>(map));
        break;
      }
      case S8: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<int8>(map));
        break;
      }
      case S32: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<int32>(map));
        break;
      }
      case S64: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<int64>(map));
        break;
      }
      case F16: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map],
                            MapImpl<Eigen::half>(map));
        break;
      }
      case F32: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<float>(map));
        break;
      }
      case F64: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<double>(map));
        break;
      }
      case C64: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<complex64>(map));
        break;
      }
      default:
        LOG(FATAL) << "HandleMap: unhandled primitive type for "
                      "input operand: "
                   << PrimitiveType_Name(
                          map->operand(0)->shape().element_type());
    }

    return Status::OK();
  }

  Status HandleReduce(HloInstruction* reduce) override {
    auto arg = reduce->operand(0);
    auto init_value = reduce->operand(1);
    tensorflow::gtl::ArraySlice<int64> dimensions(reduce->dimensions());
    HloComputation* function = reduce->to_apply();
    TF_RET_CHECK(ShapeUtil::Rank(reduce->shape()) ==
                 ShapeUtil::Rank(arg->shape()) - dimensions.size());
    TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                        ShapeInference::InferReduceShape(
                            /*arg=*/arg->shape(),
                            /*init_value=*/init_value->shape(),
                            /*dimensions_to_reduce=*/dimensions,
                            /*to_apply=*/function->ComputeProgramShape()));
    TF_RET_CHECK(ShapeUtil::Compatible(reduce->shape(), inferred_return_shape))
        << "return shape is set to: " << ShapeUtil::HumanString(reduce->shape())
        << "but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);

    const Literal& arg_literal = parent_->GetEvaluatedLiteralFor(arg);
    VLOG(3) << "HandleReduce arg_literal: " << arg_literal.ToString();
    const Literal& init_literal = parent_->GetEvaluatedLiteralFor(init_value);
    VLOG(3) << "HandleReduce init_literal: " << init_literal.ToString();
    TF_RET_CHECK(ShapeUtil::IsScalar(init_literal.shape()));
    auto init_scalar = init_literal.Get<ReturnT>({});

    auto result = Literal::CreateFromShape(reduce->shape());

    const auto arg_dimensions = AsInt64Slice(arg_literal.shape().dimensions());
    std::vector<int64> arg_dim_steps(arg_dimensions.size());
    std::vector<int64> arg_dim_counts(arg_dimensions.size());
    for (const int64 dim : dimensions) {
      arg_dim_steps[dim] = 1;
      arg_dim_counts[dim] = arg_dimensions[dim];
    }

    // Create mapping from result index to arg index.
    const int64 result_rank = ShapeUtil::Rank(result->shape());
    int64 result_dim = 0;
    std::vector<int64> result_to_arg_index(result_rank);
    for (int64 i = 0; i < arg_dimensions.size(); ++i) {
      if (arg_dim_steps[i] == 0) {
        result_to_arg_index[result_dim] = i;
        ++result_dim;
      }
    }

    HloEvaluator embedded_evaluator;
    // For each resulting dimension, calculate and assign computed value.
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
          ReturnT result_val = init_scalar;

          std::vector<int64> base(arg_dimensions.size());
          for (int64 i = 0; i < multi_index.size(); ++i) {
            base[result_to_arg_index[i]] = multi_index[i];
          }

          auto func = [&](const std::vector<int64>& input_index) {
            auto curr_val = arg_literal.Get<ReturnT>(input_index);

            // Evaluate computation with specified literal operands.
            auto curr_val_literal = Literal::CreateR0<ReturnT>(curr_val);
            auto result_val_literal = Literal::CreateR0<ReturnT>(result_val);
            std::vector<const Literal*> args = {curr_val_literal.get(),
                                                result_val_literal.get()};

            std::unique_ptr<Literal> computed_result =
                embedded_evaluator.Evaluate<const Literal*>(*function, args)
                    .ConsumeValueOrDie();
            // Clear visit states so that the we can use the evaluate again on
            // the same computation.
            embedded_evaluator.ResetVisitStates();

            // Assign computed result to result_val.
            result_val = computed_result->Get<ReturnT>({});

            return true;
          };

          ShapeUtil::ForEachIndex(arg_literal.shape(), base, arg_dim_counts,
                                  arg_dim_steps, func);

          return result_val;
        }));

    parent_->evaluated_[reduce] = std::move(result);
    return Status::OK();
  }

  Status HandleSelectAndScatter(HloInstruction* select_and_scatter) override {
    auto operand = select_and_scatter->operand(0);
    auto source = select_and_scatter->operand(1);
    const Window& window = select_and_scatter->window();

    const Literal& init_literal =
        parent_->GetEvaluatedLiteralFor(select_and_scatter->operand(2));
    TF_RET_CHECK(ShapeUtil::IsScalar(init_literal.shape()));
    auto init_scalar = init_literal.Get<ReturnT>({});

    auto result = Literal::CreateFromShape(select_and_scatter->shape());

    // Initialize result array with the init value.
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> output_index) {
          return init_scalar;
        }));

    std::vector<int64> window_dimension_sizes;
    for (const auto& window_dimension : window.dimensions()) {
      window_dimension_sizes.push_back(window_dimension.size());
    }
    const Shape window_shape = ShapeUtil::MakeShape(
        operand->shape().element_type(), window_dimension_sizes);

    HloComputation* select = select_and_scatter->select();
    HloComputation* scatter = select_and_scatter->scatter();

    const Literal& operand_literal = parent_->GetEvaluatedLiteralFor(operand);
    const Literal& source_literal = parent_->GetEvaluatedLiteralFor(source);

    int64 rank = ShapeUtil::Rank(operand_literal.shape());

    HloEvaluator embedded_evaluator;
    DimensionVector source_index(rank);

    std::fill(source_index.begin(), source_index.end(), 0);
    do {
      // For each element in `source`, we place a window in `operand`. For each
      // window placement, we iterate inside the window twice:
      //
      // 1. Find the selected index by applying `select` function to all
      // elements. E.g., If the `select` function is GreaterEqual, the first
      // iteration through the window finds the biggest value and returns its
      // index.
      //
      // 2. Using the selected index, scatter value from `source` to result. We
      // do this by iterating through the window, and compare each index with
      // the selected index.
      tensorflow::gtl::optional<ReturnT> selected_val;
      tensorflow::gtl::optional<std::vector<int64>> selected_index;

      IterateThroughWindow(
          window_shape, window, operand_literal.shape(), source_index,
          [&](const std::vector<int64>& operand_index) {
            auto curr_val = operand_literal.Get<ReturnT>(operand_index);
            if (!selected_val) {
              selected_val = curr_val;
              selected_index = operand_index;
            }
            const auto curr_val_literal = Literal::CreateR0<ReturnT>(curr_val);
            const auto selected_val_literal =
                Literal::CreateR0<ReturnT>(*selected_val);

            const std::vector<const Literal*> args = {
                curr_val_literal.get(), selected_val_literal.get()};
            std::unique_ptr<Literal> computed_result =
                embedded_evaluator.Evaluate<const Literal*>(*select, args)
                    .ConsumeValueOrDie();
            bool selected = computed_result->Get<bool>({});
            if (selected) {
              selected_val = curr_val;
              selected_index = operand_index;
            }
            embedded_evaluator.ResetVisitStates();
          });

      IterateThroughWindow(
          window_shape, window, operand_literal.shape(), source_index,
          [&](const std::vector<int64>& operand_index) {
            if (std::equal(operand_index.begin(), operand_index.end(),
                           selected_index->begin())) {
              auto source = source_literal.Get<ReturnT>(source_index);
              auto scattered = result->Get<ReturnT>(operand_index);
              const auto source_literal = Literal::CreateR0<ReturnT>(source);
              const auto scattered_literal =
                  Literal::CreateR0<ReturnT>(scattered);

              const std::vector<const Literal*> args = {
                  source_literal.get(), scattered_literal.get()};
              std::unique_ptr<Literal> computed_result =
                  embedded_evaluator.Evaluate<const Literal*>(*scatter, args)
                      .ConsumeValueOrDie();
              result->Set(operand_index, computed_result->Get<ReturnT>({}));
              // Clear visit states so that the we can use the evaluator again
              // on the same computation.
              embedded_evaluator.ResetVisitStates();
            }
          });
    } while (IndexUtil::BumpIndices(source->shape(), &source_index));

    parent_->evaluated_[select_and_scatter] = std::move(result);
    return Status::OK();
  }

  Status HandleReduceWindow(HloInstruction* reduce_window) override {
    auto operand = reduce_window->operand(0);
    const Window& window = reduce_window->window();
    HloComputation* function = reduce_window->to_apply();
    TF_ASSIGN_OR_RETURN(
        auto inferred_return_shape,
        ShapeInference::InferReduceWindowShape(
            /*operand_shape=*/reduce_window->operand(0)->shape(),
            /*init_value=*/reduce_window->operand(1)->shape(), window,
            /*to_apply_shape=*/function->ComputeProgramShape()));
    TF_RET_CHECK(
        ShapeUtil::Compatible(reduce_window->shape(), inferred_return_shape))
        << "return shape is set to: "
        << ShapeUtil::HumanStringWithLayout(reduce_window->shape())
        << "but is inferred to be: "
        << ShapeUtil::HumanStringWithLayout(inferred_return_shape);

    const Literal& operand_literal =
        parent_->GetEvaluatedLiteralFor(reduce_window->operand(0));
    VLOG(3) << "HandleReduceWindow arg_literal: " << operand_literal.ToString();
    const Literal& init_literal =
        parent_->GetEvaluatedLiteralFor(reduce_window->operand(1));
    VLOG(3) << "HandleReduceWindow init_literal: " << init_literal.ToString();
    TF_RET_CHECK(ShapeUtil::IsScalar(init_literal.shape()));
    auto init_scalar = init_literal.Get<ReturnT>({});

    auto result = Literal::CreateFromShape(reduce_window->shape());

    // Creates a Shape object from window, for iteration below.
    std::vector<int64> window_dimension_sizes;
    for (const auto& window_dimension : window.dimensions()) {
      window_dimension_sizes.push_back(window_dimension.size());
    }
    const Shape window_shape = ShapeUtil::MakeShape(
        operand->shape().element_type(), window_dimension_sizes);

    DimensionVector window_index(window.dimensions_size());
    DimensionVector operand_index(ShapeUtil::Rank(operand_literal.shape()));

    HloEvaluator embedded_evaluator;
    // For each resulting dimension, calculate and assign computed value.
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> output_index) {
          ReturnT result_val = init_scalar;

          std::fill(window_index.begin(), window_index.end(), 0);
          std::fill(operand_index.begin(), operand_index.end(), 0);

          IterateThroughWindow(
              window_shape, window, operand_literal.shape(), output_index,
              [&](const std::vector<int64>& operand_index) {
                auto curr_val = operand_literal.Get<ReturnT>(operand_index);

                // Evaluate computation with specified literal operands.
                const auto curr_val_literal =
                    Literal::CreateR0<ReturnT>(curr_val);
                const auto result_val_literal =
                    Literal::CreateR0<ReturnT>(result_val);
                const std::vector<const Literal*> args = {
                    curr_val_literal.get(), result_val_literal.get()};
                std::unique_ptr<Literal> computed_result =
                    embedded_evaluator.Evaluate<const Literal*>(*function, args)
                        .ConsumeValueOrDie();

                // Clear visit states so that the we can use the evaluate again
                // on the same computation.
                embedded_evaluator.ResetVisitStates();

                result_val = computed_result->Get<ReturnT>({});
              });

          return result_val;
        }));

    parent_->evaluated_[reduce_window] = std::move(result);
    return Status::OK();
  }

  Status HandleSlice(HloInstruction* slice) override {
    auto operand = slice->operand(0);
    const Shape& shape = slice->shape();
    TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                        ShapeInference::InferSliceShape(
                            operand->shape(), slice->slice_starts(),
                            slice->slice_limits(), slice->slice_strides()));
    TF_RET_CHECK(ShapeUtil::Compatible(shape, inferred_return_shape))
        << "return shape set to: " << ShapeUtil::HumanString(shape)
        << " but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);

    const int64 rank = ShapeUtil::Rank(operand->shape());
    const Literal& operand_literal = parent_->GetEvaluatedLiteralFor(operand);
    auto func = [&](tensorflow::gtl::ArraySlice<int64> out_index) {
      DimensionVector operand_index(rank);
      for (int64 i = 0; i < rank; ++i) {
        operand_index[i] =
            slice->slice_starts(i) + out_index[i] * slice->slice_strides(i);
      }
      return operand_literal.Get<ReturnT>(operand_index);
    };

    auto result = Literal::CreateFromDimensions(
        shape.element_type(), AsInt64Slice(shape.dimensions()));
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(func));
    parent_->evaluated_[slice] = std::move(result);
    return Status::OK();
  }

  template <typename NativeT, typename std::enable_if<std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleSin(HloInstruction* sin) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[sin],
                        ElementWiseUnaryOp(sin, [](ElementwiseT elem_operand) {
                          return std::sin(elem_operand);
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<std::is_integral<NativeT>::value ||
                              is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleSin(HloInstruction* sin) {
    return InvalidArgument("Unsupported type for Sin");
  }

  Status HandleSin(HloInstruction* sin) override {
    return HandleSin<ElementwiseT>(sin);
  }

  template <typename NativeT, typename std::enable_if<std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleCos(HloInstruction* cos) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[cos],
                        ElementWiseUnaryOp(cos, [](ElementwiseT elem_operand) {
                          return std::cos(elem_operand);
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<std::is_integral<NativeT>::value ||
                              is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleCos(HloInstruction* cos) {
    return InvalidArgument("Unsupported type for Cos");
  }

  Status HandleCos(HloInstruction* cos) override {
    return HandleCos<ElementwiseT>(cos);
  }

  template <typename NativeT, typename std::enable_if<std::is_same<
                                  float, NativeT>::value>::type* = nullptr>
  Status HandleReducePrecision(HloInstruction* reduce_precision) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[reduce_precision],
        ElementWiseUnaryOp(reduce_precision, [reduce_precision](
                                                 ElementwiseT elem) {
          uint32_t value_as_int = tensorflow::bit_cast<uint32_t>(elem);
          const uint32_t mantissa_bits = reduce_precision->mantissa_bits();
          const uint32_t exponent_bits = reduce_precision->exponent_bits();

          // Code is based on the CPU/GPU implementation in LLVM-emitting code.
          //
          // Bits in float type:
          //   mantissa : bits [0:22]
          //   exponent : bits [23:30]
          //   sign     : bits [31]
          if (mantissa_bits < 23) {
            const uint32_t last_mantissa_bit_mask = 1u << (23 - mantissa_bits);

            // Compute rounding bias for round-to-nearest with ties to even.
            // This is equal to a base value of 0111... plus one bit if the last
            // remaining mantissa bit is 1.
            const uint32_t base_rounding_bias =
                (last_mantissa_bit_mask >> 1) - 1;
            const uint32_t x_last_mantissa_bit =
                (value_as_int & last_mantissa_bit_mask) >> (23 - mantissa_bits);
            const uint32_t x_rounding_bias =
                x_last_mantissa_bit + base_rounding_bias;

            // Add rounding bias, and mask out truncated bits.  Note that the
            // case where adding the rounding bias overflows into the exponent
            // bits is correct; the non-masked mantissa bits will all be zero,
            // and the exponent will be incremented by one.
            const uint32_t truncation_mask = ~(last_mantissa_bit_mask - 1);
            value_as_int = value_as_int + x_rounding_bias;
            value_as_int = value_as_int & truncation_mask;
          }
          if (exponent_bits < 8) {
            // Masks for f32 values.
            const uint32_t f32_sign_bit_mask = 1u << 31;
            const uint32_t f32_exp_bits_mask = 0xffu << 23;

            // An exponent of 2^(n-1)-1 -- that is, 0111... with the zero in the
            // most- significant bit -- is equal to 1.0f for all exponent sizes.
            // Adding 2^(n-1)-1 to this gives us the highest non-infinite
            // exponent for a bit- size of n, and subtracting 2^(n-1)-1 from
            // this gives us the lowest' exponent (corresponding to 0.0f).
            //
            // Thus, the f32 exponent corresponding to the highest non-infinite
            // exponent for a bit size of n is (2^7-1) + 2^(n-1)-1, and the f32
            // exponent corresponding to the lowest exponent for a bit size of n
            // is (2^7-1) - 2^(n-1)-1.
            //
            // Note that we have already checked that exponents_bits >= 1.
            const uint32_t f32_exponent_bias = (1 << 7) - 1;
            const uint32_t reduced_exponent_bias =
                (1 << (exponent_bits - 1)) - 1;
            const uint32_t reduced_max_exponent =
                f32_exponent_bias + reduced_exponent_bias;
            const uint32_t reduced_min_exponent =
                f32_exponent_bias - reduced_exponent_bias;

            // Do we overflow or underflow?
            const uint32_t x_exponent = value_as_int & f32_exp_bits_mask;
            const bool x_overflows = x_exponent > (reduced_max_exponent << 23);
            const bool x_underflows =
                x_exponent <= (reduced_min_exponent << 23);

            // Compute appropriately-signed values of zero and infinity.
            const uint32_t x_signed_zero = value_as_int & f32_sign_bit_mask;
            const uint32_t x_signed_inf = x_signed_zero | f32_exp_bits_mask;

            // Force to zero or infinity if overflow or underflow.  (Note that
            // this truncates all denormal values to zero, rather than rounding
            // them.)
            value_as_int = x_overflows ? x_signed_inf : value_as_int;
            value_as_int = x_underflows ? x_signed_zero : value_as_int;
          }

          float reduced_result = tensorflow::bit_cast<float>(value_as_int);
          if (std::isnan(elem)) {
            reduced_result = mantissa_bits > 0
                                 ? elem
                                 : std::numeric_limits<float>::infinity();
          }
          return reduced_result;
        }));
    return Status::OK();
  }

  template <typename NativeT, typename std::enable_if<std::is_same<
                                  double, NativeT>::value>::type* = nullptr>
  Status HandleReducePrecision(HloInstruction* reduce_precision) {
    return InvalidArgument("Double not supported for reduce precision");
  }

  template <
      typename NativeT,
      typename std::enable_if<std::is_integral<NativeT>::value ||
                              is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleReducePrecision(HloInstruction* reduce_precision) {
    return InvalidArgument("Unsupported type for reduce precision");
  }

  Status HandleReducePrecision(HloInstruction* reduce_precision) override {
    return HandleReducePrecision<ElementwiseT>(reduce_precision);
  }

 private:
  template <typename IndexT>
  StatusOr<std::unique_ptr<Literal>> DynamicSlice(
      const Literal& operand_literal, const Literal& start_indices_literal,
      const Shape& result_shape) {
    auto start_indices_typed = start_indices_literal.data<IndexT>();
    std::vector<int64> start(start_indices_typed.begin(),
                             start_indices_typed.end());

    std::vector<int64> operand_indices(start.size());

    auto result = Literal::CreateFromShape(result_shape);
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
          for (int64 i = 0; i < operand_indices.size(); ++i) {
            CHECK_GE(multi_index[i] + start[i], 0);
            // Mod is only used here to be consistent with the existing
            // backends' behavior.
            operand_indices[i] = (multi_index[i] + start[i]) %
                                 operand_literal.shape().dimensions(i);
          }

          auto result = operand_literal.Get<ReturnT>(operand_indices);
          return result;
        }));

    return std::move(result);
  }

  template <typename IndexT>
  StatusOr<std::unique_ptr<Literal>> DynamicUpdateSlice(
      const Literal& operand_literal, const Literal& update_literal,
      const Literal& start_indices_literal) {
    auto start_indices_typed = start_indices_literal.data<IndexT>();
    const std::vector<int64> start(start_indices_typed.begin(),
                                   start_indices_typed.end());

    auto result = operand_literal.CloneToUnique();
    std::vector<int64> result_index(ShapeUtil::Rank(result->shape()), 0);

    auto func = [&](const std::vector<int64>& update_index) {
      std::transform(update_index.begin(), update_index.end(), start.begin(),
                     result_index.begin(), std::plus<int64>());

      result->Set<ReturnT>(result_index,
                           update_literal.Get<ReturnT>(update_index));
      return true;
    };

    std::vector<int64> base(update_literal.shape().dimensions_size(), 0);
    std::vector<int64> step(update_literal.shape().dimensions_size(), 1);
    ShapeUtil::ForEachIndex(update_literal.shape(), base,
                            AsInt64Slice(update_literal.shape().dimensions()),
                            step, func);

    return std::move(result);
  }

  StatusOr<std::unique_ptr<Literal>> ElementWiseUnaryOp(
      HloInstruction* instruction,
      const std::function<ElementwiseT(ElementwiseT)>& unary_op) {
    const Literal& operand_literal =
        parent_->GetEvaluatedLiteralFor(instruction->operand(0));
    TF_ASSIGN_OR_RETURN(
        auto result_literal,
        (ElementWiseUnaryOpImpl<ReturnT, ReturnT>(
            instruction, ConvertUnaryFunction(unary_op), operand_literal)));

    return std::move(result_literal);
  }

  StatusOr<std::unique_ptr<Literal>> ElementWiseBinaryOp(
      HloInstruction* instruction,
      const std::function<ElementwiseT(ElementwiseT, ElementwiseT)>&
          binary_op) {
    const auto shape = instruction->shape();
    const auto* lhs = instruction->operand(0);
    const auto* rhs = instruction->operand(1);

    // TODO(b/35950897, b/27796129): add DCHECK back once implicit broadcast
    // is removed.
    if (!(ShapeUtil::SameDimensions(shape, rhs->shape()) &&
          ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()))) {
      return Unimplemented(
          "Implicit broadcasting is currently unsupported in HLO evaluator "
          "Shape Mismatch: %s vs %s vs %s: ",
          ShapeUtil::HumanString(shape).c_str(),
          ShapeUtil::HumanString(lhs->shape()).c_str(),
          ShapeUtil::HumanString(rhs->shape()).c_str());
    }

    const Literal& lhs_literal = parent_->GetEvaluatedLiteralFor(lhs);
    const Literal& rhs_literal = parent_->GetEvaluatedLiteralFor(rhs);

    auto result = Literal::CreateFromShape(shape);

    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
          return ConvertBinaryFunction(binary_op)(
              lhs_literal.Get<ReturnT>(multi_index),
              rhs_literal.Get<ReturnT>(multi_index));
        }));
    return std::move(result);
  }

  template <typename LhsType, typename RhsType, typename EhsType>
  StatusOr<std::unique_ptr<Literal>> ElementwiseTernaryOp(
      HloInstruction* instruction,
      const std::function<ReturnT(LhsType, RhsType, EhsType)>& ternary_op) {
    const auto shape = instruction->shape();
    const auto* lhs = instruction->operand(0);
    const auto* rhs = instruction->operand(1);
    const auto* ehs = instruction->operand(2);

    // TODO(b/35950897, b/27796129): add DCHECK back once implicit
    // broadcast is removed.
    if (!(ShapeUtil::SameDimensions(shape, lhs->shape()) &&
          ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()) &&
          ShapeUtil::SameDimensions(rhs->shape(), ehs->shape()))) {
      return Unimplemented(
          "Implicit broadcasting is currently unsupported in HLO evaluator "
          "Shape Mismatch: %s vs %s vs %s vs %s: ",
          ShapeUtil::HumanString(shape).c_str(),
          ShapeUtil::HumanString(lhs->shape()).c_str(),
          ShapeUtil::HumanString(rhs->shape()).c_str(),
          ShapeUtil::HumanString(ehs->shape()).c_str());
    }

    const Literal& lhs_literal = parent_->GetEvaluatedLiteralFor(lhs);
    const Literal& rhs_literal = parent_->GetEvaluatedLiteralFor(rhs);
    const Literal& ehs_literal = parent_->GetEvaluatedLiteralFor(ehs);

    auto result = Literal::CreateFromShape(shape);

    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
          return ternary_op(lhs_literal.Get<LhsType>(multi_index),
                            rhs_literal.Get<RhsType>(multi_index),
                            ehs_literal.Get<EhsType>(multi_index));
        }));

    return std::move(result);
  }

  template <typename NativeT>
  static bool IsShiftOutOfBounds(NativeT rhs) {
    typedef typename std::make_unsigned<NativeT>::type UnsignedT;
    UnsignedT lhs_size_unsigned = sizeof(NativeT) * CHAR_BIT;
    UnsignedT rhs_unsigned = static_cast<UnsignedT>(rhs);
    return rhs_unsigned >= lhs_size_unsigned;
  }

  HloEvaluator* parent_;
};  // class HloEvaluator::TypedVisitor

HloEvaluator::HloEvaluator() {
  typed_visitors_[PRED] = MakeUnique<TypedVisitor<bool>>(this);
  typed_visitors_[U8] = MakeUnique<TypedVisitor<uint8>>(this);
  typed_visitors_[U16] = MakeUnique<FunctionVisitor>([](HloInstruction*) {
    return Unimplemented("HloEvaluator: unhandled primitive type: U16.");
  });
  typed_visitors_[U32] = MakeUnique<TypedVisitor<uint32>>(this);
  typed_visitors_[U64] = MakeUnique<TypedVisitor<uint64>>(this);
  typed_visitors_[S8] = MakeUnique<TypedVisitor<int8>>(this);
  typed_visitors_[S16] = MakeUnique<FunctionVisitor>([](HloInstruction*) {
    return Unimplemented("HloEvaluator: unhandled primitive type: S16.");
  });
  typed_visitors_[S32] = MakeUnique<TypedVisitor<int32>>(this);
  typed_visitors_[S64] = MakeUnique<TypedVisitor<int64>>(this);
  typed_visitors_[F16] = MakeUnique<TypedVisitor<Eigen::half, float>>(this);
  typed_visitors_[F32] = MakeUnique<TypedVisitor<float>>(this);
  typed_visitors_[F64] = MakeUnique<TypedVisitor<double>>(this);
  typed_visitors_[C64] = MakeUnique<TypedVisitor<complex64>>(this);

  // Most of the evaluator computations we use don't support BF16 (e.g.,
  // std::ceil, std::tanh). To make evaluator work with BF16, we set all
  // elementwise computations to be done in F32 and do BF16<->F32 conversion
  // around the input and the output of the computations.
  typed_visitors_[BF16] = MakeUnique<TypedVisitor<bfloat16, float>>(this);
  typed_visitors_[TUPLE] = MakeUnique<FunctionVisitor>([](HloInstruction*) {
    return Unimplemented("HloEvaluator: unhandled primitive type: TUPLE.");
  });
  typed_visitors_[OPAQUE] = MakeUnique<FunctionVisitor>([](HloInstruction*) {
    return Unimplemented("HloEvaluator: unhandled primitive type: OPAQUE.");
  });
}

template <typename LiteralPtr>
StatusOr<std::unique_ptr<Literal>> HloEvaluator::Evaluate(
    const HloModule& module,
    tensorflow::gtl::ArraySlice<LiteralPtr> arg_literals) {
  XLA_VLOG_LINES(2, "HloEvaluator::Evaluate module:\n" + module.ToString());

  evaluated_.clear();
  arg_literals_.clear();
  for (const auto& literal_ptr : arg_literals) {
    arg_literals_.push_back(&*literal_ptr);
  }

  TF_RETURN_IF_ERROR(module.entry_computation()->Accept(this));

  return GetEvaluatedLiteralFor(module.entry_computation()->root_instruction())
      .CloneToUnique();
}

template <typename LiteralPtr>
StatusOr<std::unique_ptr<Literal>> HloEvaluator::Evaluate(
    const HloComputation& computation,
    tensorflow::gtl::ArraySlice<LiteralPtr> arg_literals) {
  XLA_VLOG_LINES(
      2, "HloEvaluator::Evaluate computation:\n" + computation.ToString());

  evaluated_.clear();
  arg_literals_.clear();
  for (const auto& literal_ptr : arg_literals) {
    arg_literals_.push_back(&*literal_ptr);
  }

  TF_RETURN_IF_ERROR(computation.Accept(this));
  return GetEvaluatedLiteralFor(computation.root_instruction()).CloneToUnique();
}

template <typename LiteralPtr>
StatusOr<std::unique_ptr<Literal>> HloEvaluator::Evaluate(
    HloInstruction* instruction,
    tensorflow::gtl::ArraySlice<LiteralPtr> arg_literals) {
  TF_RET_CHECK(hlo_query::AllOperandsAreParametersOrConstants(*instruction));
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(instruction->shape()));

  evaluated_.clear();
  arg_literals_.clear();
  for (const auto& literal_ptr : arg_literals) {
    arg_literals_.push_back(&*literal_ptr);
  }

  // Evaluate operands of Parameter type against the input literals which
  // caches the evaluated literal results.
  for (const auto operand : instruction->operands()) {
    if (operand->opcode() == HloOpcode::kParameter) {
      const Literal* input_literal = arg_literals_[operand->parameter_number()];
      VLOG(2) << "Parameter operand evaluated to: "
              << input_literal->ToString();
      TF_RET_CHECK(ShapeUtil::Equal(operand->shape(), input_literal->shape()));

      evaluated_[operand] = input_literal->CloneToUnique();
    }
  }

  TF_RETURN_IF_ERROR(Preprocess(instruction));
  TF_RETURN_IF_ERROR(instruction->Visit(this));
  TF_RETURN_IF_ERROR(Postprocess(instruction));
  return GetEvaluatedLiteralFor(instruction).CloneToUnique();
}

StatusOr<std::unique_ptr<Literal>> HloEvaluator::Evaluate(
    HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kParameter) {
    return tensorflow::errors::FailedPrecondition(
        "Cannot evaluate a parameter.");
  }
  if (!hlo_query::AllOperandsAreConstants(*instruction)) {
    return tensorflow::errors::FailedPrecondition(
        "Not all operands are constants.");
  }
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(instruction->shape()));

  arg_literals_.clear();
  evaluated_.clear();

  TF_RETURN_IF_ERROR(Preprocess(instruction));
  TF_RETURN_IF_ERROR(instruction->Visit(this));
  TF_RETURN_IF_ERROR(Postprocess(instruction));
  return GetEvaluatedLiteralFor(instruction).CloneToUnique();
}

std::unique_ptr<Literal> HloEvaluator::TryEvaluate(
    HloInstruction* instruction) {
  auto result_or = Evaluate(instruction);
  if (!result_or.ok()) {
    VLOG(1) << "TryEvaluate failed:" << result_or.status();
    return nullptr;
  }

  return result_or.ConsumeValueOrDie();
}

StatusOr<std::unique_ptr<Literal>> HloEvaluator::EvaluateWithSubstitutions(
    const HloInstruction* instruction,
    const std::unordered_map<const HloInstruction*, const Literal*>&
        substitutions) {
  std::vector<std::unique_ptr<HloInstruction>> owned_operands;
  for (const HloInstruction* operand : instruction->operands()) {
    auto it = substitutions.find(operand);
    if (it == substitutions.end()) {
      owned_operands.push_back(operand->Clone());
    } else {
      owned_operands.push_back(
          HloInstruction::CreateConstant(it->second->CloneToUnique()));
    }
  }

  std::vector<HloInstruction*> operands;
  operands.reserve(owned_operands.size());
  for (auto& operand : owned_operands) {
    operands.push_back(operand.get());
  }

  std::unique_ptr<HloInstruction> cloned_instruction =
      instruction->CloneWithNewOperands(instruction->shape(), operands);
  auto result = Evaluate(cloned_instruction.get());

  // Clean up our cloned instructions before returning.
  cloned_instruction->DetachFromOperands();
  for (auto& operand : owned_operands) {
    operand->DetachFromOperands();
  }

  return result;
}

Status HloEvaluator::HandleParameter(HloInstruction* parameter) {
  CHECK_LT(parameter->parameter_number(), arg_literals_.size());
  const Literal* input_literal = arg_literals_[parameter->parameter_number()];
  VLOG(2) << "Parameter evaluated to: " << input_literal->ToString();
  DCHECK(ShapeUtil::Equal(parameter->shape(), input_literal->shape()))
      << "parameter shape is: " << ShapeUtil::HumanString(parameter->shape())
      << ", but input literal shape is: "
      << ShapeUtil::HumanString(input_literal->shape());

  evaluated_[parameter] = input_literal->CloneToUnique();
  return Status::OK();
}

Status HloEvaluator::HandleConstant(HloInstruction*) { return Status::OK(); }

Status HloEvaluator::HandleReshape(HloInstruction* reshape) {
  TF_ASSIGN_OR_RETURN(
      evaluated_[reshape],
      GetEvaluatedLiteralFor(reshape->operand(0))
          .Reshape(AsInt64Slice(reshape->shape().dimensions())));
  return Status::OK();
}

Status HloEvaluator::HandleTranspose(HloInstruction* transpose) {
  evaluated_[transpose] = GetEvaluatedLiteralFor(transpose->operand(0))
                              .Transpose(transpose->dimensions());
  return Status::OK();
}

Status HloEvaluator::HandleConcatenate(HloInstruction* concatenate) {
  tensorflow::gtl::ArraySlice<HloInstruction*> operands(
      concatenate->operands());
  // The result concatenate dimension is going to be the sum of all
  // concatenate dimensions of the operands taking part of the operation.
  const Shape& reference_shape = operands[0]->shape();
  CHECK(!ShapeUtil::IsTuple(reference_shape));
  const int64 rank = ShapeUtil::Rank(reference_shape);
  const int64 concat_dim = concatenate->dimensions()[0];
  CHECK_GE(concat_dim, 0);
  CHECK_LT(concat_dim, rank);

  DimensionVector concat_dimensions(reference_shape.dimensions().begin(),
                                    reference_shape.dimensions().end());

  for (int64 i = 1; i < operands.size(); ++i) {
    const Shape& operand_shape = operands[i]->shape();
    CHECK(!ShapeUtil::IsTuple(operand_shape));
    // Accumulate the concat dimension from all tensors taking part to the
    // operation.
    concat_dimensions[concat_dim] +=
        ShapeUtil::GetDimension(operand_shape, concat_dim);
  }

  auto result_literal = Literal::CreateFromDimensions(
      reference_shape.element_type(), concat_dimensions);
  DimensionVector source_indices(rank, 0);
  DimensionVector dest_indices(concat_dimensions.size(), 0);

  for (auto operand : operands) {
    const Shape& operand_shape = operand->shape();
    TF_RETURN_IF_ERROR(result_literal->CopySliceFrom(
        GetEvaluatedLiteralFor(operand), source_indices, dest_indices,
        AsInt64Slice(operand_shape.dimensions())));
    dest_indices[concat_dim] +=
        ShapeUtil::GetDimension(operand_shape, concat_dim);
  }

  evaluated_[concatenate] = std::move(result_literal);
  return Status::OK();
}

Status HloEvaluator::HandleIsFinite(HloInstruction* is_finite) {
  auto operand = is_finite->operand(0);
  if (!ShapeUtil::ElementIsFloating(operand->shape())) {
    return InvalidArgument(
        "expected element type in shape to be float for IsFinite op, got: %s",
        PrimitiveType_Name(operand->shape().element_type()).c_str());
  }

  switch (operand->shape().element_type()) {
    case F16:
      return Unimplemented("unhandled primitive type: F16.");
    case F32: {
      auto result_or = ElementWiseUnaryOpImpl<bool, float>(
          is_finite,
          [](float elem_operand) { return std::isfinite(elem_operand); },
          GetEvaluatedLiteralFor(operand));
      TF_ASSIGN_OR_RETURN(evaluated_[is_finite], std::move(result_or));
      break;
    }
    case F64: {
      auto result_or = ElementWiseUnaryOpImpl<bool, double>(
          is_finite,
          [](double elem_operand) { return std::isfinite(elem_operand); },
          GetEvaluatedLiteralFor(operand));
      TF_ASSIGN_OR_RETURN(evaluated_[is_finite], std::move(result_or));
      break;
    }
    default:
      LOG(FATAL) << "HandleIsFinite: unknown/unhandled primitive type: "
                 << PrimitiveType_Name(operand->shape().element_type());
  }

  return Status::OK();
}

Status HloEvaluator::HandleCompare(HloInstruction* compare) {
  HloOpcode opcode = compare->opcode();
  auto lhs = compare->operand(0);
  auto rhs = compare->operand(1);
  // TODO(b/35950897, b/27796129): add DCHECK back once implicit broadcast is
  // removed.
  if (!(ShapeUtil::SameDimensions(compare->shape(), rhs->shape()) &&
        ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()))) {
    return Unimplemented(
        "Implicit broadcasting is currently unsupported in HLO evaluator "
        "Shape Mismatch: %s vs %s vs %s",
        ShapeUtil::HumanString(compare->shape()).c_str(),
        ShapeUtil::HumanString(lhs->shape()).c_str(),
        ShapeUtil::HumanString(rhs->shape()).c_str());
  }

  TF_RET_CHECK(lhs->shape().element_type() == rhs->shape().element_type());

  const Literal& lhs_literal = GetEvaluatedLiteralFor(lhs);
  const Literal& rhs_literal = GetEvaluatedLiteralFor(rhs);

  // Note here we switch on the operand's type.
  switch (lhs->shape().element_type()) {
    case PRED: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<bool>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case U8: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<uint8>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case U16:
      return Unimplemented("unhandled primitive type: U16.");
    case U32: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<uint32>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case U64: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<uint64>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case S8: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<int8>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case S16:
      return Unimplemented("unhandled primitive type: S16.");
    case S32: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<int32>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case S64: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<int64>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case F16:
      return Unimplemented("unhandled primitive type: F16.");
    case F32: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<float>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case F64: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<double>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case C64: {
      TF_ASSIGN_OR_RETURN(evaluated_[compare],
                          Compare<complex64>(compare->shape(), opcode,
                                             lhs_literal, rhs_literal));
    } break;
    default:
      LOG(FATAL) << "HandleCompare: unknown primitive type: "
                 << PrimitiveType_Name(lhs->shape().element_type());
  }

  return Status::OK();
}

Status HloEvaluator::HandleTuple(HloInstruction* tuple) {
  std::vector<const Literal*> operand_literals;
  for (auto operand : tuple->operands()) {
    operand_literals.push_back(&GetEvaluatedLiteralFor(operand));
  }

  evaluated_[tuple] = Literal::MakeTuple(operand_literals);
  return Status::OK();
}

Status HloEvaluator::HandleGetTupleElement(HloInstruction* get_tuple_element) {
  const auto result_shape = get_tuple_element->shape();
  const int64 index = get_tuple_element->tuple_index();

  auto operand = get_tuple_element->operand(0);
  TF_ASSIGN_OR_RETURN(
      auto inferred_return_shape,
      ShapeInference::InferGetTupleElementShape(operand->shape(), index));
  TF_RET_CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
      << "return shape set to: " << ShapeUtil::HumanString(result_shape)
      << " but is inferred to be: "
      << ShapeUtil::HumanString(inferred_return_shape);

  const Literal& operand_tuple_literal = GetEvaluatedLiteralFor(operand);

  evaluated_[get_tuple_element] = MakeUnique<Literal>(
      ShapeUtil::GetTupleElementShape(operand->shape(), index));
  return evaluated_[get_tuple_element]->CopyFrom(operand_tuple_literal,
                                                 /*dest_shape_index=*/{},
                                                 /*src_shape_index=*/{index});
}

Status HloEvaluator::HandleCopy(HloInstruction* copy) {
  TF_RET_CHECK(ShapeUtil::Compatible(copy->shape(), copy->operand(0)->shape()));

  auto result = GetEvaluatedLiteralFor(copy->operand(0)).CloneToUnique();
  evaluated_[copy] = std::move(result);
  return Status::OK();
}

Status HloEvaluator::HandleCall(HloInstruction* call) {
  auto* computation = call->to_apply();
  auto operands = call->operands();

  std::vector<const Literal*> arg_literals;
  arg_literals.reserve(operands.size());
  for (auto operand : operands) {
    const Literal& arg_literal = GetEvaluatedLiteralFor(operand);
    arg_literals.push_back(&arg_literal);
  }

  HloEvaluator embedded_evaluator;
  std::unique_ptr<Literal> result =
      embedded_evaluator.Evaluate<const Literal*>(*computation, arg_literals)
          .ConsumeValueOrDie();

  evaluated_[call] = std::move(result);
  return Status::OK();
}

Status HloEvaluator::HandleConditional(HloInstruction* conditional) {
  const auto& pred = GetEvaluatedLiteralFor(conditional->operand(0));
  const auto& true_computation_arg =
      GetEvaluatedLiteralFor(conditional->operand(1));
  const auto& false_computation_arg =
      GetEvaluatedLiteralFor(conditional->operand(2));

  auto* true_computation = conditional->true_computation();
  auto* false_computation = conditional->false_computation();

  auto result = Literal::CreateFromShape(conditional->shape());
  HloEvaluator embedded_evaluator;
  if (pred.Get<bool>({})) {
    result = embedded_evaluator
                 .Evaluate<const Literal*>(*true_computation,
                                           {&true_computation_arg})
                 .ConsumeValueOrDie();
  } else {
    result = embedded_evaluator
                 .Evaluate<const Literal*>(*false_computation,
                                           {&false_computation_arg})
                 .ConsumeValueOrDie();
  }

  evaluated_[conditional] = std::move(result);
  return Status::OK();
}

Status HloEvaluator::Preprocess(HloInstruction* hlo) {
  VLOG(2) << "About to visit HLO: " << hlo->ToString();
  return Status::OK();
}

Status HloEvaluator::Postprocess(HloInstruction* hlo) {
  VLOG(2) << "Finished visiting " << hlo->ToString()
          << "; evaluated value is: " << GetEvaluatedLiteralFor(hlo).ToString();
  return Status::OK();
}

// Explicit instantiation of templatized Evaluate* methods.
//
template StatusOr<std::unique_ptr<Literal>> HloEvaluator::Evaluate<
    const Literal*>(const HloModule& module,
                    tensorflow::gtl::ArraySlice<const Literal*> arg_literals);
template StatusOr<std::unique_ptr<Literal>>
HloEvaluator::Evaluate<std::unique_ptr<Literal>>(
    const HloModule& module,
    tensorflow::gtl::ArraySlice<std::unique_ptr<Literal>> arg_literals);

template StatusOr<std::unique_ptr<Literal>> HloEvaluator::Evaluate<
    const Literal*>(const HloComputation& computation,
                    tensorflow::gtl::ArraySlice<const Literal*> arg_literals);
template StatusOr<std::unique_ptr<Literal>>
HloEvaluator::Evaluate<std::unique_ptr<Literal>>(
    const HloComputation& computation,
    tensorflow::gtl::ArraySlice<std::unique_ptr<Literal>> arg_literals);

template StatusOr<std::unique_ptr<Literal>> HloEvaluator::Evaluate<
    const Literal*>(HloInstruction* instruction,
                    tensorflow::gtl::ArraySlice<const Literal*> arg_literals);
template StatusOr<std::unique_ptr<Literal>>
HloEvaluator::Evaluate<std::unique_ptr<Literal>>(
    HloInstruction* instruction,
    tensorflow::gtl::ArraySlice<std::unique_ptr<Literal>> arg_literals);

}  // namespace xla
