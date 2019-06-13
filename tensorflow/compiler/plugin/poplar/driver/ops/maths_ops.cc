#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/dot_graph_caching.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/classification_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/util/bcast.h"

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>

#include <poplin/MatMul.hpp>
#include <popnn/NonLinearity.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {

static const std::string a_conn("a");
static const std::string b_conn("b");
static const std::string c_conn("c");
static const std::string out_conn("out");

#define POPLAR_OPCODE(O, N) \
  case HloOpcode::O:        \
    return std::string(N)
#define UNUSED_OPCODE(O) \
  case HloOpcode::O:     \
    break;

StatusOr<popops::expr::UnaryOpType> LookupUnaryFn(const HloInstruction* inst) {
  HloOpcode opcode = inst->opcode();
  switch (opcode) {
    case HloOpcode::kAbs:
      return popops::expr::UnaryOpType::ABSOLUTE;
    case HloOpcode::kCeil:
      return popops::expr::UnaryOpType::CEIL;
    case HloOpcode::kClz:
      return popops::expr::UnaryOpType::COUNT_LEADING_ZEROS;
    case HloOpcode::kCos:
      return popops::expr::UnaryOpType::COS;
    case HloOpcode::kExp:
      return popops::expr::UnaryOpType::EXPONENT;
    case HloOpcode::kExpm1:
      return popops::expr::UnaryOpType::EXPONENT_MINUS_ONE;
    case HloOpcode::kFloor:
      return popops::expr::UnaryOpType::FLOOR;
    case HloOpcode::kLog:
      return popops::expr::UnaryOpType::LOGARITHM;
    case HloOpcode::kLog1p:
      return popops::expr::UnaryOpType::LOGARITHM_ONE_PLUS;
    case HloOpcode::kNegate:
      return popops::expr::UnaryOpType::NEGATE;
    case HloOpcode::kPopulationCount:
      return popops::expr::UnaryOpType::POPCOUNT;
    case HloOpcode::kRoundNearestAfz:
      return popops::expr::UnaryOpType::ROUND;
    case HloOpcode::kRsqrt:
      return popops::expr::UnaryOpType::RSQRT;
    case HloOpcode::kSign:
      return popops::expr::UnaryOpType::SIGNUM;
    case HloOpcode::kSin:
      return popops::expr::UnaryOpType::SIN;
    case HloOpcode::kSqrt:
      return popops::expr::UnaryOpType::SQRT;
    case HloOpcode::kTanh:
      return popops::expr::UnaryOpType::TANH;
    case HloOpcode::kIsFinite:
      return popops::expr::UnaryOpType::IS_FINITE;
    default:
      break;
  }

  if (opcode == HloOpcode::kNot) {
    if (inst->shape().element_type() == PRED) {
      return popops::expr::UnaryOpType::LOGICAL_NOT;
    } else {
      return popops::expr::UnaryOpType::BITWISE_NOT;
    }
  }

  return tensorflow::errors::Unknown(
      StrCat("[Poplar] Invalid opcode lookup ", HloOpcodeString(opcode)));
}

StatusOr<popops::expr::BinaryOpType> LookupComparisonFn(
    const HloInstruction* inst) {
  auto direction = inst->comparison_direction();
  switch (direction) {
    case ComparisonDirection::kEq:
      return popops::expr::BinaryOpType::EQUAL;
    case ComparisonDirection::kGt:
      return popops::expr::BinaryOpType::GREATER_THAN;
    case ComparisonDirection::kGe:
      return popops::expr::BinaryOpType::GREATER_THAN_EQUAL;
    case ComparisonDirection::kLt:
      return popops::expr::BinaryOpType::LESS_THAN;
    case ComparisonDirection::kLe:
      return popops::expr::BinaryOpType::LESS_THAN_EQUAL;
    case ComparisonDirection::kNe:
      return popops::expr::BinaryOpType::NOT_EQUAL;
    default:
      break;
  }

  return tensorflow::errors::Unknown(
      StrCat("[Poplar] Invalid opcode lookup ",
             ComparisonDirectionToString(direction)));
}

StatusOr<popops::expr::BinaryOpType> LookupBinaryFn(
    const HloInstruction* inst) {
  HloOpcode opcode = inst->opcode();
  switch (opcode) {
    case HloOpcode::kAdd:
      return popops::expr::BinaryOpType::ADD;
    case HloOpcode::kAtan2:
      return popops::expr::BinaryOpType::ATAN2;
    case HloOpcode::kDivide:
      return popops::expr::BinaryOpType::DIVIDE;
    case HloOpcode::kMaximum:
      return popops::expr::BinaryOpType::MAXIMUM;
    case HloOpcode::kMinimum:
      return popops::expr::BinaryOpType::MINIMUM;
    case HloOpcode::kMultiply:
      return popops::expr::BinaryOpType::MULTIPLY;
    case HloOpcode::kPower:
      return popops::expr::BinaryOpType::POWER;
    case HloOpcode::kRemainder:
      return popops::expr::BinaryOpType::REMAINDER;
    case HloOpcode::kShiftLeft:
      return popops::expr::BinaryOpType::SHIFT_LEFT;
    case HloOpcode::kShiftRightArithmetic:
      return popops::expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND;
    case HloOpcode::kShiftRightLogical:
      return popops::expr::BinaryOpType::SHIFT_RIGHT;
    case HloOpcode::kSubtract:
      return popops::expr::BinaryOpType::SUBTRACT;
    case HloOpcode::kCompare:
      return LookupComparisonFn(inst);
    default:
      break;
  }

  if (opcode == HloOpcode::kAnd) {
    if (inst->shape().element_type() == PRED) {
      return popops::expr::BinaryOpType::LOGICAL_AND;
    } else {
      return popops::expr::BinaryOpType::BITWISE_AND;
    }
  }

  if (opcode == HloOpcode::kOr) {
    if (inst->shape().element_type() == PRED) {
      return popops::expr::BinaryOpType::LOGICAL_OR;
    } else {
      return popops::expr::BinaryOpType::BITWISE_OR;
    }
  }

  if (opcode == HloOpcode::kXor) {
    if (inst->shape().element_type() == PRED) {
      return popops::expr::BinaryOpType::NOT_EQUAL;
    } else {
      return popops::expr::BinaryOpType::BITWISE_XOR;
    }
  }

  return tensorflow::errors::Unknown(
      StrCat("[Poplar] Invalid opcode lookup ", HloOpcodeString(opcode)));
}

static dot_graph_caching::MatMulPass GetMatMulPass(
    const HloInstruction* inst, const CompilerAnnotations& annotations) {
  if (IsForward(inst, annotations)) {
    return dot_graph_caching::MatMulPass::TRAINING_FWD;
  }
  if (IsBackpropInput(inst, annotations)) {
    return dot_graph_caching::MatMulPass::TRAINING_BWD;
  }
  if (IsBackpropFilter(inst, annotations)) {
    return dot_graph_caching::MatMulPass::TRAINING_WU;
  }
  return dot_graph_caching::MatMulPass::INFERENCE_FWD;
}

namespace {
// Helper struct for generating the popops expression for elementwise operation.
struct ExpressionInput {
  std::unique_ptr<popops::expr::Expr> expr;
  absl::optional<poplar::Tensor> tensor;
  ExpressionInput() = delete;

  ExpressionInput(std::unique_ptr<popops::expr::Expr> expr,
                  poplar::Tensor& tensor)
      : expr(std::move(expr)), tensor(tensor) {}
  ExpressionInput(std::unique_ptr<popops::expr::Expr> expr)
      : expr(std::move(expr)), tensor(absl::nullopt) {}

  ExpressionInput(const ExpressionInput& other) {
    expr = other.expr->clone();
    tensor = other.tensor;
  }
};
using ExpressionInputs = std::vector<ExpressionInput>;

std::vector<poplar::Tensor> GetTensorsFromExpressionInputs(
    ExpressionInputs& expression_inputs) {
  std::vector<poplar::Tensor> tensors;
  for (auto& expression_input : expression_inputs) {
    if (expression_input.tensor) {
      tensors.push_back(*expression_input.tensor);
    }
  }
  return tensors;
}

// Get the elementwise instruction when the instruction can be a fused
// instruction indicating implicit broadcasting op.
const HloInstruction* GetElementwiseOp(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kFusion
             ? inst->fused_instructions_computation()->root_instruction()
             : inst;
}

// Get the input tensor and create a PlaceHolder Expression.
StatusOr<ExpressionInput> GetTensorInput(CompilerResources& res,
                                         const HloInstruction* inst,
                                         TensorMap& tensor_map,
                                         int64 operand_idx, int64 input_idx,
                                         poplar::program::Sequence& seq) {
  // For elementwise, operand 0 might be inplace.
  poplar::Tensor tensor;
  if (operand_idx == 0 && AreInplaceOutputTensorsWritable(tensor_map, inst)) {
    TF_ASSIGN_OR_RETURN(
        ArgVectors inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, seq, false));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    tensor = inputs[0][0];
  } else {
    TF_ASSIGN_OR_RETURN(tensor, FindInstructionInput(tensor_map, res, inst,
                                                     input_idx, seq, false));
  }
  // Poplar PlaceHolders start at 1
  auto expr = absl::make_unique<popops::expr::PlaceHolder>(input_idx + 1);
  return ExpressionInput(std::move(expr), tensor);
}

StatusOr<ExpressionInput> GetConstantInput(const HloInstruction* inst) {
  auto type = inst->shape().element_type();
  switch (type) {
#define GET_CONST_EXPRESSION(XLA_TYPE, NATIVE_TYPE)                         \
  case XLA_TYPE: {                                                          \
    TF_ASSIGN_OR_RETURN(                                                    \
        auto val, LiteralScalarToNativeType<NATIVE_TYPE>(inst->literal())); \
    return ExpressionInput(absl::make_unique<popops::expr::Const>(val));    \
  }
    GET_CONST_EXPRESSION(PRED, bool)
    GET_CONST_EXPRESSION(S8, int8)
    GET_CONST_EXPRESSION(U8, uint8)
    GET_CONST_EXPRESSION(S16, int16)
    GET_CONST_EXPRESSION(U16, uint16)
    GET_CONST_EXPRESSION(S32, int32)
    GET_CONST_EXPRESSION(U32, uint32)
    GET_CONST_EXPRESSION(S64, int64)
    GET_CONST_EXPRESSION(U64, uint64)
    GET_CONST_EXPRESSION(F32, float)
#undef GET_CONST_EXPRESSION
    case F16: {
      // Poplar doesn't support half as a native type, use the ConstHalf
      // expression.
      TF_ASSIGN_OR_RETURN(auto val,
                          LiteralScalarToNativeType<float>(inst->literal()));
      return ExpressionInput(absl::make_unique<popops::expr::ConstHalf>(val));
    }
    default:
      return xla::FailedPrecondition(
          "Unsupported primitive type %s.",
          primitive_util::LowercasePrimitiveTypeName(type).c_str());
  }
}

StatusOr<ExpressionInput> GetElementwiseInput(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
    int64 operand_idx, int64 input_idx, poplar::program::Sequence& seq) {
  if (inst->opcode() == HloOpcode::kFusion) {
    // Fusion indicates implicit broadcasting.
    const auto* root_inst =
        inst->fused_instructions_computation()->root_instruction();
    const auto* input = root_inst->operand(operand_idx);
    if (input->opcode() == HloOpcode::kBroadcast) {
      // We either have a broadcast of a constant or another tensor.
      if (input->operand(0)->opcode() == HloOpcode::kConstant) {
        // Input is a constant, create a constant popops expression.
        return GetConstantInput(input->operand(0));
      } else {
        // Input is not constant.
        CHECK_EQ(input->operand(0)->opcode(), HloOpcode::kParameter);
        TF_ASSIGN_OR_RETURN(
            auto expr_input,
            GetTensorInput(res, inst, tensor_map, operand_idx, input_idx, seq));
        // Broadcast the tensor internally to the right shape.
        TF_ASSIGN_OR_RETURN(expr_input.tensor,
                            BroadcastTensor(*expr_input.tensor, input->shape(),
                                            input->dimensions()));
        return expr_input;
      }
    } else {
      // The input is not broadcasted - just get the tensor.
      CHECK_EQ(input->opcode(), HloOpcode::kParameter);
      return GetTensorInput(res, inst, tensor_map, operand_idx, input_idx, seq);
    }
  } else {
    // Explicit version - just get the tensor.
    return GetTensorInput(res, inst, tensor_map, operand_idx, input_idx, seq);
  }
}

// Get all the elementwise input expression and tensors.
StatusOr<ExpressionInputs> GetElementwiseInputs(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
    poplar::program::Sequence& seq) {
  std::vector<ExpressionInput> expression_inputs;

  auto operation = GetElementwiseOp(inst);

  int64 input_idx = 0;
  // Go over all the inputs to the operation, and figure out what type they are.
  for (int64 operand_idx = 0; operand_idx < operation->operand_count();
       operand_idx++) {
    TF_ASSIGN_OR_RETURN(auto expression_input,
                        GetElementwiseInput(res, inst, tensor_map, operand_idx,
                                            input_idx, seq));
    expression_inputs.push_back(expression_input);
    if (expression_input.tensor) {
      input_idx++;
    }
  }
  return expression_inputs;
}
}  // namespace

StatusOr<poplar::program::Program> CreateUnaryElementwiseOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence seq;
  TF_ASSIGN_OR_RETURN(auto expression_inputs,
                      GetElementwiseInputs(res, inst, tensor_map, seq));
  auto input_tensors = GetTensorsFromExpressionInputs(expression_inputs);

  auto operation = GetElementwiseOp(inst);
  TF_ASSIGN_OR_RETURN(popops::expr::UnaryOpType op, LookupUnaryFn(operation));
  auto expr = popops::expr::UnaryOp(op, *expression_inputs[0].expr);

  poplar::Tensor out;
  const bool is_inplace = AreInplaceOutputTensorsWritable(tensor_map, inst);
  if (is_inplace) {
    popops::mapInPlace(graph, expr, input_tensors, seq, GetDebugName(inst));
    out = input_tensors[0];
  } else {
    out = popops::map(graph, expr, input_tensors, seq, GetDebugName(inst));
  }

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

  return seq;
}

StatusOr<poplar::program::Program> CreateBinaryElementwiseOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence seq;
  TF_ASSIGN_OR_RETURN(auto expression_inputs,
                      GetElementwiseInputs(res, inst, tensor_map, seq));
  auto input_tensors = GetTensorsFromExpressionInputs(expression_inputs);

  auto operation = GetElementwiseOp(inst);
  TF_ASSIGN_OR_RETURN(popops::expr::BinaryOpType op, LookupBinaryFn(operation));
  auto expr = popops::expr::BinaryOp(op, *expression_inputs[0].expr,
                                     *expression_inputs[1].expr);

  poplar::Tensor out;
  const bool is_inplace = AreInplaceOutputTensorsWritable(tensor_map, inst);
  if (is_inplace) {
    switch (operation->opcode()) {
      case HloOpcode::kAdd:
      case HloOpcode::kSubtract: {
        // Specialize for add and subtract when all inputs are tensors.
        const bool all_inputs_tensors = input_tensors.size() == 2;
        if (all_inputs_tensors) {
          ScaledInplaceConstantOrTensor(
              graph, input_tensors[0], input_tensors[1], 1.0f, seq,
              operation->opcode(), GetDebugName(inst));
          break;
        }
        // Fall through if we can't specialize.
      }
      default: {
        popops::mapInPlace(graph, expr, input_tensors, seq, GetDebugName(inst));
        break;
      }
    }
    out = input_tensors[0];
  } else {
    out = popops::map(graph, expr, input_tensors, seq, GetDebugName(inst));
  }

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return seq;
}

StatusOr<poplar::program::Program> CreateTernaryElementwiseOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  // Non of the ternary ops currently support in-placing.
  const bool is_inplace = AreInplaceOutputTensorsWritable(tensor_map, inst);
  CHECK(!is_inplace);

  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence seq;
  TF_ASSIGN_OR_RETURN(auto expression_inputs,
                      GetElementwiseInputs(res, inst, tensor_map, seq));
  auto input_tensors = GetTensorsFromExpressionInputs(expression_inputs);

  // Get the actual ternary operation.
  auto operation = GetElementwiseOp(inst);

  // Create the expression depending on the operation.
  std::unique_ptr<popops::expr::TernaryOp> expr;
  switch (operation->opcode()) {
    case HloOpcode::kClamp: {
      expr = absl::make_unique<popops::expr::TernaryOp>(
          popops::expr::TernaryOpType::CLAMP, *expression_inputs[1].expr,
          *expression_inputs[0].expr, *expression_inputs[2].expr);
      break;
    }
    case HloOpcode::kSelect: {
      expr = absl::make_unique<popops::expr::TernaryOp>(
          popops::expr::TernaryOpType::SELECT, *expression_inputs[1].expr,
          *expression_inputs[2].expr, *expression_inputs[0].expr);
      break;
    }
    default: {
      return xla::FailedPrecondition(
          "Trying to process %s as a ternary operation.",
          operation->ToString().c_str());
    }
  }

  poplar::Tensor out =
      popops::map(graph, *expr, input_tensors, seq, GetDebugName(inst));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return seq;
}

StatusOr<poplar::program::Program> CreateTupleSelectOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence seq;

  TF_ASSIGN_OR_RETURN(
      poplar::Tensor pred,
      FindInstructionInput(tensor_map, res, inst, 0, seq, false));

  ArgVector in0 = FindInstructionInputs(tensor_map, res, inst, 1, seq, false);
  ArgVector in1 = FindInstructionInputs(tensor_map, res, inst, 2, seq, false);

  if (in0.size() != in1.size()) {
    return xla::FailedPrecondition("Mismatching tuple sizes on %s",
                                   inst->name().c_str());
  }

  for (unsigned int i = 0; i < in0.size(); i++) {
    poplar::Tensor p = pred;
    poplar::Tensor i0 = in0[i];
    poplar::Tensor i1 = in1[i];

    if (p.numElements() == 1) {
      p = p.reshape({1});
      p = p.broadcast(i0.numElements(), 0);
      p = p.reshape(i0.shape());
    }

    poplar::Tensor out = popops::map(graph, popops::expr::TernaryOpType::SELECT,
                                     i0, i1, p, seq, GetDebugName(inst));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, out));
  }

  return seq;
}

namespace {
template <typename T>
Status DoScaledInplaceConstantOrTensor(poplar::Graph& graph,
                                       poplar::Tensor& lhs, poplar::Tensor& rhs,
                                       T scale, poplar::program::Sequence& prog,
                                       const HloOpcode op_type,
                                       const std::string& name) {
  // Call the inplace op
  switch (op_type) {
    case HloOpcode::kAdd: {
      popops::scaledAddTo(graph, lhs, rhs, scale, prog, name);
      break;
    }
    case HloOpcode::kSubtract: {
      popops::scaledSubtractFrom(graph, lhs, rhs, scale, prog, name);
      break;
    }
    default: {
      return xla::FailedPrecondition("Unsupported scaled inplace op: %s",
                                     name.c_str());
    }
  }
  return Status::OK();
}
}  // namespace

Status ScaledInplaceConstantOrTensor(poplar::Graph& graph, poplar::Tensor& lhs,
                                     poplar::Tensor& rhs, const double scale,
                                     poplar::program::Sequence& prog,
                                     const HloOpcode op_type,
                                     const std::string& name) {
  return DoScaledInplaceConstantOrTensor(graph, lhs, rhs, scale, prog, op_type,
                                         name);
}

Status ScaledInplaceConstantOrTensor(poplar::Graph& graph, poplar::Tensor& lhs,
                                     poplar::Tensor& rhs, poplar::Tensor& scale,
                                     poplar::program::Sequence& prog,
                                     const HloOpcode op_type,
                                     const std::string& name) {
  return DoScaledInplaceConstantOrTensor(graph, lhs, rhs, scale, prog, op_type,
                                         name);
}

StatusOr<poplar::program::Program> CreateScaledInplace(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence seq;
  TF_ASSIGN_OR_RETURN(
      ArgVectors inputs,
      FindInplaceOutputTensors(tensor_map, res, inst, seq, false));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor in0 = inputs[0][0];

  TF_ASSIGN_OR_RETURN(
      poplar::Tensor in1,
      FindInstructionInput(tensor_map, res, inst, 1, seq, false));

  const auto* root_inst =
      inst->fused_instructions_computation()->root_instruction();

  if (inst->operand_count() == 2) {
    const auto* const_inst = root_inst->operand(1)->operand(1)->operand(0);
    CHECK_EQ(const_inst->opcode(), HloOpcode::kConstant);
    // Get the scalar multiplier
    TF_ASSIGN_OR_RETURN(
        double scale, LiteralScalarToNativeType<double>(const_inst->literal()));

    TF_CHECK_OK(ScaledInplaceConstantOrTensor(
        graph, in0, in1, scale, seq, root_inst->opcode(), GetDebugName(inst)));
  } else if (inst->operand_count() == 3) {
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor scale,
        FindInstructionInput(tensor_map, res, inst, 2, seq, false));
    TF_CHECK_OK(ScaledInplaceConstantOrTensor(
        graph, in0, in1, scale, seq, root_inst->opcode(), GetDebugName(inst)));
  } else {
    return xla::FailedPrecondition("Unsupported use of scaled inplace op: %s",
                                   root_inst->name().c_str());
  }

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, in0));
  return seq;
}

StatusOr<poplar::program::Program> CreateMatMulForDotOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence seq;

  CHECK_EQ(inst->opcode(), HloOpcode::kDot);
  TF_ASSIGN_OR_RETURN(poplar::Tensor in0,
                      FindInstructionInput(tensor_map, res, inst, 0, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor in1,
                      FindInstructionInput(tensor_map, res, inst, 1, seq));

  const DotDimensionNumbers& dot_dims = inst->dot_dimension_numbers();
  auto lhs_reduction_dimensions = dot_dims.lhs_contracting_dimensions();
  auto rhs_reduction_dimensions = dot_dims.rhs_contracting_dimensions();
  auto lhs_batch_dimensions = dot_dims.lhs_batch_dimensions();
  auto rhs_batch_dimensions = dot_dims.rhs_batch_dimensions();

  // DimShuffle the LHS to [Batch..., M..., Contracting...]
  std::vector<unsigned> lhs_permutation;
  lhs_permutation.reserve(in0.rank());

  absl::c_copy(lhs_batch_dimensions, std::back_inserter(lhs_permutation));
  for (int i = 0; i < in0.rank(); ++i) {
    if (absl::c_find(lhs_reduction_dimensions, i) ==
            lhs_reduction_dimensions.end() &&
        absl::c_find(lhs_batch_dimensions, i) == lhs_batch_dimensions.end()) {
      lhs_permutation.push_back(i);
    }
  }
  absl::c_copy(lhs_reduction_dimensions, std::back_inserter(lhs_permutation));
  in0 = in0.dimShuffle(lhs_permutation);

  // DimShuffle the RHS to [Contracting..., Batch..., N...]
  std::vector<unsigned> rhs_permutation;
  rhs_permutation.reserve(in1.rank());

  absl::c_copy(rhs_batch_dimensions, std::back_inserter(rhs_permutation));
  absl::c_copy(rhs_reduction_dimensions, std::back_inserter(rhs_permutation));
  for (int i = 0; i < in1.rank(); ++i) {
    if (absl::c_find(rhs_reduction_dimensions, i) ==
            rhs_reduction_dimensions.end() &&
        absl::c_find(rhs_batch_dimensions, i) == rhs_batch_dimensions.end()) {
      rhs_permutation.push_back(i);
    }
  }
  in1 = in1.dimShuffle(rhs_permutation);

  // Collapse the LHS dimensions down to [Batch, M, Contracting]
  const auto lhs_shape = in0.shape();
  const auto lhs_shape_itr_begin = lhs_shape.begin();
  const auto lhs_shape_itr_a =
      lhs_shape_itr_begin + lhs_batch_dimensions.size();
  const auto lhs_shape_itr_end = lhs_shape.end();
  const auto lhs_shape_itr_b =
      lhs_shape_itr_end - lhs_reduction_dimensions.size();

  const auto lhs_b =
      std::accumulate(lhs_shape_itr_begin, lhs_shape_itr_a, std::size_t(1),
                      std::multiplies<std::size_t>());
  const auto lhs_m =
      std::accumulate(lhs_shape_itr_a, lhs_shape_itr_b, std::size_t(1),
                      std::multiplies<std::size_t>());
  const auto lhs_k =
      std::accumulate(lhs_shape_itr_b, lhs_shape_itr_end, std::size_t(1),
                      std::multiplies<std::size_t>());
  in0 = in0.reshape({lhs_b, lhs_m, lhs_k});

  // Collapse the RHS dimensions down to [Batch, Contracting, N]
  const auto rhs_shape = in1.shape();
  const auto rhs_shape_itr_begin = rhs_shape.begin();
  const auto rhs_shape_itr_a =
      rhs_shape_itr_begin + lhs_batch_dimensions.size();
  const auto rhs_shape_itr_b =
      rhs_shape_itr_a + rhs_reduction_dimensions.size();
  const auto rhs_shape_itr_end = rhs_shape.end();

  const auto rhs_b =
      std::accumulate(rhs_shape_itr_begin, rhs_shape_itr_a, std::size_t(1),
                      std::multiplies<std::size_t>());
  const auto rhs_k =
      std::accumulate(rhs_shape_itr_a, rhs_shape_itr_b, std::size_t(1),
                      std::multiplies<std::size_t>());
  const auto rhs_n =
      std::accumulate(rhs_shape_itr_b, rhs_shape_itr_end, std::size_t(1),
                      std::multiplies<std::size_t>());
  in1 = in1.reshape({rhs_b, rhs_k, rhs_n});

  poplar::Tensor out = dot_graph_caching::DoCachedDot(
      graph, res, in0, in1, seq, GetMatMulPass(inst, res.annotations),
      GetSingleShardingDeviceId(inst), GetDebugName(inst));

  // Reshape to XLA shape
  out = out.reshape(PoplarShapeFromXlaShape(output_shape));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

  return seq;
}

StatusOr<poplar::program::Program> CreateMatMulBiasAddOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  // Get the broadcast instruction which is required to get the bias size.
  const HloInstruction* root =
      inst->fused_instructions_computation()->root_instruction();
  const HloInstruction* broadcast = root->operand(1);
  CHECK_EQ(broadcast->opcode(), HloOpcode::kBroadcast);

  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence prog;

  TF_ASSIGN_OR_RETURN(
      ArgVectors inputs,
      FindInplaceOutputTensors(tensor_map, res, inst, prog, false));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor in = inputs[0][0];

  TF_ASSIGN_OR_RETURN(
      poplar::Tensor bias,
      FindInstructionInput(tensor_map, res, inst, 1, prog, false));

  TF_ASSIGN_OR_RETURN(
      bias, BroadcastTensor(bias, broadcast->shape(), broadcast->dimensions()));
  popops::addInPlace(graph, in, bias, prog, GetDebugName(inst));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, in));
  return prog;
}

StatusOr<poplar::program::Program> CreateCastOp(CompilerResources& res,
                                                const HloInstruction* inst,
                                                const xla::Shape& output_shape,
                                                TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence seq;

  TF_ASSIGN_OR_RETURN(poplar::Tensor in,
                      FindInstructionInput(tensor_map, res, inst, 0, seq));

  TF_ASSIGN_OR_RETURN(poplar::Type poplar_type, PoplarDataType(output_shape));

  poplar::Tensor out =
      popops::cast(graph, in, poplar_type, seq, GetDebugName(inst));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

  return seq;
}

StatusOr<poplar::program::Program> CreateNonLinearityOp(
    CompilerResources& res, const HloInstruction* inst,
    popnn::NonLinearityType non_linearity_type, const xla::Shape& output_shape,
    TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence seq;
  TF_ASSIGN_OR_RETURN(ArgVectors inputs,
                      FindInplaceOutputTensors(tensor_map, res, inst, seq));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor t = inputs[0][0];
  popnn::nonLinearityInPlace(graph, non_linearity_type, t, seq,
                             GetDebugName(inst));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, t));

  return seq;
}

StatusOr<poplar::program::Program> CreateNonLinearityGradOp(
    CompilerResources& res, const HloInstruction* inst,
    popnn::NonLinearityType non_linearity_type, const xla::Shape& output_shape,
    TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence seq;

  TF_ASSIGN_OR_RETURN(poplar::Tensor out,
                      FindInstructionInput(tensor_map, res, inst, 0, seq));

  TF_ASSIGN_OR_RETURN(poplar::Tensor outgrad,
                      FindInstructionInput(tensor_map, res, inst, 1, seq));

  poplar::Tensor t = popnn::nonLinearityInputGradient(
      graph, non_linearity_type, out, outgrad, seq, GetDebugName(inst));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, t));

  return seq;
}

StatusOr<poplar::program::Program> CreateReluOp(CompilerResources& res,
                                                const HloInstruction* inst,
                                                const xla::Shape& output_shape,
                                                TensorMap& tensor_map) {
  return CreateNonLinearityOp(res, inst, popnn::NonLinearityType::RELU,
                              output_shape, tensor_map);
}

StatusOr<poplar::program::Program> CreateReluGradOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  return CreateNonLinearityGradOp(res, inst, popnn::NonLinearityType::RELU,
                                  output_shape, tensor_map);
}

StatusOr<poplar::program::Program> CreateSigmoidOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  return CreateNonLinearityOp(res, inst, popnn::NonLinearityType::SIGMOID,
                              output_shape, tensor_map);
}

StatusOr<poplar::program::Program> CreateSigmoidGradOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  return CreateNonLinearityGradOp(res, inst, popnn::NonLinearityType::SIGMOID,
                                  output_shape, tensor_map);
}

StatusOr<poplar::program::Program> CreateTanhOp(CompilerResources& res,
                                                const HloInstruction* inst,
                                                const xla::Shape& output_shape,
                                                TensorMap& tensor_map) {
  return CreateNonLinearityOp(res, inst, popnn::NonLinearityType::TANH,
                              output_shape, tensor_map);
}

StatusOr<poplar::program::Program> CreateTanhGradOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  return CreateNonLinearityGradOp(res, inst, popnn::NonLinearityType::TANH,
                                  output_shape, tensor_map);
}

}  // namespace poplarplugin
}  // namespace xla
