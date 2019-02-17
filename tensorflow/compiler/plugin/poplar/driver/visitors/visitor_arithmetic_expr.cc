/* Copyright 2017 Graphcore Ltd
 */

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

#include "absl/strings/str_cat.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/visitor_arithmetic_expr.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"

#include "tensorflow/core/lib/core/errors.h"

#include <map>
#include <popops/ElementWise.hpp>

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {

ArithmeticExprVisitor::ArithmeticExprVisitor(CompilerResources& res,
                                             const ArgVectors& inputs)
    : FullVisitor(res), inputs_(std::move(inputs)) {}

StatusOr<std::unique_ptr<popops::expr::Expr>>
ArithmeticExprVisitor::FindExpressionInput(const HloInstruction* inst) {
  // When finding an expression, need to diffrentiate between parameters and
  // actual expressions
  if (inst->opcode() == HloOpcode::kParameter) {
    // Find the input tensor - tuples are not supported
    poplar::Tensor* in0 = &inputs_[inst->parameter_number()][0];
    // Check if an expression exists
    if (tensors_map_.count(in0) == 0) {
      // If the tensor has not been mapped yet
      // add it to tensor inputs
      ts_.push_back(*in0);
      // and create an expression
      tensors_map_[in0] = std::unique_ptr<popops::expr::PlaceHolder>(
          new popops::expr::PlaceHolder(ts_.size()));
    }
    return tensors_map_[in0]->clone();
  } else {
    auto it = expressions_map_.find(inst);
    if (it == expressions_map_.end()) {
      return tensorflow::errors::Unknown(
          StrCat("[Poplar] Couldn't find expression for %s", inst->name()));
    }
    return it->second->clone();
  }
}

Status ArithmeticExprVisitor::HandleElementwiseUnary(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  // find the op
  popops::expr::UnaryOpType op;
  TF_ASSIGN_OR_RETURN(op, LookupUnaryFn(inst));

  // get the input
  std::unique_ptr<popops::expr::Expr> in;
  TF_ASSIGN_OR_RETURN(in, FindExpressionInput(inst->operand(0)));

  // create new expression
  expressions_map_[inst] = std::unique_ptr<popops::expr::UnaryOp>(
      new popops::expr::UnaryOp(op, *in.get()));
  return Status::OK();
}

Status ArithmeticExprVisitor::HandleElementwiseBinary(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  // find the op
  popops::expr::BinaryOpType op;
  TF_ASSIGN_OR_RETURN(op, LookupBinaryFn(inst));

  // get the inputs
  std::unique_ptr<popops::expr::Expr> in0;
  TF_ASSIGN_OR_RETURN(in0, FindExpressionInput(inst->operand(0)));
  std::unique_ptr<popops::expr::Expr> in1;
  TF_ASSIGN_OR_RETURN(in1, FindExpressionInput(inst->operand(1)));

  // create new expression
  expressions_map_[inst] = std::unique_ptr<popops::expr::BinaryOp>(
      new popops::expr::BinaryOp(op, *in0.get(), *in1.get()));
  return Status::OK();
}

Status ArithmeticExprVisitor::HandleSelect(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  // set the op
  const popops::expr::TernaryOpType op = popops::expr::TernaryOpType::SELECT;

  std::unique_ptr<popops::expr::Expr> pred;
  TF_ASSIGN_OR_RETURN(pred, FindExpressionInput(inst->operand(0)));

  std::unique_ptr<popops::expr::Expr> in0;
  TF_ASSIGN_OR_RETURN(in0, FindExpressionInput(inst->operand(1)));
  std::unique_ptr<popops::expr::Expr> in1;
  TF_ASSIGN_OR_RETURN(in1, FindExpressionInput(inst->operand(2)));
  // create new expression
  expressions_map_[inst] = std::unique_ptr<popops::expr::TernaryOp>(
      new popops::expr::TernaryOp(op, *in0.get(), *in1.get(), *pred.get()));
  return Status::OK();
}

Status ArithmeticExprVisitor::HandleClamp(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  // set the op
  const popops::expr::TernaryOpType op = popops::expr::TernaryOpType::CLAMP;

  std::unique_ptr<popops::expr::Expr> min;
  TF_ASSIGN_OR_RETURN(min, FindExpressionInput(inst->operand(0)));
  std::unique_ptr<popops::expr::Expr> arg;
  TF_ASSIGN_OR_RETURN(arg, FindExpressionInput(inst->operand(1)));
  std::unique_ptr<popops::expr::Expr> max;
  TF_ASSIGN_OR_RETURN(max, FindExpressionInput(inst->operand(2)));

  // create new expression
  expressions_map_[inst] = std::unique_ptr<popops::expr::TernaryOp>(
      new popops::expr::TernaryOp(op, *arg.get(), *min.get(), *max.get()));
  return Status::OK();
}

Status ArithmeticExprVisitor::HandleParameter(HloInstruction* inst) {
  // TODO ArithmeticExprVisitor does not support tuples
  if (inputs_[inst->parameter_number()].size() > 1)
    return xla::Unimplemented(
        "Support for tuples in outlined arithmetic expressions is not "
        "implemented");
  return Status::OK();
}

Status ArithmeticExprVisitor::FinishVisit(HloInstruction* inst) {
  poplar::Graph& graph = GetGraph(resources_, inst);

  // get the expression
  std::unique_ptr<popops::expr::Expr> expr;
  TF_ASSIGN_OR_RETURN(expr, FindExpressionInput(inst));
  // map expression with the tensors
  poplar::Tensor out = popops::map(graph, *expr, ts_, sequence,
                                   GetDebugName(inst) + "_expression");
  TF_ASSIGN_OR_RETURN(out, BroadcastTensor(out, GetOutputShape(inst)));
  outputs_.push_back(out);

  resources_.tensor_maps[inst->parent()->name()] = std::move(tensor_map);

  return Status::OK();
}

}  // namespace poplarplugin
}  // namespace xla
