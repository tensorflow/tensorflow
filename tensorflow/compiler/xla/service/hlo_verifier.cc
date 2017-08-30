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

#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/flatmap.h"

namespace xla {

namespace {

// Visitor which verifies that the output shape is correctly set. Verifies
// against the inferred shape for the instruction.
// TODO(b/26024837): Check output shape for all instruction types.
class ShapeVerifier : public DfsHloVisitor {
 public:
  explicit ShapeVerifier(
      const std::function<int64(const Shape&)>& shape_size_fn)
      : shape_size_fn_(shape_size_fn) {}

  Status HandleElementwiseUnary(HloInstruction* hlo) override {
    return CheckUnaryShape(hlo);
  }

  Status HandleElementwiseBinary(HloInstruction* hlo) override {
    return CheckBinaryShape(hlo);
  }

  Status HandleClamp(HloInstruction* clamp, HloInstruction* min,
                     HloInstruction* arg, HloInstruction* max) override {
    return CheckTernaryShape(clamp);
  }

  Status HandleSelect(HloInstruction* select, HloInstruction* pred,
                      HloInstruction* on_true,
                      HloInstruction* on_false) override {
    return CheckTernaryShape(select);
  }

  Status HandleConcatenate(
      HloInstruction* concatenate,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands) override {
    return tensorflow::Status::OK();
  }

  Status HandleConvert(HloInstruction* convert) override {
    return tensorflow::Status::OK();
  }

  Status HandleCopy(HloInstruction* copy) override {
    return CheckUnaryShape(copy);
  }

  Status HandleDot(HloInstruction* dot, HloInstruction* lhs,
                   HloInstruction* rhs) override {
    return CheckBinaryShape(dot);
  }

  Status HandleConvolution(HloInstruction* convolution, HloInstruction* lhs,
                           HloInstruction* rhs, const Window& window) override {
    return tensorflow::Status::OK();
  }

  Status HandleCrossReplicaSum(HloInstruction* crs) override {
    return tensorflow::Status::OK();
  }

  Status HandleReducePrecision(HloInstruction* reduce_precision) override {
    return tensorflow::Status::OK();
  }

  Status HandleInfeed(HloInstruction* infeed) override {
    return tensorflow::Status::OK();
  }

  Status HandleOutfeed(HloInstruction* outfeed) override {
    return tensorflow::Status::OK();
  }

  Status HandleRng(HloInstruction* random,
                   RandomDistribution distribution) override {
    return tensorflow::Status::OK();
  }

  Status HandleReverse(HloInstruction* reverse,
                       HloInstruction* operand) override {
    return tensorflow::Status::OK();
  }

  Status HandleSort(HloInstruction* sort, HloInstruction* operand) override {
    return tensorflow::Status::OK();
  }

  Status HandleConstant(HloInstruction* constant,
                        const Literal& literal) override {
    return tensorflow::Status::OK();
  }

  Status HandleGetTupleElement(HloInstruction* get_tuple_element,
                               HloInstruction* operand) override {
    return tensorflow::Status::OK();
  }

  Status HandleReduce(HloInstruction* reduce, HloInstruction* arg,
                      HloInstruction* init_value,
                      tensorflow::gtl::ArraySlice<int64> dimensions,
                      HloComputation* function) override {
    return tensorflow::Status::OK();
  }

  Status HandleBitcast(HloInstruction* bitcast) override {
    // Bitcasts can be any shape, as long as the size matches the operand size.
    TF_RET_CHECK(shape_size_fn_(bitcast->shape()) ==
                 shape_size_fn_(bitcast->operand(0)->shape()));
    return tensorflow::Status::OK();
  }

  Status HandleBroadcast(HloInstruction* broadcast) override {
    return tensorflow::Status::OK();
  }

  Status HandleReshape(HloInstruction* reshape) override {
    return tensorflow::Status::OK();
  }

  Status HandleTranspose(HloInstruction* transpose) override {
    return tensorflow::Status::OK();
  }

  Status HandleParameter(HloInstruction* parameter) override {
    return tensorflow::Status::OK();
  }

  Status HandleFusion(HloInstruction* fusion) override {
    return tensorflow::Status::OK();
  }

  Status HandleCall(HloInstruction* call) override {
    return tensorflow::Status::OK();
  }

  Status HandleCustomCall(HloInstruction* custom_call,
                          tensorflow::gtl::ArraySlice<HloInstruction*> operands,
                          tensorflow::StringPiece custom_call_target) override {
    return tensorflow::Status::OK();
  }

  Status HandleSlice(HloInstruction* slice, HloInstruction* operand) override {
    return tensorflow::Status::OK();
  }

  Status HandleDynamicSlice(HloInstruction* dynamic_slice,
                            HloInstruction* operand,
                            HloInstruction* start_indices) override {
    return tensorflow::Status::OK();
  }

  Status HandleDynamicUpdateSlice(HloInstruction* dynamic_update_slice,
                                  HloInstruction* operand,
                                  HloInstruction* update,
                                  HloInstruction* start_indices) override {
    return tensorflow::Status::OK();
  }

  Status HandleTuple(
      HloInstruction* tuple,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands) override {
    return CheckVariadicShape(tuple);
  }

  Status HandleMap(
      HloInstruction* map,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands,
      HloComputation* function,
      tensorflow::gtl::ArraySlice<HloInstruction*> static_operands) override {
    return tensorflow::Status::OK();
  }

  Status HandleReduceWindow(HloInstruction* reduce_window,
                            HloInstruction* operand, const Window& window,
                            HloComputation* function) override {
    return tensorflow::Status::OK();
  }

  Status HandleSelectAndScatter(HloInstruction* instruction) override {
    return tensorflow::Status::OK();
  }

  Status HandleWhile(HloInstruction* xla_while) override {
    return tensorflow::Status::OK();
  }

  Status HandlePad(HloInstruction* pad) override {
    return tensorflow::Status::OK();
  }

  Status HandleSend(HloInstruction* send) override {
    return tensorflow::Status::OK();
  }

  Status HandleRecv(HloInstruction* recv) override {
    return tensorflow::Status::OK();
  }

  Status HandleBatchNormTraining(HloInstruction* batchNormTraining) override {
    return tensorflow::Status::OK();
  }

  Status HandleBatchNormInference(HloInstruction* batchNormInference) override {
    return tensorflow::Status::OK();
  }

  Status HandleBatchNormGrad(HloInstruction* batchNormGrad) override {
    return tensorflow::Status::OK();
  }

  Status FinishVisit(HloInstruction* root) override {
    return tensorflow::Status::OK();
  }

 private:
  // Check the instruction's shape against the given expected shape and return
  // an appropriate error if there is a mismatch.
  Status CheckShape(const HloInstruction* instruction,
                    const Shape& expected_shape) {
    if (!ShapeUtil::Compatible(instruction->shape(), expected_shape)) {
      return InvalidArgument(
          "Expected instruction to have shape compatible with %s, actual "
          "shape is %s:\n%s",
          ShapeUtil::HumanString(expected_shape).c_str(),
          ShapeUtil::HumanString(instruction->shape()).c_str(),
          instruction->ToString().c_str());
    }
    return tensorflow::Status::OK();
  }

  // Check a unary (binary, etc) instruction's shape against the inferred shape.
  Status CheckUnaryShape(const HloInstruction* instruction) {
    TF_ASSIGN_OR_RETURN(const Shape expected,
                        ShapeInference::InferUnaryOpShape(
                            instruction->opcode(), instruction->operand(0)));
    return CheckShape(instruction, expected);
  }
  Status CheckBinaryShape(const HloInstruction* instruction) {
    TF_ASSIGN_OR_RETURN(const Shape expected,
                        ShapeInference::InferBinaryOpShape(
                            instruction->opcode(), instruction->operand(0),
                            instruction->operand(1)));
    return CheckShape(instruction, expected);
  }
  Status CheckTernaryShape(const HloInstruction* instruction) {
    TF_ASSIGN_OR_RETURN(const Shape expected,
                        ShapeInference::InferTernaryOpShape(
                            instruction->opcode(), instruction->operand(0),
                            instruction->operand(1), instruction->operand(2)));
    return CheckShape(instruction, expected);
  }
  Status CheckVariadicShape(const HloInstruction* instruction) {
    TF_ASSIGN_OR_RETURN(const Shape expected,
                        ShapeInference::InferVariadicOpShape(
                            instruction->opcode(), instruction->operands()));
    return CheckShape(instruction, expected);
  }

  // Returns the size of a Shape in bytes.
  const std::function<int64(const Shape&)> shape_size_fn_;
};

}  // namespace

StatusOr<bool> HloVerifier::Run(HloModule* module) {
  tensorflow::gtl::FlatMap<string, const HloInstruction*> instructions;
  ShapeVerifier shape_verifier(shape_size_fn_);

  for (auto& computation : module->computations()) {
    for (const auto& instruction : computation->instructions()) {
      TF_RET_CHECK(instruction->parent() == computation.get());
      if (instruction->opcode() == HloOpcode::kFusion) {
        for (const auto& fused : instruction->fused_instructions()) {
          TF_RET_CHECK(fused->parent() ==
                       instruction->fused_instructions_computation())
              << "Fused HLO was missing a parent: " << fused->ToString()
              << " parent: " << fused->parent()
              << " computation: " << computation.get();
        }
      }

      auto previous = instructions.find(instruction->name());
      TF_RET_CHECK(previous == instructions.end())
          << "HLO has name that is not unique within module:\n"
          << instruction->ToString()
          << " in computation: " << computation->name()
          << "\nPrevious HLO with same name:\n"
          << previous->second->ToString()
          << " in computation: " << previous->second->parent()->name();
      instructions[instruction->name()] = instruction.get();
    }

    TF_RETURN_IF_ERROR(computation->Accept(&shape_verifier));
  }

  return false;
}

}  // namespace xla
