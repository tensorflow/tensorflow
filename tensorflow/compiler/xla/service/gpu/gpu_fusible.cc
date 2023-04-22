/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"

#include <algorithm>
#include <iterator>
#include <stack>
#include <vector>

#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace gpu {
namespace {

// The amount of shared memory a CUDA kernel can use.
//
// Stay on the conservative side, this is smaller than full 64kB, but allows
// some extra space for cache.
int64_t kSharedMemoryBudgetInBytes = 40000;

void AppendParams(const HloInstruction& instr,
                  std::vector<HloInstruction*>* params) {
  if (instr.opcode() == HloOpcode::kFusion) {
    params->insert(std::end(*params), std::begin(instr.fused_parameters()),
                   std::end(instr.fused_parameters()));
  } else {
    for (HloInstruction* operand : instr.operands()) {
      params->push_back(operand);
    }
  }
}

bool IfFusedReadsElementsMultipleTimes(const HloInstruction& instr) {
  CHECK_NE(instr.opcode(), HloOpcode::kFusion) << "`instr` has to be unfused.";
  if (instr.opcode() == HloOpcode::kReduce &&
      !IsReductionFromOrToContiguousDimensions(instr)) {
    return true;
  }
  // Avoid fusing reduce-window when stride is less than window size to minimize
  // the number of reads of the same elements.
  if (instr.opcode() == HloOpcode::kReduceWindow) {
    for (const auto& dim : instr.window().dimensions()) {
      if (dim.size() > dim.stride()) {
        return true;
      }
    }
  }
  return false;
}

std::vector<int64> ExtractRelativeOrderOfNontrivialDims(const Shape& shape) {
  std::vector<int64> relative_order;
  for (int64_t dim : LayoutUtil::MinorToMajor(shape)) {
    if (shape.dimensions(dim) > 1) {
      relative_order.push_back(dim);
    }
  }
  // Now normalize the dimensions to values between 0 and true rank - 1.
  std::vector<int64> sorted_dims = relative_order;
  std::sort(sorted_dims.begin(), sorted_dims.end());
  for (int64& dim : relative_order) {
    int64_t sorted_index = std::distance(
        sorted_dims.begin(),
        std::lower_bound(sorted_dims.begin(), sorted_dims.end(), dim));
    dim = sorted_index;
  }
  return relative_order;
}

}  // namespace

bool LayoutsAreReduceInputFusionFriendly(const HloInstruction& producer,
                                         const HloInstruction& reduce) {
  std::vector<HloInstruction*> params;
  AppendParams(producer, &params);
  AppendParams(reduce, &params);
  int64_t max_true_rank = -1;
  std::vector<int64> max_rank_order;
  for (HloInstruction* param : params) {
    if (param->shape().IsArray() &&
        ShapeUtil::TrueRank(param->shape()) > max_true_rank) {
      max_true_rank = ShapeUtil::TrueRank(param->shape());
      max_rank_order = ExtractRelativeOrderOfNontrivialDims(param->shape());
    }
  }
  return absl::c_all_of(params, [&](HloInstruction* param) {
    return !param->shape().IsArray() ||
           ShapeUtil::TrueRank(param->shape()) < max_true_rank ||
           ExtractRelativeOrderOfNontrivialDims(param->shape()) ==
               max_rank_order;
  });
}

bool IsReduceInputFusion(const HloInstruction& instr) {
  if (instr.IsMultiOutputFusion()) {
    for (const HloInstruction* operand :
         instr.fused_expression_root()->operands()) {
      if (IsReductionFromOrToContiguousDimensions(*operand)) {
        CHECK(instr.IsInputFusion())
            << " Multi-output fusion rooted at reduction-to-vector ops must be "
               "of kind kInput: "
            << instr.ToString();
        return true;
      }
    }
  } else if (instr.opcode() == HloOpcode::kFusion &&
             IsReductionFromOrToContiguousDimensions(
                 *instr.fused_expression_root())) {
    CHECK(instr.IsInputFusion())
        << " Fusion rooted at reduction-to-vector op must be of kind kInput: "
        << instr.ToString();
    return true;
  }
  return false;
}

bool IsInputFusibleReduction(const HloInstruction& instr) {
  // TODO(b/129089333): Don't fuse variadic reduce.
  if (instr.opcode() == HloOpcode::kReduce && instr.shape().IsTuple()) {
    return false;
  }

  return IsReduceInputFusion(instr) ||
         IsReductionFromOrToContiguousDimensions(instr);
}

const HloInstruction* GetRealHeroForMultiOutputFusion(
    const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return &instr;
  }
  auto fused_expression_root = instr.fused_expression_root();
  if (!instr.IsMultiOutputFusion()) {
    return fused_expression_root;
  }
  // If possible, we want to pick a reduction-from-or-to-contiguous-dims
  // operand of the fusion root, because it has the most constraints.
  for (const auto* inst : fused_expression_root->operands()) {
    if (IsReductionFromOrToContiguousDimensions(*inst)) {
      return inst;
    }
  }
  return fused_expression_root->operands()[0];
}

bool ShapesCompatibleForMultiOutputFusion(const HloInstruction& instr1,
                                          const HloInstruction& instr2) {
  // Multi-output fusion kernels share a common parallel loop. The loop
  // dimensions are determined by instruction shapes.
  auto get_loop_shape = [&](const HloInstruction* element_instr) {
    // Special-case reduction-to-vector ops: The loop dimensions are determined
    // by the shape of the first operand.
    if (IsReductionFromOrToContiguousDimensions(*element_instr)) {
      return element_instr->operand(0)->shape();
    }
    return element_instr->shape();
  };

  // All shapes of the root tuple of multi-output fusions should agree, i.e. all
  // root ops should have equal output shapes. An exception are
  // reduction-to-vector ops. Here the input shapes of the reduction (first
  // operand shape) and the reduction dimensions need to match.
  auto* instr_1 = GetRealHeroForMultiOutputFusion(instr1);
  auto* instr_2 = GetRealHeroForMultiOutputFusion(instr2);
  if (IsReductionFromOrToContiguousDimensions(*instr_1) &&
      IsReductionFromOrToContiguousDimensions(*instr_2) &&
      !AreFusedReductionOutputsConsistent({instr_1, instr_2}, instr_1)) {
    return false;
  }
  // The elementwise output shapes must be the same (including layout).
  return ShapeUtil::EqualIgnoringElementType(get_loop_shape(instr_1),
                                             get_loop_shape(instr_2));
}

bool IsInputFusibleScatter(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kScatter ||
      (instr.opcode() == HloOpcode::kFusion &&
       instr.fusion_kind() == HloInstruction::FusionKind::kInput &&
       instr.fused_expression_root()->opcode() == HloOpcode::kScatter)) {
    return true;
  }
  return false;
}

bool IsInputFusible(const HloInstruction& instr) {
  // Input fusion only handles non-elemental reduction and scatter operations.
  return instr.IsFusible() &&
         (IsInputFusibleReduction(instr) || IsInputFusibleScatter(instr));
}

bool IsLoopFusible(const HloInstruction& instr) {
  // Don't fuse get-tuple-element on GPU: We can, but it's slower than not
  // fusing.  We never generate kernels for unfused GTEs.  Instead, if an
  // unfused GTE is an input to a kernel (including a fusion kernel), we
  // compute the address of the GTE at the top of the kernel.  Often we know the
  // address of the GTE result statically, so we can do this without chasing any
  // pointers.
  return instr.IsFusible() &&
         ((instr.IsElementwise() && instr.operand_count() > 0) ||
          instr.opcode() == HloOpcode::kBitcast ||
          instr.opcode() == HloOpcode::kBroadcast ||
          instr.opcode() == HloOpcode::kConcatenate ||
          instr.opcode() == HloOpcode::kDynamicSlice ||
          instr.opcode() == HloOpcode::kDynamicUpdateSlice ||
          (instr.opcode() == HloOpcode::kFusion &&
           instr.fusion_kind() == HloInstruction::FusionKind::kLoop) ||
          instr.opcode() == HloOpcode::kGather ||
          instr.opcode() == HloOpcode::kIota ||
          instr.opcode() == HloOpcode::kPad ||
          (instr.opcode() == HloOpcode::kReduce &&
           !IsReductionFromOrToContiguousDimensions(instr) &&
           !instr.shape().IsTuple()) ||  // TODO(b/129089333): Don't fuse
                                         // variadic reductions.
          instr.opcode() == HloOpcode::kReduceWindow ||
          instr.opcode() == HloOpcode::kReshape ||
          instr.opcode() == HloOpcode::kReverse ||
          instr.opcode() == HloOpcode::kSlice ||
          instr.opcode() == HloOpcode::kConstant ||
          instr.opcode() == HloOpcode::kTranspose);
}

bool IsProducerConsumerFusible(const HloInstruction& producer,
                               const HloInstruction& consumer) {
  if (!IsLoopFusible(producer)) {
    VLOG(5) << "Producer " << producer.name() << " is not loop-fusible";
    return false;
  }

  if (!IsInputFusible(consumer) && !IsLoopFusible(consumer)) {
    VLOG(5) << "Consumer " << consumer.name()
            << "is not input-fusible and not loop-fusible";
    return false;
  }

  // Skip multiple output fusion. It's not yet supported.
  if (producer.IsMultiOutputFusion()) {
    VLOG(5) << "Producer " << producer.name()
            << " is not fusible as it is a multi-output fusion";
    return false;
  }

  if (CreatesNestedLoop(producer, consumer)) {
    VLOG(5) << "Fusing " << producer.name() << " into " << consumer.name()
            << " creates nested loop";
    return false;
  }

  // Do not fuse into reduce input fusions if the resulting kernel would suffer
  // from poor data locality (due to unfriendly input layouts).
  if (IsInputFusibleReduction(consumer) &&
      !LayoutsAreReduceInputFusionFriendly(producer, consumer)) {
    VLOG(5) << "Layout of " << producer.name()
            << " is not fusion-friendly for consumer reduction "
            << consumer.name();
    return false;
  }

  // Fuse scalar constants into loop fusion nodes. This reduces the number of
  // parameters and makes matching scalar broadcasts easier.
  //
  // Don't fuse other constants: Unfused constants in GPU land can be
  // represented as an external constant (i.e. not emitted in LLVM IR / PTX),
  // but fused constants are handled by shrared CPU/GPU code and always emitted
  // in the IR/PTX.  The external constant representation makes for faster
  // compiles and significantly smaller assembly code.
  if (producer.opcode() == HloOpcode::kConstant &&
      (!ShapeUtil::IsEffectiveScalar(producer.shape()) ||
       consumer.opcode() != HloOpcode::kFusion)) {
    VLOG(5) << "Not fusing constant " << producer.name() << " into "
            << consumer.name();
    return false;
  }

  return true;
}

bool IsProducerConsumerMultiOutputFusible(const HloInstruction& producer,
                                          const HloInstruction& consumer) {
  // Skip multiple output fusion. It's not yet supported.
  if (producer.IsMultiOutputFusion()) {
    return false;
  }

  if (!IsLoopFusible(producer) || !IsFusibleAsMultiOutputFusionRoot(consumer)) {
    return false;
  }
  if (CreatesNestedLoop(producer, consumer)) {
    return false;
  }
  if (!ShapesCompatibleForMultiOutputFusion(producer, consumer)) {
    return false;
  }
  if (!LayoutsAreReduceInputFusionFriendly(producer, consumer)) {
    return false;
  }
  return true;
}

// Returns shared memory usage for a given instruction in bytes.
static int64 SharedMemoryUsage(const HloInstruction& instr) {
  // For now we are only fusing reductions.
  if (instr.opcode() == HloOpcode::kReduce &&
      IsReductionFromOrToContiguousDimensions(instr)) {
    ReductionDimensions reduction_info =
        GetReductionKindAndContiguousComponents(instr);
    int64_t primitive_size =
        ShapeUtil::ByteSizeOfPrimitiveType(instr.shape().element_type());
    if (reduction_info.is_row_reduction) {
      // __shared__[32] is used for row reduction.
      return 32 * primitive_size;
    } else {
      // __shared__[2][32][33] cache is used for column reduction ("2" comes
      // from potential x-tiling).
      return 2 * 32 * 33 * primitive_size;
    }
  } else if (instr.opcode() == HloOpcode::kFusion) {
    int64_t sum = 0;
    for (const HloInstruction* hlo :
         instr.fused_instructions_computation()->MakeInstructionPostOrder()) {
      sum += SharedMemoryUsage(*hlo);
    }
    return sum;
  }
  // Other fused expressions for now don't need the shared memory budget.
  return 0;
}

// This function limits the maximum number of operands to a fusion, and the
// amount of shared memory which can be consumed by the fusion.
//
// There's a cap on how many parameters we can pass to a CUDA kernel, but
// exactly what that limit is hazy, as it depends on (among other things) how
// much GPU constant memory is in use for other purposes.
//
// Moreover, we don't even know at the point that we're running fusion how many
// arguments the CUDA kernel for a fusion node will have: It depends on buffer
// assignment, where we will decide which of the fusion's operands live in XLA's
// big temp buffer versus in other allocations.
//
// As a heuristic, we simply cap the number of fusion operands plus outputs at
// kMaxOperandsAndOutputsPerFusion.  This puts an upper bound on the number of
// parameters to the kernel, working around the correctness problem.
//
// This limit is also often good for performance.  In a fusion with many
// operands, each GPU thread likely has to do a lot of work, and so possibly
// uses a lot of registers, thus limiting occupancy.
//
// If the fusion is a producer/consumer fusion and instr1 is the
// consumer and instr2 is the producer, set is_consumer_producer_fusion
// to true to enable more fusion.
bool FusionWouldBeTooLarge(const HloInstruction& instr1,
                           const HloInstruction& instr2,
                           bool is_consumer_producer_fusion) {
  if (SharedMemoryUsage(instr1) + SharedMemoryUsage(instr2) >
      kSharedMemoryBudgetInBytes) {
    VLOG(5) << "Shared memory usage of fusion of " << instr1.ToString()
            << " and " << instr2.ToString() << " would be over the budget of "
            << kSharedMemoryBudgetInBytes << "B";
    return true;
  }

  // Compute the number of outputs of the (possibly multi-output) fusion node
  // we're considering creating.
  //
  // This isn't precise; we may be off by one if
  //  - We're creating a multi-output fusion out of two non-MOFs.  Creating a
  //    MOF adds a new buffer, namely, the tuple buffer.
  //  - We're merging two MOFs.  In this case, we should count the tuple buffer
  //    only once.
  //  - WLOG there's an edge from `a` to `b` and `b` is the only consumer of
  //    `a`.  In this case the result of `a` is not part of the output of the
  //    fusion.
  //
  // But because this is a heuristic and our limit
  // kMaxOperandsAndOutputsPerFusion is a large value (so +/- 1 doesn't make a
  // big difference), we ignore this small inaccuracy in favor of simplicity.
  int64_t num_output_buffers = ShapeUtil::SubshapeCount(instr1.shape()) +
                               ShapeUtil::SubshapeCount(instr2.shape());

  // The new fusion will have no more operands and outputs than
  //   producer_operands + consumer_operands - 1 + num_output_buffers
  // (minus one because we may be fusing a producer->consumer edge between `a`
  // and `b`).
  //
  // This fact may be enough to let us avoid having to compute the true total
  // number of operands, which can be expensive.
  if (instr1.operand_count() + instr2.operand_count() - 1 +
          num_output_buffers <=
      kMaxOperandsAndOutputsPerFusion) {
    return false;
  } else {
    VLOG(5) << "Operand count of "
            << "(" << instr1.ToString() << " ) = " << instr1.operand_count()
            << " and ( " << instr2.ToString()
            << " ) = " << instr2.operand_count()
            << " and num_output_buffers = " << num_output_buffers
            << " is bigger than the bound of "
            << kMaxOperandsAndOutputsPerFusion;
  }

  // Compute the precise number of operands to the new fusion.
  absl::flat_hash_set<const HloInstruction*> operands(instr1.operands().begin(),
                                                      instr1.operands().end());
  operands.insert(instr2.operands().begin(), instr2.operands().end());
  // If there's an edge between `a` and `b`, don't count it: We're fusing that
  // producer -> consumer relationship.
  operands.erase(&instr1);
  operands.erase(&instr2);

  // If we generate the same numbers of inputs and outputs as
  // before, it won't be bigger after fusion. So accept the fusion.
  // As this is a consumer_producer fusion, this does not change the
  // consumer numbers of output. So no need to check it.
  if (is_consumer_producer_fusion &&
      operands.size() <= instr1.operands().size()) {
    return false;
  }

  // Does the new fusion have more operands and outputs than the max?
  return operands.size() + num_output_buffers > kMaxOperandsAndOutputsPerFusion;
}

bool CreatesNestedLoop(const HloInstruction& producer,
                       const HloInstruction& consumer) {
  // If producer does not have an instruction that codegens a loop then there is
  // nothing to do.
  auto producer_has_loop_codegen = [&](const HloInstruction& instr) {
    if (producer.opcode() != HloOpcode::kFusion) {
      return IfFusedReadsElementsMultipleTimes(producer);
    }
    for (const auto& instr : producer.fused_instructions()) {
      if (IfFusedReadsElementsMultipleTimes(*instr)) {
        return true;
      }
    }
    return false;
  };
  if (!producer_has_loop_codegen(producer)) {
    return false;
  }

  // If consumer is a non-fusion instruction then we have to check if it
  // generates a loop.
  if (consumer.opcode() != HloOpcode::kFusion) {
    return IfFusedReadsElementsMultipleTimes(consumer);
  }

  // If consumer is a fusion then we have to check if the output of producer is
  // used directly or indirectly as an input to an HLO instruction that
  // generates a loop, i.e. there is a path in the graph from an operand
  // corresponding to the producer to an HLO instruction generating a loop in
  // the consumer.
  for (const HloInstruction* operand : consumer.operands()) {
    if (operand != &producer) {
      continue;
    }

    const HloInstruction* root =
        consumer.fused_instructions_computation()->parameter_instruction(
            consumer.operand_index(operand));

    std::stack<const HloInstruction*> dfs;
    dfs.push(root);
    absl::flat_hash_set<const HloInstruction*> visited;
    while (!dfs.empty()) {
      const HloInstruction* cur = dfs.top();
      dfs.pop();

      if (visited.contains(cur)) {
        continue;
      }
      visited.insert(cur);

      if (IfFusedReadsElementsMultipleTimes(*cur)) {
        return true;
      }
      for (const auto& user : cur->users()) {
        if (visited.contains(user)) {
          continue;
        }
        dfs.push(user);
      }
    }
  }
  return false;
}

bool IsFusibleAsMultiOutputFusionRoot(const HloInstruction& instr) {
  // We can fuse reduces and loop fusions. Elementwise instructions can be fused
  // with any other instruction.
  // Note that scatter cannot be the root of a multi-output fusion because
  // its emitter doesn't support it.

  return instr.IsFusible() &&
         (IsInputFusibleReduction(instr) ||
          instr.IsLoopFusion() ||  // TODO(b/130013493): Use IsLoopFusible here.
          instr.IsElementwise());
}

HloInstruction::FusionKind ChooseFusionKind(const HloInstruction& /*producer*/,
                                            const HloInstruction& consumer) {
  return IsInputFusible(consumer) ? HloInstruction::FusionKind::kInput
                                  : HloInstruction::FusionKind::kLoop;
}

bool IsConsumerTheOnlyNonRootUser(const HloInstruction& instr,
                                  const HloInstruction& consumer) {
  return absl::c_all_of(instr.users(), [&](const HloInstruction* user) {
    if (user->opcode() == HloOpcode::kGetTupleElement) {
      // Skip GTE.
      return IsConsumerTheOnlyNonRootUser(*user, consumer);
    }
    if (user == &consumer) {
      // `user` is `consumer`.
      return true;
    }
    if (user == user->parent()->root_instruction()) {
      // Consumed by ROOT.
      return true;
    }
    return false;
  });
}

size_t GetInstrCountOfFusible(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return 1;
  } else {
    return instr.fused_instruction_count();
  }
}

absl::InlinedVector<const HloInstruction*, 2> GetOutputsOfFusible(
    const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return {&instr};
  }

  HloInstruction* root = instr.fused_expression_root();
  if (root->opcode() != HloOpcode::kTuple) {
    return {root};
  } else {
    auto v = root->operands();
    return absl::InlinedVector<const HloInstruction*, 2>(v.begin(), v.end());
  }
}

size_t GetOutputSizeOfFusible(const HloInstruction& instr) {
  if (!instr.IsMultiOutputFusion()) {
    return 1;
  }
  const HloInstruction* root = instr.fused_expression_root();
  return ShapeUtil::TupleElementCount(root->shape());
}

}  // namespace gpu
}  // namespace xla
