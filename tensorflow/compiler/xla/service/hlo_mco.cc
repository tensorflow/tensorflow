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

// MCO is an XLA pass used to optimise matrix multiplication chain. It has two
// phases: matrix chain detection and matrix chain optimisation. But in order to
// optimise more matrix chains, some extra processes are applied.

#include "tensorflow/compiler/xla/service/hlo_mco.h"

#include <set>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
namespace xla {

namespace {
namespace m = match;

using DFSStack = absl::InlinedVector<std::pair<int, HloInstruction*>, 16>;
void DebugPrint(std::string functionName, std::string content) {
  LOG(INFO) << "[" << functionName << "]: " << content;
}
void PrintCycle(const HloInstruction* child, DFSStack* dfs_stack) {
  // This set contains HloInstructions from the top of `DFSStack` that might
  // belong to the cycle, i.e. if  DFSStack :=[back,...,child,...,top], then
  // `subgraph` := {child,...,top}.
  absl::flat_hash_set<const HloInstruction*> subgraph;
  while (!dfs_stack->empty() && dfs_stack->back().second != child) {
    subgraph.insert(dfs_stack->back().second);
    dfs_stack->pop_back();
  }
  // Start dfs at `child` and find a cycle with all nodes in `subgraph`.
  absl::flat_hash_set<const HloInstruction*> visited;
  absl::InlinedVector<const HloInstruction*, 16> dfs;
  dfs.push_back(child);
  while (!dfs.empty()) {
    bool found_next_instr = false;
    for (const auto& user : dfs.back()->users()) {
      if (user == child) {
        dfs.push_back(child);
        LOG(INFO) << "\n\nDirected cycle:\n  "
                  << absl::StrJoin(
                         dfs, "\n  ",
                         [](std::string* out, const HloInstruction* instr) {
                           out->append(instr->name());
                         });
        return;
      }
      if (!subgraph.contains(user) || visited.contains(user)) {
        continue;
      }
      visited.insert(user);
      dfs.push_back(user);
      found_next_instr = true;
    }
    if (!found_next_instr) {
      dfs.pop_back();
    }
  }
}

// Push "child" onto the dfs_stack if not already visited.  Returns false if a
// cycle was detected, and true otherwise.
template <typename Visitor>
inline bool PushDFSChild(Visitor* visitor, DFSStack* dfs_stack,
                         HloInstruction* child) {
  CHECK(child != nullptr);
  const int id = child->unique_id();
  CHECK_GE(id, 0) << "instruction may not have a parent computation";
  switch (visitor->GetVisitState(id)) {
    case Visitor::kVisiting:
      return false;

    case Visitor::kVisited:
      // Nothing to do
      return true;

    case Visitor::kNotVisited:
      dfs_stack->push_back(std::make_pair(id, child));
      return true;
  }
}

// perform a pre-order DFS to find an existing chain
template <typename Visitor>
static Status DetectMatrixChainPreorderDFS(
    HloInstruction* root, Visitor* visitor,
    bool ignore_control_predecessors = false) {
  // Calculating the instruction count within a module can be expensive on large
  // models so only do it if the visit state is empty. This will help when the
  // same visitor is reused across many computations of a single module.
  if (visitor->VisitStateCapacity() == 0) {
    visitor->ReserveVisitStates(root->GetModule()->instruction_count());
  }

  // dfs_stack holds pairs of <HloInstruction*->unique_id(), HloInstruction*>.
  //
  // We need to keep track of both the id and the instruction because
  // instructions can get deleted while they are on the stack, so we
  // can't always use the (potentially dead) instruction object to grab
  // its id.
  DFSStack dfs_stack;
  dfs_stack.emplace_back(root->unique_id(), root);

  do {
    DCHECK(!dfs_stack.empty());

    int current_id = dfs_stack.back().first;
    HloInstruction* current_node = dfs_stack.back().second;
    CHECK_GE(current_id, 0)
        << "[DetectMatrixChainPreorderDFS] " << current_id << ": "
        << current_node << ": instruction may not have parent computation";
    typename Visitor::VisitState visit_state =
        visitor->GetVisitState(current_id);
    if (visit_state == Visitor::kVisited) {
      dfs_stack.pop_back();
      continue;
    }

    dfs_stack.pop_back();
    bool is_matmul_node = MatrixChainDetector::CheckRealDot(current_node);
    TF_RETURN_IF_ERROR(visitor->Preprocess(current_node));
    visitor->SetVisitState(current_id, Visitor::kVisited);
    if (!is_matmul_node) {
      // for ohter op, we just target the its child nodes in the current branch
      // as a single operand
      continue;
    }

    const size_t old_dfs_stack_size = dfs_stack.size();
    // current_node is a dot op, must have 2 operands
    CHECK_EQ(current_node->operands().size(), 2);
    for (HloInstruction* child : current_node->operands()) {
      if (!TF_PREDICT_TRUE(PushDFSChild(visitor, &dfs_stack, child))) {
        PrintCycle(child, &dfs_stack);
        return FailedPrecondition(
            "DetectMatrixChainPreorderDFS A cycle is detected while visiting "
            "instruction %s",
            current_node->ToString());
      }
    }

    // This makes the traversal order the same as what you'd expect
    // out of a recursive algorithm.
    std::reverse(dfs_stack.begin() + old_dfs_stack_size, dfs_stack.end());
  } while (!dfs_stack.empty());

  return OkStatus();
}

StatusOr<Literal> CreateOneConstVector(const PrimitiveType primitive_type,
                                       int64_t length) {
  switch (primitive_type) {
    case PrimitiveType::F16: {
      std::vector<half> raw_one_vector(length, half(1));
      return LiteralUtil::CreateR1<half>(raw_one_vector);
    }
    case PrimitiveType::BF16: {
      std::vector<bfloat16> raw_one_vector(length, bfloat16(1));
      return LiteralUtil::CreateR1<bfloat16>(raw_one_vector);
    }
    case PrimitiveType::F32: {
      std::vector<float> raw_one_vector(length, float(1));
      return LiteralUtil::CreateR1<float>(raw_one_vector);
      ;
    }
    case PrimitiveType::F64: {
      std::vector<double> raw_one_vector(length, double(1));
      return LiteralUtil::CreateR1<double>(raw_one_vector);
    }
    case PrimitiveType::S8: {
      std::vector<int8_t> raw_one_vector(length, int8_t(1));
      return LiteralUtil::CreateR1<int8_t>(raw_one_vector);
    }
    case PrimitiveType::S16: {
      std::vector<int16_t> raw_one_vector(length, int16_t(1));
      return LiteralUtil::CreateR1<int16_t>(raw_one_vector);
    }
    case PrimitiveType::S32: {
      std::vector<int32_t> raw_one_vector(length, int32_t(1));
      return LiteralUtil::CreateR1<int32_t>(raw_one_vector);
    }
    case PrimitiveType::S64: {
      std::vector<int64_t> raw_one_vector(length, int64_t(1));
      return LiteralUtil::CreateR1<int64_t>(raw_one_vector);
    }
    case PrimitiveType::U8: {
      std::vector<uint8_t> raw_one_vector(length, uint8_t(1));
      return LiteralUtil::CreateR1<uint8_t>(raw_one_vector);
    }
    case PrimitiveType::U16: {
      std::vector<uint16_t> raw_one_vector(length, uint16_t(1));
      return LiteralUtil::CreateR1<uint16_t>(raw_one_vector);
    }
    case PrimitiveType::U32: {
      std::vector<uint32_t> raw_one_vector(length, uint32_t(1));
      return LiteralUtil::CreateR1<uint32_t>(raw_one_vector);
    }
    case PrimitiveType::U64: {
      std::vector<uint64_t> raw_one_vector(length, uint64_t(1));
      return LiteralUtil::CreateR1<uint64_t>(raw_one_vector);
    }
    default:
      return tensorflow::errors::Internal(absl::StrCat(
          "Unsupported type: ", PrimitiveType_Name(primitive_type)));
  }
}

}  // namespace

Status ChainRecorder::Preprocess(HloInstruction* hlo) {
  if (!MatrixChainDetector::CheckRealDot(hlo)) {
    chain_map[chain_root].emplace_back(hlo);
    DebugPrint("ChainRecorder::Preprocess",
               "Add node: " + hlo->name() + " to root: " + chain_root->name() +
                   " chain_map[" + chain_root->name() +
                   "].size = " + std::to_string(chain_map[chain_root].size()));
  }
  return OkStatus();
}

// For matrix multiplication chains, due to associativity, we do not need to
// care the order in which the matrix multiplication chain is computed, we only
// need to record all the matrices involved in the matrix multiplication chain.
// Take the following graph as an example, where dot.11 represents the root node
// of a subgraph of a matrix multiplication chain, and Parameter 0, Parameter 1,
// Parameter 2 are the three matrices involved in the chain. We can store this
// chain as {dot.11 : Parameter 0, Parameter 1, Parameter 2}.
// The basic idea is we need to perform a post-order DFS algorithm to traverse a
// computational graph from its root node, which is the final result of the
// computational graph. We perform a post-order DFS on the computational graph,
// checking all its operands if the node currently being processed is not a dot.
// If an operand of the node is a dot, it means that this dot is the root node
// of a matrix multiplication chain. Then we use start from the dot to record
// the chain, whenever a node is encountered that is not a dot, it is an operand
// in the matrix multiplication chain and is recorded in the hash table,
// otherwise it continues to iterate through all the operands of the dot.
Status MatrixChainDetector::DetectMatrixChain(HloInstruction* chain_root) {
  std::deque<HloInstruction*> chain_roots{chain_root};
  while (!chain_roots.empty()) {
    HloInstruction* cur_root = chain_roots.front();
    chain_roots.pop_front();
    DebugPrint("MatrixChainDetector::MatrixChainDetector",
               "Find a new chain_root:" + cur_root->name());
    ChainRecorder chain_recorder(cur_root);
    auto status = DetectMatrixChainPreorderDFS(cur_root, &chain_recorder);

    auto chain = chain_recorder.GetChain(cur_root);
    chain_map.insert({cur_root, chain});
  }
  DebugPrint("MatrixChainDetector::MatrixChainDetector",
             "Finished chain_map.size() = " + std::to_string(chain_map.size()));
  return OkStatus();
}

bool MatrixChainDetector::CheckRealDot(HloInstruction* hlo) {
  if (hlo->opcode() != HloOpcode::kDot) {
    return false;
  }
  HloInstruction *lhs, *rhs;
  Match(hlo, m::Dot(m::Op(&lhs), m::Op(&rhs)));
  if (lhs->shape().dimensions_size() == 1 &&
      rhs->shape().dimensions_size() == 1) {
    // skip v-v dot
    return false;
  }
  const DotDimensionNumbers& dnums = hlo->dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.rhs_contracting_dimensions_size() != 1 ||
      dnums.lhs_batch_dimensions_size() != 0 ||
      dnums.rhs_batch_dimensions_size() != 0 ||
      hlo->shape().dimensions_size() > 2 ||
      hlo->shape().dimensions_size() == 0) {
    // not m-m, m-v dot
    // do not need to handle v-v inner/outer product, since both of them won't
    // lead to better solution for inner-product, the result is a scalar for
    // outer-product, although the result is a matrix, but if we split the chain
    // from here and let two vector enter 2 sub-chain, the optimal solution is
    // the same
    return false;
  }

  if (*(dnums.lhs_contracting_dimensions().begin()) != 1 ||
      *(dnums.rhs_contracting_dimensions().begin()) != 0) {
    // since transpose op may be rewritten to dot op with
    // lhs_contracting_dimension = {0} rhs_contracting_dimension = {1}
    // such dot is actuall a tranpose and is not associative with other dots.

    return false;
  }
  return true;
}
Status MatrixChainDetector::Preprocess(HloInstruction* hlo) {
  DebugPrint("MatrixChainDetector::Preprocess", "Start " + hlo->ToString());
  // skip 2D dot op but if it is the root_instruction, then it must be the root
  // of a matrix chain

  if (CheckRealDot(hlo)) {
    if (hlo != hlo->parent()->root_instruction()) {
      return OkStatus();
    } else {
      DebugPrint("MatrixChainDetector::Preprocess",
                 "root_instruction is dot, inst.name:" + hlo->name() +
                     " opcode = " + HloOpcodeString(hlo->opcode()));
      TF_RETURN_IF_ERROR(DetectMatrixChain(hlo));
      return OkStatus();
    }
  }

  for (auto op : hlo->operands()) {
    if (Match(op, m::Dot())) {
      if (!CheckRealDot(op)) {
        // Only consider 2D dot product for now
        continue;
      }
      // current node != kDot, child op = kDot, child op is the root of a matrix
      // chain
      if (chain_map.contains(op)) {
        // find a reused chain which has already been added into chain_map
        continue;
      }
      TF_RETURN_IF_ERROR(DetectMatrixChain(op));
      DebugPrint(
          "MatrixChainDetector::Preprocess",
          "After DetectMatrixChain(" + op->name() +
              ") chain_map.size() = " + std::to_string(chain_map.size()));
    }
  }
  return OkStatus();
}

StatusOr<HloInstruction*> HloMCO::ConstructOptimalChain(
    HloInstruction* orig_root, std::vector<std::vector<int64_t>>& solution,
    std::vector<HloInstruction*>& chain_instructions,
    absl::flat_hash_map<HloInstruction*, HloInstruction*>&
        reduce_one_vector_to_orig_init_val) {
  DebugPrint("HloMCO::ConstructOptimalChain",
             "Start, root = " + orig_root->name());
  HloInstruction* optimal_root = nullptr;
  std::vector<HloInstruction*> subgraph_stack;
  TF_RETURN_IF_ERROR(ConstructOptimalChainHelper(
      orig_root, solution, chain_instructions, 0, chain_instructions.size() - 1,
      subgraph_stack, reduce_one_vector_to_orig_init_val));
  CHECK_EQ(subgraph_stack.size(), 1);
  optimal_root = subgraph_stack.back();
  return optimal_root;
}

Status HloMCO::ConstructOptimalChainHelper(
    HloInstruction* orig_root, std::vector<std::vector<int64_t>>& solution,
    std::vector<HloInstruction*>& chain_instructions, int64_t start_index,
    int64_t end_index, std::vector<HloInstruction*>& subgraph_stack,
    absl::flat_hash_map<HloInstruction*, HloInstruction*>&
        reduce_one_vector_to_orig_init_val) {
  auto create_dot = [&](HloInstruction* l, HloInstruction* r) {
    std::string temp_string = "Start:  operand1 = " + l->name() +
                              " shape = " + l->shape().ToString() +
                              " operand2 = " + r->name() +
                              " shape = " + r->shape().ToString();
    const Shape lhs_shape = l->shape();
    DotDimensionNumbers dimension_numbers;
    dimension_numbers.add_lhs_contracting_dimensions(
        lhs_shape.dimensions_size() == 1 ? 0 : 1);
    dimension_numbers.add_rhs_contracting_dimensions(0);
    auto status_or = ShapeInference::InferDotOpShape(
        l->shape(), r->shape(), dimension_numbers, l->shape().element_type());
    Shape output_shape = std::move(status_or).ValueOrDie();

    temp_string = "InferDotOpShape:  operand1 = " + l->name() +
                  " shape = " + l->shape().ToString() +
                  " operand2 = " + r->name() +
                  " shape = " + r->shape().ToString() +
                  " inferred_output_shape = " + output_shape.ToString();

    // for newly created instruction, we need to save it to the computation
    HloInstruction* new_matmul_inst_ptr = l->parent()->AddInstruction(
        HloInstruction::CreateDot(output_shape, l, r, dimension_numbers,
                                  orig_root->precision_config()));
    subgraph_stack.emplace_back(new_matmul_inst_ptr);
  };

  if (start_index == end_index) {
    // for single operand, it has already been stored in the compoutation
    subgraph_stack.emplace_back(chain_instructions[start_index]);
    return OkStatus();
  }

  if (start_index == end_index - 1) {
    // construction a new matmul op
    if (reduce_one_vector_to_orig_init_val.contains(
            chain_instructions[start_index])) {
      if (chain_instructions[start_index]->shape().dimensions(0) ==
          chain_instructions[end_index]->shape().dimensions(0)) {
        auto init_val =
            reduce_one_vector_to_orig_init_val[chain_instructions[start_index]];
        std::vector<int64_t> reduce_dims;
        reduce_dims.push_back(0);
        LOG(INFO) << "reduce.operand="
                  << chain_instructions[end_index]->ToString()
                  << " reduce.dim=" << 0
                  << " reduce.init_val=" << init_val->ToString();
        TF_ASSIGN_OR_RETURN(
            auto new_reduce,
            MakeReduceHlo(chain_instructions[end_index], init_val, reduce_dims,
                          HloOpcode::kAdd));
        subgraph_stack.emplace_back(new_reduce);
      } else {
        auto init_val =
            reduce_one_vector_to_orig_init_val[chain_instructions[start_index]];
        std::vector<int64_t> reduce_dims;
        reduce_dims.push_back(1);
        LOG(INFO) << "reduce.operand="
                  << chain_instructions[end_index]->ToString()
                  << " reduce.dim=" << 1
                  << " reduce.init_val=" << init_val->ToString();
        TF_ASSIGN_OR_RETURN(
            auto new_reduce,
            MakeReduceHlo(chain_instructions[end_index], init_val, reduce_dims,
                          HloOpcode::kAdd));
        subgraph_stack.emplace_back(new_reduce);
      }
    } else if (reduce_one_vector_to_orig_init_val.contains(
                   chain_instructions[end_index])) {
      if (chain_instructions[end_index]->shape().dimensions(0) ==
          chain_instructions[start_index]->shape().dimensions(0)) {
        auto init_val =
            reduce_one_vector_to_orig_init_val[chain_instructions[end_index]];
        std::vector<int64_t> reduce_dims;
        reduce_dims.push_back(0);
        LOG(INFO) << "reduce.operand="
                  << chain_instructions[start_index]->ToString()
                  << " reduce.dim=" << 0
                  << " reduce.init_val=" << init_val->ToString();
        TF_ASSIGN_OR_RETURN(
            auto new_reduce,
            MakeReduceHlo(chain_instructions[start_index], init_val,
                          reduce_dims, HloOpcode::kAdd));
        subgraph_stack.emplace_back(new_reduce);
      } else {
        auto init_val =
            reduce_one_vector_to_orig_init_val[chain_instructions[end_index]];
        std::vector<int64_t> reduce_dims;
        reduce_dims.push_back(1);
        LOG(INFO) << "reduce.operand="
                  << chain_instructions[start_index]->ToString()
                  << " reduce.dim=" << 1
                  << " reduce.init_val=" << init_val->ToString();
        TF_ASSIGN_OR_RETURN(
            auto new_reduce,
            MakeReduceHlo(chain_instructions[start_index], init_val,
                          reduce_dims, HloOpcode::kAdd));
        subgraph_stack.emplace_back(new_reduce);
      }
    } else {
      create_dot(chain_instructions[start_index],
                 chain_instructions[end_index]);
    }

    return OkStatus();
  }
  DebugPrint("HloMCO::ConstructOptimalChainHelper",
             "First interval = [" + std::to_string(start_index) + "," +
                 std::to_string(solution[start_index][end_index]) + "] " +
                 "Second interval = [" +
                 std::to_string(solution[start_index][end_index] + 1) + "," +
                 std::to_string(end_index) + "]");
  TF_RETURN_IF_ERROR(ConstructOptimalChainHelper(
      orig_root, solution, chain_instructions, start_index,
      solution[start_index][end_index], subgraph_stack,
      reduce_one_vector_to_orig_init_val));
  TF_RETURN_IF_ERROR(ConstructOptimalChainHelper(
      orig_root, solution, chain_instructions,
      solution[start_index][end_index] + 1, end_index, subgraph_stack,
      reduce_one_vector_to_orig_init_val));

  // since this is a stack, the right_operand is on the top of left_operand
  HloInstruction* right_operand = subgraph_stack.back();
  subgraph_stack.pop_back();
  HloInstruction* left_operand = subgraph_stack.back();
  subgraph_stack.pop_back();
  DebugPrint(
      "HloMCO::ConstructOptimalChainHelper",
      "combile left_operand =" + left_operand->name() +
          " subgraph_stack.size= " + std::to_string(subgraph_stack.size()));

  if (reduce_one_vector_to_orig_init_val.contains(left_operand)) {
    if (left_operand->shape().dimensions(0) ==
        right_operand->shape().dimensions(0)) {
      auto init_val = reduce_one_vector_to_orig_init_val[left_operand];
      std::vector<int64_t> reduce_dims;
      reduce_dims.push_back(0);
      LOG(INFO) << "reduce.operand=" << right_operand->ToString()
                << " reduce.dim=" << 0
                << " reduce.init_val=" << init_val->ToString();
      TF_ASSIGN_OR_RETURN(
          auto new_reduce,
          MakeReduceHlo(right_operand, init_val, reduce_dims, HloOpcode::kAdd));
      subgraph_stack.emplace_back(new_reduce);
    } else {
      auto init_val = reduce_one_vector_to_orig_init_val[left_operand];
      std::vector<int64_t> reduce_dims;
      reduce_dims.push_back(1);
      LOG(INFO) << "reduce.operand=" << right_operand->ToString()
                << " reduce.dim=" << 1
                << " reduce.init_val=" << init_val->ToString();
      TF_ASSIGN_OR_RETURN(
          auto new_reduce,
          MakeReduceHlo(right_operand, init_val, reduce_dims, HloOpcode::kAdd));
      subgraph_stack.emplace_back(new_reduce);
    }
  } else if (reduce_one_vector_to_orig_init_val.contains(right_operand)) {
    if (right_operand->shape().dimensions(0) ==
        left_operand->shape().dimensions(0)) {
      auto init_val = reduce_one_vector_to_orig_init_val[right_operand];
      std::vector<int64_t> reduce_dims;
      reduce_dims.push_back(0);
      LOG(INFO) << "reduce.operand=" << left_operand->ToString()
                << " reduce.dim=" << 0
                << " reduce.init_val=" << init_val->ToString();
      TF_ASSIGN_OR_RETURN(
          auto new_reduce,
          MakeReduceHlo(left_operand, init_val, reduce_dims, HloOpcode::kAdd));
      subgraph_stack.emplace_back(new_reduce);
    } else {
      auto init_val = reduce_one_vector_to_orig_init_val[right_operand];
      std::vector<int64_t> reduce_dims;
      reduce_dims.push_back(1);
      LOG(INFO) << "reduce.operand=" << left_operand->ToString()
                << " reduce.dim=" << 1
                << " reduce.init_val=" << init_val->ToString();
      TF_ASSIGN_OR_RETURN(
          auto new_reduce,
          MakeReduceHlo(left_operand, init_val, reduce_dims, HloOpcode::kAdd));
      subgraph_stack.emplace_back(new_reduce);
    }
  } else {
    create_dot(left_operand, right_operand);
  }
  return OkStatus();
}

StatusOr<HloInstruction*> HloMCO::ComputeOptimalChainOrder(
    HloInstruction* root, std::vector<HloInstruction*>& chain,
    absl::flat_hash_map<HloInstruction*, HloInstruction*>&
        reduce_one_vector_to_orig_init_val) {
  DebugPrint("HloMCO::ComputeOptimalChainOrder",
             "chain_root = " + root->name() +
                 " chain_length = " + std::to_string(chain.size()));
  HloInstruction* optimal_root = nullptr;
  int64_t chain_length = chain.size();
  // sizes[i] stores the number of rows of operand[i]
  // sizes[i+1] stores the number of columns of operand[i]
  std::vector<int64_t> sizes(chain_length + 1, 0);
  for (auto i = 0; i < chain_length; ++i) {
    CHECK_LE(chain[i]->shape().rank(), 2);
    if (chain[i]->shape().rank() == 1) {
      // vector operand
      sizes[i] = chain[i]->shape().dimensions(0);
      sizes[i + 1] = 1;
    } else if (chain[i]->shape().rank() == 2) {
      // matrix operand
      sizes[i] = chain[i]->shape().dimensions(0);
      sizes[i + 1] = chain[i]->shape().dimensions(1);
    }
  }
  // solution[i][j] stores optimal break point in
  // subexpression from i to j.
  std::vector<std::vector<int64_t>> solution(
      chain_length, std::vector<int64_t>(chain_length, 0));
  /* costs[i,j] = Minimum number of scalar multiplications
        needed to compute the matrix A[i]A[i+1]...A[j] =
        A[i..j] */
  std::vector<std::vector<int64_t>> costs(
      chain_length,
      std::vector<int64_t>(chain_length, std::numeric_limits<int64_t>::max()));
  // cost is zero when multiplying one matrix.
  for (int64_t i = 0; i < chain_length; i++) costs[i][i] = 0;

  // L is chain length.
  // Dynamic Programming to find the optimal computing order
  for (int64_t L = 2; L <= chain_length; L++) {
    for (int64_t i = 0; i <= chain_length - L; i++) {
      // L = 2:             L = n:
      // i = 0 -> n-2       i = 0 -> 0
      // j = 1 -> n-1       j = n-1 -> n-1
      int64_t j = i + L - 1;
      // [i,j] is the [start,end] index of the current subchain
      for (int64_t k = i; k <= j - 1; k++) {
        // compute
        int64_t cost = costs[i][k] + costs[k + 1][j] +
                       sizes[i] * sizes[k + 1] * sizes[j + 1];
        if (cost < costs[i][j]) {
          costs[i][j] = cost;
          // Each entry solution[i,j]=k shows
          // where to split the product arr
          // i,i+1....j to [i,k] * [k+1,j] for the minimum cost.
          solution[i][j] = k;
        }
      }
    }
  }
  auto status_or = ConstructOptimalChain(root, solution, chain,
                                         reduce_one_vector_to_orig_init_val);
  optimal_root = std::move(status_or).ValueOrDie();
  return optimal_root;
}

// Iterate all chains recorded in the hash table, compute its
// optimal order and then reconstruct that subgraph using the optimal computing
// order. The standard bottom-up dynamic programming MCO algorithm is used to
// calculate the optimal computing order(http://arxiv.org/abs/1804.04021). Then
// a recursive funciton is used to construct a new chain based on the optimal
// order. And the original chain is replaced by the optimised chain.
// The reason why we cannot merge matrix chain detection and matrix chain
// optimisation into a single pass is some intermediate result of a matrix chain
// may be used by other opsrations in the compoutational graph. If the two
// procedure are merged, once we detect a matrix chain, we don't know wheter
// other operations we haven't visited are using some intermediate results of
// the chain.
StatusOr<bool> HloMCO::ChainOptimize(
    HloComputation* computation,
    absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>&
        chain_map,
    absl::flat_hash_map<HloInstruction*, HloInstruction*>&
        reduce_one_vector_to_orig_init_val) {
  bool changed = false;
  for (auto& item : chain_map) {
    DebugPrint("HloMCO::ChainOptimize",
               "optimize chain_root = " + item.first->name());
    auto status_or = ComputeOptimalChainOrder(
        item.first, item.second, reduce_one_vector_to_orig_init_val);
    HloInstruction* new_instruction = std::move(status_or).ValueOrDie();
    DebugPrint(
        "HloMCO::ChainOptimize",
        "Finish optimization, new_chain_root = " + new_instruction->name());

    item.first->ReplaceAllUsesWith(new_instruction);
    if (new_instruction == item.first->parent()->root_instruction()) {
      DebugPrint("HloMCO::ChainOptimize",
                 "Replace computation root success, new_root: " +
                     new_instruction->name());
    }

    changed = true;
  }
  return changed;
}

// Transpose Unfolder
// In practice, there are often cases where the chain contains other operations
// such as *transpose*. Take $(ABC(D(JH)^T)^T)^T$ for example. If we directly
// apply MCO on this chain, this chain will be seen as three separate matrix
// multiplication chains: $(ABCX)^T$, $(DY)^T$ and $(JH)^T$, where
// $X=(D(JH)^T)^T, Y=(JH)^T$. However,the optimisation result is not optimal. By
// the property of transposition we know that expression $(ABC(D(JH)^T)^T)^T$ is
// actually equivalent to $(ABCJHD^T)^T$. That means we need to unfold such
// transposes before optimising matrix chains, which allows optimising longer
// chains and thus getting more speed gains and reducing memory consumption.
Status EinSumReduceSumConverter::TransposeSinker(
    std::stack<HloInstruction*> trans_stack) {
  std::string prefix = "[TransposeSinker] ";
  while (!trans_stack.empty()) {
    HloInstruction* cur_trans = trans_stack.top();
    trans_stack.pop();
    HloInstruction* cur_operand;
    CHECK(Match(cur_trans, m::Transpose(m::Op(&cur_operand))));
    auto dnum_size = cur_trans->shape().dimensions_size();
    if (dnum_size > 2) {
      continue;
    }
    if (IsTrivialTranspose(cur_trans)) {
      // skip trivial transpose and transpose of vector
      TF_RETURN_IF_ERROR(ReplaceInstruction(cur_trans, cur_operand));
      if (Match(cur_operand, m::Transpose())) trans_stack.push(cur_operand);
      continue;
    }

    HloInstruction *lhs, *rhs, *next_trans_operand;
    if (Match(cur_operand, m::Dot(m::Op(&lhs), m::Op(&rhs)))) {
      // must be m-m dot
      StatusOr<HloInstruction*> status_or = MakeTransposeHlo(rhs, {1, 0});
      auto new_lhs = std::move(status_or).ValueOrDie();
      status_or = MakeTransposeHlo(lhs, {1, 0});
      auto new_rhs = std::move(status_or).ValueOrDie();

      DotDimensionNumbers dimension_numbers;
      dimension_numbers.add_lhs_contracting_dimensions(1);
      dimension_numbers.add_rhs_contracting_dimensions(0);
      status_or = MakeDotHlo(new_lhs, new_rhs, dimension_numbers,
                             cur_operand->precision_config(),
                             cur_operand->shape().element_type());
      HloInstruction* new_dot = std::move(status_or).ValueOrDie();
      LOG(INFO) << "cur_transpose=" << cur_trans->ToString()
                << " new_dot=" << new_dot->ToString();
      TF_RETURN_IF_ERROR(ReplaceInstruction(cur_trans, new_dot));
      trans_stack.push(new_rhs);
      trans_stack.push(new_lhs);
    } else if (Match(cur_operand, m::Transpose(m::Op(&next_trans_operand)))) {
      if (cur_trans->dimensions(0) == cur_operand->dimensions(0) &&
          cur_trans->dimensions(1) == cur_operand->dimensions(1)) {
        // 2 transposes counteract
        LOG(INFO) << "cur_transpose=" << cur_trans->ToString()
                  << "cur_operand=" << cur_operand->ToString()
                  << " next_trans_operand=" << next_trans_operand->ToString();
        TF_RETURN_IF_ERROR(ReplaceInstruction(cur_trans, next_trans_operand));
        if (Match(next_trans_operand, m::Transpose())) {
          trans_stack.push(next_trans_operand);
        }
      }
    }
  }
  return OkStatus();
}

//  Einsum Converter
//  Einsum(Einstein summation convention) is an elegant symbolic representation
//  that sums elementwise products of the tensors involved in the operation on
//  user-specified dimensions. For example, a matrix multiplication $AB$ can be
//  expressed in einsum as $einsum('ik,kj \rightarrow ij', A, B)$. While einsum
//  can be very convenient for users, allowing them to write more elegant
//  algorithms, it can cause a lot of problems for matrix chain optimisation.
//  Let $A \in R^{20 \times 30}, B \in R^{20 \times 40}, C \in R^{10 \times
//  30}$. and consider an expression:
//  $$
//  reduce\_sum(einsum('ki,jk \rightarrow ij',einsum('ki,kj \rightarrow
//  ij',A,B),C),axis=1)
//  $$
//  This expression is actually equivalent to $reduce\_sum((A^TB)^TC^T,axis=1)$.
//  For a valid matrix multiplication chain, the contracting dimensions of
//  neighbouring matrices must match, i.e. the number of columns of one matrix
//  must be equal to the number of rows of the next matrix. This means that the
//  valid matrix multiplication chain for this example is $B^TAC^T$. In order to
//  be able to correctly optimise the chain of matrix multiplications expressed
//  in einsums, we need to convert einsum into ordinary matrix multiplication
//  and add the appropriate transpose operations.
//  All of the above pre-processing are done in `EinSumReduceSumConverter`.
Status EinSumReduceSumConverter::HandleDot(HloInstruction* dot) {
  std::string prefix = "[EinSumReduceSumConverter::HandleDot] ";
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));
  if (lhs->shape().dimensions_size() == 1 &&
      rhs->shape().dimensions_size() == 1) {
    // skip v-v outer/inner dot
    return OkStatus();
  }
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.rhs_contracting_dimensions_size() != 1 ||
      dnums.lhs_batch_dimensions_size() != 0 ||
      dnums.rhs_batch_dimensions_size() != 0 ||
      dot->shape().dimensions_size() > 2 ||
      dot->shape().dimensions_size() == 0) {
    // not m-m, m-v dot
    // do not need to handle v-v inner/outer product, since both of them won't
    // lead to better solution for inner-product, the result is a scalar for
    // outer-product, although the result is a matrix, but if we split the chain
    // from here and let two vector enter 2 sub-chain, the optimal solution is
    // the same
    return OkStatus();
  }
  int lhs_contracting_dim = *(dnums.lhs_contracting_dimensions().begin());
  int rhs_contracting_dim = *(dnums.rhs_contracting_dimensions().begin());

  bool lhs_is_vector = lhs->shape().dimensions_size() == 1;
  bool rhs_is_vector = rhs->shape().dimensions_size() == 1;
  bool lhs_is_matrix = lhs->shape().dimensions_size() == 2;
  bool rhs_is_matrix = rhs->shape().dimensions_size() == 2;
  bool is_regular_dot = (lhs_contracting_dim == 1 && rhs_contracting_dim == 0);
  HloInstruction* new_dot = dot;
  if (!is_regular_dot) {
    // need to convert
    if (lhs_is_vector && rhs_is_matrix) {
      if (lhs_contracting_dim == 0 && rhs_contracting_dim == 0) {
        // einsum(v,M) = M^T*v
        StatusOr<HloInstruction*> status_or = MakeTransposeHlo(rhs, {1, 0});
        HloInstruction* new_lhs = std::move(status_or).ValueOrDie();
        HloInstruction* new_rhs = lhs;
        DotDimensionNumbers dimension_numbers;
        dimension_numbers.add_lhs_contracting_dimensions(1);
        dimension_numbers.add_rhs_contracting_dimensions(0);
        status_or =
            MakeDotHlo(new_lhs, new_rhs, dimension_numbers,
                       dot->precision_config(), dot->shape().element_type());
        new_dot = std::move(status_or).ValueOrDie();
      } else if (lhs_contracting_dim == 0 && rhs_contracting_dim == 1) {
        // einsum(v,M) = M*v
        HloInstruction* new_lhs = rhs;
        HloInstruction* new_rhs = lhs;
        DotDimensionNumbers dimension_numbers;
        dimension_numbers.add_lhs_contracting_dimensions(1);
        dimension_numbers.add_rhs_contracting_dimensions(0);
        StatusOr<HloInstruction*> status_or =
            MakeDotHlo(new_lhs, new_rhs, dimension_numbers,
                       dot->precision_config(), dot->shape().element_type());
        new_dot = std::move(status_or).ValueOrDie();
      }

    } else if (lhs_is_matrix && rhs_is_vector) {
      if (lhs_contracting_dim == 0 && rhs_contracting_dim == 0) {
        // einsum(M,v) = M^T*v
        StatusOr<HloInstruction*> status_or = MakeTransposeHlo(lhs, {1, 0});
        HloInstruction* new_lhs = std::move(status_or).ValueOrDie();
        HloInstruction* new_rhs = rhs;
        DotDimensionNumbers dimension_numbers;
        dimension_numbers.add_lhs_contracting_dimensions(1);
        dimension_numbers.add_rhs_contracting_dimensions(0);
        status_or =
            MakeDotHlo(new_lhs, new_rhs, dimension_numbers,
                       dot->precision_config(), dot->shape().element_type());
        new_dot = std::move(status_or).ValueOrDie();
      }

    } else if (lhs_is_matrix && rhs_is_matrix) {
      if (lhs_contracting_dim == 0 && rhs_contracting_dim == 0) {
        // einsum(M,N) = M^T*N
        StatusOr<HloInstruction*> status_or = MakeTransposeHlo(lhs, {1, 0});
        HloInstruction* new_lhs = std::move(status_or).ValueOrDie();
        HloInstruction* new_rhs = rhs;
        DotDimensionNumbers dimension_numbers;
        dimension_numbers.add_lhs_contracting_dimensions(1);
        dimension_numbers.add_rhs_contracting_dimensions(0);
        status_or =
            MakeDotHlo(new_lhs, new_rhs, dimension_numbers,
                       dot->precision_config(), dot->shape().element_type());
        new_dot = std::move(status_or).ValueOrDie();
      } else if (lhs_contracting_dim == 0 && rhs_contracting_dim == 1) {
        // einsum(M,N) = M^T*N^T
        StatusOr<HloInstruction*> status_or = MakeTransposeHlo(lhs, {1, 0});
        HloInstruction* new_lhs = std::move(status_or).ValueOrDie();
        status_or = MakeTransposeHlo(rhs, {1, 0});
        HloInstruction* new_rhs = std::move(status_or).ValueOrDie();
        DotDimensionNumbers dimension_numbers;
        dimension_numbers.add_lhs_contracting_dimensions(1);
        dimension_numbers.add_rhs_contracting_dimensions(0);
        status_or =
            MakeDotHlo(new_lhs, new_rhs, dimension_numbers,
                       dot->precision_config(), dot->shape().element_type());
        new_dot = std::move(status_or).ValueOrDie();
      } else if (lhs_contracting_dim == 1 && rhs_contracting_dim == 1) {
        // einsum(M,N) = M-N^T
        HloInstruction* new_lhs = lhs;
        StatusOr<HloInstruction*> status_or = MakeTransposeHlo(rhs, {1, 0});
        HloInstruction* new_rhs = std::move(status_or).ValueOrDie();
        DotDimensionNumbers dimension_numbers;
        dimension_numbers.add_lhs_contracting_dimensions(1);
        dimension_numbers.add_rhs_contracting_dimensions(0);
        status_or =
            MakeDotHlo(new_lhs, new_rhs, dimension_numbers,
                       dot->precision_config(), dot->shape().element_type());
        new_dot = std::move(status_or).ValueOrDie();
      }
    }
  }
  std::stack<HloInstruction*> trans_stack;
  HloInstruction *new_lhs, *new_rhs;
  CHECK(Match(new_dot, m::Dot(m::Op(&new_lhs), m::Op(&new_rhs))));
  if (Match(new_rhs, m::Transpose())) {
    trans_stack.push(new_rhs);
  }
  if (Match(new_lhs, m::Transpose())) {
    trans_stack.push(new_lhs);
  }
  if (!trans_stack.empty()) TF_RETURN_IF_ERROR(TransposeSinker(trans_stack));

  if (new_dot != dot) {
    LOG(INFO) << prefix << " old_dot=" << dot->ToString()
              << " new_dot=" << new_dot->ToString();
    return ReplaceInstruction(dot, new_dot);
  } else {
    return OkStatus();
  }
}
bool EinSumReduceSumConverter::IsReduceSumDot(const HloInstruction* reduce) {
  if (reduce->opcode() != HloOpcode::kReduce) {
    return false;
  }
  int64_t op_count = reduce->operand_count() / 2;
  if (op_count > 1) return false;
  HloInstruction* reduce_function = reduce->to_apply()->root_instruction();
  if (reduce->shape().IsTuple() ||
      reduce_function->opcode() != HloOpcode::kAdd ||
      !reduce->shape().IsArray() || reduce->dimensions().size() != 1 ||
      reduce->shape().dimensions_size() != 1) {
    // only consider reduce_sum(matrix) -> vector
    return false;
  }
  auto zero = LiteralUtil::CreateR0<float>(0);
  if (reduce->operand(1)->opcode() != HloOpcode::kConstant ||
      (reduce->operand(1)->literal() != zero)) {
    // If Reduce Sum has non-zero init_value, such
    // reduce sum cannot be target as matrix multiplication
    return false;
  }

  DebugPrint("EinSumReduceSumConverter::IsReduceSumDot",
             reduce->name() + " is reduce_sum");
  return true;
}

// Reducesum Converter
// reduce_sum is also an extremely common operator. The function of this
// operator is to sum all the elements of the specified dimension on the
// operand. If the operand of reduce_sum is a two-dimensional matrix, this
// operation can actually be seen as a multiplication of that matrix with a
// vector whose elements are all 1s. Suppose there is a matrix $M \in R^{m
// \times n}$. We can perform the reduce_sum operation on both dimensions of the
// matrix. The operations in each of these two dimensions are equal to the
// following matrix-vector multiplications respectively:
// $$
//  reduce\_sum(M,0) = M^Tv
//  \\reduce\_sum(M,1) = Mv
// $$
// So if we can convert these *reducesum*s, which are equivalent to
// matrix-vector multiplications, into their corresponding matrix-vector
// multiplications, our optimisation algorithm can optimise more and longer
// matrix multiplication chains. Of course, in order not to affect the
// processing of reducesum by other eXLA optimisers, we also need to convert the
// matrix vector multiplication from reducesum back to the corresponding
// reducesum operation after performing matrix chain optimisation. For example,
// if we want to calculate $reduce\_sum(ABC,axis=1)$,where  $A \in R^{40 \times
// 20}, B \in R^{20 \times 30}, C \in R^{30 \times 10}$. By transforming
// reducesum into matrix vector multiplication, completing matrix chain
// optimisation and recovering reducesum, we can obtain the following equation:
// $reduce\_sum(ABC,axis=1) = ABCv = A(B(Cv))=A(B\cdot reduce\_sum(C,axis=1))$.

Status EinSumReduceSumConverter::HandleReduce(HloInstruction* reduce) {
  if (!IsReduceSumDot(reduce)) {
    return OkStatus();
  }
  DebugPrint(
      "EinSumReduceSumConverter::HandleReduce",
      "Start " + reduce->name() + " dimension_size = " +
          std::to_string(reduce->shape().dimensions_size()) +
          " reduce_func = " +
          HloOpcodeString(reduce->to_apply()->root_instruction()->opcode()) +
          " operand.name = " + reduce->operand(0)->name() +
          " operand.dimension_size = " +
          std::to_string(reduce->operand(0)->shape().dimensions_size()) +
          " init_value.name = " + reduce->operand(1)->name());

  // replace reduce with dot
  HloInstruction* operand = reduce->mutable_operand(0);
  auto reduce_dim = reduce->dimensions(0);
  PrecisionConfig precision_config;
  precision_config.mutable_operand_precision()->Resize(
      2, PrecisionConfig::DEFAULT);
  HloInstruction* new_dot = nullptr;
  if (reduce_dim == 0) {
    // reduce(M) = M^T*v
    DebugPrint("EinSumReduceSumConverter::HandleReduce",
               "Is M^T-v dot, reduce_dim = " + std::to_string(reduce_dim));
    auto raw_one_vector_status_or = CreateOneConstVector(
        reduce->shape().element_type(), operand->shape().dimensions(0));
    auto raw_one_vector = std::move(raw_one_vector_status_or).ValueOrDie();
    HloInstruction* one_vector_inst = reduce->parent()->AddInstruction(
        HloInstruction::CreateConstant(std::move(raw_one_vector)));
    DebugPrint(
        "EinSumReduceSumConverter::HandleReduce",
        "Is M^T-v dot, one_vector_inst = " + one_vector_inst->ToString());
    StatusOr<HloInstruction*> status_or = MakeTransposeHlo(operand, {1, 0});
    HloInstruction* lhs = std::move(status_or).ValueOrDie();
    HloInstruction* rhs = one_vector_inst;
    reduce_one_vector_to_orig_init_val[one_vector_inst] =
        reduce->parent()->AddInstruction(reduce->mutable_operand(1)->Clone());

    DotDimensionNumbers dimension_numbers;
    dimension_numbers.add_lhs_contracting_dimensions(1);
    dimension_numbers.add_rhs_contracting_dimensions(0);
    status_or = MakeDotHlo(lhs, rhs, dimension_numbers, precision_config,
                           reduce->shape().element_type());
    new_dot = std::move(status_or).ValueOrDie();

  } else if (reduce_dim == 1) {
    // reduce(M) = M*v
    DebugPrint("EinSumReduceSumConverter::HandleReduce",
               "Is M-v dot, reduce_dim = " + std::to_string(reduce_dim));
    auto raw_one_vector_status_or = CreateOneConstVector(
        reduce->shape().element_type(), operand->shape().dimensions(1));
    auto raw_one_vector = std::move(raw_one_vector_status_or).ValueOrDie();
    HloInstruction* one_vector_inst = reduce->parent()->AddInstruction(
        HloInstruction::CreateConstant(std::move(raw_one_vector)));
    DebugPrint("EinSumReduceSumConverter::HandleReduce",
               "Is M-v dot, one_vector_inst = " + one_vector_inst->ToString());
    HloInstruction* lhs = operand;
    HloInstruction* rhs = one_vector_inst;
    reduce_one_vector_to_orig_init_val[one_vector_inst] =
        reduce->parent()->AddInstruction(reduce->mutable_operand(1)->Clone());
    DotDimensionNumbers dimension_numbers;
    dimension_numbers.add_lhs_contracting_dimensions(1);
    dimension_numbers.add_rhs_contracting_dimensions(0);
    StatusOr<HloInstruction*> status_or =
        MakeDotHlo(lhs, rhs, dimension_numbers, precision_config,
                   reduce->shape().element_type());
    new_dot = std::move(status_or).ValueOrDie();
  }
  std::stack<HloInstruction*> trans_stack;
  HloInstruction *new_lhs, *new_rhs;
  CHECK(Match(new_dot, m::Dot(m::Op(&new_lhs), m::Op(&new_rhs))));
  if (Match(new_rhs, m::Transpose())) {
    trans_stack.push(new_rhs);
  }
  if (Match(new_lhs, m::Transpose())) {
    trans_stack.push(new_lhs);
  }
  if (!trans_stack.empty()) TF_RETURN_IF_ERROR(TransposeSinker(trans_stack));

  if (new_dot != nullptr) {
    LOG(INFO) << "EinSumReduceSumConverter::HandleReduce"
              << " Replace reduce.name=" << reduce->name()
              << " reduce.shape=" << reduce->shape().ToString()
              << " with new_dot.name=" << new_dot->name()
              << " new_dot.shape=" << new_dot->shape().ToString();
    return ReplaceInstruction(reduce, new_dot);
  }
  return OkStatus();
}

StatusOr<bool> HloMCO::Run(HloModule* module) {
  bool changed = false;
  TF_RET_CHECK(!module->name().empty());

  if (module->entry_computation()->IsFusionComputation()) {
    return InvalidArgument(
        "Module entry computation cannot be a fusion computation");
  }
  DebugPrint("HloMCO::Run", "Start Run");
  for (auto* computation : module->MakeNonfusionComputations()) {
    DebugPrint("HloMCO::Run", "computation: " + computation->ToString());
    DebugPrint("HloMCO::Run", "start EinSumReduceSumConverter");
    EinSumReduceSumConverter converter;
    TF_RETURN_IF_ERROR(computation->Accept(&converter));

    DebugPrint("HloMCO::Run", "start matriox_chain_detector");
    MatrixChainDetector matrix_chain_detector;
    // detection matrix chain on the whithin the computation;
    TF_RETURN_IF_ERROR(computation->Accept(&matrix_chain_detector));
    DebugPrint("HloMCO::Run", "finish matriox_chain_detector");

    auto chain_map = matrix_chain_detector.GetChainMap();
    auto reduce_one_vector_to_orig_init_val = converter.GetReduceOneVetorSet();
    DebugPrint("HloMCO::Run",
               "chain_map.size = " + std::to_string(chain_map.size()));
    TF_ASSIGN_OR_RETURN(bool changed_for_computation,
                        ChainOptimize(computation, chain_map,
                                      reduce_one_vector_to_orig_init_val));
    changed |= changed_for_computation;
    computation->Cleanup();
    DebugPrint("HloMCO::Run",
               "After optimization computation: " + computation->ToString());
  }
  return changed;
}

}  // namespace xla
