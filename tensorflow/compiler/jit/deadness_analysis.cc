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

#include "tensorflow/compiler/jit/deadness_analysis.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/jit/deadness_analysis_internal.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/hash/hash.h"

// ALGORITHM OVERVIEW
// ==================
//
// We map every output produced by each node in the TensorFlow graph (including
// control dependence) into an instance of the Predicate class.  Instances of
// Predicate denote logical formulas and mapping a node `n` to a predicate
// `pred` implies that `n` is live whenever `pred` is true.  Then we can deduce
// mismatching liveness in the inputs to node by comparing the predicate those
// inputs are mapped to.  The core logic of this pass resides in creating the
// map from TensorFlow nodes to predicates.
//
//
// MAPPING NODES TO PREDICATES, MODULO CYCLES
// ------------------------------------------
//
// If we ignore cycles for a moment, computing predicates is fairly
// straightforward.  We traverse the graph in RPO, mapping each node to a
// predicate based on the predicates its inputs are mapped to.  For instance a
// Merge(X, Y) node will be mapped to OR(PredicateFor(X), PredicateFor(Y)).
// Roughtly speaking, we abstract interpret each node on the "liveness" domain,
// where values in the domain represent if a tensor carries a dead signal or
// not.
//
//
// DEALING WITH CYCLES
// -------------------
//
// We map Merge nodes that are the target of a backedge to AndRecurrence
// instances.  An AndRecurrence with start() = S and step() = X, printed as
// {S,&,X}, *roughly* represents the infinite list of predicates
// [S,S&X,S&X&X,S&X&X, ...].  So {S,&,X} can be used to represent the predicate
// for Merge in a graph like:
//
//     Init
//       |
//       v
//     Merge <-----------+
//       |               |
//       v               |
//      Incr             |
//       |               |
//       v               |
//      Switch <- Cond   |
//       |               |
//       v (oidx: 1)     |
//       |               |
//       +---------------+
//
// Where S is the predicate for Init and X is the predicate that asserts that
// Cond is true.  {S,&,X} states that Merge is live on the first "iteration" iff
// S is true, live on the second iteration iff "S&X" is true, live on the third
// iteration iff "S&X&X" is true etc.  There is a subtlety here, S&X&X would
// normally be equivalent to S&X which isn't quite what we want to represent.
// Instead we want {S,&,X} to denote the infinite list [S, S&X,
// S&X&X',S&X&X'&X'', ...] where X, X', X'' are predicates that assert Cond is
// true on iteration 0, 1, 2 respectively.  This is made more precise in the
// comment on the AndRecurrence class.
//
// The general algorithm that deals with cycles does two RPO (reverse post
// order) passes over the graph.  On the first pass it assigns a symbolic
// predicate to merge nodes with backedges.  On the second pass it tries to
// pattern matche the predicates for the backedges of these merges and infer an
// AndRecurrence for the merge.
//
// In other words, we do a pessimistic data flow analysis where the data-flow
// lattice has two elements, Symbolic and NonSymbolic with Symbolic >
// NonSymbolic. The lattice has height = 2 so two iterations are sufficient to
// converge.  We don't do an optimistic data flow analysis to make pattern
// matching easier: if we assigned the predicate of the initial value to the
// merge during the first pass, on the second pass the backedge may see a
// simplified value that would be difficult to pattern match.
//
// We still use symbolic predicates for merges for which we can't pattern match
// on the backedge predicate.  This is conservatively correct.

namespace tensorflow {

namespace {

// Represents a logical predicate, used as described in the algorithm overview
// above.
class Predicate {
 public:
  enum class Kind { kAnd, kOr, kNot, kAndRecurrence, kSymbol };

  virtual string ToString() const = 0;
  int64 hash() const { return hash_; }
  virtual absl::Span<Predicate* const> GetOperands() const = 0;

  virtual Kind kind() const = 0;
  virtual ~Predicate() {}

  // Invokes func on p and on all of its operands recursively.  Does not invoke
  // `func` on the same Predicate instance twice.  Aborts the search if `func`
  // returns true.
  template <typename FunctionTy>
  static void Visit(Predicate* p, const FunctionTy& func);

 protected:
  explicit Predicate(int64 hash) : hash_(hash) {}

 private:
  const int64 hash_;

  TF_DISALLOW_COPY_AND_ASSIGN(Predicate);
};

int64 HashPredicateSequence(Predicate::Kind kind,
                            absl::Span<Predicate* const> preds) {
  int64 hash = ::tensorflow::hash<Predicate::Kind>()(kind);
  for (Predicate* pred : preds) {
    hash = Hash64Combine(hash, pred->hash());
  }
  return hash;
}

// Represents a logical conjunction of a set of predicates.
class AndPredicate : public Predicate {
 public:
  explicit AndPredicate(std::vector<Predicate*> operands)
      : Predicate(HashPredicateSequence(Kind::kAnd, operands)),
        operands_(std::move(operands)) {}

  string ToString() const override {
    if (operands().empty()) {
      return "#true";
    }

    std::vector<string> operands_str;
    std::transform(operands().begin(), operands().end(),
                   std::back_inserter(operands_str),
                   [](Predicate* pred) { return pred->ToString(); });

    return absl::StrCat("(", absl::StrJoin(operands_str, " & "), ")");
  }

  Kind kind() const override { return Kind::kAnd; }

  absl::Span<Predicate* const> GetOperands() const override {
    return operands_;
  }
  absl::Span<Predicate* const> operands() const { return operands_; }

 private:
  std::vector<Predicate*> operands_;
};

// Represents a logical disjunction of a set of predicates.
class OrPredicate : public Predicate {
 public:
  explicit OrPredicate(std::vector<Predicate*> operands)
      : Predicate(HashPredicateSequence(Kind::kOr, operands)),
        operands_(std::move(operands)) {}

  string ToString() const override {
    if (operands().empty()) {
      return "#false";
    }

    std::vector<string> operands_str;
    std::transform(operands().begin(), operands().end(),
                   std::back_inserter(operands_str),
                   [](Predicate* pred) { return pred->ToString(); });

    return absl::StrCat("(", absl::StrJoin(operands_str, " | "), ")");
  }

  Kind kind() const override { return Kind::kOr; }
  absl::Span<Predicate* const> GetOperands() const override {
    return operands_;
  }
  absl::Span<Predicate* const> operands() const { return operands_; }

 private:
  std::vector<Predicate*> operands_;
};

// Represents a logical negation of a set of predicates.
class NotPredicate : public Predicate {
 public:
  explicit NotPredicate(Predicate* operand)
      : Predicate(HashPredicateSequence(Kind::kNot, {operand})),
        operands_({operand}) {}

  string ToString() const override {
    return absl::StrCat("~", operand()->ToString());
  }

  Kind kind() const override { return Kind::kNot; }
  Predicate* operand() const { return operands_[0]; }
  absl::Span<Predicate* const> GetOperands() const override {
    return operands_;
  }

 private:
  std::array<Predicate*, 1> operands_;
};

// Represents an infinite list of predicates.
//
// An AndRecurrence with start = S and step = X is printed as {S,&,X} and stands
// for the list of predicates:
//
//   S, S & GenSym(X,1), S & GenSym(X,1) & GenSym(X,2), ...
//
// where GenSym(<expression>, <id>) renames every SymbolPredicate in
// <expression> by appending <id> to it, in effect creating a "fresh" symbol.
// This means {P,&,Q} is not equal to "P on the first iteration; P&Q on
// subsequent iterations".
class AndRecurrencePredicate : public Predicate {
 public:
  explicit AndRecurrencePredicate(Predicate* start, Predicate* step)
      : Predicate(HashPredicateSequence(Kind::kAndRecurrence, {start, step})),
        operands_({start, step}) {}

  Predicate* start() const { return operands_[0]; }
  Predicate* step() const { return operands_[1]; }

  string ToString() const override {
    return absl::StrCat("{", start()->ToString(), ",&,", step()->ToString(),
                        "}");
  }

  Kind kind() const override { return Kind::kAndRecurrence; }

  absl::Span<Predicate* const> GetOperands() const override {
    return operands_;
  }

 private:
  std::array<Predicate*, 2> operands_;
};

// Represents an uninterpreted symbol in a logical predicate.
//
// Two predicates are equivalent iff they are equivalent for all assignments to
// the symbols contained in them, i.e. predicates are forall qualified over
// symbols.
class SymbolPredicate : public Predicate {
 public:
  explicit SymbolPredicate(TensorId tensor_id, bool must_be_true)
      : Predicate(Hash(tensor_id, must_be_true)),
        tensor_id_(std::move(tensor_id)),
        must_be_true_(must_be_true) {}

  string ToString() const override {
    return must_be_true() ? absl::StrCat("*", tensor_id_.ToString())
                          : tensor_id_.ToString();
  }

  Kind kind() const override { return Kind::kSymbol; }
  absl::Span<Predicate* const> GetOperands() const override { return {}; }

  // If `must_be_true()` is true this SymbolPredicate represents the proposition
  // "tensor_id() is live and evaluates to true".
  //
  // If `must_be_true()` is false then this SymbolPredicate represents the
  // proposition "tensor_id() is live (and may evalutate to any value)"
  TensorId tensor_id() const { return tensor_id_; }
  bool must_be_true() const { return must_be_true_; }

 private:
  TensorId tensor_id_;
  bool must_be_true_;

  static int64 Hash(const TensorId tensor_id, bool must_be_true) {
    return Hash64Combine(
        ::tensorflow::hash<bool>()(must_be_true),
        Hash64Combine(::tensorflow::hash<Predicate::Kind>()(Kind::kSymbol),
                      TensorId::Hasher{}(tensor_id)));
  }
};

template <typename FunctionTy>
/*static*/ void Predicate::Visit(Predicate* p, const FunctionTy& func) {
  absl::flat_hash_set<Predicate*> visited;
  std::vector<Predicate*> stack;

  stack.push_back(p);
  visited.insert(p);

  while (!stack.empty()) {
    Predicate* current = stack.back();
    stack.pop_back();
    bool done = func(current);
    if (done) {
      return;
    }
    for (Predicate* op : current->GetOperands()) {
      if (visited.insert(op).second) {
        stack.push_back(op);
      }
    }
  }
}

// Creates and owns Predicate instances.  Simplifies predicates as it creates
// them.
class PredicateFactory {
 public:
  Predicate* MakeAndPredicate(absl::Span<Predicate* const> operands) {
    return MakeAndOrImpl(operands, /*is_and=*/true);
  }

  Predicate* MakeOrPredicate(absl::Span<Predicate* const> operands) {
    return MakeAndOrImpl(operands, /*is_and=*/false);
  }

  Predicate* MakeNotPredicate(Predicate* pred) {
    SignatureForNot signature = pred;
    auto it = interned_not_instances_.find(signature);
    if (it == interned_not_instances_.end()) {
      std::unique_ptr<Predicate> new_pred = Make<NotPredicate>(pred);
      Predicate* new_pred_ptr = new_pred.get();
      interned_not_instances_.emplace(signature, std::move(new_pred));
      return new_pred_ptr;
    } else {
      return it->second.get();
    }
  }

  Predicate* MakeAndRecurrencePredicate(Predicate* start, Predicate* step) {
    auto it = interned_and_rec_instances_.find({start, step});
    if (it != interned_and_rec_instances_.end()) {
      return it->second.get();
    }

    std::unique_ptr<Predicate> new_pred =
        Make<AndRecurrencePredicate>(start, step);
    Predicate* new_pred_ptr = new_pred.get();
    CHECK(interned_and_rec_instances_
              .emplace(SignatureForAndRec(start, step), std::move(new_pred))
              .second);
    return new_pred_ptr;
  }

  Predicate* MakeSymbolPredicate(TensorId tensor_id, bool must_be_true) {
    SignatureForSymbol signature = {tensor_id, must_be_true};
    auto it = interned_symbol_instances_.find(signature);
    if (it == interned_symbol_instances_.end()) {
      std::unique_ptr<Predicate> new_pred =
          Make<SymbolPredicate>(tensor_id, must_be_true);
      Predicate* new_pred_ptr = new_pred.get();
      interned_symbol_instances_.emplace(std::move(signature),
                                         std::move(new_pred));
      return new_pred_ptr;
    } else {
      return it->second.get();
    }
  }

  Predicate* MakeTrue() { return MakeAndPredicate({}); }
  Predicate* MakeFalse() { return MakeOrPredicate({}); }

 private:
  template <typename PredicateT, typename... Args>
  std::unique_ptr<Predicate> Make(Args&&... args) {
    return std::unique_ptr<PredicateT>(
        new PredicateT(std::forward<Args>(args)...));
  }

  Predicate* MakeAndOrImpl(absl::Span<Predicate* const> operands, bool is_and);
  Predicate* MakeInternedAndOr(std::vector<Predicate*> simplified_ops,
                               Predicate::Kind pred_kind);

  // Predicate instances are interned, meaning that there is only a single
  // instance of a Predicate object with a given content.  This makes checking
  // for structural equality super-cheap -- we can just compare pointers.
  //
  // We intern predicates by maintaining a map from the content of a Predicate
  // to the only instance of said predicate we allow to exist in the
  // interned_and_or_instances_, interned_not_instances_ and
  // interned_symbol_instances_ fields.  These maps also double up as storage
  // for the owning pointers to predicate instances.

  using SignatureForAndOr =
      std::pair<Predicate::Kind, absl::Span<Predicate* const>>;
  using SignatureForNot = Predicate*;
  using SignatureForAndRec = std::pair<Predicate*, Predicate*>;
  using SignatureForSymbol = std::pair<SafeTensorId, bool>;

  struct HashSignatureForAndOr {
    size_t operator()(const SignatureForAndOr& signature) const {
      size_t hash = ::tensorflow::hash<Predicate::Kind>()(signature.first);
      for (Predicate* p : signature.second) {
        hash = Hash64Combine(hash, ::tensorflow::hash<Predicate*>()(p));
      }
      return hash;
    }
  };

  struct HashSignatureForSymbol {
    size_t operator()(const SignatureForSymbol& signature) const {
      return Hash64Combine(SafeTensorId::Hasher()(signature.first),
                           ::tensorflow::hash<bool>()(signature.second));
    }
  };

  absl::flat_hash_map<SignatureForAndOr, std::unique_ptr<Predicate>,
                      HashSignatureForAndOr>
      interned_and_or_instances_;
  absl::flat_hash_map<SignatureForNot, std::unique_ptr<Predicate>>
      interned_not_instances_;
  absl::flat_hash_map<SignatureForAndRec, std::unique_ptr<Predicate>>
      interned_and_rec_instances_;
  absl::flat_hash_map<SignatureForSymbol, std::unique_ptr<Predicate>,
                      HashSignatureForSymbol>
      interned_symbol_instances_;
};

Predicate* PredicateFactory::MakeInternedAndOr(
    std::vector<Predicate*> simplified_ops, Predicate::Kind pred_kind) {
  std::stable_sort(
      simplified_ops.begin(), simplified_ops.end(),
      [](Predicate* a, Predicate* b) { return a->hash() < b->hash(); });

  auto it = interned_and_or_instances_.find({pred_kind, simplified_ops});
  if (it != interned_and_or_instances_.end()) {
    return it->second.get();
  }

  simplified_ops.shrink_to_fit();
  // NB!  Because we'll use a non-owning reference to simplified_ops in the
  // key for interned_and_or_instances_ we need to be careful to std::move()
  // it all the way through.
  absl::Span<Predicate* const> operands_slice = simplified_ops;
  std::unique_ptr<Predicate> new_pred =
      pred_kind == Predicate::Kind::kAnd
          ? Make<AndPredicate>(std::move(simplified_ops))
          : Make<OrPredicate>(std::move(simplified_ops));

  Predicate* new_pred_ptr = new_pred.get();
  interned_and_or_instances_.emplace(
      SignatureForAndOr(pred_kind, operands_slice), std::move(new_pred));
  return new_pred_ptr;
}

// Common code to create AndPredicate or OrPredicate instances.
Predicate* PredicateFactory::MakeAndOrImpl(
    absl::Span<Predicate* const> operands, bool is_and) {
  Predicate::Kind pred_kind =
      is_and ? Predicate::Kind::kAnd : Predicate::Kind::kOr;
  Predicate::Kind other_pred_kind =
      is_and ? Predicate::Kind::kOr : Predicate::Kind::kAnd;
  absl::flat_hash_set<Predicate*> simplified_ops_set;
  std::vector<Predicate*> simplified_ops;
  for (Predicate* op : operands) {
    // Simplify A&A => A and  A|A => A.
    if (!simplified_ops_set.insert(op).second) {
      continue;
    }

    if (op->kind() == pred_kind) {
      // "Inline" the operands of an inner And/Or into the parent And/Or.
      for (Predicate* subop : op->GetOperands()) {
        if (simplified_ops_set.insert(subop).second) {
          simplified_ops.push_back(subop);
        }
      }
    } else {
      simplified_ops.push_back(op);
    }
  }

  if (simplified_ops.size() == 1) {
    return simplified_ops[0];
  }

  // Simplify "A&~A=>False" and "A|~A=>True".
  absl::flat_hash_set<Predicate*> negated_ops;
  for (Predicate* op : simplified_ops) {
    if (op->kind() == Predicate::Kind::kNot) {
      negated_ops.insert(dynamic_cast<NotPredicate&>(*op).operand());
    }
  }

  for (Predicate* op : simplified_ops) {
    if (negated_ops.count(op)) {
      return is_and ? MakeFalse() : MakeTrue();
    }
  }

  // If all ops contain the same subop, then factor it out thanks to the
  // distributive property. Such as:
  // - (A & B) | (A & C) | (A & D) => A & (B | C | D)
  // - (A | B) & (A | C) & (A | D) => A | (B & C & D)
  //
  // First find any predicates contained in all subops.
  std::vector<Predicate*> common_inner_operands;
  absl::flat_hash_set<Predicate*> common_inner_operands_set;
  for (Predicate* op : simplified_ops) {
    if (op->kind() != other_pred_kind) {
      common_inner_operands.clear();
      break;
    }

    if (common_inner_operands.empty()) {
      common_inner_operands.insert(common_inner_operands.end(),
                                   op->GetOperands().begin(),
                                   op->GetOperands().end());
    } else {
      common_inner_operands.clear();
      absl::c_copy_if(op->GetOperands(),
                      std::back_inserter(common_inner_operands),
                      [&](Predicate* sub_op) {
                        return common_inner_operands_set.count(sub_op) == 1;
                      });
    }
    if (common_inner_operands.empty()) break;
    common_inner_operands_set.clear();
    common_inner_operands_set.insert(common_inner_operands.begin(),
                                     common_inner_operands.end());
  }

  if (common_inner_operands.empty()) {
    return MakeInternedAndOr(std::move(simplified_ops), pred_kind);
  }

  // For all predicates that can be factored out, remove them and recreate the
  // subops.
  std::vector<Predicate*> factored_ops;
  for (Predicate* op : simplified_ops) {
    std::vector<Predicate*> new_sub_op_ops;
    absl::c_copy_if(op->GetOperands(), std::back_inserter(new_sub_op_ops),
                    [&](Predicate* sub_op) {
                      return std::find(common_inner_operands.begin(),
                                       common_inner_operands.end(),
                                       sub_op) == common_inner_operands.end();
                    });
    factored_ops.push_back(MakeAndOrImpl(new_sub_op_ops, !is_and));
  }

  Predicate* new_inner_op = MakeAndOrImpl(factored_ops, is_and);
  std::vector<Predicate*> outer_ops;
  outer_ops.push_back(new_inner_op);
  outer_ops.insert(outer_ops.end(), common_inner_operands.begin(),
                   common_inner_operands.end());
  return MakeAndOrImpl(outer_ops, !is_and);
}

class DeadnessAnalysisImpl : public DeadnessAnalysis {
 public:
  explicit DeadnessAnalysisImpl(const Graph* graph)
      : graph_(*graph), vlog_(VLOG_IS_ON(2)) {}

  Status Populate();
  Status PopulateWithReversePostOrder(absl::Span<Node* const> rpo);
  bool HasInputsWithMismatchingDeadness(const Node& node) override;
  void Print() const override;
  absl::flat_hash_map<TensorId, string, TensorId::Hasher> PredicateMapAsString()
      const;

 private:
  enum class EdgeKind { kDataAndControl, kDataOnly, kControlOnly };

  Status GetInputPreds(Node* n, EdgeKind edge_kind,
                       std::vector<Predicate*>* result);

  // Sets the predicate for output `output_idx` of `n` to `pred`.  Sets the i'th
  // bit of `should_revisit` if `pred` is different from the current predicate
  // for the `output_idx` output of `n`.
  void SetPredicate(Node* n, int output_idx, Predicate* pred,
                    std::vector<bool>* should_revisit) {
    auto insert_result =
        predicate_map_.insert({TensorId(n->name(), output_idx), pred});
    if (!insert_result.second && insert_result.first->second != pred) {
      VLOG(4) << "For " << n->name() << ":" << output_idx << " from "
              << insert_result.first->second->ToString() << " "
              << insert_result.first->second << " to " << pred->ToString()
              << " " << pred;
      insert_result.first->second = pred;
      if (should_revisit != nullptr) {
        for (const Edge* e : n->out_edges()) {
          (*should_revisit)[e->dst()->id()] = true;
        }
      }
    }
  }

  void SetPredicate(Node* n, absl::Span<const int> output_idxs, Predicate* pred,
                    std::vector<bool>* should_revisit) {
    for (int output_idx : output_idxs) {
      SetPredicate(n, output_idx, pred, should_revisit);
    }
  }

  Status HandleSwitch(Node* n, std::vector<bool>* should_revisit);
  Status HandleMerge(Node* n, std::vector<bool>* should_revisit);
  Status HandleRecv(Node* n, std::vector<bool>* should_revisit);
  Status HandleGeneric(Node* n, std::vector<bool>* should_revisit);
  Status HandleNode(Node* n, std::vector<bool>* should_revisit);

  const Graph& graph_;
  absl::flat_hash_map<TensorId, Predicate*, TensorId::Hasher> predicate_map_;
  PredicateFactory predicate_factory_;
  bool vlog_;
};

TensorId InputEdgeToTensorId(const Edge* e) {
  return TensorId(e->src()->name(), e->src_output());
}

Status DeadnessAnalysisImpl::GetInputPreds(
    Node* n, DeadnessAnalysisImpl::EdgeKind edge_kind,
    std::vector<Predicate*>* result) {
  result->clear();
  for (const Edge* in_edge : n->in_edges()) {
    bool should_process =
        edge_kind == EdgeKind::kDataAndControl ||
        (in_edge->IsControlEdge() && edge_kind == EdgeKind::kControlOnly) ||
        (!in_edge->IsControlEdge() && edge_kind == EdgeKind::kDataOnly);

    if (should_process) {
      auto it = predicate_map_.find(InputEdgeToTensorId(in_edge));
      if (it == predicate_map_.end()) {
        GraphCycles graph_cycles;
        TF_RETURN_IF_ERROR(CreateCycleDetectionGraph(&graph_, &graph_cycles));

        // If we didn't return with an error above then the graph is probably
        // fine and we have a bug in deadness analysis.
        return errors::Internal("Could not find input ", in_edge->DebugString(),
                                " to ", n->name(),
                                " when visiting the graph in post-order.  Most "
                                "likely indicates a bug in deadness analysis.");
      }
      result->push_back(it->second);
    }
  }
  return Status::OK();
}

Status DeadnessAnalysisImpl::HandleSwitch(Node* n,
                                          std::vector<bool>* should_revisit) {
  std::vector<Predicate*> input_preds;
  TF_RETURN_IF_ERROR(GetInputPreds(n, EdgeKind::kDataAndControl, &input_preds));
  const Edge* pred_edge;
  TF_RETURN_IF_ERROR(n->input_edge(1, &pred_edge));
  Predicate* true_switch = predicate_factory_.MakeSymbolPredicate(
      TensorId(pred_edge->src()->name(), pred_edge->src_output()),
      /*must_be_true=*/true);
  Predicate* false_switch = predicate_factory_.MakeNotPredicate(true_switch);

  // Output 0 is alive iff all inputs are alive and the condition is false.
  input_preds.push_back(false_switch);
  SetPredicate(n, 0, predicate_factory_.MakeAndPredicate(input_preds),
               should_revisit);
  input_preds.pop_back();

  // Output 1 is alive iff all inputs are alive and the condition is true.
  input_preds.push_back(true_switch);
  SetPredicate(n, 1, predicate_factory_.MakeAndPredicate(input_preds),
               should_revisit);
  input_preds.pop_back();

  // Control is alive iff all inputs are alive.
  SetPredicate(n, Graph::kControlSlot,
               predicate_factory_.MakeAndPredicate(input_preds),
               should_revisit);

  return Status::OK();
}

namespace {
Status CreateMultipleNextIterationInputsError(Node* merge) {
  std::vector<string> backedges;
  for (const Edge* backedge : merge->in_edges()) {
    if (backedge->src()->IsNextIteration()) {
      backedges.push_back(absl::StrCat("  ", SummarizeNode(*backedge->src())));
    }
  }
  return errors::InvalidArgument(
      "Multiple NextIteration inputs to merge node ",
      FormatNodeForError(*merge), ": \n", absl::StrJoin(backedges, "\n"),
      "\nMerge nodes can have at most one incoming NextIteration edge.");
}

Status FindUniqueBackedge(Node* merge, const Edge** result) {
  *result = nullptr;
  CHECK(merge->IsMerge());
  for (const Edge* e : merge->in_edges()) {
    if (e->src()->IsNextIteration()) {
      if (*result != nullptr) {
        return CreateMultipleNextIterationInputsError(merge);
      }
      *result = e;
    }
  }
  return Status::OK();
}

// If `backedge_predicate` is equal to `symbolic_predicate` & Step where Step
// does not contain `symbolic_predicate` as an inner (not top-level) operand
// then returns `Step`.  Otherwise returns nullptr.
Predicate* DeduceStepPredicate(PredicateFactory* predicate_factory,
                               Predicate* symbolic_predicate,
                               Predicate* backedge_predicate) {
  CHECK(dynamic_cast<SymbolPredicate*>(symbolic_predicate));
  if (backedge_predicate->kind() != Predicate::Kind::kAnd) {
    return nullptr;
  }

  std::vector<Predicate*> and_ops;
  absl::Span<Predicate* const> recurrent_pred_ops =
      backedge_predicate->GetOperands();

  bool found_sym = false;
  for (Predicate* and_op : recurrent_pred_ops) {
    // We want the `symbol_predicate` to be the one of the operands of
    // `backedge_predicate`,
    if (and_op == symbolic_predicate) {
      found_sym = true;
      continue;
    }

    // but we don't want it to be present anywhere else in the formula.  E.g. we
    // don't want the recurrent predicate to be
    // symbol_predicate&(X|symbol_predicate).
    bool found_sym_as_inner_operand = false;
    auto has_self_as_inner_operand = [&](Predicate* p) {
      if (p == symbolic_predicate) {
        found_sym_as_inner_operand = true;
        return true;  // Stop searching, we're done.
      }

      // Continue searching.
      return false;
    };

    Predicate::Visit(and_op, has_self_as_inner_operand);
    if (found_sym_as_inner_operand) {
      return nullptr;
    }
    and_ops.push_back(and_op);
  }

  return found_sym ? predicate_factory->MakeAndPredicate(and_ops) : nullptr;
}
}  // namespace

Status DeadnessAnalysisImpl::HandleMerge(Node* n,
                                         std::vector<bool>* should_revisit) {
  // Merge ignores deadness of its control inputs.  A merge that isn't the
  // target of a backedge has is alive iff any of its data inputs are.  The
  // liveness of a merge that is the target of a backedge can sometimes be
  // represented using a AndRecurrencePredicate.  If neither apply, we represent
  // the liveness of the merge symbolically.

  bool has_unvisited_backedge = false;
  for (const Edge* e : n->in_edges()) {
    if (!e->IsControlEdge() && e->src()->IsNextIteration()) {
      has_unvisited_backedge |= !predicate_map_.count(InputEdgeToTensorId(e));
    }
  }

  auto it = predicate_map_.find(TensorId(n->name(), 0));
  if (it == predicate_map_.end()) {
    if (has_unvisited_backedge) {
      // We're visiting this merge for the first time and it has an unvisited
      // backedge.
      Predicate* input_data_pred = predicate_factory_.MakeSymbolPredicate(
          TensorId(n->name(), 0), /*must_be_true=*/false);
      SetPredicate(n, {0, 1, Graph::kControlSlot}, input_data_pred,
                   should_revisit);
      return Status::OK();
    }

    std::vector<Predicate*> input_preds;
    TF_RETURN_IF_ERROR(GetInputPreds(n, EdgeKind::kDataOnly, &input_preds));

    // We're visiting this merge for the first time and it is a acyclic merge.
    Predicate* input_data_pred =
        predicate_factory_.MakeOrPredicate(input_preds);
    SetPredicate(n, {0, 1, Graph::kControlSlot}, input_data_pred,
                 should_revisit);
    return Status::OK();
  }

  if (it->second->kind() == Predicate::Kind::kSymbol) {
    // Last time we visited this merge we only got a symbolic predicate because
    // of an unvisited backedge.  Try to pattern match the predicate expression
    // for that backedge (which should be visited now) into an and recurrence
    // for the merge node.
    const Edge* unique_backedge;
    TF_RETURN_IF_ERROR(FindUniqueBackedge(n, &unique_backedge));
    if (unique_backedge) {
      if (Predicate* step = DeduceStepPredicate(
              &predicate_factory_, it->second,
              predicate_map_[InputEdgeToTensorId(unique_backedge)])) {
        // If the predicate for the backedge is "Sym&X" where "Sym" is the
        // predicate for the merge then the merge has predicate {S,&,X} where S
        // is the predicate for the merge ignoring the backedge.
        std::vector<Predicate*> non_recurrent_inputs;
        for (const Edge* e : n->in_edges()) {
          if (e != unique_backedge) {
            non_recurrent_inputs.push_back(
                predicate_map_[InputEdgeToTensorId(e)]);
          }
        }

        Predicate* start =
            predicate_factory_.MakeOrPredicate(non_recurrent_inputs);
        Predicate* and_rec =
            predicate_factory_.MakeAndRecurrencePredicate(start, step);
        SetPredicate(n, {0, 1, Graph::kControlSlot}, and_rec, should_revisit);
        return Status::OK();
      }
    }
  }
  return Status::OK();
}

Status DeadnessAnalysisImpl::HandleRecv(Node* n,
                                        std::vector<bool>* should_revisit) {
  // In addition to being alive or dead based on the inputs, a _Recv can also
  // acquire a dead signal from a _Send.
  std::vector<Predicate*> input_preds;
  TF_RETURN_IF_ERROR(GetInputPreds(n, EdgeKind::kDataAndControl, &input_preds));
  input_preds.push_back(predicate_factory_.MakeSymbolPredicate(
      TensorId(n->name(), 0), /*must_be_true=*/false));
  SetPredicate(n, {0, Graph::kControlSlot},
               predicate_factory_.MakeAndPredicate(input_preds),
               should_revisit);
  return Status::OK();
}

Status DeadnessAnalysisImpl::HandleGeneric(Node* n,
                                           std::vector<bool>* should_revisit) {
  // Generally nodes are alive iff all their inputs are alive.
  std::vector<Predicate*> input_preds;
  TF_RETURN_IF_ERROR(GetInputPreds(n, EdgeKind::kDataAndControl, &input_preds));
  Predicate* pred = predicate_factory_.MakeAndPredicate(input_preds);
  for (int output_idx = 0; output_idx < n->num_outputs(); output_idx++) {
    SetPredicate(n, output_idx, pred, should_revisit);
  }
  SetPredicate(n, Graph::kControlSlot, pred, should_revisit);
  return Status::OK();
}

Status DeadnessAnalysisImpl::HandleNode(Node* n,
                                        std::vector<bool>* should_revisit) {
  if (n->IsSwitch()) {
    TF_RETURN_IF_ERROR(HandleSwitch(n, should_revisit));
  } else if (n->IsMerge()) {
    TF_RETURN_IF_ERROR(HandleMerge(n, should_revisit));
  } else if (n->IsControlTrigger()) {
    SetPredicate(n, Graph::kControlSlot, predicate_factory_.MakeTrue(),
                 nullptr);
  } else if (n->IsRecv() || n->IsHostRecv()) {
    TF_RETURN_IF_ERROR(HandleRecv(n, should_revisit));
  } else if (n->IsNextIteration()) {
    TF_RETURN_IF_ERROR(HandleGeneric(n, should_revisit));
  } else {
    TF_RETURN_IF_ERROR(HandleGeneric(n, should_revisit));
  }
  return Status::OK();
}

Status DeadnessAnalysisImpl::Populate() {
  std::vector<Node*> rpo;
  GetReversePostOrder(graph_, &rpo, /*stable_comparator=*/NodeComparatorName(),
                      /*edge_filter=*/[](const Edge& edge) {
                        return !edge.src()->IsNextIteration();
                      });
  return PopulateWithReversePostOrder(rpo);
}

Status DeadnessAnalysisImpl::PopulateWithReversePostOrder(
    absl::Span<Node* const> rpo) {
  // This an abstract interpretation over the deadness propagation semantics of
  // the graph executor.
  //
  // We iterate over the graph twice, each time in RPO.  On the first iteration
  // merge nodes with backedges are mapped to symbolic predicates.  On the
  // second iteration we use the predicates assigned to the backedges in the
  // previous iteration to infer a more precise predicate for the backedge merge
  // nodes and all the nodes that transitively use it.
  //
  // We don't track the output indices for should_revisit.  Instead, putting a
  // node in `should_revisit` denotes that the deadness flowing out from any
  // output from said node may have changed.  This is fine; only switches
  // propagate different deadness along different output edges, and since the
  // delta is solely due to the input *values* (and not input deadness), the
  // delta should not change in the second iteration.
  std::vector<bool> should_revisit;
  should_revisit.resize(graph_.num_node_ids());
  for (Node* n : rpo) {
    VLOG(4) << "Visiting " << n->name();
    TF_RETURN_IF_ERROR(HandleNode(n, /*should_revisit=*/nullptr));
    if (n->IsNextIteration()) {
      // If this is a backedge for a merge node then remember to reprocess the
      // merge the next time we run.
      for (const Edge* e : n->out_edges()) {
        if (e->dst()->IsMerge()) {
          should_revisit[e->dst()->id()] = true;
        }
      }
    }
  }

  for (Node* n : rpo) {
    // The nodes added to should_revisit in the previous loop need to be
    // revisited now.  Reprocesing these initial nodes may add *their* consumers
    // to should_revisit, and these newly added nodes will also be processed by
    // this very same loop.  Since we're traversing the graph in reverse post
    // order (producers before consumers) and HandleNode(n) can only ever add
    // n's consumers to should_revisit, we won't "miss" an addition to
    // should_revisit.
    if (should_revisit[n->id()]) {
      VLOG(4) << "Revisiting " << n->name();
      TF_RETURN_IF_ERROR(HandleNode(n, &should_revisit));
    }
  }

  return Status::OK();
}

bool DeadnessAnalysisImpl::HasInputsWithMismatchingDeadness(const Node& node) {
  CHECK(!node.IsMerge());

  if (vlog_) {
    VLOG(2) << "HasInputsWithMismatchingDeadness(" << node.name() << ")";
  }

  Predicate* pred = nullptr;
  for (const Edge* edge : node.in_edges()) {
    auto it = predicate_map_.find(InputEdgeToTensorId(edge));
    CHECK(it != predicate_map_.end());
    if (vlog_) {
      VLOG(2) << "  " << InputEdgeToTensorId(edge).ToString() << ": "
              << it->second->ToString();
    }

    // Today we just compare the predicates for equality (with some
    // canonicalization/simplification happening before) but we could be more
    // sophisticated here if need be.  Comparing pointers is sufficient because
    // we intern Predicate instances by their content.
    if (pred != nullptr && pred != it->second) {
      if (vlog_) {
        VLOG(2) << "HasInputsWithMismatchingDeadness(" << node.name()
                << ") -> true";
      }
      return true;
    }
    pred = it->second;
  }

  if (vlog_) {
    VLOG(2) << "HasInputsWithMismatchingDeadness(" << node.name()
            << ") -> false";
  }

  return false;
}

void DeadnessAnalysisImpl::Print() const {
  std::vector<TensorId> tensor_ids;
  for (const auto& kv_pair : predicate_map_) {
    tensor_ids.push_back(kv_pair.first);
  }

  std::sort(tensor_ids.begin(), tensor_ids.end());

  for (TensorId tensor_id : tensor_ids) {
    auto it = predicate_map_.find(tensor_id);
    CHECK(it != predicate_map_.end()) << tensor_id.ToString();
    VLOG(2) << tensor_id.ToString() << " -> " << it->second->ToString();
  }
}

}  // namespace

DeadnessAnalysis::~DeadnessAnalysis() {}

/*static*/ Status DeadnessAnalysis::Run(
    const Graph& graph, std::unique_ptr<DeadnessAnalysis>* result) {
  std::unique_ptr<DeadnessAnalysisImpl> analysis(
      new DeadnessAnalysisImpl(&graph));
  TF_RETURN_IF_ERROR(analysis->Populate());

  if (VLOG_IS_ON(2)) {
    analysis->Print();
  }

  *result = std::move(analysis);
  return Status::OK();
}

absl::flat_hash_map<TensorId, string, TensorId::Hasher>
DeadnessAnalysisImpl::PredicateMapAsString() const {
  absl::flat_hash_map<TensorId, string, TensorId::Hasher> result;
  std::vector<TensorId> tensor_ids;
  for (const auto& kv_pair : predicate_map_) {
    CHECK(result.insert({kv_pair.first, kv_pair.second->ToString()}).second);
  }
  return result;
}

namespace deadness_analysis_internal {
Status ComputePredicates(const Graph& graph,
                         PredicateMapTy* out_predicate_map) {
  DeadnessAnalysisImpl impl(&graph);
  TF_RETURN_IF_ERROR(impl.Populate());
  *out_predicate_map = impl.PredicateMapAsString();
  return Status::OK();
}

Status ComputePredicates(const Graph& graph,
                         absl::Span<Node* const> reverse_post_order,
                         PredicateMapTy* out_predicate_map) {
  DeadnessAnalysisImpl impl(&graph);
  TF_RETURN_IF_ERROR(impl.PopulateWithReversePostOrder(reverse_post_order));
  *out_predicate_map = impl.PredicateMapAsString();
  return Status::OK();
}
}  // namespace deadness_analysis_internal

}  // namespace tensorflow
