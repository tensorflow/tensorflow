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
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/hash/hash.h"

// ALGORITHM OVERVIEW
//
// We map every output produced by each node in the TensorFlow graph (including
// control dependence) into an instance of the Predicate class.  Instances of
// Predicate denote logical formulas and mapping a node `n` to a predicate
// `pred` implies that `n` is executed whenver `pred` is true.  Then we can
// deduce mismatching liveness in the inputs to node by comparing the predicate
// those inputs are mapped to.
//
// Loops are handled pessimistically -- we map Merge nodes with backedges to
// uninterpreted symbols (the same kind we use to represent Switch and _Recv).
// Predicate equality has to hold over all possible assignments to these
// uninterpreted symbols.

namespace tensorflow {

namespace {

// Represents a logical predicate, used as described in the algorithm overview
// above.
class Predicate {
 public:
  enum class Kind { kAnd, kOr, kNot, kSymbol };

  virtual string ToString() const = 0;
  int64 hash() const { return hash_; }

  virtual Kind kind() const = 0;
  virtual ~Predicate() {}

 protected:
  explicit Predicate(int64 hash) : hash_(hash) {}

 private:
  const int64 hash_;

  TF_DISALLOW_COPY_AND_ASSIGN(Predicate);
};

int64 HashPredicateSequence(Predicate::Kind kind,
                            gtl::ArraySlice<Predicate*> preds) {
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

    return strings::StrCat("(", str_util::Join(operands_str, " & "), ")");
  }

  Kind kind() const override { return Kind::kAnd; }

  const gtl::ArraySlice<Predicate*> operands() const { return operands_; }

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

    return strings::StrCat("(", str_util::Join(operands_str, " | "), ")");
  }

  Kind kind() const override { return Kind::kOr; }
  const gtl::ArraySlice<Predicate*> operands() const { return operands_; }

 private:
  std::vector<Predicate*> operands_;
};

// Represents a logical negation of a set of predicates.
class NotPredicate : public Predicate {
 public:
  explicit NotPredicate(Predicate* operand)
      : Predicate(HashPredicateSequence(Kind::kNot, {operand})),
        operand_(operand) {}

  string ToString() const override {
    return strings::StrCat("~", operand()->ToString());
  }

  Kind kind() const override { return Kind::kNot; }
  Predicate* operand() const { return operand_; }

 private:
  Predicate* operand_;
};

// Represents an uninterpreted symbol in a logical predicate.
//
// Two predicates are equivalent iff they are equivalent for all assignments to
// the symbols contained in them.
class SymbolPredicate : public Predicate {
 public:
  explicit SymbolPredicate(TensorId tensor_id, bool must_be_true)
      : Predicate(Hash(tensor_id, must_be_true)),
        tensor_id_(std::move(tensor_id)),
        must_be_true_(must_be_true) {}

  string ToString() const override { return tensor_id_.ToString(); }
  Kind kind() const override { return Kind::kSymbol; }

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

// Creates and owns Predicate instances.  Simplifies predicates as it creates
// them.
class PredicateFactory {
 public:
  Predicate* MakeAndPredicate(gtl::ArraySlice<Predicate*> operands) {
    return MakeAndOrImpl(operands, /*is_and=*/true);
  }

  Predicate* MakeOrPredicate(gtl::ArraySlice<Predicate*> operands) {
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

  Predicate* MakeAndOrImpl(gtl::ArraySlice<Predicate*> operands, bool is_and);

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
      std::pair<Predicate::Kind, gtl::ArraySlice<Predicate*>>;
  using SignatureForNot = Predicate*;
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

  gtl::FlatMap<SignatureForAndOr, std::unique_ptr<Predicate>,
               HashSignatureForAndOr>
      interned_and_or_instances_;
  gtl::FlatMap<SignatureForNot, std::unique_ptr<Predicate>>
      interned_not_instances_;
  gtl::FlatMap<SignatureForSymbol, std::unique_ptr<Predicate>,
               HashSignatureForSymbol>
      interned_symbol_instances_;
};

// Common code to create AndPredicate or OrPredicate instances.
Predicate* PredicateFactory::MakeAndOrImpl(gtl::ArraySlice<Predicate*> operands,
                                           bool is_and) {
  Predicate::Kind pred_kind =
      is_and ? Predicate::Kind::kAnd : Predicate::Kind::kOr;
  gtl::FlatSet<Predicate*> simplified_ops_set;
  std::vector<Predicate*> simplified_ops;
  for (Predicate* op : operands) {
    // Simplify A&A => A and  A|A => A.
    if (!simplified_ops_set.insert(op).second) {
      continue;
    }

    if (op->kind() == pred_kind) {
      // "Inline" the operands of an inner And/Or into the parent And/Or.
      gtl::ArraySlice<Predicate*> operands =
          is_and ? dynamic_cast<AndPredicate*>(op)->operands()
                 : dynamic_cast<OrPredicate*>(op)->operands();
      for (Predicate* subop : operands) {
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
  gtl::FlatSet<Predicate*> negated_ops;
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

  std::stable_sort(
      simplified_ops.begin(), simplified_ops.end(),
      [](Predicate* a, Predicate* b) { return a->hash() < b->hash(); });

  auto it = interned_and_or_instances_.find({pred_kind, simplified_ops});
  if (it == interned_and_or_instances_.end()) {
    simplified_ops.shrink_to_fit();
    // NB!  Because we'll use a non-owning reference to simplified_ops in the
    // key for interned_and_or_instances_ we need to be careful to std::move()
    // it all the way through.
    gtl::ArraySlice<Predicate*> operands_slice = simplified_ops;
    std::unique_ptr<Predicate> new_pred =
        is_and ? Make<AndPredicate>(std::move(simplified_ops))
               : Make<OrPredicate>(std::move(simplified_ops));

    Predicate* new_pred_ptr = new_pred.get();
    CHECK(interned_and_or_instances_
              .emplace(SignatureForAndOr(pred_kind, operands_slice),
                       std::move(new_pred))
              .second);
    return new_pred_ptr;
  } else {
    return it->second.get();
  }
}

class DeadnessAnalysisImpl : public DeadnessAnalysis {
 public:
  explicit DeadnessAnalysisImpl(const Graph* graph)
      : graph_(*graph), vlog_(VLOG_IS_ON(2)) {}

  Status Populate();
  bool HasInputsWithMismatchingDeadness(const Node& node) override;
  void Print() const override;

 private:
  enum class EdgeKind { kDataAndControl, kDataOnly, kControlOnly };

  std::vector<Predicate*> GetIncomingPreds(Node* n, EdgeKind edge_kind);
  void SetPred(Node* n, int output_idx, Predicate* pred) {
    CHECK(
        predicate_map_.insert({TensorId(n->name(), output_idx), pred}).second);
  }
  void SetPred(Node* n, gtl::ArraySlice<int> output_idxs, Predicate* pred) {
    for (int output_idx : output_idxs) {
      SetPred(n, output_idx, pred);
    }
  }

  Status HandleSwitch(Node* n);
  Status HandleMerge(Node* n);
  Status HandleRecv(Node* n);
  Status HandleGeneric(Node* n);

  const Graph& graph_;
  gtl::FlatMap<TensorId, Predicate*, TensorId::Hasher> predicate_map_;
  PredicateFactory predicate_factory_;
  bool vlog_;
};

TensorId InputEdgeToTensorId(const Edge* e) {
  return TensorId(e->src()->name(), e->src_output());
}

std::vector<Predicate*> DeadnessAnalysisImpl::GetIncomingPreds(
    Node* n, DeadnessAnalysisImpl::EdgeKind edge_kind) {
  std::vector<Predicate*> incoming_preds;
  for (const Edge* in_edge : n->in_edges()) {
    bool should_process =
        edge_kind == EdgeKind::kDataAndControl ||
        (in_edge->IsControlEdge() && edge_kind == EdgeKind::kControlOnly) ||
        (!in_edge->IsControlEdge() && edge_kind == EdgeKind::kDataOnly);

    if (should_process) {
      auto it = predicate_map_.find(InputEdgeToTensorId(in_edge));
      CHECK(it != predicate_map_.end());
      incoming_preds.push_back(it->second);
    }
  }
  return incoming_preds;
}

Status DeadnessAnalysisImpl::HandleSwitch(Node* n) {
  std::vector<Predicate*> input_preds =
      GetIncomingPreds(n, EdgeKind::kDataAndControl);
  const Edge* pred_edge;
  TF_RETURN_IF_ERROR(n->input_edge(1, &pred_edge));
  Predicate* true_switch = predicate_factory_.MakeSymbolPredicate(
      TensorId(pred_edge->src()->name(), pred_edge->src_output()),
      /*must_be_true=*/true);
  Predicate* false_switch = predicate_factory_.MakeNotPredicate(true_switch);

  // Output 0 is alive iff all inputs are alive and the condition is false.
  input_preds.push_back(false_switch);
  SetPred(n, 0, predicate_factory_.MakeAndPredicate(input_preds));
  input_preds.pop_back();

  // Output 1 is alive iff all inputs are alive and the condition is true.
  input_preds.push_back(true_switch);
  SetPred(n, 1, predicate_factory_.MakeAndPredicate(input_preds));
  input_preds.pop_back();

  // Control is alive iff any inputs are alive.
  SetPred(n, Graph::kControlSlot,
          predicate_factory_.MakeAndPredicate(input_preds));

  return Status::OK();
}

Status DeadnessAnalysisImpl::HandleMerge(Node* n) {
  // Merge ignores deadness of its control inputs.  A merge that isn't the
  // target of a backedge has is alive iff any of its data inputs are.  We treat
  // the liveness of a merge that is the target of a backedge symbolically.

  bool has_backedge = std::any_of(
      n->in_edges().begin(), n->in_edges().end(), [](const Edge* e) {
        return !e->IsControlEdge() && e->src()->IsNextIteration();
      });

  Predicate* input_data_pred =
      has_backedge ? predicate_factory_.MakeSymbolPredicate(
                         TensorId(n->name(), 0), /*must_be_true=*/false)
                   : predicate_factory_.MakeOrPredicate(
                         GetIncomingPreds(n, EdgeKind::kDataOnly));

  SetPred(n, {0, 1, Graph::kControlSlot}, input_data_pred);
  return Status::OK();
}

Status DeadnessAnalysisImpl::HandleRecv(Node* n) {
  // In addition to being alive or dead based on the inputs, a _Recv can also
  // acquire a dead signal from a _Send.
  std::vector<Predicate*> input_preds =
      GetIncomingPreds(n, EdgeKind::kDataAndControl);
  input_preds.push_back(predicate_factory_.MakeSymbolPredicate(
      TensorId(n->name(), 0), /*must_be_true=*/false));
  SetPred(n, {0, Graph::kControlSlot},
          predicate_factory_.MakeAndPredicate(input_preds));
  return Status::OK();
}

Status DeadnessAnalysisImpl::HandleGeneric(Node* n) {
  // Generally nodes are alive iff all their inputs are alive.
  Predicate* pred = predicate_factory_.MakeAndPredicate(
      GetIncomingPreds(n, EdgeKind::kDataAndControl));
  for (int output_idx = 0; output_idx < n->num_outputs(); output_idx++) {
    SetPred(n, output_idx, pred);
  }
  SetPred(n, Graph::kControlSlot, pred);
  return Status::OK();
}

Status DeadnessAnalysisImpl::Populate() {
  std::vector<Node*> rpo;
  GetReversePostOrder(graph_, &rpo, /*stable_comparator=*/{},
                      /*edge_filter=*/[](const Edge& edge) {
                        return !edge.src()->IsNextIteration();
                      });

  // This an abstract interpretation over the deadness propagation semantics of
  // the graph executor.
  for (Node* n : rpo) {
    if (n->IsSwitch()) {
      TF_RETURN_IF_ERROR(HandleSwitch(n));
    } else if (n->IsMerge()) {
      TF_RETURN_IF_ERROR(HandleMerge(n));
    } else if (n->IsControlTrigger()) {
      SetPred(n, Graph::kControlSlot, predicate_factory_.MakeTrue());
    } else if (n->IsRecv() || n->IsHostRecv()) {
      TF_RETURN_IF_ERROR(HandleRecv(n));
    } else {
      TF_RETURN_IF_ERROR(HandleGeneric(n));
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

}  // namespace tensorflow
