//===- ConstraintAnalysisGraph.h - Graphs type for constraints --*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides graph-based data structures for representing anchors
// and constraints between them.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_QUANTIZER_SUPPORT_CONSTRAINTANALYSISGRAPH_H
#define MLIR_QUANTIZER_SUPPORT_CONSTRAINTANALYSISGRAPH_H

#include <utility>
#include <vector>

#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Quantizer/Support/Metadata.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace quantizer {

class CAGNode;
class CAGSlice;
class TargetConfiguration;

/// A node in the Constraint Analysis Graph.
/// Nodes are either anchors (representing results and operands) or constraints.
/// Anchor nodes are connected to other anchor nodes via constraints.
/// Nodes exist within graph slices, which are typically analyses attached to
/// the function or module. Slices can contain other slices, which mirrors
/// the nesting of analyses.
///
/// Nodes have directed relationships which propagate successor-ward when dirty.
/// Relationships can be bi-directional, in which case, the constraint's
/// propagation mechanism must ensure convergence.
class CAGNode {
public:
  enum class Kind {
    /// Anchors.
    Anchor,
    OperandAnchor,
    ResultAnchor,
    LastAnchor = ResultAnchor,

    /// Constraints.
    Constraint,
    SolveUniformConstraint,
    UniformPropagateExplicitScale,
    LastConstraint = UniformPropagateExplicitScale,
  };

  // Vector and iterator over nodes.
  using node_vector = SmallVector<CAGNode *, 1>;
  using iterator = node_vector::iterator;
  using const_iterator = node_vector::const_iterator;

  virtual ~CAGNode() = default;

  Kind getKind() const { return kind; }

  /// Unique id of the node within the slice.
  int getNodeId() const { return nodeId; }

  /// Whether the node is dirty, requiring one or more calls to propagate().
  bool isDirty() const { return dirty; }
  void markDirty() { dirty = true; }
  void clearDirty() { dirty = false; }

  /// Iterator over this node's children (outgoing) nodes.
  const_iterator begin() const { return outgoing.begin(); }
  const_iterator end() const { return outgoing.end(); }
  iterator begin() { return outgoing.begin(); }
  iterator end() { return outgoing.end(); }

  /// Iterator over this parents (incoming) nodes.
  const_iterator incoming_begin() const { return incoming.begin(); }
  const_iterator incoming_end() const { return incoming.end(); }
  iterator incoming_begin() { return incoming.begin(); }
  iterator incoming_end() { return incoming.end(); }

  virtual void propagate(SolverContext &solverContext,
                         const TargetConfiguration &config) {}

  /// Prints the node label, suitable for one-line display.
  virtual void printLabel(raw_ostream &os) const;

  template <typename T> void findChildrenOfKind(SmallVectorImpl<T *> &found) {
    for (CAGNode *child : *this) {
      T *ofKind = dyn_cast<T>(child);
      if (ofKind) {
        found.push_back(ofKind);
      }
    }
  }

  /// Replaces this node by rerouting any parent nodes to have otherNode
  /// as a child.
  void replaceIncoming(CAGNode *otherNode);

  /// Adds an outgoing connection to this node (and corresponding back
  /// incoming connection).
  void addOutgoing(CAGNode *toNode);

  /// Whether this node is an orphan (has no incoming or outgoing connections).
  bool isOrphan() const { return incoming.empty() && outgoing.empty(); }

protected:
  CAGNode(Kind kind) : kind(kind) {}

private:
  Kind kind;
  int nodeId = -1;
  node_vector outgoing;
  node_vector incoming;
  bool dirty = false;

  friend class CAGSlice;
};

/// Anchor nodes represent points in the source IR where we may choose to
/// introduce a type transition. These include operands, results, arguments
/// returns, etc.
class CAGAnchorNode : public CAGNode {
public:
  enum class TypeTransformRule {
    /// The owning op directly supports all transformed types. In practice,
    /// this means that the op supports QuantizedType for this anchor.
    Direct,

    /// The type of this anchor should be set to the QuantizedType storage
    /// type. This will only be valid if constraints are such that all
    /// inputs/outputs converge to the same storage type (i.e. coupled).
    DirectStorage,

    /// The anchor must only be typed based on the expressed type. This is
    /// used for ops that do not natively support quantization, and suitable
    /// casts will be inserted.
    ExpressedOnly,
  };

  /// Metadata for solving uniform quantization params.
  CAGUniformMetadata &getUniformMetadata() { return uniformMetadata; }
  const CAGUniformMetadata &getUniformMetadata() const {
    return uniformMetadata;
  }

  virtual Operation *getOp() const = 0;
  virtual Value getValue() const = 0;

  static bool classof(const CAGNode *n) {
    return n->getKind() >= Kind::Anchor && n->getKind() <= Kind::LastAnchor;
  }

  void propagate(SolverContext &solverContext,
                 const TargetConfiguration &config) override;

  void printLabel(raw_ostream &os) const override;

  /// Given the anchor metadata and resolved solutions, chooses the most
  /// salient and returns an appropriate type to represent it.
  Type getTransformedType();

  TypeTransformRule getTypeTransformRule() const { return typeTransformRule; }

  void setTypeTransformRule(TypeTransformRule r) { typeTransformRule = r; }

  /// Gets the Type that was defined for this anchor at the time of
  /// construction.
  Type getOriginalType() const { return originalType; }

protected:
  CAGAnchorNode(Kind kind, Type originalType)
      : CAGNode(kind), originalType(originalType) {}

private:
  CAGUniformMetadata uniformMetadata;
  Type originalType;
  TypeTransformRule typeTransformRule = TypeTransformRule::Direct;
};

/// An anchor tied to a specific operand.
/// Since operand anchors can be rewritten so that the operand refers to
/// a new result, they are maintained by reference (to the op and index).
class CAGOperandAnchor : public CAGAnchorNode {
public:
  CAGOperandAnchor(Operation *op, unsigned operandIdx);

  Operation *getOp() const final { return op; }
  unsigned getOperandIdx() const { return operandIdx; }

  static bool classof(const CAGNode *n) {
    return n->getKind() == Kind::Anchor || n->getKind() == Kind::OperandAnchor;
  }

  Value getValue() const final { return op->getOperand(operandIdx); }

  void printLabel(raw_ostream &os) const override;

private:
  Operation *op;
  unsigned operandIdx;
};

/// An anchor tied to a specific result.
/// Since a result is already anchored to its defining op, result anchors refer
/// directly to the underlying Value.
class CAGResultAnchor : public CAGAnchorNode {
public:
  CAGResultAnchor(Operation *op, unsigned resultIdx);

  static bool classof(const CAGNode *n) {
    return n->getKind() == Kind::Anchor || n->getKind() == Kind::ResultAnchor;
  }

  Operation *getOp() const final { return resultValue->getDefiningOp(); }
  Value getValue() const final { return resultValue; }

  void printLabel(raw_ostream &os) const override;

private:
  Value resultValue;
};

/// Base class for constraint nodes.
class CAGConstraintNode : public CAGNode {
public:
  CAGConstraintNode(Kind kind) : CAGNode(kind) {}

  static bool classof(const CAGNode *n) {
    return n->getKind() >= Kind::Constraint &&
           n->getKind() <= Kind::LastConstraint;
  }
};

/// A slice of a CAG (which may be the whole graph).
class CAGSlice {
public:
  CAGSlice(SolverContext &context);
  ~CAGSlice();

  using node_vector = std::vector<CAGNode *>;
  using iterator = node_vector::iterator;
  using const_iterator = node_vector::const_iterator;

  iterator begin() { return allNodes.begin(); }
  iterator end() { return allNodes.end(); }
  const_iterator begin() const { return allNodes.begin(); }
  const_iterator end() const { return allNodes.end(); }

  /// Gets an operand anchor node.
  CAGOperandAnchor *getOperandAnchor(Operation *op, unsigned operandIdx);

  /// Gets a result anchor node.
  CAGResultAnchor *getResultAnchor(Operation *op, unsigned resultIdx);

  /// Adds a relation constraint with incoming 'from' anchors and outgoing 'to'
  /// anchors.
  template <typename T, typename... Args>
  T *addUniqueConstraint(ArrayRef<CAGAnchorNode *> anchors, Args... args) {
    static_assert(std::is_convertible<T *, CAGConstraintNode *>(),
                  "T must be a CAGConstraingNode");
    T *constraintNode = addNode(std::make_unique<T>(args...));
    for (auto *anchor : anchors)
      anchor->addOutgoing(constraintNode);
    return constraintNode;
  }

  /// Adds a unidirectional constraint from a node to an array of target nodes.
  template <typename T, typename... Args>
  T *addUnidirectionalConstraint(CAGAnchorNode *fromAnchor,
                                 ArrayRef<CAGAnchorNode *> toAnchors,
                                 Args... args) {
    static_assert(std::is_convertible<T *, CAGConstraintNode *>(),
                  "T must be a CAGConstraingNode");
    T *constraintNode = addNode(std::make_unique<T>(args...));
    fromAnchor->addOutgoing(constraintNode);
    for (auto *toAnchor : toAnchors) {
      constraintNode->addOutgoing(toAnchor);
    }
    return constraintNode;
  }

  template <typename T>
  T *addClusteredConstraint(ArrayRef<CAGAnchorNode *> anchors) {
    static_assert(std::is_convertible<T *, CAGConstraintNode *>(),
                  "T must be a CAGConstraingNode");
    SmallVector<T *, 8> cluster;
    for (auto *anchor : anchors) {
      anchor->findChildrenOfKind<T>(cluster);
    }

    T *constraintNode;
    if (cluster.empty()) {
      // Create new.
      constraintNode = addNode(std::make_unique<T>());
    } else {
      // Merge existing.
      constraintNode = cluster[0];
      for (size_t i = 1, e = cluster.size(); i < e; ++i) {
        cluster[i]->replaceIncoming(constraintNode);
      }
    }
    for (auto *anchor : anchors) {
      anchor->addOutgoing(constraintNode);
    }
    return constraintNode;
  }

  /// Enumerates all implied connections in the slice.
  /// An implied connection is any two nodes that physically refer to the
  /// same value in the IR, such as result->operand.
  /// Typically this will be modeled with some kind of strong or weak
  /// identity constraint such that types propagate.
  /// This is usually called when the slice has been fully constructed in
  /// order to add final constraints.
  /// It is legal for the callback to modify the graph by adding constraints.
  void enumerateImpliedConnections(
      std::function<void(CAGAnchorNode *from, CAGAnchorNode *to)> callback);

  /// Performs one round of propagation, returning the number of nodes
  /// propagates. If returns > 0, then additional propagate() rounds are
  /// required.
  unsigned propagate(const TargetConfiguration &config);

private:
  /// Adds a node to the graph.
  /// The node should be a subclass of TransformNode.
  /// Returns the raw pointer to the node.
  template <typename T>
  T *addNode(std::unique_ptr<T> node) {
    node->nodeId = allNodes.size();
    T *unownedNode = node.release();
    allNodes.push_back(unownedNode);
    return unownedNode;
  }

  SolverContext &context;
  std::vector<CAGNode *> allNodes;
  DenseMap<std::pair<Operation *, unsigned>, CAGOperandAnchor *> operandAnchors;
  DenseMap<std::pair<Operation *, unsigned>, CAGResultAnchor *> resultAnchors;
};

inline raw_ostream &operator<<(raw_ostream &os, const CAGNode &node) {
  node.printLabel(os);
  return os;
}

} // namespace quantizer
} // namespace mlir

#endif // MLIR_QUANTIZER_SUPPORT_CONSTRAINTANALYSISGRAPH_H
