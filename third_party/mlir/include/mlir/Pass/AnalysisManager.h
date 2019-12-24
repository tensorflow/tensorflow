//===- AnalysisManager.h - Analysis Management Infrastructure ---*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PASS_ANALYSISMANAGER_H
#define MLIR_PASS_ANALYSISMANAGER_H

#include "mlir/IR/Module.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/TypeName.h"

namespace mlir {
/// A special type used by analyses to provide an address that identifies a
/// particular analysis set or a concrete analysis type.
using AnalysisID = ClassID;

//===----------------------------------------------------------------------===//
// Analysis Preservation and Concept Modeling
//===----------------------------------------------------------------------===//

namespace detail {
/// A utility class to represent the analyses that are known to be preserved.
class PreservedAnalyses {
public:
  /// Mark all analyses as preserved.
  void preserveAll() { preservedIDs.insert(&allAnalysesID); }

  /// Returns true if all analyses were marked preserved.
  bool isAll() const { return preservedIDs.count(&allAnalysesID); }

  /// Returns true if no analyses were marked preserved.
  bool isNone() const { return preservedIDs.empty(); }

  /// Preserve the given analyses.
  template <typename AnalysisT> void preserve() {
    preserve(AnalysisID::getID<AnalysisT>());
  }
  template <typename AnalysisT, typename AnalysisT2, typename... OtherAnalysesT>
  void preserve() {
    preserve<AnalysisT>();
    preserve<AnalysisT2, OtherAnalysesT...>();
  }
  void preserve(const AnalysisID *id) { preservedIDs.insert(id); }

  /// Returns if the given analysis has been marked as preserved. Note that this
  /// simply checks for the presence of a given analysis ID and should not be
  /// used as a general preservation checker.
  template <typename AnalysisT> bool isPreserved() const {
    return isPreserved(AnalysisID::getID<AnalysisT>());
  }
  bool isPreserved(const AnalysisID *id) const {
    return preservedIDs.count(id);
  }

private:
  /// An identifier used to represent all potential analyses.
  constexpr static AnalysisID allAnalysesID = {};

  /// The set of analyses that are known to be preserved.
  SmallPtrSet<const void *, 2> preservedIDs;
};

namespace analysis_impl {
/// Trait to check if T provides a static 'isInvalidated' method.
template <typename T, typename... Args>
using has_is_invalidated = decltype(std::declval<T &>().isInvalidated(
    std::declval<const PreservedAnalyses &>()));

/// Implementation of 'isInvalidated' if the analysis provides a definition.
template <typename AnalysisT>
std::enable_if_t<is_detected<has_is_invalidated, AnalysisT>::value, bool>
isInvalidated(AnalysisT &analysis, const PreservedAnalyses &pa) {
  return analysis.isInvalidated(pa);
}
/// Default implementation of 'isInvalidated'.
template <typename AnalysisT>
std::enable_if_t<!is_detected<has_is_invalidated, AnalysisT>::value, bool>
isInvalidated(AnalysisT &analysis, const PreservedAnalyses &pa) {
  return !pa.isPreserved<AnalysisT>();
}
} // end namespace analysis_impl

/// The abstract polymorphic base class representing an analysis.
struct AnalysisConcept {
  virtual ~AnalysisConcept() = default;

  /// A hook used to query analyses for invalidation. Given a preserved analysis
  /// set, returns true if it should truly be invalidated. This allows for more
  /// fine-tuned invalidation in cases where an analysis wasn't explicitly
  /// marked preserved, but may be preserved(or invalidated) based upon other
  /// properties such as analyses sets.
  virtual bool isInvalidated(const PreservedAnalyses &pa) = 0;
};

/// A derived analysis model used to hold a specific analysis object.
template <typename AnalysisT> struct AnalysisModel : public AnalysisConcept {
  template <typename... Args>
  explicit AnalysisModel(Args &&... args)
      : analysis(std::forward<Args>(args)...) {}

  /// A hook used to query analyses for invalidation.
  bool isInvalidated(const PreservedAnalyses &pa) final {
    return analysis_impl::isInvalidated(analysis, pa);
  }

  /// The actual analysis object.
  AnalysisT analysis;
};

/// This class represents a cache of analyses for a single operation. All
/// computation, caching, and invalidation of analyses takes place here.
class AnalysisMap {
  /// A mapping between an analysis id and an existing analysis instance.
  using ConceptMap =
      DenseMap<const AnalysisID *, std::unique_ptr<AnalysisConcept>>;

  /// Utility to return the name of the given analysis class.
  template <typename AnalysisT> static StringRef getAnalysisName() {
    StringRef name = llvm::getTypeName<AnalysisT>();
    if (!name.consume_front("mlir::"))
      name.consume_front("(anonymous namespace)::");
    return name;
  }

public:
  explicit AnalysisMap(Operation *ir) : ir(ir) {}

  /// Get an analysis for the current IR unit, computing it if necessary.
  template <typename AnalysisT> AnalysisT &getAnalysis(PassInstrumentor *pi) {
    auto *id = AnalysisID::getID<AnalysisT>();

    typename ConceptMap::iterator it;
    bool wasInserted;
    std::tie(it, wasInserted) = analyses.try_emplace(id);

    // If we don't have a cached analysis for this function, compute it directly
    // and add it to the cache.
    if (wasInserted) {
      if (pi)
        pi->runBeforeAnalysis(getAnalysisName<AnalysisT>(), id, ir);

      it->second = std::make_unique<AnalysisModel<AnalysisT>>(ir);

      if (pi)
        pi->runAfterAnalysis(getAnalysisName<AnalysisT>(), id, ir);
    }
    return static_cast<AnalysisModel<AnalysisT> &>(*it->second).analysis;
  }

  /// Get a cached analysis instance if one exists, otherwise return null.
  template <typename AnalysisT>
  Optional<std::reference_wrapper<AnalysisT>> getCachedAnalysis() const {
    auto res = analyses.find(AnalysisID::getID<AnalysisT>());
    if (res == analyses.end())
      return llvm::None;
    return {static_cast<AnalysisModel<AnalysisT> &>(*res->second).analysis};
  }

  /// Returns the operation that this analysis map represents.
  Operation *getOperation() const { return ir; }

  /// Clear any held analyses.
  void clear() { analyses.clear(); }

  /// Invalidate any cached analyses based upon the given set of preserved
  /// analyses.
  void invalidate(const PreservedAnalyses &pa) {
    // Remove any analyses that were invalidated.
    for (auto it = analyses.begin(), e = analyses.end(); it != e;) {
      auto curIt = it++;
      if (curIt->second->isInvalidated(pa))
        analyses.erase(curIt);
    }
  }

private:
  Operation *ir;
  ConceptMap analyses;
};

/// An analysis map that contains a map for the current operation, and a set of
/// maps for any child operations.
struct NestedAnalysisMap {
  NestedAnalysisMap(Operation *op) : analyses(op) {}

  /// Get the operation for this analysis map.
  Operation *getOperation() const { return analyses.getOperation(); }

  /// Invalidate any non preserved analyses.
  void invalidate(const PreservedAnalyses &pa);

  /// The cached analyses for nested operations.
  DenseMap<Operation *, std::unique_ptr<NestedAnalysisMap>> childAnalyses;

  /// The analyses for the owning module.
  detail::AnalysisMap analyses;
};
} // namespace detail

//===----------------------------------------------------------------------===//
// Analysis Management
//===----------------------------------------------------------------------===//
class ModuleAnalysisManager;

/// This class represents an analysis manager for a particular operation
/// instance. It is used to manage and cache analyses on the operation as well
/// as those for child operations, via nested AnalysisManager instances
/// accessible via 'slice'. This class is intended to be passed around by value,
/// and cannot be constructed directly.
class AnalysisManager {
  using ParentPointerT =
      PointerUnion<const ModuleAnalysisManager *, const AnalysisManager *>;

public:
  using PreservedAnalyses = detail::PreservedAnalyses;

  // Query for a cached analysis on the given parent operation. The analysis may
  // not exist and if it does it may be out-of-date.
  template <typename AnalysisT>
  Optional<std::reference_wrapper<AnalysisT>>
  getCachedParentAnalysis(Operation *parentOp) const {
    ParentPointerT curParent = parent;
    while (auto *parentAM = curParent.dyn_cast<const AnalysisManager *>()) {
      if (parentAM->impl->getOperation() == parentOp)
        return parentAM->getCachedAnalysis<AnalysisT>();
      curParent = parentAM->parent;
    }
    return None;
  }

  // Query for the given analysis for the current operation.
  template <typename AnalysisT> AnalysisT &getAnalysis() {
    return impl->analyses.getAnalysis<AnalysisT>(getPassInstrumentor());
  }

  // Query for a cached entry of the given analysis on the current operation.
  template <typename AnalysisT>
  Optional<std::reference_wrapper<AnalysisT>> getCachedAnalysis() const {
    return impl->analyses.getCachedAnalysis<AnalysisT>();
  }

  /// Query for a analysis of a child operation, constructing it if necessary.
  template <typename AnalysisT> AnalysisT &getChildAnalysis(Operation *op) {
    return slice(op).template getAnalysis<AnalysisT>();
  }

  /// Query for a cached analysis of a child operation, or return null.
  template <typename AnalysisT>
  Optional<std::reference_wrapper<AnalysisT>>
  getCachedChildAnalysis(Operation *op) const {
    assert(op->getParentOp() == impl->getOperation());
    auto it = impl->childAnalyses.find(op);
    if (it == impl->childAnalyses.end())
      return llvm::None;
    return it->second->analyses.getCachedAnalysis<AnalysisT>();
  }

  /// Get an analysis manager for the given child operation.
  AnalysisManager slice(Operation *op);

  /// Invalidate any non preserved analyses,
  void invalidate(const PreservedAnalyses &pa) { impl->invalidate(pa); }

  /// Clear any held analyses.
  void clear() {
    impl->analyses.clear();
    impl->childAnalyses.clear();
  }

  /// Returns a pass instrumentation object for the current operation. This
  /// value may be null.
  PassInstrumentor *getPassInstrumentor() const;

private:
  AnalysisManager(const AnalysisManager *parent,
                  detail::NestedAnalysisMap *impl)
      : parent(parent), impl(impl) {}
  AnalysisManager(const ModuleAnalysisManager *parent,
                  detail::NestedAnalysisMap *impl)
      : parent(parent), impl(impl) {}

  /// A reference to the parent analysis manager, or the top-level module
  /// analysis manager.
  ParentPointerT parent;

  /// A reference to the impl analysis map within the parent analysis manager.
  detail::NestedAnalysisMap *impl;

  /// Allow access to the constructor.
  friend class ModuleAnalysisManager;
};

/// An analysis manager class specifically for the top-level module operation.
/// This class contains the memory allocations for all nested analysis managers,
/// and provides an anchor point. This is necessary because AnalysisManager is
/// designed to be a thin wrapper around an existing analysis map instance.
class ModuleAnalysisManager {
public:
  ModuleAnalysisManager(ModuleOp module, PassInstrumentor *passInstrumentor)
      : analyses(module), passInstrumentor(passInstrumentor) {}
  ModuleAnalysisManager(const ModuleAnalysisManager &) = delete;
  ModuleAnalysisManager &operator=(const ModuleAnalysisManager &) = delete;

  /// Returns a pass instrumentation object for the current module. This value
  /// may be null.
  PassInstrumentor *getPassInstrumentor() const { return passInstrumentor; }

  /// Returns an analysis manager for the current top-level module.
  operator AnalysisManager() { return AnalysisManager(this, &analyses); }

private:
  /// The analyses for the owning module.
  detail::NestedAnalysisMap analyses;

  /// An optional instrumentation object.
  PassInstrumentor *passInstrumentor;
};

} // end namespace mlir

#endif // MLIR_PASS_ANALYSISMANAGER_H
