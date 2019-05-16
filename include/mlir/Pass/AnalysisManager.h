//===- AnalysisManager.h - Analysis Management Infrastructure ---*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

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

/// The abstract polymorphic base class representing an analysis.
struct AnalysisConcept {
  virtual ~AnalysisConcept() = default;
};

/// A derived analysis model used to hold a specific analysis object.
template <typename AnalysisT> struct AnalysisModel : public AnalysisConcept {
  template <typename... Args>
  explicit AnalysisModel(Args &&... args)
      : analysis(std::forward<Args>(args)...) {}

  AnalysisT analysis;
};

/// This class represents a cache of analyses for a single IR unit. All
/// computation, caching, and invalidation of analyses takes place here.
template <typename IRUnitT> class AnalysisMap {
  /// A mapping between an analysis id and an existing analysis instance.
  using ConceptMap =
      llvm::DenseMap<const AnalysisID *, std::unique_ptr<AnalysisConcept>>;

  /// Utility to return the name of the given analysis class.
  template <typename AnalysisT> static llvm::StringRef getAnalysisName() {
    StringRef name = llvm::getTypeName<AnalysisT>();
    if (!name.consume_front("mlir::"))
      name.consume_front("(anonymous namespace)::");
    return name;
  }

public:
  explicit AnalysisMap(IRUnitT *ir) : ir(ir) {}

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

      it->second = llvm::make_unique<AnalysisModel<AnalysisT>>(ir);

      if (pi)
        pi->runAfterAnalysis(getAnalysisName<AnalysisT>(), id, ir);
    }
    return static_cast<AnalysisModel<AnalysisT> &>(*it->second).analysis;
  }

  /// Get a cached analysis instance if one exists, otherwise return null.
  template <typename AnalysisT>
  llvm::Optional<std::reference_wrapper<AnalysisT>> getCachedAnalysis() const {
    auto res = analyses.find(AnalysisID::getID<AnalysisT>());
    if (res == analyses.end())
      return llvm::None;
    return {static_cast<AnalysisModel<AnalysisT> &>(*res->second).analysis};
  }

  /// Returns the IR unit that this analysis map represents.
  IRUnitT *getIRUnit() { return ir; }
  const IRUnitT *getIRUnit() const { return ir; }

  /// Clear any held analyses.
  void clear() { analyses.clear(); }

  /// Invalidate any cached analyses based upon the given set of preserved
  /// analyses.
  void invalidate(const detail::PreservedAnalyses &pa) {
    // Remove any analyses not marked as preserved.
    for (auto it = analyses.begin(), e = analyses.end(); it != e;) {
      auto curIt = it++;
      if (!pa.isPreserved(curIt->first))
        analyses.erase(curIt);
    }
  }

private:
  IRUnitT *ir;
  ConceptMap analyses;
};

} // namespace detail

//===----------------------------------------------------------------------===//
// Analysis Management
//===----------------------------------------------------------------------===//
class ModuleAnalysisManager;

/// An analysis manager for a specific function instance. This class can only be
/// constructed from a ModuleAnalysisManager instance.
class FunctionAnalysisManager {
public:
  // Query for a cached analysis on the parent Module. The analysis may not
  // exist and if it does it may be stale.
  template <typename AnalysisT>
  llvm::Optional<std::reference_wrapper<AnalysisT>>
  getCachedModuleAnalysis() const;

  // Query for the given analysis for the current function.
  template <typename AnalysisT> AnalysisT &getAnalysis() {
    return impl->getAnalysis<AnalysisT>(getPassInstrumentor());
  }

  // Query for a cached entry of the given analysis on the current function.
  template <typename AnalysisT>
  llvm::Optional<std::reference_wrapper<AnalysisT>> getCachedAnalysis() const {
    return impl->getCachedAnalysis<AnalysisT>();
  }

  /// Invalidate any non preserved analyses,
  void invalidate(const detail::PreservedAnalyses &pa) {
    // If all analyses were preserved, then there is nothing to do here.
    if (pa.isAll())
      return;
    impl->invalidate(pa);
  }

  /// Clear any held analyses.
  void clear() { impl->clear(); }

  /// Returns a pass instrumentation object for the current function. This value
  /// may be null.
  PassInstrumentor *getPassInstrumentor() const;

private:
  FunctionAnalysisManager(const ModuleAnalysisManager *parent,
                          detail::AnalysisMap<Function> *impl)
      : parent(parent), impl(impl) {}

  /// A reference to the parent analysis manager.
  const ModuleAnalysisManager *parent;

  /// A reference to the impl analysis map within the owning analysis manager.
  detail::AnalysisMap<Function> *impl;

  /// Allow access to the constructor.
  friend class ModuleAnalysisManager;
};

/// An analysis manager for a specific module instance.
class ModuleAnalysisManager {
public:
  ModuleAnalysisManager(Module *module, PassInstrumentor *passInstrumentor)
      : moduleAnalyses(module), passInstrumentor(passInstrumentor) {}
  ModuleAnalysisManager(const ModuleAnalysisManager &) = delete;
  ModuleAnalysisManager &operator=(const ModuleAnalysisManager &) = delete;

  /// Query for the analysis of a function. The analysis is computed if it does
  /// not exist.
  template <typename AnalysisT>
  AnalysisT &getFunctionAnalysis(Function *function) {
    return slice(function).getAnalysis<AnalysisT>();
  }

  /// Query for a cached analysis of a child function, or return null.
  template <typename AnalysisT>
  llvm::Optional<std::reference_wrapper<AnalysisT>>
  getCachedFunctionAnalysis(Function *function) const {
    auto it = functionAnalyses.find(function);
    if (it == functionAnalyses.end())
      return llvm::None;
    return it->second->getCachedAnalysis<AnalysisT>();
  }

  /// Query for the analysis for the module. The analysis is computed if it does
  /// not exist.
  template <typename AnalysisT> AnalysisT &getAnalysis() {
    return moduleAnalyses.getAnalysis<AnalysisT>(getPassInstrumentor());
  }

  /// Query for a cached analysis for the module, or return null.
  template <typename AnalysisT>
  llvm::Optional<std::reference_wrapper<AnalysisT>> getCachedAnalysis() const {
    return moduleAnalyses.getCachedAnalysis<AnalysisT>();
  }

  /// Create an analysis slice for the given child function.
  FunctionAnalysisManager slice(Function *function);

  /// Invalidate any non preserved analyses.
  void invalidate(const detail::PreservedAnalyses &pa);

  /// Returns a pass instrumentation object for the current module. This value
  /// may be null.
  PassInstrumentor *getPassInstrumentor() const { return passInstrumentor; }

private:
  /// The cached analyses for functions within the current module.
  llvm::DenseMap<Function *, std::unique_ptr<detail::AnalysisMap<Function>>>
      functionAnalyses;

  /// The analyses for the owning module.
  detail::AnalysisMap<Module> moduleAnalyses;

  /// An optional instrumentation object.
  PassInstrumentor *passInstrumentor;
};

// Query for a cached analysis on the parent Module. The analysis may not exist
// and if it does it may be stale.
template <typename AnalysisT>
llvm::Optional<std::reference_wrapper<AnalysisT>>
FunctionAnalysisManager::getCachedModuleAnalysis() const {
  return parent->getCachedAnalysis<AnalysisT>();
}

} // end namespace mlir

#endif // MLIR_PASS_ANALYSISMANAGER_H
