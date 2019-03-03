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
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
/// A special type used by analyses to provide an address that identifies a
/// particular analysis set or a concrete analysis type.
struct AnalysisID {
  template <typename AnalysisT> static AnalysisID *getID() {
    static AnalysisID id;
    return &id;
  }
};

//===----------------------------------------------------------------------===//
// Analysis Preservation and Result Modeling
//===----------------------------------------------------------------------===//

namespace detail {
/// A utility class to represent the analyses that are known to be preserved.
class PreservedAnalyses {
public:
  /// Mark all analyses as preserved.
  void preserveAll() { preservedIDs.insert(&allAnalysesID); }

  /// Returns if all analyses were marked preserved.
  bool isAll() const { return preservedIDs.count(&allAnalysesID); }

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

/// This class represents a cache of analysis results for a single IR unit. All
/// computation, caching, and invalidation of analyses takes place here.
template <typename IRUnitT> class AnalysisResultMap {
  /// A mapping between an analysis id and an existing analysis instance.
  using ResultMap =
      DenseMap<const AnalysisID *, std::unique_ptr<AnalysisConcept>>;

public:
  explicit AnalysisResultMap(IRUnitT *ir) : ir(ir) {}

  /// Get an analysis for the current IR unit, computing it if necessary.
  template <typename AnalysisT> AnalysisT &getResult() {
    typename ResultMap::iterator it;
    bool wasInserted;
    std::tie(it, wasInserted) =
        results.try_emplace(AnalysisID::getID<AnalysisT>());

    // If we don't have a cached result for this function, compute it directly
    // and add it to the cache.
    if (wasInserted)
      it->second = llvm::make_unique<AnalysisModel<AnalysisT>>(ir);
    return static_cast<AnalysisModel<AnalysisT> &>(*it->second).analysis;
  }

  /// Get a cached analysis instance if one exists, otherwise return null.
  template <typename AnalysisT>
  llvm::Optional<std::reference_wrapper<AnalysisT>> getCachedResult() const {
    auto res = results.find(AnalysisID::getID<AnalysisT>());
    if (res == results.end())
      return llvm::None;
    return {static_cast<AnalysisModel<AnalysisT> &>(*res->second).analysis};
  }

  /// Returns the IR unit that this result map represents.
  IRUnitT *getIRUnit() { return ir; }
  const IRUnitT *getIRUnit() const { return ir; }

  /// Clear any held analysis results.
  void clear() { results.clear(); }

  /// Invalidate any cached analyses based upon the given set of preserved
  /// analyses.
  void invalidate(const detail::PreservedAnalyses &pa) {
    // If all analyses were preserved, then there is nothing to do here.
    if (pa.isAll())
      return;
    // TODO: Fine grain invalidation of analyses.
    clear();
  }

private:
  IRUnitT *ir;
  ResultMap results;
};

} // namespace detail

//===----------------------------------------------------------------------===//
// Analysis Management
//===----------------------------------------------------------------------===//

/// An analysis manager for a specific function instance. This class can only be
/// constructed from a ModuleAnalysisManager instance.
class FunctionAnalysisManager {
public:
  // Query for a cached analysis on the parent Module. The analysis may not
  // exist and if it does it may be stale.
  template <typename AnalysisT>
  llvm::Optional<std::reference_wrapper<AnalysisT>>
  getCachedModuleResult() const {
    return parentImpl->getCachedResult<AnalysisT>();
  }

  // Query for the given analysis for the current function.
  template <typename AnalysisT> AnalysisT &getResult() {
    return impl->getResult<AnalysisT>();
  }

  // Query for a cached entry of the given analysis on the current function.
  template <typename AnalysisT>
  llvm::Optional<std::reference_wrapper<AnalysisT>> getCachedResult() const {
    return impl->getCachedResult<AnalysisT>();
  }

  /// Invalidate any non preserved analyses,
  void invalidate(const detail::PreservedAnalyses &pa) { impl->invalidate(pa); }

  /// Clear any held analyses.
  void clear() { impl->clear(); }

private:
  FunctionAnalysisManager(const detail::AnalysisResultMap<Module> *parentImpl,
                          detail::AnalysisResultMap<Function> *impl)
      : parentImpl(parentImpl), impl(impl) {}

  /// A reference to the results map of the parent module within the owning
  /// analysis manager.
  const detail::AnalysisResultMap<Module> *parentImpl;

  /// A reference to the results map within the owning analysis manager.
  detail::AnalysisResultMap<Function> *impl;

  /// Allow access to the constructor.
  friend class ModuleAnalysisManager;
};

/// An analysis manager for a specific module instance.
class ModuleAnalysisManager {
public:
  ModuleAnalysisManager(Module *module) : moduleAnalyses(module) {}
  ModuleAnalysisManager(const ModuleAnalysisManager &) = delete;
  ModuleAnalysisManager &operator=(const ModuleAnalysisManager &) = delete;

  /// Query for the analysis of a function. The analysis is computed if it does
  /// not exist.
  template <typename AnalysisT>
  AnalysisT &getFunctionResult(Function *function) {
    return slice(function).getResult<AnalysisT>();
  }

  /// Query for the analysis of a module. The analysis is computed if it does
  /// not exist.
  template <typename AnalysisT> AnalysisT &getResult() {
    return moduleAnalyses.getResult<AnalysisT>();
  }

  /// Query for a cached analysis for the module, or return nullptr.
  template <typename AnalysisT>
  llvm::Optional<std::reference_wrapper<AnalysisT>> getCachedResult() const {
    return moduleAnalyses.getCachedResult<AnalysisT>();
  }

  /// Create an analysis slice for the given child function.
  FunctionAnalysisManager slice(Function *function);

  /// Invalidate any non preserved analyses.
  void invalidate(const detail::PreservedAnalyses &pa);

private:
  /// The cached analyses for functions within the current module.
  DenseMap<Function *, detail::AnalysisResultMap<Function>> functionAnalyses;

  /// The analyses for the owning module.
  detail::AnalysisResultMap<Module> moduleAnalyses;
};

} // end namespace mlir

#endif // MLIR_PASS_ANALYSISMANAGER_H
