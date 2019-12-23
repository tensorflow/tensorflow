//===- Configuration.h - Configuration object base classes ------*- C++ -*-===//
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
//
// The quantizer is relatively agnostic to source and target dialects, with
// the specific represented by configuration policy objects derived from
// classes in this file.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_QUANTIZER_SUPPORT_CONFIGURATION_H
#define MLIR_QUANTIZER_SUPPORT_CONFIGURATION_H

#include <functional>

#include "mlir/Dialect/QuantOps/QuantTypes.h"
#include "mlir/IR/Identifier.h"
#include "mlir/Quantizer/Support/ConstraintAnalysisGraph.h"
#include "mlir/Quantizer/Support/Metadata.h"
#include "mlir/Quantizer/Support/Rules.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/StringSet.h"

namespace mlir {
class Operation;

namespace quantizer {

class CAGSlice;

/// Defines quantization configuration for the target.
/// The settings here depend on a variety of details about the deployment
/// environment, although, where we have control over such things, we do
/// try to standardize as possible.
///
/// Non-const methods are used to setup the configuration. It is expected that
/// const instances/references are used post-build.
class TargetConfiguration {
public:
  static constexpr size_t MaxSchemeIndex = 31;
  using OpHandlerFn = std::function<void(Operation *op, CAGSlice &cag)>;

  TargetConfiguration(SolverContext &context);
  virtual ~TargetConfiguration() = default;

  /// Adds a candidate type, returning its ordinal.
  unsigned addCandidateType(quant::AnyQuantizedType quantizedType,
                            CandidateQuantizedType::Scheme scheme) {
    unsigned ordinal = candidateTypes.size();
    assert(allCandidateTypesMask.size() == ordinal);
    CandidateQuantizedType ct{ordinal, quantizedType, scheme};
    candidateTypes.push_back(ct);
    allCandidateTypesMask.push_back(true);
    return ordinal;
  }

  /// Gets a prototype scheme by index.
  const CandidateQuantizedType &getCandidateType(unsigned index) const {
    assert(index < candidateTypes.size());
    return candidateTypes[index];
  }

  ArrayRef<CandidateQuantizedType> getCandidateTypes() const {
    return candidateTypes;
  }

  /// Gets a mask of all enabled candidate types by ordinal.
  llvm::SmallBitVector getAllCandidateTypesMask() const {
    return allCandidateTypesMask;
  }

  /// Gets a mask with every candidate type except those in the given mask.
  llvm::SmallBitVector
  getCandidateTypeDisabledExceptMask(ArrayRef<unsigned> exceptOrdinals) const {
    llvm::SmallBitVector disabled(allCandidateTypesMask);
    for (unsigned ordinal : exceptOrdinals) {
      disabled.reset(ordinal);
    }
    return disabled;
  }

  /// Adds an op handler.
  template <typename OpTy>
  void addOpHandler(OpHandlerFn fn) {
    addOpHandlerByName(OpTy::getOperationName(), fn);
  }

  /// Adds an operation which requires statistics at its result nodes for
  /// best quantization performance. Note that the opName StringRef is
  /// expected to come from getOperationName() and be static.
  template <typename OpTy>
  void addRequireStatsOp() {
    addRequireStatsOpByName(OpTy::getOperationName());
  }

  /// Returns whether opName is a RequireStatsOp.
  bool isRequireStatsOp(Operation *op) const;

  /// Adds an op which does not mutate its values but may mutate its shape
  /// or combine its operands in an arbitrary way.
  /// Such ops are expected to have the same types for operands and results
  /// and must be capable of operating on storage types.
  template <typename OpTy>
  void addValueIdentityOp() {
    addValueIdentityOpByName(OpTy::getOperationName());
  }

  /// Handles the operation if a handler is defined for it.
  void handleOp(Operation *op, CAGSlice &cag) const;

  /// Finalizes the CAG after all anchors have been added.
  virtual void finalizeAnchors(CAGSlice &cag) const {}

  /// Whether an operand or result type is subject to analysis by this config.
  virtual bool isHandledType(Type t) const = 0;

protected:
  virtual void addValueIdentityOpByName(StringRef opName) = 0;
  void addOpHandlerByName(StringRef name, OpHandlerFn fn);

private:
  void addRequireStatsOpByName(StringRef opName);

  /// Vector of all candidate type constraints, indexed by ordinal.
  std::vector<CandidateQuantizedType> candidateTypes;

  // A SmallBoolVector with bits set for all known candidate types.
  llvm::SmallBitVector allCandidateTypesMask;

  /// Map of all op handlers.
  llvm::StringMap<OpHandlerFn> opHandlers;

  /// Names of operations which should have their results annotated with
  /// statistics.
  llvm::StringSet<> requireStatsOpNames;
};

} // namespace quantizer
} // namespace mlir

#endif // MLIR_QUANTIZER_SUPPORT_CONFIGURATION_H
