//===- PassDetail.h - MLIR Pass details -------------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_PASS_PASSDETAIL_H_
#define MLIR_PASS_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace detail {

//===----------------------------------------------------------------------===//
// Verifier Pass
//===----------------------------------------------------------------------===//

/// Pass to verify an operation and signal failure if necessary.
class VerifierPass : public OperationPass<VerifierPass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// OpToOpPassAdaptor
//===----------------------------------------------------------------------===//

/// A base class for Op-to-Op adaptor passes.
class OpToOpPassAdaptorBase {
public:
  OpToOpPassAdaptorBase(OpPassManager &&mgr);
  OpToOpPassAdaptorBase(const OpToOpPassAdaptorBase &rhs) = default;

  /// Merge the current pass adaptor into given 'rhs'.
  void mergeInto(OpToOpPassAdaptorBase &rhs);

  /// Returns the pass managers held by this adaptor.
  MutableArrayRef<OpPassManager> getPassManagers() { return mgrs; }

  /// Returns the adaptor pass name.
  std::string getName();

protected:
  // A set of adaptors to run.
  SmallVector<OpPassManager, 1> mgrs;
};

/// An adaptor pass used to run operation passes over nested operations
/// synchronously on a single thread.
class OpToOpPassAdaptor : public OperationPass<OpToOpPassAdaptor>,
                          public OpToOpPassAdaptorBase {
public:
  OpToOpPassAdaptor(OpPassManager &&mgr);

  /// Run the held pipeline over all operations.
  void runOnOperation() override;
};

/// An adaptor pass used to run operation passes over nested operations
/// asynchronously across multiple threads.
class OpToOpPassAdaptorParallel
    : public OperationPass<OpToOpPassAdaptorParallel>,
      public OpToOpPassAdaptorBase {
public:
  OpToOpPassAdaptorParallel(OpPassManager &&mgr);

  /// Run the held pipeline over all operations.
  void runOnOperation() override;

  /// Return the async pass managers held by this parallel adaptor.
  MutableArrayRef<SmallVector<OpPassManager, 1>> getParallelPassManagers() {
    return asyncExecutors;
  }

private:
  // A set of executors, cloned from the main executor, that run asynchronously
  // on different threads.
  SmallVector<SmallVector<OpPassManager, 1>, 8> asyncExecutors;
};

/// Utility function to convert the given class to the base adaptor it is an
/// adaptor pass, returns nullptr otherwise.
OpToOpPassAdaptorBase *getAdaptorPassBase(Pass *pass);

/// Utility function to return if a pass refers to an adaptor pass. Adaptor
/// passes are those that internally execute a pipeline.
inline bool isAdaptorPass(Pass *pass) {
  return isa<OpToOpPassAdaptorParallel>(pass) || isa<OpToOpPassAdaptor>(pass);
}

} // end namespace detail
} // end namespace mlir
#endif // MLIR_PASS_PASSDETAIL_H_
