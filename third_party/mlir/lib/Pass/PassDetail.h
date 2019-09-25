//===- PassDetail.h - MLIR Pass details -------------------------*- C++ -*-===//
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
