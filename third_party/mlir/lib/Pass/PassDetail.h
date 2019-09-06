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

namespace mlir {
class OpPassManager;

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

/// An adaptor pass used to run operation passes over nested operations
/// synchronously on a single thread.
class OpToOpPassAdaptor : public OperationPass<OpToOpPassAdaptor> {
public:
  OpToOpPassAdaptor(std::unique_ptr<OpPassManager> mgr);
  OpToOpPassAdaptor(const OpToOpPassAdaptor &rhs);

  /// Run the held pipeline over all operations.
  void runOnOperation() override;

  /// Returns the nested pass manager for this adaptor.
  OpPassManager &getPassManager() { return *mgr; }

private:
  std::unique_ptr<OpPassManager> mgr;
};

/// An adaptor pass used to run operation passes over nested operations
/// asynchronously across multiple threads.
class OpToOpPassAdaptorParallel
    : public OperationPass<OpToOpPassAdaptorParallel> {
public:
  OpToOpPassAdaptorParallel(std::unique_ptr<OpPassManager> mgr);
  OpToOpPassAdaptorParallel(const OpToOpPassAdaptorParallel &rhs);

  /// Run the held pipeline over all operations.
  void runOnOperation() override;

  /// Returns the nested pass manager for this adaptor.
  OpPassManager &getPassManager() { return *mgr; }

private:
  // The main pass executor for this adaptor.
  std::unique_ptr<OpPassManager> mgr;

  // A set of executors, cloned from the main executor, that run asynchronously
  // on different threads.
  std::vector<OpPassManager> asyncExecutors;
};

/// Utility function to return if a pass refers to an OpToOpAdaptorPass
/// instance.
inline bool isOpToOpAdaptorPass(Pass *pass) {
  return isa<OpToOpPassAdaptorParallel>(pass) || isa<OpToOpPassAdaptor>(pass);
}

/// Utility function to return if a pass refers to an adaptor pass. Adaptor
/// passes are those that internally execute a pipeline, such as the
/// OpToOpPassAdaptor.
inline bool isAdaptorPass(Pass *pass) { return isOpToOpAdaptorPass(pass); }

/// Utility function to return the operation name that the given adaptor pass
/// operates on. Return None if the given pass is not an adaptor pass.
Optional<StringRef> getAdaptorPassOpName(Pass *pass);

} // end namespace detail
} // end namespace mlir
#endif // MLIR_PASS_PASSDETAIL_H_
