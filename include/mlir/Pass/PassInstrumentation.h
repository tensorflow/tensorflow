//===- PassInstrumentation.h ------------------------------------*- C++ -*-===//
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

#ifndef MLIR_PASS_PASSINSTRUMENTATION_H_
#define MLIR_PASS_PASSINSTRUMENTATION_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
struct AnalysisID;
class Pass;

namespace detail {
struct PassInstrumentorImpl;
} // end namespace detail

/// PassInstrumentation provdes several entry points into the pass manager
/// infrastructure. Instrumentations should be added directly to a PassManager
/// before running a pipeline.
class PassInstrumentation {
public:
  virtual ~PassInstrumentation() = 0;

  /// A callback to run before a pass is executed. This function takes a pointer
  /// to the pass to be executed, as well as an llvm::Any holding a pointer to
  /// the IR unit being transformed on.
  virtual void runBeforePass(Pass *pass, const llvm::Any &ir) {}

  /// A callback to run after a pass is successfully executed. This function
  /// takes a pointer to the pass to be executed, as well as an llvm::Any
  /// holding a pointer to the IR unit being transformed on.
  virtual void runAfterPass(Pass *pass, const llvm::Any &ir) {}

  /// A callback to run when a pass execution fails. This function takes a
  /// pointer to the pass that was being executed, as well as an llvm::Any
  /// holding a pointer to the IR unit that was being transformed. Note
  /// that the ir unit may be in an invalid state.
  virtual void runAfterPassFailed(Pass *pass, const llvm::Any &ir) {}

  /// A callback to run before an analysis is computed. This function takes the
  /// name of the analysis to be computed, its AnalysisID, as well as an
  /// llvm::Any holding a pointer to the IR unit being analyzed on.
  virtual void runBeforeAnalysis(llvm::StringRef name, AnalysisID *id,
                                 const llvm::Any &ir) {}

  /// A callback to run before an analysis is computed. This function takes the
  /// name of the analysis that was computed, its AnalysisID, as well as an
  /// llvm::Any holding a pointer to the IR unit that was analyzed.
  virtual void runAfterAnalysis(llvm::StringRef name, AnalysisID *id,
                                const llvm::Any &ir) {}
};

/// This class holds a collection of PassInstrumentation objects, and invokes
/// their respective call backs.
class PassInstrumentor {
public:
  PassInstrumentor();
  PassInstrumentor(PassInstrumentor &&) = delete;
  PassInstrumentor(const PassInstrumentor &) = delete;
  ~PassInstrumentor();

  /// See PassInstrumentation::runBeforePass for details.
  template <typename IRUnitT> void runBeforePass(Pass *pass, IRUnitT *ir) {
    runBeforePass(pass, llvm::Any(ir));
  }

  /// See PassInstrumentation::runAfterPass for details.
  template <typename IRUnitT> void runAfterPass(Pass *pass, IRUnitT *ir) {
    runAfterPass(pass, llvm::Any(ir));
  }

  /// See PassInstrumentation::runAfterPassFailed for details.
  template <typename IRUnitT> void runAfterPassFailed(Pass *pass, IRUnitT *ir) {
    runAfterPassFailed(pass, llvm::Any(ir));
  }

  /// See PassInstrumentation::runBeforeAnalysis for details.
  template <typename IRUnitT>
  void runBeforeAnalysis(llvm::StringRef name, AnalysisID *id, IRUnitT *ir) {
    runBeforeAnalysis(name, id, llvm::Any(ir));
  }

  /// See PassInstrumentation::runAfterAnalysis for details.
  template <typename IRUnitT>
  void runAfterAnalysis(llvm::StringRef name, AnalysisID *id, IRUnitT *ir) {
    runAfterAnalysis(name, id, llvm::Any(ir));
  }

  /// Add the given instrumentation to the collection. This takes ownership over
  /// the given pointer.
  void addInstrumentation(PassInstrumentation *pi);

private:
  /// See PassInstrumentation::runBeforePass for details.
  void runBeforePass(Pass *pass, const llvm::Any &ir);

  /// See PassInstrumentation::runAfterPass for details.
  void runAfterPass(Pass *pass, const llvm::Any &ir);

  /// See PassInstrumentation::runAfterPassFailed for details.
  void runAfterPassFailed(Pass *pass, const llvm::Any &ir);

  /// See PassInstrumentation::runBeforeAnalysis for details.
  void runBeforeAnalysis(llvm::StringRef name, AnalysisID *id,
                         const llvm::Any &ir);

  /// See PassInstrumentation::runAfterAnalysis for details.
  void runAfterAnalysis(llvm::StringRef name, AnalysisID *id,
                        const llvm::Any &ir);

  std::unique_ptr<detail::PassInstrumentorImpl> impl;
};

} // end namespace mlir

#endif // MLIR_PASS_PASSINSTRUMENTATION_H_
