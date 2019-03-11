//===- PassTiming.h ---------------------------------------------*- C++ -*-===//
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
// Pass and analysis execution timing instrumentation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PASS_PASSTIMING_H_
#define MLIR_PASS_PASSTIMING_H_

#include "mlir/Pass/PassInstrumentation.h"
#include "llvm/ADT/MapVector.h"

namespace llvm {
class Timer;
} // end namespace llvm

namespace mlir {
class Module;
class Pass;

class PassTiming : public PassInstrumentation {
public:
  enum DisplayMode {
    // In this mode the results are displayed in a list sorted by total time,
    // with each pass/analysis instance aggregated into one unique result.
    List,

    // In this mode the results are displayed in a nested pipeline view that
    // mirrors the internal pass pipeline that is being executed in the pass
    // manager.
    Pipeline,
  };

  PassTiming(DisplayMode displayMode);
  ~PassTiming();

  /// Print and clear the timing results.
  void print();

private:
  /// Setup the instrumentation hooks.
  void runBeforePass(Pass *pass, const llvm::Any &) override {
    startPassTimer(pass);
  }
  void runAfterPass(Pass *pass, const llvm::Any &) override {
    stopPassTimer(pass);
  }
  void runAfterPassFailed(Pass *pass, const llvm::Any &) override {
    stopPassTimer(pass);
  }
  void runBeforeAnalysis(llvm::StringRef name, AnalysisID *id,
                         const llvm::Any &) override {
    startAnalysisTimer(name, id);
  }
  void runAfterAnalysis(llvm::StringRef name, AnalysisID *id,
                        const llvm::Any &) override {
    stopAnalysisTimer(name, id);
  }

  /// Start a new timer for the given pass.
  void startPassTimer(Pass *pass);

  /// Stop a timer for the given pass.
  void stopPassTimer(Pass *pass);

  /// Start a new timer for the given analysis.
  void startAnalysisTimer(llvm::StringRef name, AnalysisID *id);

  /// Stop a timer for the given analysis.
  void stopAnalysisTimer(llvm::StringRef name, AnalysisID *id);

  /// Print the timing result in list mode.
  void printResultsAsList(llvm::raw_ostream &os);

  /// Print the timing result in pipeline mode.
  void printResultsAsPipeline(llvm::raw_ostream &os);

  /// Mapping between pass and a respective timer.
  llvm::MapVector<Pass *, std::unique_ptr<llvm::Timer>> passTimers;

  /// Mapping between [analysis id, pass] and a respective timer.
  llvm::DenseMap<std::pair<AnalysisID *, Pass *>, std::unique_ptr<llvm::Timer>>
      analysisTimers;

  /// A pointer to the currently active pass, or null.
  Pass *activePass = nullptr;

  /// The display mode to use when printing the timing results.
  DisplayMode displayMode;
};

} // end namespace mlir

#endif // MLIR_PASS_PASSTIMING_H_
