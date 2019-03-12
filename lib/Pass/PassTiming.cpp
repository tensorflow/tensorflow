//===- PassTiming.cpp -----------------------------------------------------===//
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

#include "PassDetail.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Timer.h"

using namespace mlir;
using namespace mlir::detail;

constexpr llvm::StringLiteral kPassTimingDescription =
    "... Pass execution timing report ...";

/// Utility function to return if a pass refers to an adaptor pass. Adaptor
/// passes are those that internally execute a pipeline, such as the
/// ModuleToFunctionPassAdaptor.
static bool isAdaptorPass(Pass *pass) {
  return isa<ModuleToFunctionPassAdaptor>(pass);
}

namespace {
struct PassTiming : public PassInstrumentation {
  PassTiming(PassTimingDisplayMode displayMode) : displayMode(displayMode) {}
  ~PassTiming() { print(); }

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

  /// Print and clear the timing results.
  void print();

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
  PassTimingDisplayMode displayMode;
};
} // end anonymous namespace

/// Print out the current timing information.
void PassTiming::print() {
  // Don't print anything if there is no timing data.
  if (passTimers.empty() && analysisTimers.empty())
    return;

  switch (displayMode) {
  case PassTimingDisplayMode::List:
    printResultsAsList(*llvm::CreateInfoOutputFile());
    break;
  case PassTimingDisplayMode::Pipeline:
    printResultsAsPipeline(*llvm::CreateInfoOutputFile());
    break;
  }

  // Reset and clear the timers.
  for (auto &timerPair : passTimers)
    timerPair.second->clear();
  for (auto &timerPair : analysisTimers)
    timerPair.second->clear();
  passTimers.clear();
  analysisTimers.clear();
}

/// Start a new timer for the given pass.
void PassTiming::startPassTimer(Pass *pass) {
  std::unique_ptr<llvm::Timer> &timer = passTimers[pass];

  /// If the timer doesn't exist then create a new one.
  if (!timer) {
    auto passName = pass->getName();
    timer.reset(new llvm::Timer(passName, passName));
  }

  activePass = pass;
  timer->startTimer();
}

/// Stop a timer for the given pass.
void PassTiming::stopPassTimer(Pass *pass) {
  std::unique_ptr<llvm::Timer> &timer = passTimers[pass];
  assert(timer && "expected valid timer to stop");
  timer->stopTimer();
  activePass = nullptr;
}

/// Start a new timer for the given analysis.
void PassTiming::startAnalysisTimer(llvm::StringRef name, AnalysisID *id) {
  auto &timer = analysisTimers[std::make_pair(id, activePass)];

  /// If the timer doesn't exist then create a new one.
  if (!timer)
    timer.reset(new llvm::Timer(name, Twine("(A) " + name).str()));
  timer->startTimer();
}

/// Stop a timer for the given analysis.
void PassTiming::stopAnalysisTimer(llvm::StringRef name, AnalysisID *id) {
  auto &timer = analysisTimers[std::make_pair(id, activePass)];
  assert(timer && "expected a valid timer to stop");
  timer->stopTimer();
}

/// Print the timing result in list mode.
void PassTiming::printResultsAsList(llvm::raw_ostream &os) {
  // Build a map of timer records uniqued by the timer name.
  llvm::StringMap<llvm::TimeRecord> records;
  auto addTimer = [&](llvm::Timer *timer) {
    auto it = records.try_emplace(timer->getName(), timer->getTotalTime());
    if (!it.second)
      it.first->second += timer->getTotalTime();
  };

  // Add all non-adaptor classes to the time records.
  for (auto &timerPair : passTimers)
    if (!isAdaptorPass(timerPair.first))
      addTimer(timerPair.second.get());

  // Add the analysis timers.
  for (auto &timerPair : analysisTimers)
    addTimer(timerPair.second.get());

  // Create a timer group for the records and print it out.
  llvm::TimerGroup timerGroup("pass", kPassTimingDescription, records);
  timerGroup.print(os);
}

/// Utility to print the heading information for the pipeline timer.
/// Note: This is a replication of the header generated by llvm::TimerGroup.
static void printPipelineTimerHeader(llvm::raw_ostream &os,
                                     llvm::TimeRecord &pipelineTotal) {
  os << "===" << std::string(73, '-') << "===\n";
  // Figure out how many spaces to description name.
  unsigned Padding = (80 - kPassTimingDescription.size()) / 2;
  os.indent(Padding) << kPassTimingDescription << '\n';
  os << "===" << std::string(73, '-') << "===\n";

  // Print the total time.
  os << llvm::format(
      "  Total Execution Time: %5.4f seconds (%5.4f wall clock)\n\n",
      pipelineTotal.getProcessTime(), pipelineTotal.getWallTime());

  // Add the headers for each time section.
  if (pipelineTotal.getUserTime())
    os << "   ---User Time---";
  if (pipelineTotal.getSystemTime())
    os << "   --System Time--";
  if (pipelineTotal.getProcessTime())
    os << "   --User+System--";
  os << "   ---Wall Time---";
  if (pipelineTotal.getMemUsed())
    os << "  ---Mem---";
  os << "  --- Name ---\n";
}

/// Print the timing result in pipeline mode.
void PassTiming::printResultsAsPipeline(llvm::raw_ostream &os) {
  // Collect the total time information for each of the non-adaptor passes.
  llvm::TimeRecord pipelineTotal;
  for (auto &timerPair : passTimers)
    if (!isAdaptorPass(timerPair.first))
      pipelineTotal += timerPair.second->getTotalTime();

  // Collect the analysis time records for each pass.
  llvm::DenseMap<Pass *, std::vector<llvm::Timer *>> passAnalyses;
  for (auto &timerPair : analysisTimers) {
    passAnalyses[timerPair.first.second].push_back(timerPair.second.get());
    pipelineTotal += timerPair.second->getTotalTime();
  }
  // Sort each of the analysis timers.
  for (auto &analysisPair : passAnalyses) {
    llvm::array_pod_sort(
        analysisPair.second.begin(), analysisPair.second.end(),
        [](llvm::Timer *const *lhsTimer, llvm::Timer *const *rhsTimer) -> int {
          return (*lhsTimer)->getDescription().compare(
              (*rhsTimer)->getDescription());
        });
  }

  // Print out timing header.
  printPipelineTimerHeader(os, pipelineTotal);

  // Print the formatted timing record.
  unsigned currentIndent = 0;
  auto printTimer = [&](llvm::StringRef name, llvm::TimeRecord timeRecord) {
    timeRecord.print(pipelineTotal, os);
    os.indent(currentIndent) << name;
    os << "\n";
  };

  // Utility to print the timing information for a pass and its analyses.
  auto printPassTimer = [&](Pass *pass, llvm::Timer *passTimer) {
    printTimer(passTimer->getDescription(), passTimer->getTotalTime());

    // Print the computed analyses for this pass.
    currentIndent += 2;
    for (llvm::Timer *timer : passAnalyses[pass])
      printTimer(timer->getDescription(), timer->getTotalTime());
    currentIndent -= 2;
  };

  // Print the total execution time.
  for (auto it = passTimers.begin(), e = passTimers.end(); it != e;) {
    // Handle a ModuleToFunctionAdaptor pass.
    if (isa<ModuleToFunctionPassAdaptor>(it->first)) {
      // Print the time for this adaptor as the accumulation of each of the
      // nested function passes.
      llvm::TimeRecord total;
      for (auto fpIt = ++it; fpIt != e && isa<FunctionPassBase>(fpIt->first);) {
        total += fpIt->second->getTotalTime();
        for (llvm::Timer *timer : passAnalyses[(fpIt++)->first])
          total += timer->getTotalTime();
      }
      printTimer("Function Pipeline", total);

      // Update the indent and print the time for each of the function passes
      // within the pipeline.
      currentIndent += 2;
      for (; it != e && isa<FunctionPassBase>(it->first); ++it)
        printPassTimer(it->first, it->second.get());
      currentIndent -= 2;
      continue;
    }

    // Otherwise, we print the pass timer directly.
    printPassTimer(it->first, it->second.get());
    ++it;
  }

  printTimer("Total\n", pipelineTotal);
  os.flush();
}

//===----------------------------------------------------------------------===//
// PassManager
//===----------------------------------------------------------------------===//

/// Add an instrumentation to time the execution of passes and the computation
/// of analyses.
void PassManager::enableTiming(PassTimingDisplayMode displayMode) {
  // Check if pass timing is already enabled.
  if (passTiming)
    return;
  addInstrumentation(new PassTiming(displayMode));
  passTiming = true;
}
