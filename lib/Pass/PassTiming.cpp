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
#include <chrono>

using namespace mlir;
using namespace mlir::detail;

constexpr llvm::StringLiteral kPassTimingDescription =
    "... Pass execution timing report ...";

namespace {
struct Timer {
  explicit Timer(std::string &&name) : name(std::move(name)) {}

  /// Start the timer.
  void start() { startTime = std::chrono::system_clock::now(); }

  /// Stop the timer.
  void stop() { total += (std::chrono::system_clock::now() - startTime); }

  /// Get or create a child timer with the provided name and id.
  Timer *getChildTimer(const void *id,
                       std::function<std::string()> &&nameBuilder) {
    auto &child = children[id];
    if (!child)
      child.reset(new Timer(nameBuilder()));
    return child.get();
  }

  /// Returns the total time for this timer in seconds.
  double getTotalTime() {
    // If the total has a count, then we directly compute the seconds.
    if (total.count()) {
      return std::chrono::duration_cast<std::chrono::duration<double>>(total)
          .count();
    }

    // Otheriwse, accumulate the timing from each of the children.
    double totalTime = 0.0;
    for (auto &child : children)
      totalTime += child.second->getTotalTime();
    return totalTime;
  }

  /// Raw timing information.
  std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
  std::chrono::nanoseconds total = std::chrono::nanoseconds(0);

  /// A map of unique identifiers to child timers.
  llvm::MapVector<const void *, std::unique_ptr<Timer>> children;

  /// A descriptive name for this timer.
  std::string name;
};

struct PassTiming : public PassInstrumentation {
  PassTiming(PassTimingDisplayMode displayMode) : displayMode(displayMode) {}
  ~PassTiming() { print(); }

  /// Setup the instrumentation hooks.
  void runBeforePass(Pass *pass, const llvm::Any &) override {
    startPassTimer(pass);
  }
  void runAfterPass(Pass *pass, const llvm::Any &) override;
  void runAfterPassFailed(Pass *pass, const llvm::Any &ir) override {
    runAfterPass(pass, ir);
  }
  void runBeforeAnalysis(llvm::StringRef name, AnalysisID *id,
                         const llvm::Any &) override {
    startAnalysisTimer(name, id);
  }
  void runAfterAnalysis(llvm::StringRef, AnalysisID *,
                        const llvm::Any &) override;

  /// Print and clear the timing results.
  void print();

  /// Start a new timer for the given pass.
  void startPassTimer(Pass *pass);

  /// Start a new timer for the given analysis.
  void startAnalysisTimer(llvm::StringRef name, AnalysisID *id);

  /// Stop a pass timer.
  void stopPassTimer(Pass *pass);

  /// Stop the last active timer.
  void stopTimer();

  /// Print the timing result in list mode.
  void printResultsAsList(llvm::raw_ostream &os, double totalTime);

  /// Print the timing result in pipeline mode.
  void printResultsAsPipeline(llvm::raw_ostream &os, double totalTime);

  /// Returns a timer for the provided identifier and name.
  Timer *getTimer(const void *id, std::function<std::string()> &&nameBuilder) {
    if (activeTimers.empty())
      return rootTimer.getChildTimer(id, std::move(nameBuilder));
    return activeTimers.back()->getChildTimer(id, std::move(nameBuilder));
  }

  /// The root top level timer.
  Timer rootTimer = Timer("root");

  /// A stack of the currently active pass timers.
  SmallVector<Timer *, 4> activeTimers;

  /// The display mode to use when printing the timing results.
  PassTimingDisplayMode displayMode;
};
} // end anonymous namespace

/// Start a new timer for the given pass.
void PassTiming::startPassTimer(Pass *pass) {
  Timer *timer = getTimer(pass, [pass] {
    if (isModuleToFunctionAdaptorPass(pass))
      return StringRef("Function Pipeline");
    return pass->getName();
  });
  activeTimers.push_back(timer);

  // We don't actually want to time the adaptor passes, they gather their total
  // from their held passes.
  if (!isAdaptorPass(pass))
    timer->start();
}

/// Start a new timer for the given analysis.
void PassTiming::startAnalysisTimer(llvm::StringRef name, AnalysisID *id) {
  Timer *timer = getTimer(id, [name] { return "(A) " + name.str(); });
  activeTimers.push_back(timer);
  timer->start();
}

/// Stop a pass timer.
void PassTiming::runAfterPass(Pass *pass, const llvm::Any &) {
  assert(!activeTimers.empty() && "expected active timer");
  Timer *timer = activeTimers.pop_back_val();

  // Adapator passes aren't timed directly, so we don't need to stop their
  // timers.
  if (!isAdaptorPass(pass))
    timer->stop();
}

/// Stop a timer.
void PassTiming::runAfterAnalysis(llvm::StringRef, AnalysisID *,
                                  const llvm::Any &) {
  assert(!activeTimers.empty() && "expected active timer");
  Timer *timer = activeTimers.pop_back_val();
  timer->stop();
}

/// Utility to print the timer heading information.
static void printTimerHeader(llvm::raw_ostream &os, double total) {
  os << "===" << std::string(73, '-') << "===\n";
  // Figure out how many spaces to description name.
  unsigned Padding = (80 - kPassTimingDescription.size()) / 2;
  os.indent(Padding) << kPassTimingDescription << '\n';
  os << "===" << std::string(73, '-') << "===\n";

  // Print the total time followed by the section headers.
  os << llvm::format("  Total Execution Time: %5.4f seconds\n\n", total);
  os << "   ---Wall Time---  --- Name ---\n";
}

/// Utility to print a single line entry in the timer output.
static void printTimeEntry(raw_ostream &os, unsigned indent, StringRef name,
                           double time, double totalTime) {
  os << llvm::format("  %7.4f (%5.1f%%)  ", time, 100.0 * time / totalTime);
  os.indent(indent) << name << "\n";
}

/// Print out the current timing information.
void PassTiming::print() {
  // Don't print anything if there is no timing data.
  if (rootTimer.children.empty())
    return;
  auto os = llvm::CreateInfoOutputFile();

  // Print the timer header.
  double totalTime = rootTimer.getTotalTime();
  printTimerHeader(*os, totalTime);

  // Defer to a specialized printer for each display mode.
  switch (displayMode) {
  case PassTimingDisplayMode::List:
    printResultsAsList(*os, totalTime);
    break;
  case PassTimingDisplayMode::Pipeline:
    printResultsAsPipeline(*os, totalTime);
    break;
  }
  printTimeEntry(*os, 0, "Total", totalTime, totalTime);
  os->flush();

  // Reset root timer.
  rootTimer.children.clear();
}

/// Print the timing result in list mode.
void PassTiming::printResultsAsList(llvm::raw_ostream &os, double totalTime) {
  llvm::StringMap<double> mergedTimings;

  std::function<void(Timer *)> addTimer = [&](Timer *timer) {
    // Check for timing information.
    if (timer->total.count())
      mergedTimings[timer->name] += timer->getTotalTime();
    for (auto &children : timer->children)
      addTimer(children.second.get());
  };

  // Add each of the top level timers.
  for (auto &topLevelTimer : rootTimer.children)
    addTimer(topLevelTimer.second.get());

  // Sort the timing information.
  std::vector<std::pair<StringRef, double>> timerNameAndTime;
  for (auto &it : mergedTimings)
    timerNameAndTime.emplace_back(it.first(), it.second);
  llvm::array_pod_sort(timerNameAndTime.begin(), timerNameAndTime.end(),
                       [](const std::pair<StringRef, double> *lhs,
                          const std::pair<StringRef, double> *rhs) {
                         return llvm::array_pod_sort_comparator<double>(
                             &rhs->second, &lhs->second);
                       });

  // Print the timing information sequentially.
  for (auto &timeData : timerNameAndTime)
    printTimeEntry(os, 0, timeData.first, timeData.second, totalTime);
}

/// Print the timing result in pipeline mode.
void PassTiming::printResultsAsPipeline(llvm::raw_ostream &os,
                                        double totalTime) {
  std::function<void(unsigned, Timer *)> printTimer = [&](unsigned indent,
                                                          Timer *timer) {
    // Check for timing information.
    printTimeEntry(os, indent, timer->name, timer->getTotalTime(), totalTime);
    for (auto &children : timer->children)
      printTimer(indent + 2, children.second.get());
  };

  // Print each of the top level timers.
  for (auto &topLevelTimer : rootTimer.children)
    printTimer(0, topLevelTimer.second.get());
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
