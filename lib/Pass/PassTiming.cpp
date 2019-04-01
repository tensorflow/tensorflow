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
#include "llvm/Support/Threading.h"
#include <chrono>

using namespace mlir;
using namespace mlir::detail;

constexpr llvm::StringLiteral kPassTimingDescription =
    "... Pass execution timing report ...";

namespace {
/// Simple record class to record timing information.
struct TimeRecord {
  TimeRecord(double wall = 0.0, double user = 0.0) : wall(wall), user(user) {}

  TimeRecord &operator+=(const TimeRecord &other) {
    wall += other.wall;
    user += other.user;
    return *this;
  }

  /// Print the current time record to 'os', with a breakdown showing
  /// contributions to the give 'total' time record.
  void print(raw_ostream &os, const TimeRecord &total) {
    if (total.user != total.wall)
      os << llvm::format("  %7.4f (%5.1f%%)  ", user,
                         100.0 * user / total.user);
    os << llvm::format("  %7.4f (%5.1f%%)  ", wall, 100.0 * wall / total.wall);
  }

  double wall, user;
};

struct Timer {
  explicit Timer(std::string &&name) : name(std::move(name)) {}

  /// Start the timer.
  void start() { startTime = std::chrono::system_clock::now(); }

  /// Stop the timer.
  void stop() {
    auto newTime = std::chrono::system_clock::now() - startTime;
    wallTime += newTime;
    userTime += newTime;
  }

  /// Get or create a child timer with the provided name and id.
  Timer *getChildTimer(const void *id,
                       std::function<std::string()> &&nameBuilder) {
    auto &child = children[id];
    if (!child)
      child.reset(new Timer(nameBuilder()));
    return child.get();
  }

  /// Returns the total time for this timer in seconds.
  TimeRecord getTotalTime() {
    // If we have a valid wall time, then we directly compute the seconds.
    if (wallTime.count()) {
      return TimeRecord(
          std::chrono::duration_cast<std::chrono::duration<double>>(wallTime)
              .count(),
          std::chrono::duration_cast<std::chrono::duration<double>>(userTime)
              .count());
    }

    // Otheriwse, accumulate the timing from each of the children.
    TimeRecord totalTime;
    for (auto &child : children)
      totalTime += child.second->getTotalTime();
    return totalTime;
  }

  /// A map of unique identifiers to child timers.
  using ChildrenMap = llvm::MapVector<const void *, std::unique_ptr<Timer>>;

  /// Merge the timing data from 'other' into this timer.
  void merge(Timer &&other) {
    if (wallTime < other.wallTime)
      wallTime = other.wallTime;
    userTime += other.userTime;
    mergeChildren(std::move(other.children), /*isStructural=*/false);
  }

  /// Merge the timer chilren in 'otherChildren' with the children of this
  /// timer. If 'isStructural' is true, the children are merged lexographically
  /// and 'otherChildren' must have the same number of elements as the children
  /// of this timer. Otherwise, the timer children are merged based upon the
  /// given timer key.
  void mergeChildren(ChildrenMap &&otherChildren, bool isStructural) {
    // Check for an empty children list.
    if (children.empty()) {
      children = std::move(otherChildren);
      return;
    }

    if (isStructural) {
      // If this is a structural merge, the number of children must be the same.
      assert(children.size() == otherChildren.size() &&
             "structural merge requires the same number of children");
      auto it = children.begin(), otherIt = otherChildren.begin();
      for (auto e = children.end(); it != e; ++it, ++otherIt)
        it->second->merge(std::move(*otherIt->second));
      return;
    }

    // Otherwise, we merge based upon the child timers key.
    for (auto &otherChild : otherChildren) {
      auto &child = children[otherChild.first];
      if (!child)
        child = std::move(otherChild.second);
      else
        child->merge(std::move(*otherChild.second));
    }
  }

  /// Raw timing information.
  std::chrono::time_point<std::chrono::system_clock> startTime;
  std::chrono::nanoseconds wallTime = std::chrono::nanoseconds(0);
  std::chrono::nanoseconds userTime = std::chrono::nanoseconds(0);

  /// A map of unique identifiers to child timers.
  ChildrenMap children;

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
  void printResultsAsList(raw_ostream &os, Timer *root, TimeRecord totalTime);

  /// Print the timing result in pipeline mode.
  void printResultsAsPipeline(raw_ostream &os, Timer *root,
                              TimeRecord totalTime);

  /// Returns a timer for the provided identifier and name.
  Timer *getTimer(const void *id, std::function<std::string()> &&nameBuilder) {
    auto tid = llvm::get_threadid();

    // If there is no active timer then add to the root timer.
    auto &activeTimers = activeThreadTimers[tid];
    if (activeTimers.empty()) {
      auto &rootTimer = rootTimers[tid];
      if (!rootTimer)
        rootTimer.reset(new Timer("root"));
      auto *timer = rootTimer->getChildTimer(id, std::move(nameBuilder));
      activeTimers.push_back(timer);
      return timer;
    }

    // Otherwise, add this to the active timer.
    auto timer = activeTimers.back()->getChildTimer(id, std::move(nameBuilder));
    activeTimers.push_back(timer);
    return timer;
  }

  /// The root top level timers for each thread.
  DenseMap<uint64_t, std::unique_ptr<Timer>> rootTimers;

  /// A stack of the currently active pass timers per thread.
  DenseMap<uint64_t, SmallVector<Timer *, 4>> activeThreadTimers;

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

  // We don't actually want to time the adaptor passes, they gather their total
  // from their held passes.
  if (!isAdaptorPass(pass))
    timer->start();
}

/// Start a new timer for the given analysis.
void PassTiming::startAnalysisTimer(llvm::StringRef name, AnalysisID *id) {
  Timer *timer = getTimer(id, [name] { return "(A) " + name.str(); });
  timer->start();
}

/// Stop a pass timer.
void PassTiming::runAfterPass(Pass *pass, const llvm::Any &) {
  auto tid = llvm::get_threadid();
  auto &activeTimers = activeThreadTimers[tid];
  assert(!activeTimers.empty() && "expected active timer");
  Timer *timer = activeTimers.pop_back_val();

  // If this is an ModuleToFunctionPassAdaptorParallel, then we need to merge in
  // the timing data for the other threads.
  if (auto *asyncMTFPass =
          dyn_cast<ModuleToFunctionPassAdaptorParallel>(pass)) {
    // The asychronous pipeline timers should exist as children of root timers
    // for other threads.
    for (auto &rootTimer : llvm::make_early_inc_range(rootTimers)) {
      // Skip the current thread.
      if (rootTimer.first == tid)
        continue;
      // Check that this thread has no active timers.
      assert(activeThreadTimers[tid].empty() && "expected no active timers");

      // Structurally merge this timers children into the parallel
      // module-to-function pass timer.
      timer->mergeChildren(std::move(rootTimer.second->children),
                           /*isStructural=*/true);
      rootTimers.erase(rootTimer.first);
    }
    return;
  }

  // Adapator passes aren't timed directly, so we don't need to stop their
  // timers.
  if (!isAdaptorPass(pass))
    timer->stop();
}

/// Stop a timer.
void PassTiming::runAfterAnalysis(llvm::StringRef, AnalysisID *,
                                  const llvm::Any &) {
  auto &activeTimers = activeThreadTimers[llvm::get_threadid()];
  assert(!activeTimers.empty() && "expected active timer");
  Timer *timer = activeTimers.pop_back_val();
  timer->stop();
}

/// Utility to print the timer heading information.
static void printTimerHeader(llvm::raw_ostream &os, TimeRecord total) {
  os << "===" << std::string(73, '-') << "===\n";
  // Figure out how many spaces to description name.
  unsigned Padding = (80 - kPassTimingDescription.size()) / 2;
  os.indent(Padding) << kPassTimingDescription << '\n';
  os << "===" << std::string(73, '-') << "===\n";

  // Print the total time followed by the section headers.
  os << llvm::format("  Total Execution Time: %5.4f seconds\n\n", total.wall);
  if (total.user != total.wall)
    os << "   ---User Time---";
  os << "   ---Wall Time---  --- Name ---\n";
}

/// Utility to print a single line entry in the timer output.
static void printTimeEntry(raw_ostream &os, unsigned indent, StringRef name,
                           TimeRecord time, TimeRecord totalTime) {
  time.print(os, totalTime);
  os.indent(indent) << name << "\n";
}

/// Print out the current timing information.
void PassTiming::print() {
  // Don't print anything if there is no timing data.
  if (rootTimers.empty())
    return;

  assert(rootTimers.size() == 1 && "expected one remaining root timer");
  auto &rootTimer = rootTimers.begin()->second;
  auto os = llvm::CreateInfoOutputFile();

  // Print the timer header.
  TimeRecord totalTime = rootTimer->getTotalTime();
  printTimerHeader(*os, totalTime);

  // Defer to a specialized printer for each display mode.
  switch (displayMode) {
  case PassTimingDisplayMode::List:
    printResultsAsList(*os, rootTimer.get(), totalTime);
    break;
  case PassTimingDisplayMode::Pipeline:
    printResultsAsPipeline(*os, rootTimer.get(), totalTime);
    break;
  }
  printTimeEntry(*os, 0, "Total", totalTime, totalTime);
  os->flush();

  // Reset root timers.
  rootTimers.clear();
  activeThreadTimers.clear();
}

/// Print the timing result in list mode.
void PassTiming::printResultsAsList(raw_ostream &os, Timer *root,
                                    TimeRecord totalTime) {
  llvm::StringMap<TimeRecord> mergedTimings;

  std::function<void(Timer *)> addTimer = [&](Timer *timer) {
    // Check for timing information.
    if (timer->wallTime.count())
      mergedTimings[timer->name] += timer->getTotalTime();
    for (auto &children : timer->children)
      addTimer(children.second.get());
  };

  // Add each of the top level timers.
  for (auto &topLevelTimer : root->children)
    addTimer(topLevelTimer.second.get());

  // Sort the timing information by wall time.
  std::vector<std::pair<StringRef, TimeRecord>> timerNameAndTime;
  for (auto &it : mergedTimings)
    timerNameAndTime.emplace_back(it.first(), it.second);
  llvm::array_pod_sort(timerNameAndTime.begin(), timerNameAndTime.end(),
                       [](const std::pair<StringRef, TimeRecord> *lhs,
                          const std::pair<StringRef, TimeRecord> *rhs) {
                         return llvm::array_pod_sort_comparator<double>(
                             &rhs->second.wall, &lhs->second.wall);
                       });

  // Print the timing information sequentially.
  for (auto &timeData : timerNameAndTime)
    printTimeEntry(os, 0, timeData.first, timeData.second, totalTime);
}

/// Print the timing result in pipeline mode.
void PassTiming::printResultsAsPipeline(raw_ostream &os, Timer *root,
                                        TimeRecord totalTime) {
  std::function<void(unsigned, Timer *)> printTimer = [&](unsigned indent,
                                                          Timer *timer) {
    printTimeEntry(os, indent, timer->name, timer->getTotalTime(), totalTime);
    for (auto &children : timer->children)
      printTimer(indent + 2, children.second.get());
  };

  // Print each of the top level timers.
  for (auto &topLevelTimer : root->children)
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
