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
#include "llvm/ADT/STLExtras.h"
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

/// An enumeration of the different types of timers.
enum class TimerKind {
  /// This timer represents an ordered collection of pass timers, corresponding
  /// to a pass pipeline.
  Pipeline,

  /// This timer represents a collection of pipeline timers.
  PipelineCollection,

  /// This timer represents an analysis or pass timer.
  PassOrAnalysis
};

struct Timer {
  explicit Timer(std::string &&name, TimerKind kind)
      : name(std::move(name)), kind(kind) {}

  /// Start the timer.
  void start() { startTime = std::chrono::system_clock::now(); }

  /// Stop the timer.
  void stop() {
    auto newTime = std::chrono::system_clock::now() - startTime;
    wallTime += newTime;
    userTime += newTime;
  }

  /// Get or create a child timer with the provided name and id.
  Timer *getChildTimer(const void *id, TimerKind kind,
                       std::function<std::string()> &&nameBuilder) {
    auto &child = children[id];
    if (!child)
      child = std::make_unique<Timer>(nameBuilder(), kind);
    return child.get();
  }

  /// Returns the total time for this timer in seconds.
  TimeRecord getTotalTime() {
    // If this is a pass or analysis timer, use the recorded time directly.
    if (kind == TimerKind::PassOrAnalysis) {
      return TimeRecord(
          std::chrono::duration_cast<std::chrono::duration<double>>(wallTime)
              .count(),
          std::chrono::duration_cast<std::chrono::duration<double>>(userTime)
              .count());
    }

    // Otherwise, accumulate the timing from each of the children.
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
    mergeChildren(std::move(other.children));
  }

  /// Merge the timer children in 'otherChildren' with the children of this
  /// timer.
  void mergeChildren(ChildrenMap &&otherChildren) {
    // Check for an empty children list.
    if (children.empty()) {
      children = std::move(otherChildren);
      return;
    }

    // Pipeline merges are handled separately as the children are merged
    // lexicographically.
    if (kind == TimerKind::Pipeline) {
      assert(children.size() == otherChildren.size() &&
             "pipeline merge requires the same number of children");
      for (auto it : llvm::zip(children, otherChildren))
        std::get<0>(it).second->merge(std::move(*std::get<1>(it).second));
      return;
    }

    // Otherwise, we merge children based upon their timer key.
    for (auto &otherChild : otherChildren)
      mergeChild(std::move(otherChild));
  }

  /// Merge in the given child timer and id into this timer.
  void mergeChild(ChildrenMap::value_type &&childIt) {
    auto &child = children[childIt.first];
    if (!child)
      child = std::move(childIt.second);
    else
      child->merge(std::move(*childIt.second));
  }

  /// Raw timing information.
  std::chrono::time_point<std::chrono::system_clock> startTime;
  std::chrono::nanoseconds wallTime = std::chrono::nanoseconds(0);
  std::chrono::nanoseconds userTime = std::chrono::nanoseconds(0);

  /// A map of unique identifiers to child timers.
  ChildrenMap children;

  /// A descriptive name for this timer.
  std::string name;

  /// The type of timer this instance represents.
  TimerKind kind;
};

struct PassTiming : public PassInstrumentation {
  PassTiming(PassDisplayMode displayMode) : displayMode(displayMode) {}
  ~PassTiming() override { print(); }

  /// Setup the instrumentation hooks.
  void runBeforePipeline(const OperationName &name,
                         const PipelineParentInfo &parentInfo) override;
  void runAfterPipeline(const OperationName &name,
                        const PipelineParentInfo &parentInfo) override;
  void runBeforePass(Pass *pass, Operation *) override { startPassTimer(pass); }
  void runAfterPass(Pass *pass, Operation *) override;
  void runAfterPassFailed(Pass *pass, Operation *op) override {
    runAfterPass(pass, op);
  }
  void runBeforeAnalysis(llvm::StringRef name, AnalysisID *id,
                         Operation *) override {
    startAnalysisTimer(name, id);
  }
  void runAfterAnalysis(llvm::StringRef, AnalysisID *, Operation *) override;

  /// Print and clear the timing results.
  void print();

  /// Start a new timer for the given pass.
  void startPassTimer(Pass *pass);

  /// Start a new timer for the given analysis.
  void startAnalysisTimer(llvm::StringRef name, AnalysisID *id);

  /// Pop the last active timer for the current thread.
  Timer *popLastActiveTimer() {
    auto tid = llvm::get_threadid();
    auto &activeTimers = activeThreadTimers[tid];
    assert(!activeTimers.empty() && "expected active timer");
    return activeTimers.pop_back_val();
  }

  /// Print the timing result in list mode.
  void printResultsAsList(raw_ostream &os, Timer *root, TimeRecord totalTime);

  /// Print the timing result in pipeline mode.
  void printResultsAsPipeline(raw_ostream &os, Timer *root,
                              TimeRecord totalTime);

  /// Returns a timer for the provided identifier and name.
  Timer *getTimer(const void *id, TimerKind kind,
                  std::function<std::string()> &&nameBuilder) {
    auto tid = llvm::get_threadid();

    // If there is no active timer then add to the root timer.
    auto &activeTimers = activeThreadTimers[tid];
    Timer *parentTimer;
    if (activeTimers.empty()) {
      auto &rootTimer = rootTimers[tid];
      if (!rootTimer)
        rootTimer = std::make_unique<Timer>("root", TimerKind::Pipeline);
      parentTimer = rootTimer.get();
    } else {
      // Otherwise, add this to the active timer.
      parentTimer = activeTimers.back();
    }

    auto timer = parentTimer->getChildTimer(id, kind, std::move(nameBuilder));
    activeTimers.push_back(timer);
    return timer;
  }

  /// The root top level timers for each thread.
  DenseMap<uint64_t, std::unique_ptr<Timer>> rootTimers;

  /// A stack of the currently active pass timers per thread.
  DenseMap<uint64_t, SmallVector<Timer *, 4>> activeThreadTimers;

  /// The display mode to use when printing the timing results.
  PassDisplayMode displayMode;

  /// A mapping of pipeline timers that need to be merged into the parent
  /// collection. The timers are mapped to the parent info to merge into.
  DenseMap<PipelineParentInfo, SmallVector<Timer::ChildrenMap::value_type, 4>>
      pipelinesToMerge;
};
} // end anonymous namespace

void PassTiming::runBeforePipeline(const OperationName &name,
                                   const PipelineParentInfo &parentInfo) {
  // We don't actually want to time the piplelines, they gather their total
  // from their held passes.
  getTimer(name.getAsOpaquePointer(), TimerKind::Pipeline,
           [&] { return ("'" + name.getStringRef() + "' Pipeline").str(); });
}

void PassTiming::runAfterPipeline(const OperationName &name,
                                  const PipelineParentInfo &parentInfo) {
  // Pop the timer for the pipeline.
  auto tid = llvm::get_threadid();
  auto &activeTimers = activeThreadTimers[tid];
  assert(!activeTimers.empty() && "expected active timer");
  activeTimers.pop_back();

  // If the current thread is the same as the parent, there is nothing left to
  // do.
  if (tid == parentInfo.parentThreadID)
    return;

  // Otherwise, mark the pipeline timer for merging into the correct parent
  // thread.
  assert(activeTimers.empty() && "expected parent timer to be root");
  auto *parentTimer = rootTimers[tid].get();
  assert(parentTimer->children.size() == 1 &&
         parentTimer->children.count(name.getAsOpaquePointer()) &&
         "expected a single pipeline timer");
  pipelinesToMerge[parentInfo].push_back(
      std::move(*parentTimer->children.begin()));
  rootTimers.erase(tid);
}

/// Start a new timer for the given pass.
void PassTiming::startPassTimer(Pass *pass) {
  auto kind = isAdaptorPass(pass) ? TimerKind::PipelineCollection
                                  : TimerKind::PassOrAnalysis;
  Timer *timer = getTimer(pass, kind, [pass]() -> std::string {
    if (auto *adaptor = getAdaptorPassBase(pass))
      return adaptor->getName();
    return pass->getName();
  });

  // We don't actually want to time the adaptor passes, they gather their total
  // from their held passes.
  if (!isAdaptorPass(pass))
    timer->start();
}

/// Start a new timer for the given analysis.
void PassTiming::startAnalysisTimer(llvm::StringRef name, AnalysisID *id) {
  Timer *timer = getTimer(id, TimerKind::PassOrAnalysis,
                          [name] { return "(A) " + name.str(); });
  timer->start();
}

/// Stop a pass timer.
void PassTiming::runAfterPass(Pass *pass, Operation *) {
  Timer *timer = popLastActiveTimer();

  // If this is an OpToOpPassAdaptorParallel, then we need to merge in the
  // timing data for the pipelines running on other threads.
  if (isa<OpToOpPassAdaptorParallel>(pass)) {
    auto toMerge = pipelinesToMerge.find({llvm::get_threadid(), pass});
    if (toMerge != pipelinesToMerge.end()) {
      for (auto &it : toMerge->second)
        timer->mergeChild(std::move(it));
      pipelinesToMerge.erase(toMerge);
    }
    return;
  }

  // Adaptor passes aren't timed directly, so we don't need to stop their
  // timers.
  if (!isAdaptorPass(pass))
    timer->stop();
}

/// Stop a timer.
void PassTiming::runAfterAnalysis(llvm::StringRef, AnalysisID *, Operation *) {
  popLastActiveTimer()->stop();
}

/// Utility to print the timer heading information.
static void printTimerHeader(llvm::raw_ostream &os, TimeRecord total) {
  os << "===" << std::string(73, '-') << "===\n";
  // Figure out how many spaces to description name.
  unsigned padding = (80 - kPassTimingDescription.size()) / 2;
  os.indent(padding) << kPassTimingDescription << '\n';
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
  case PassDisplayMode::List:
    printResultsAsList(*os, rootTimer.get(), totalTime);
    break;
  case PassDisplayMode::Pipeline:
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
    // Only add timing information for passes and analyses.
    if (timer->kind == TimerKind::PassOrAnalysis)
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
    // If this is a timer for a pipeline collection and the collection only has
    // one pipeline child, then only print the child.
    if (timer->kind == TimerKind::PipelineCollection &&
        timer->children.size() == 1)
      return printTimer(indent, timer->children.begin()->second.get());

    printTimeEntry(os, indent, timer->name, timer->getTotalTime(), totalTime);

    // If this timer is a pipeline, then print the children in-order.
    if (timer->kind == TimerKind::Pipeline) {
      for (auto &child : timer->children)
        printTimer(indent + 2, child.second.get());
      return;
    }

    // Otherwise, sort the children by name to give a deterministic ordering
    // when emitting the time.
    SmallVector<Timer *, 4> children;
    children.reserve(timer->children.size());
    for (auto &child : timer->children)
      children.push_back(child.second.get());
    llvm::array_pod_sort(children.begin(), children.end(),
                         [](Timer *const *lhs, Timer *const *rhs) {
                           return (*lhs)->name.compare((*rhs)->name);
                         });
    for (auto &child : children)
      printTimer(indent + 2, child);
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
void PassManager::enableTiming(PassDisplayMode displayMode) {
  // Check if pass timing is already enabled.
  if (passTiming)
    return;
  addInstrumentation(std::make_unique<PassTiming>(displayMode));
  passTiming = true;
}
