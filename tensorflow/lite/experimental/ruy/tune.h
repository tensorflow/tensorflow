/* Copyright 2019 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Library doing minimal CPU detection to decide what to tune asm code for.
//
// # Tuning vs Path
//
// Tunings are merely local variations of optimized code paths, that are
// drop-in replacements for each other --- the input and output data layouts
// are identical.  By contrast, what ruy calls a Path dictates its own
// data layouts. For example, Path::kNeonDotprod will use different
// layouts compared to Path::kNeon; but within each, different tunings
// will share that same layout.
//
// # Tuning is for now only based on 1 bit: OutOfOrder / InOrder
//
// In practice, each of our asm code paths only needs one bit information to
// decide on tuning: whether the CPU is out-of-order or in-order.
// That is because out-of-order CPUs are by definition relatively insensitive
// to small-scale asm details (which is what "tuning" is about); and for each
// asm code path, there tends to be one main in-order CPU architecture that
// we focus our tuning effort on. Examples:
//  * For Path::kNeon, the main in-order CPU is Cortex-A53/A55 (pre-dotprod)
//  * For Path::kNeonDotprod, the main in-order CPU is Cortex-A55r1 (dotprod)
//
// Because having tuned code paths is a compromise of efficiency gains
// versus implementation effort and code size, we are happy to stop at just this
// single bit of information, OutOfOrder/InOrder, at least in the current CPU
// landscape. This could change in the future.
//
// # Implementation notes and alternatives.
//
// The current implementation uses a nano-benchmark, see tune.cc.
// That is why it's quite expensive, making caching /
// statefulness necessary (see TuningResolver class comment).
//
// An interesting alternative, which was explained to us by Marat Dukhan
// (maratek@) after this was implemented, would be to use the
// getcpu(2) system call on Linux. This returns a
// numeric CPU identifier that could be mapped to a OutOfOrder/InOrder
// classification given additional information about the CPU.  Such
// additional information could be obtained by the cpuinfo library,
//   https://github.com/pytorch/cpuinfo
// which obtains this information mainly from parsing /proc/cpuinfo.
// Pros:
//   * Would remove the need for the relatively expensive nano-benchmark
//     (dozens of microseconds, which have to be reevaluated again several
//     times per second).
//   * Would conceivably be more reliable.
// Cons:
//   * Linux-specific.
//   * Modest binary size increase (Marat mentioned the cpuinfo lib is 20k).
//   * Won't support exactly 100% of devices (nonstandard /proc/cpuinfo etc).
//
// We could also have both:
//  * Maybe by trying getcpu first if supported, then falling back to a
//    nano-benchmark.
//  * Maybe using getcpu in conjunction with the nano-benchmark to cache
//    per-CPU-id nano-benchmark results.
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_TUNE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_TUNE_H_

#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/platform.h"
#include "tensorflow/lite/experimental/ruy/time.h"

// Tuning only implemented on NEON_64 at the moment (see assembly code
// in the nano-benchmark) and not on Apple (some Apple CPUs produce incorrect
// results on in-order-tuned kernels combining ARM and NEON load instructions
// and NEON `ins` instructions).
//
// When tuning is not implemented, we simply always use Tuning::kOutOfOrder.
#if RUY_OPT_ENABLED(RUY_OPT_TUNING) && RUY_PLATFORM(NEON_64) && \
    !RUY_PLATFORM(APPLE)
#define RUY_IMPLEMENT_TUNING
#endif

namespace ruy {

enum class Tuning {
  // kAuto means please use auto-detection. It's the default in the
  // user-visible parts (see Context). It's meant to be resolved to an
  // actual tuning at some point by means of TuningResolver.
  kAuto,
  // Target an out-order CPU. Example: ARM Cortex-A75.
  kOutOfOrder,
  // Target an in-order CPU. Example: ARM Cortex-A55.
  kInOrder
};

// Why a TuningResolver class?
//
// Ideally, this Library would offer a single function,
//   Tuning GetCurrentCPUTuning();
//
// However, determining information about the current CPU is not necessarily,
// cheap, so we currently cache that and only invalidate/reevaluate after
// a fixed amount of time. This need to store state is why this library
// has to expose a class, TuningResolver, not just a function.
class TuningResolver {
 public:
  TuningResolver();

  // Allows the user to specify an explicit Tuning value, bypassing auto
  // detection; or to specify Tuning::kAuto, reverting to auto detection.
  void SetTuning(Tuning tuning) { unresolved_tuning_ = tuning; }

  // Get an actual tuning --- that is the function that this class wanted to be.
  Tuning Resolve();

 private:
  TuningResolver(const TuningResolver&) = delete;

  // TuningTool is a demo/tool used to tweak the tuning implementation to
  // specific devices. It needs to access some finer granularity information
  // than just the Tuning returned by Resolve. Nothing else should need
  // access to that.
  friend class TuneTool;
  // Actually runs a nano-benchmark, producing a real number called 'ratio'
  // whose meaning is generally opaque / implementation defined. Typically,
  // this would be the ratio between the latencies of two different
  // pieces of asm code differing only by the ordering of instructions,
  // revealing whether the CPU cares about such ordering details.
  // An implementation may just return a dummy value if it is not based on
  // such nanobenchmarking / ratio evaluation.
  float EvalRatio();
  // Empirically determined threshold on ratio values delineating
  // out-of-order (ratios closer to 1) from in-order (ratios farther from 1).
  // An implementation may just return a dummy value if it is not based on
  // such nanobenchmarking / ratio evaluation.
  float ThresholdRatio();
  // Perform the tuning resolution now. That may typically use EvalRatio and
  // ThresholdRatio, but an implementation may use a different approach instead.
  Tuning ResolveNow();

  // The tuning as specified by the user, before actual resolution happens
  // i.e. before querying any specifics of the current CPU.
  // The default value kAuto means try to auto-detect. Other values mean
  // bypass auto-detect, use explicit value instead. See SetTuning().
  Tuning unresolved_tuning_ = Tuning::kAuto;
  // Cached last resolved tuning.
  Tuning last_resolved_tuning_ = Tuning::kAuto;
  // Timepoint of cached last resolved tuning, for invalidation purposes.
  TimePoint last_resolved_timepoint_;
  // Cached last resolved tunings that are older than this age are invalid.
  const Duration expiry_duration_;
};

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_TUNE_H_
