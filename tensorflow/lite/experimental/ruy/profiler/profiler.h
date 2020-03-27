/* Copyright 2020 Google LLC. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_PROFILER_PROFILER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_PROFILER_PROFILER_H_

#include <cstdio>

#ifdef RUY_PROFILER
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#endif

#include "tensorflow/lite/experimental/ruy/profiler/instrumentation.h"
#include "tensorflow/lite/experimental/ruy/profiler/treeview.h"

namespace ruy {
namespace profiler {

#ifdef RUY_PROFILER

// RAII user-facing way to create a profiler and let it profile a code scope,
// and print out an ASCII/MarkDown treeview upon leaving the scope.
class ScopeProfile {
 public:
  // Default constructor, unconditionally profiling.
  ScopeProfile();

  // Constructor allowing to choose at runtime whether to profile.
  explicit ScopeProfile(bool enable);

  // Destructor. It's where the profile is reported.
  ~ScopeProfile();

  // See treeview_ member.
  void SetUserTreeView(TreeView* treeview) { user_treeview_ = treeview; }

 private:
  void Start();

  // Thread entry point function for the profiler thread. This thread is
  // created on construction.
  void ThreadFunc();

  // Record a stack as a sample.
  void Sample(const detail::ThreadStack& stack);

  // Finalize the profile. Called on destruction.
  // If user_treeview_ is non-null, it will receive the treeview.
  // Otherwise the treeview will just be printed.
  void Finish();

  // Buffer where samples are recorded during profiling.
  std::vector<char> samples_buf_;

  // Used to synchronize thread termination.
  std::atomic<bool> finishing_;

  // Underlying profiler thread, which will perform the sampling.
  // This profiler approach relies on a thread rather than on signals.
  std::unique_ptr<std::thread> thread_;

  // TreeView to populate upon destruction. If left null (the default),
  // a temporary treeview will be used and dumped on stdout. The user
  // may override that by passing their own TreeView object for other
  // output options or to directly inspect the TreeView.
  TreeView* user_treeview_ = nullptr;
};

#else  // no RUY_PROFILER

struct ScopeProfile {
  ScopeProfile() {
#ifdef GEMMLOWP_PROFILING
    fprintf(
        stderr,
        "\n\n\n**********\n\nWARNING:\n\nLooks like you defined "
        "GEMMLOWP_PROFILING, but this code has been ported to the new ruy "
        "profiler replacing the old gemmlowp profiler. You should now be "
        "defining RUY_PROFILER and not GEMMLOWP_PROFILING. When building using "
        "Bazel, just pass --define=ruy_profiler=true.\n\n**********\n\n\n");
#endif
  }
  explicit ScopeProfile(bool) {}
};

#endif

}  // namespace profiler
}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_PROFILER_PROFILER_H_
