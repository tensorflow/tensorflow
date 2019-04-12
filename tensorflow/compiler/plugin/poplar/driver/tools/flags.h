/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DIRVER_TOOLS_FALGS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DIRVER_TOOLS_FALGS_H_

#include <vector>

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {

struct PoplarXlaFlags {
  // Display all the flags infos.
  bool help;

  // If enabled, there will be no data transfers between the host and the
  // IPU(s).
  bool use_synthetic_data;

  // If enabled, this computation will be executed on the IPU model.
  bool use_ipu_model;

  // If enabled, we allow replicated graphs with no AllReduce operations in them
  // to still run in replicated mode.
  bool force_replicated_mode;

  // When trying to convert a while loop to a repeat loop, we can try and use a
  // brute force method to simulate the conditional part of the while and find
  // the number of iterations. This flag sets how many iterations of the while
  // loop we should try and brute force it for (default 128).
  int64 while_loop_brute_force_max_trip_count;

  // The maximum number of threads Poplar should use during compilation of the
  // graph.
  int64 max_compilation_threads;

  // Path to a file where the profiling information is saved to when an Out Of
  // Memory occurs.
  std::string save_oom_profiler;

  // Path to a file where the Poplar vertex graph should be saved to.
  std::string save_vertex_graph;

  // Path to the executable cache.
  std::string executable_cache_path;
};

// Getters for flags structs defined above.  The first call to any of these
// parses TF_POPLAR_FLAGS for all of them.
const PoplarXlaFlags& GetPoplarXlaFlags();

// Getter for the flag usage string.
const std::string GetFlagUsageString();
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DIRVER_TOOLS_FALGS_H_
