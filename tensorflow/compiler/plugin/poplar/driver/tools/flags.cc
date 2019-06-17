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

#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hash.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "tensorflow/compiler/xla/parse_flags_from_env.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <mutex>
#include <sstream>

namespace xla {
namespace poplarplugin {
namespace {

absl::flat_hash_map<std::string, std::string> GetFlagUsage() {
  static absl::flat_hash_map<std::string, std::string> flag_usage = {
      {"help", "Display all the flags infos. (bool)"},
      {"use_synthetic_data",
       "If enabled, there will be no data transfers between the host and the "
       "IPU(s). (bool)"},
      {"synthetic_data_initializer",
       "If set when using synthetic data, all the inputs to the graph can be "
       "initialized directly on the IPU either randomly "
       "(synthetic_data_initializer=random, uniform, normal) or to a constant "
       "value X (synthetic_data_initializer=int)."},
      {"use_ipu_model",
       "If enabled, this computation will be executed on the IPU model. "
       "(bool)"},
      {"force_replicated_mode",
       "If enabled, we allow replicated graphs with no AllReduce operations in "
       "them to still run in replicated mode. (bool)"},
      {"while_loop_brute_force_max_trip_count",
       "When trying to convert a while loop to a repeat loop, we can try and "
       "use a brute force method to simulate the conditional part of the while "
       "and find the number of iterations. This flag sets how many iterations "
       "of the while loop we should try and brute force it for. (int=128)"},
      {"max_compilation_threads",
       "The maximum number of threads Poplar should use during compilation of "
       "the graph. Negative value allows Poplar to pick the number of threads "
       "automatically. (int=-1)"},
      {"save_oom_profiler",
       "Path to a file where the profiling information is saved to when an Out "
       "Of Memory error occurs. (path)"},
      {"save_vertex_graph",
       "Path to a directory where the Poplar vertex graphs should be saved to. "
       "(path)"},
      {"save_interval_report",
       "Path to a directory where the Poplar interval reports should be saved "
       "to. (path)"},
      {"executable_cache_path", "Path to the executable cache. (path)"},
      {"dump_schedule_as_dot", "Dumps the scheduler graph as a dot file."},
      {"tensor_map_file_path", "Directory for tensor map dump files."},
      {"fallback_scheduler",
       "Use the sync list scheduler rather than the default one."}};
  return flag_usage;
}
}  // namespace

PoplarXlaFlags::PoplarXlaFlags() {
  // Struct for deprecated flags.
  struct DeprecatedFlags {
    bool add_all_reduce_copies = false;
  };

  DeprecatedFlags deprecated_flags;
  auto flag_usage = GetFlagUsage();

  std::vector<tensorflow::Flag> flag_list = {
#define ADD_FLAG(FLAG_NAME) \
  tensorflow::Flag(#FLAG_NAME, &FLAG_NAME, flag_usage.at(#FLAG_NAME)),
#define ADD_DEPRECATED_FLAG(FLAG_NAME) \
  tensorflow::Flag(#FLAG_NAME, &deprecated_flags.FLAG_NAME, ""),
      // clang-format off
    ADD_FLAG(help)
    ADD_FLAG(use_synthetic_data)
    ADD_FLAG(synthetic_data_initializer)
    ADD_FLAG(use_ipu_model)
    ADD_FLAG(force_replicated_mode)
    ADD_FLAG(while_loop_brute_force_max_trip_count)
    ADD_FLAG(max_compilation_threads)
    ADD_FLAG(save_oom_profiler)
    ADD_FLAG(save_vertex_graph)
    ADD_FLAG(save_interval_report)
    ADD_FLAG(executable_cache_path)
    ADD_FLAG(dump_schedule_as_dot)
    ADD_FLAG(tensor_map_file_path)
    ADD_FLAG(fallback_scheduler)

    // Deprecated flags.
    ADD_DEPRECATED_FLAG(add_all_reduce_copies)

// clang-format on
#undef ADD_FLAG
#undef ADD_DEPRECATED_FLAG
  };
  xla::ParseFlagsFromEnvAndDieIfUnknown("TF_POPLAR_FLAGS", flag_list);

  // Store all the flags as a string.
  as_string = "";
  if (const char* flag_buffer = std::getenv("TF_POPLAR_FLAGS")) {
    as_string = flag_buffer;
  }

  if (!use_synthetic_data && !synthetic_data_initializer.empty()) {
    LOG(FATAL) << "The flag \"synthetic_data_initializer\" can only be used "
                  "in combination with \"use_synthetic_data\".";
  }

  if (deprecated_flags.add_all_reduce_copies) {
    LOG(INFO)
        << "The TensorFlow Poplar flag \"add_all_reduce_copies\" is "
           "deprecated, has no effect and it will be removed in the future.";
  }

  // Hash all the flags which affect the graph generation and compilation only.
  hlo_hash = hash_util::hash(use_synthetic_data, synthetic_data_initializer,
                             use_ipu_model, force_replicated_mode,
                             while_loop_brute_force_max_trip_count,
                             executable_cache_path, fallback_scheduler);
}

const PoplarXlaFlags& PoplarXlaFlags::Get() {
  static PoplarXlaFlags poplar_xla_flags;
  return poplar_xla_flags;
}

const std::string PoplarXlaFlags::GetFlagUsageString() {
  auto flag_usage = GetFlagUsage();
  std::stringstream usage_stream;
  usage_stream << "Usage for TF_POPLAR_FLAGS is:" << std::endl;
  for (auto pair : flag_usage) {
    usage_stream << "\t--" << pair.first << ": " << pair.second << std::endl;
  }
  return usage_stream.str();
}
}  // namespace poplarplugin
}  // namespace xla
