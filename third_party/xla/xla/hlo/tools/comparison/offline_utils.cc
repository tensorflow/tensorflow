/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/tools/comparison/offline_utils.h"

#include <unistd.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "re2/re2.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/comparison/comparison_result.pb.h"
#include "xla/hlo/tools/comparison/launch_info_compat.pb.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tsl/platform/env.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"

namespace xla::numerics::comparison {

ScopeInstructionProto ProtoFromScopeInstruction(const ScopeInstruction& si) {
  ScopeInstructionProto p;
  p.set_instruction_name(si.instruction_name);
  p.set_iteration_index(si.iteration_index);
  return p;
}

// File utilities
absl::StatusOr<std::vector<std::string>> FindFiles(
    absl::Span<const std::string> dirs, absl::string_view pattern) {
  std::vector<std::string> result;
  for (const auto& dir : dirs) {
    std::vector<std::string> matches;
    ABSL_RETURN_IF_ERROR(tsl::Env::Default()->GetMatchingPaths(
        tsl::io::JoinPath(dir, pattern), &matches));
    result.insert(result.end(), matches.begin(), matches.end());
  }
  return result;
}

absl::StatusOr<std::string> FindOneFile(absl::Span<const std::string> dirs,
                                        absl::string_view pattern) {
  ABSL_ASSIGN_OR_RETURN(std::vector<std::string> files, FindFiles(dirs, pattern));
  if (files.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No file found matching pattern: ", pattern));
  }
  return files[0];
}

absl::StatusOr<LaunchInfoResult> FindLaunchInfo(
    absl::Span<const std::string> dirs, absl::string_view module_name,
    std::optional<int64_t> launch_barrier_id) {
  if (launch_barrier_id.has_value()) {
    std::string pattern = absl::StrFormat("*module_*.%s.%d.launch_info.pbtxt",
                                          module_name, *launch_barrier_id);
    ABSL_ASSIGN_OR_RETURN(std::string path, FindOneFile(dirs, pattern));
    return LaunchInfoResult{path, *launch_barrier_id};
  }

  // If no ID provided, find one and parse ID from filename.
  std::string pattern =
      absl::StrFormat("*module_*.%s.*.launch_info.pbtxt", module_name);
  ABSL_ASSIGN_OR_RETURN(std::vector<std::string> files, FindFiles(dirs, pattern));
  if (files.empty()) {
    return absl::NotFoundError(absl::StrFormat(
        "No launch info file found for module '%s'. The file name should match "
        "pattern '*module_*.%s.*.launch_info.pbtxt'. Searched in the "
        "following "
        "directories:\n  %s",
        module_name, module_name, absl::StrJoin(dirs, "\n  ")));
  }
  std::string path;
  if (files.size() > 1) {
    absl::c_sort(files);
    if (isatty(STDIN_FILENO)) {
      std::cerr << "Multiple launch info files found for module " << module_name
                << ":\n";
      for (int i = 0; i < files.size(); ++i) {
        std::cerr << "[" << i << "] " << files[i] << "\n";
      }
      std::cerr << "Please enter the index of the file to use: ";
      int choice = -1;
      std::cin >> choice;
      // Ignore the rest of the line to consume the newline character.
      std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      if (std::cin.fail() || choice < 0 || choice >= files.size()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Invalid choice. Please choose an index between 0 and ",
            files.size() - 1, "."));
      }
      path = files[choice];
    } else {
      path = files[0];
      std::cerr << "WARNING: Multiple launch info files found for module "
                << module_name
                << ", and stdin is not a TTY. Using the first file: " << path
                << "\n";
    }
  } else {
    path = files[0];
  }
  int64_t parsed_id;
  static constexpr LazyRE2 kLaunchInfoRe = {
      R"(.*module_\d+\..*?\.(-?\d+)\.launch_info\.pbtxt)"};
  if (!RE2::FullMatch(tsl::io::Basename(path), *kLaunchInfoRe, &parsed_id)) {
    return absl::InternalError(
        absl::StrCat("Could not parse launch barrier id from file: ", path));
  }
  std::cerr << "Using launch_barrier_id " << parsed_id << " for module "
            << module_name << " from file " << path << "\n";
  return LaunchInfoResult{path, parsed_id};
}

absl::StatusOr<std::unique_ptr<HloModule>> ReadHloModuleFromFile(
    absl::string_view path, xla::StackFrameIndexProto* stack_frame_index) {
  HloModuleProto proto;
  ABSL_RETURN_IF_ERROR(
      tsl::ReadBinaryProto(tsl::Env::Default(), std::string(path), &proto));
  if (stack_frame_index != nullptr && proto.has_stack_frame_index()) {
    *stack_frame_index = proto.stack_frame_index();
  }
  xla::DebugOptions debug_options;
  ABSL_ASSIGN_OR_RETURN(
      xla::HloModuleConfig module_config,
      xla::HloModule::CreateModuleConfigFromProto(proto, debug_options));
  return HloModule::CreateFromProto(proto, module_config);
}

absl::StatusOr<RunData> LoadRunData(absl::Span<const std::string> dirs,
                                    absl::string_view module_name,
                                    std::optional<int64_t> launch_barrier_id) {
  RunData run_data;
  run_data.module_name = module_name;

  // 1. Load launch info and get device assignment.
  ABSL_ASSIGN_OR_RETURN(LaunchInfoResult launch_info,
                   FindLaunchInfo(dirs, module_name, launch_barrier_id));
  run_data.launch_barrier_id = launch_info.launch_barrier_id;

  LaunchInfo launch_info_proto;
  ABSL_RETURN_IF_ERROR(tsl::ReadTextProto(tsl::Env::Default(), launch_info.path,
                                     &launch_info_proto));
  ABSL_ASSIGN_OR_RETURN(
      run_data.device_assignment,
      DeviceAssignment::Deserialize(launch_info_proto.device_assignment()));

  // 2. Find and parse HLO modules.
  ABSL_ASSIGN_OR_RETURN(
      std::string original_hlo_path,
      FindOneFile(dirs,
                  absl::StrFormat("*module_*.%s.%d.original_hlo_module.pb",
                                  module_name, run_data.launch_barrier_id)));
  ABSL_ASSIGN_OR_RETURN(
      run_data.original_module,
      ReadHloModuleFromFile(original_hlo_path, &run_data.stack_frame_index));

  ABSL_ASSIGN_OR_RETURN(
      std::string optimized_hlo_path,
      FindOneFile(dirs,
                  absl::StrFormat("*module_*.%s.%d.optimized_hlo_module.pb",
                                  module_name, run_data.launch_barrier_id)));
  ABSL_ASSIGN_OR_RETURN(run_data.optimized_module,
                   ReadHloModuleFromFile(optimized_hlo_path));

  // 3. Find log files.
  ABSL_ASSIGN_OR_RETURN(
      std::vector<std::string> all_log_files,
      FindFiles(dirs, absl::StrFormat("*%s*tpu_log-*_%d-*.riegeli", module_name,
                                      launch_info.launch_barrier_id)));

  absl::flat_hash_map<int64_t, std::vector<std::pair<int, std::string>>>
      device_logs;  // device_id -> list<pair<seq_no, path>>
  static constexpr LazyRE2 kLogFileRe = {
      R"(.+?\.tpu_log-(\d+)_-?\d+-(\d+)\.riegeli)"};
  for (const std::string& path : all_log_files) {
    int64_t dev_id_in_file;
    int seq_no;
    if (RE2::FullMatch(tsl::io::Basename(path), *kLogFileRe, &dev_id_in_file,
                       &seq_no)) {
      device_logs[dev_id_in_file].push_back({seq_no, path});
    }
  }

  for (int64_t replica = 0;
       replica < run_data.device_assignment->replica_count(); ++replica) {
    for (int64_t computation = 0;
         computation < run_data.device_assignment->computation_count();
         ++computation) {
      int64_t physical_id =
          run_data.device_assignment->DeviceId(replica, computation);
      auto it = device_logs.find(physical_id);
      if (it == device_logs.end() || it->second.empty()) {
        return absl::NotFoundError(absl::StrFormat(
            "Log files for device %d not found for module %s launch %d",
            physical_id, module_name, run_data.launch_barrier_id));
      }
      // Sort files by sequence number.
      absl::c_sort(it->second);
      for (const auto& pair : it->second) {
        run_data.log_files.push_back(pair.second);
        run_data.device_ids_for_log_files.push_back(physical_id);
      }
    }
  }

  return run_data;
}

}  // namespace xla::numerics::comparison
