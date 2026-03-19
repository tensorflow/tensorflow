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

#ifndef XLA_SERVICE_DUMP_OPTIONS_H_
#define XLA_SERVICE_DUMP_OPTIONS_H_

#include <cstdint>
#include <functional>
#include <string>

#include "absl/strings/string_view.h"
#include "xla/xla.pb.h"

namespace xla {

// Canonicalized form of DebugOptions for dumping.
struct DumpOptions {
  explicit DumpOptions(const DebugOptions& opts);

  bool dumping_to_stdout() const { return dump_to == "-"; }

  std::string dump_to;
  std::function<bool(absl::string_view module_name)> should_dump_module;
  std::function<bool(absl::string_view pass_name)> should_dump_pass;
  std::function<bool(absl::string_view emitter_name)> should_dump_emitter;
  std::function<bool(absl::string_view pipeline_name)> should_dump_pipeline;

  bool dump_as_text;
  bool dump_as_proto;
  bool dump_as_dot;
  bool dump_as_html;
  bool dump_as_url;
  bool dump_fusion_visualization;
  bool dump_snapshots;
  bool dump_unoptimized_snapshots;
  bool dump_include_timestamp;
  int64_t dump_max_hlo_modules;
  bool dump_compress_protos;
  bool dump_fdo_profiles;
  bool dump_mlir_pretty_form;
};

}  // namespace xla

#endif  // XLA_SERVICE_DUMP_OPTIONS_H_
