/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xrt/xrt_util.h"

#include <stdlib.h>
#include <string.h>

#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {

bool DebugOptionsPassThroughEnabled() {
  const char* env = getenv("TF_XLA_DEBUG_OPTIONS_PASSTHROUGH");
  bool enabled =
      env != nullptr && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0);
  if (enabled) {
    LOG(WARNING) << "Passing through XLA debug options!";
  } else {
    LOG(WARNING) << "TF_XLA_DEBUG_OPTIONS_PASSTHROUGH not set, not all options "
                    "will be retained";
  }
  return enabled;
}

string SafeDebugPath(const string& path) {
  if (path.empty() || path.compare(0, 5, "gs://") == 0 ||
      path.compare(0, 11, "bigstore://") == 0) {
    return path;
  }
  LOG(WARNING) << "Invalid config path (will be dropped): " << path;
  return string();
}

}  // namespace

xla::DebugOptions BuildXlaDebugOptions(const xla::DebugOptions& ref_options) {
  static const bool options_passthrough = DebugOptionsPassThroughEnabled();
  if (options_passthrough) {
    return ref_options;
  }
  xla::DebugOptions options = xla::GetDebugOptionsFromFlags();
  options.set_xla_generate_hlo_text_to(
      SafeDebugPath(ref_options.xla_generate_hlo_text_to()));
  options.set_xla_dump_optimized_hlo_proto_to(
      SafeDebugPath(ref_options.xla_dump_optimized_hlo_proto_to()));
  options.set_xla_dump_computations_to(
      SafeDebugPath(ref_options.xla_dump_computations_to()));
  options.set_xla_dump_executions_to(
      SafeDebugPath(ref_options.xla_dump_executions_to()));
  for (auto& pass : ref_options.xla_disable_hlo_passes()) {
    options.add_xla_disable_hlo_passes(pass);
  }
  options.set_xla_dump_unoptimized_hlo_proto_to(
      SafeDebugPath(ref_options.xla_dump_unoptimized_hlo_proto_to()));
  options.set_xla_dump_per_pass_hlo_proto_to(
      SafeDebugPath(ref_options.xla_dump_per_pass_hlo_proto_to()));
  return options;
}

}  // namespace tensorflow
