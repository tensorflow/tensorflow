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

#include <mutex>  // NOLINT

#include "tensorflow/compiler/jit/legacy_flags/build_xla_ops_pass_flags.h"
#include "tensorflow/compiler/xla/legacy_flags/parse_flags_from_env.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace legacy_flags {
namespace {

BuildXlaOpsPassFlags* flags;
std::vector<Flag>* flag_list;
std::once_flag flags_init;

void AllocateAndParseFlags() {
  flags = new BuildXlaOpsPassFlags;
  flags->tf_xla_enable_lazy_compilation = true;
  flag_list = new std::vector<Flag>({
      Flag("tf_xla_enable_lazy_compilation",
           &flags->tf_xla_enable_lazy_compilation, ""),
  });
  xla::legacy_flags::ParseFlagsFromEnv(*flag_list);
}

}  // namespace

const BuildXlaOpsPassFlags& GetBuildXlaOpsPassFlags() {
  std::call_once(flags_init, &AllocateAndParseFlags);
  return *flags;
}
}  // namespace legacy_flags
}  // namespace tensorflow
