/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/ops/gen/common/path_config.h"

#include <algorithm>
#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace generator {

PathConfig::PathConfig(const string& output_dir, const string& source_dir,
                       const string& api_dir_list,
                       const std::vector<string> op_names)
    : output_path(output_dir), op_names(op_names) {
  api_dirs = str_util::Split(api_dir_list, ",", str_util::SkipEmpty());

  // Decompose the directory components given the output/source directories.
  //
  // Be flexible; accept any path string with a "tensorflow" directory name.
  // (It's hard to construct a location-agnostic include path string using the
  // build system, so we accept an example, such as the source build target.)

  tf_root_dir = "tensorflow";

  // Prefix, e.g. "third_party" given root_dir "third_party/tensorflow/...."
  std::vector<string> source_path_components =
      tensorflow::str_util::Split(source_dir, "/");
  auto source_tfroot_pos = std::find(source_path_components.begin(),
                                     source_path_components.end(), tf_root_dir);
  if (source_tfroot_pos != source_path_components.end()) {
    tf_prefix_dir =
        absl::StrJoin(source_path_components.begin(), source_tfroot_pos, "/");
  } else {
    tf_prefix_dir = source_dir;
  }

  // TF subdir, e.g. "c/ops" given output_dir "blah/blah/tensorflow/c/ops"
  std::vector<string> output_path_components =
      tensorflow::str_util::Split(output_dir, "/");
  auto output_tfroot_pos = std::find(output_path_components.begin(),
                                     output_path_components.end(), tf_root_dir);
  if (output_tfroot_pos != output_path_components.end()) {
    tf_output_dir =
        absl::StrJoin(output_tfroot_pos + 1, output_path_components.end(), "/");
  } else {
    tf_output_dir = output_dir;
  }
}

}  // namespace generator
}  // namespace tensorflow
