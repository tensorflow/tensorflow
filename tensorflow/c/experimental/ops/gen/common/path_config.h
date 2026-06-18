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
#ifndef TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_COMMON_PATH_CONFIG_H_
#define TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_COMMON_PATH_CONFIG_H_

#include <vector>

#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace generator {

struct PathConfig {
  std::string output_path;
  std::vector<std::string> op_names;
  std::vector<std::string> api_dirs;
  std::string tf_prefix_dir;
  std::string tf_root_dir;
  std::string tf_output_dir;

  explicit PathConfig() = default;
  explicit PathConfig(const std::string& output_dir,
                      const std::string& source_dir,
                      const std::string& api_dir_list,
                      const std::vector<std::string> op_names);
};

}  // namespace generator
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_COMMON_PATH_CONFIG_H_
