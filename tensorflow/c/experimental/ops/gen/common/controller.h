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
#ifndef TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_COMMON_CONTROLLER_H_
#define TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_COMMON_CONTROLLER_H_

#include "tensorflow/c/experimental/ops/gen/common/path_config.h"
#include "tensorflow/c/experimental/ops/gen/common/source_code.h"
#include "tensorflow/c/experimental/ops/gen/model/op_spec.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace generator {

class Controller {
 public:
  explicit Controller(PathConfig path_config, Env* env = Env::Default());
  virtual ~Controller();
  const void WriteFile(const string& file_path, const SourceCode& code) const;
  const std::vector<OpSpec>& GetModelOps() const;

 private:
  void InitializeOpApi();
  void BuildModel();

  // Data model: Ops to generate
  std::vector<OpSpec> operators_;

  // Configuration
  Env* env_;
  PathConfig path_config_;

  // Initialized TensorFlow Op/API definitions
  OpList op_list_;
  ApiDefMap* api_def_map_;
};

}  // namespace generator
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_COMMON_CONTROLLER_H_
