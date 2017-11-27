/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/python/eager/python_eager_op_gen.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"

namespace tensorflow {
namespace {

constexpr char kBaseApiDef[] =
    "tensorflow/core/api_def/base_api/*.pbtxt";
constexpr char kPythonApiDef[] =
    "tensorflow/core/api_def/python_api/*.pbtxt";
constexpr bool kUseApiDef = false;

void PrintAllPythonOps(const std::vector<string>& hidden_ops) {
  OpList ops;
  OpRegistry::Global()->Export(false, &ops);

  ApiDefMap api_def_map(ops);
  if (kUseApiDef) {
    Env* env = Env::Default();

    std::vector<string> base_api_files;
    std::vector<string> python_api_files;
    TF_CHECK_OK(env->GetMatchingPaths(kBaseApiDef, &base_api_files));
    TF_CHECK_OK(env->GetMatchingPaths(kPythonApiDef, &python_api_files));

    TF_CHECK_OK(api_def_map.LoadFileList(env, base_api_files));
    TF_CHECK_OK(api_def_map.LoadFileList(env, python_api_files));
  }
  PrintEagerPythonOps(ops, api_def_map, hidden_ops, true /* require_shapes */);
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  if (argc == 1) {
    tensorflow::PrintAllPythonOps({});
  } else {
    return -1;
  }
  return 0;
}
