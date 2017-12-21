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
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"

namespace tensorflow {
namespace {

void PrintAllPythonOps(const std::vector<string>& hidden_ops,
                       const std::vector<string>& api_def_dirs) {
  OpList ops;
  OpRegistry::Global()->Export(false, &ops);

  ApiDefMap api_def_map(ops);
  if (!api_def_dirs.empty()) {
    Env* env = Env::Default();

    for (const auto& api_def_dir : api_def_dirs) {
      std::vector<string> api_files;
      TF_CHECK_OK(env->GetMatchingPaths(io::JoinPath(api_def_dir, "*.pbtxt"),
                                        &api_files));
      TF_CHECK_OK(api_def_map.LoadFileList(env, api_files));
    }
    api_def_map.UpdateDocs();
  }

  PrintEagerPythonOps(ops, api_def_map, hidden_ops, true /* require_shapes */);
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  // Usage:
  //   python_eager_op_gen_main api_def_dir1,api_def_dir2,...
  if (argc == 1) {
    tensorflow::PrintAllPythonOps({}, {});
  } else if (argc == 2) {
    const std::vector<tensorflow::string> api_def_dirs =
        tensorflow::str_util::Split(argv[1], ",",
                                    tensorflow::str_util::SkipEmpty());
    tensorflow::PrintAllPythonOps({}, api_def_dirs);
  } else {
    return -1;
  }
  return 0;
}
