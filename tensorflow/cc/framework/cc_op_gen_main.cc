/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/framework/cc_op_gen.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

void PrintAllCCOps(const std::string& dot_h, const std::string& dot_cc,
                   const std::string& overrides_fnames, bool include_internal,
                   const std::vector<string>& api_def_dirs) {
  OpList ops;
  OpRegistry::Global()->Export(include_internal, &ops);
  ApiDefMap api_def_map(ops);
  if (!api_def_dirs.empty()) {
    Env* env = Env::Default();
    // Only load files that correspond to "ops".
    for (const auto& op : ops.op()) {
      for (const auto& api_def_dir : api_def_dirs) {
        const std::string api_def_file_pattern =
            io::JoinPath(api_def_dir, "api_def_" + op.name() + ".pbtxt");
        if (env->FileExists(api_def_file_pattern).ok()) {
          TF_CHECK_OK(api_def_map.LoadFile(env, api_def_file_pattern));
        }
      }
    }
  }

  api_def_map.UpdateDocs();

  WriteCCOps(ops, api_def_map, dot_h, dot_cc, overrides_fnames);
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  // TODO(annarev): Update this file to no longer take op_gen_overrides.pbtxt
  // as an argument.
  if (argc != 6) {
    for (int i = 1; i < argc; ++i) {
      fprintf(stderr, "Arg %d = %s\n", i, argv[i]);
    }
    fprintf(stderr,
            "Usage: %s out.h out.cc overrides1.pbtxt,2.pbtxt include_internal "
            "api_def_dirs1,api_def_dir2 ...\n"
            "  include_internal: 1 means include internal ops\n",
            argv[0]);
    exit(1);
  }

  bool include_internal = tensorflow::StringPiece("1") == argv[4];
  std::vector<tensorflow::string> api_def_dirs = tensorflow::str_util::Split(
      argv[5], ",", tensorflow::str_util::SkipEmpty());
  tensorflow::PrintAllCCOps(argv[1], argv[2], argv[3], include_internal,
                            api_def_dirs);
  return 0;
}
