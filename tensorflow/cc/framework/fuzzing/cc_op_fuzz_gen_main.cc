// Main executable to generate op fuzzers

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
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/cc/framework/cc_op_gen_util.h"
#include "tensorflow/cc/framework/fuzzing/cc_op_fuzz_gen.h"
#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace cc_op {
namespace {

void WriteAllFuzzers(string root_location, std::vector<string> api_def_dirs,
                     std::vector<string> op_names) {
  OpList ops;
  absl::StatusOr<ApiDefMap> api_def_map =
      LoadOpsAndApiDefs(ops, false, api_def_dirs);

  TF_CHECK_OK(api_def_map.status());

  Env* env = Env::Default();
  absl::Status status;
  std::unique_ptr<WritableFile> fuzz_file = nullptr;
  for (const OpDef& op_def : ops.op()) {
    if (std::find(op_names.begin(), op_names.end(), op_def.name()) ==
        op_names.end())
      continue;

    const ApiDef* api_def = api_def_map->GetApiDef(op_def.name());
    if (api_def == nullptr) {
      continue;
    }

    OpInfo op_info(op_def, *api_def, std::vector<string>());
    status.Update(env->NewWritableFile(
        root_location + "/" + op_def.name() + "_fuzz.cc", &fuzz_file));
    status.Update(
        fuzz_file->Append(WriteSingleFuzzer(op_info, OpFuzzingIsOk(op_info))));
    status.Update(fuzz_file->Close());
  }
  TF_CHECK_OK(status);
}

}  // namespace
}  // namespace cc_op
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc != 4) {
    for (int i = 1; i < argc; ++i) {
      fprintf(stderr, "Arg %d = %s\n", i, argv[i]);
    }
    fprintf(stderr, "Usage: %s location api_def1,api_def2 op1,op2,op3\n",
            argv[0]);
    exit(1);
  }
  for (int i = 1; i < argc; ++i) {
    fprintf(stdout, "Arg %d = %s\n", i, argv[i]);
  }
  std::vector<tensorflow::string> api_def_srcs = tensorflow::str_util::Split(
      argv[2], ",", tensorflow::str_util::SkipEmpty());
  std::vector<tensorflow::string> op_names = tensorflow::str_util::Split(
      argv[3], ",", tensorflow::str_util::SkipEmpty());
  tensorflow::cc_op::WriteAllFuzzers(argv[1], api_def_srcs, op_names);
  return 0;
}
