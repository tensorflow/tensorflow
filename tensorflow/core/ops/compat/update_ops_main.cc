/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <stdio.h>

#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/ops/compat/op_compatibility_lib.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/public/release_version.h"
#include "tsl/platform/protobuf.h"

namespace tensorflow {
namespace {

void WriteUpdateTo(const string& directory) {
  OpCompatibilityLib compatibility(
      directory, strings::StrCat("v", TF_MAJOR_VERSION), nullptr);

  // Write full copy of all ops to ops.pbtxt.
  Env* env = Env::Default();
  {
    const string& ops_file = compatibility.ops_file();
    printf("Writing ops to %s...\n", ops_file.c_str());
    TF_QCHECK_OK(WriteStringToFile(env, ops_file, compatibility.OpsString()));
  }

  // Make sure the current version of ops are compatible with the
  // historical versions, and generate a new history adding all
  // changed ops.
  OpCompatibilityLib::OpHistory out_op_history;
  int changed_ops = 0;
  int added_ops = 0;
  TF_QCHECK_OK(compatibility.ValidateCompatible(env, &changed_ops, &added_ops,
                                                &out_op_history));
  printf("%d changed ops\n%d added ops\n", changed_ops, added_ops);

  const string& history_dir = compatibility.op_history_directory();
  absl::Status status = env->CreateDir(history_dir);
  if (!absl::IsAlreadyExists(status)) {
    TF_QCHECK_OK(status);
  }
  if (changed_ops + added_ops > 0 || !absl::IsAlreadyExists(status)) {
    // Write out new op history.
    printf("Writing updated op history to %s/...\n", history_dir.c_str());
    for (const auto& op_file : out_op_history) {
      TF_QCHECK_OK(
          WriteStringToFile(env, io::JoinPath(history_dir, op_file.first),
                            tsl::LegacyUnredactedDebugString(op_file.second)));
    }
  }
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc != 2) {
    printf("Usage: %s <core/ops directory>\n", argv[0]);
    return 1;
  }
  printf("TensorFlow version: %s\n", TF_VERSION_STRING);
  tensorflow::WriteUpdateTo(argv[1]);
  return 0;
}
