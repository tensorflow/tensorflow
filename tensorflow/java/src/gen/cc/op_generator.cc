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

#include <string>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/java/src/gen/cc/op_generator.h"

namespace tensorflow {

OpGenerator::OpGenerator(Env* env, const string& output_dir)
  : env(env), output_path(output_dir + "/src/main/java/") {
}

OpGenerator::~OpGenerator() {}

Status OpGenerator::Run(const string& ops_file, const OpList& ops) {
  const string& lib_name = ops_file.substr(0, ops_file.find_last_of('_'));
  const string package_name =
      str_util::StringReplace("org.tensorflow.op." + lib_name, "_", "", true);
  const string package_path =
      output_path + str_util::StringReplace(package_name, ".", "/", true);

  if (!env->FileExists(package_path).ok()) {
    TF_CHECK_OK(env->RecursivelyCreateDir(package_path));
  }

  LOG(INFO) << "Generating Java wrappers for \"" << lib_name << "\" operations";
  // TODO(karllessard) generate wrappers from list of ops

  return Status::OK();
}

}  // namespace tensorflow
