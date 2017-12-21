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

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/java/src/gen/cc/op_generator.h"

namespace tensorflow {
namespace {

string CamelCase(const string& str, char delimiter, bool upper) {
  string result;
  bool cap = upper;
  for (string::const_iterator it = str.begin(); it != str.end(); ++it) {
    const char c = *it;
    if (c == delimiter) {
      cap = true;
    } else if (cap) {
      result += toupper(c);
      cap = false;
    } else {
      result += c;
    }
  }
  return result;
}

}  // namespace

OpGenerator::OpGenerator() : env(Env::Default()) {}

OpGenerator::~OpGenerator() {}

Status OpGenerator::Run(const OpList& ops, const string& lib_name,
                        const string& base_package, const string& output_dir) {
  const string package =
      base_package + '.' + str_util::StringReplace(lib_name, "_", "", true);
  const string package_path =
      output_dir + '/' + str_util::StringReplace(package, ".", "/", true);
  const string group = CamelCase(lib_name, '_', false);

  if (!env->FileExists(package_path).ok()) {
    TF_CHECK_OK(env->RecursivelyCreateDir(package_path));
  }

  LOG(INFO) << "Generating Java wrappers for '" << lib_name << "' operations";
  // TODO(karllessard) generate wrappers from list of ops

  return Status::OK();
}

}  // namespace tensorflow
