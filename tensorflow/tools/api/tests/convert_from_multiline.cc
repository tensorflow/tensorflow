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
// Converts all *.pbtxt files in a directory from Multiline to proto format.
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"

namespace tensorflow {

namespace {
constexpr char kApiDefFilePattern[] = "*.pbtxt";

Status ConvertFilesFromMultiline(const string& input_dir,
                                 const string& output_dir) {
  Env* env = Env::Default();

  const string file_pattern = io::JoinPath(input_dir, kApiDefFilePattern);
  std::vector<string> matching_paths;
  TF_CHECK_OK(env->GetMatchingPaths(file_pattern, &matching_paths));

  if (!env->IsDirectory(output_dir).ok()) {
    TF_RETURN_IF_ERROR(env->CreateDir(output_dir));
  }

  for (const auto& path : matching_paths) {
    string contents;
    TF_RETURN_IF_ERROR(tensorflow::ReadFileToString(env, path, &contents));
    contents = tensorflow::PBTxtFromMultiline(contents);
    string output_path = io::JoinPath(output_dir, io::Basename(path));
    // Write contents to output_path
    TF_RETURN_IF_ERROR(
        tensorflow::WriteStringToFile(env, output_path, contents));
  }
  return Status::OK();
}
}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  const std::string usage =
      "Usage: convert_from_multiline input_dir output_dir";
  if (argc != 3) {
    std::cerr << usage << std::endl;
    return -1;
  }
  TF_CHECK_OK(tensorflow::ConvertFilesFromMultiline(argv[1], argv[2]));
  return 0;
}
