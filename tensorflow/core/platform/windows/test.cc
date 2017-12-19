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

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/net.h"

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace testing {

std::unique_ptr<SubProcess> CreateSubProcess(const std::vector<string>& argv) {
  LOG(FATAL) << "CreateSubProcess NOT IMPLEMENTED for Windows yet ! ";
  return nullptr;
}

int PickUnusedPortOrDie() { return internal::PickUnusedPortOrDie(); }

static void fill_manifest(std::unordered_map<string, string>& out,
                          const std::string& line, size_t line_begin,
                          size_t line_end) {
  // line[line_end] == '\0' or line[line_end] == '\n'
  if (line[line_end - 1] == '\r') {
    --line_end;
  }
  size_t space_location = string::npos;
  for (size_t i = line_begin; i != line_end; ++i) {
    if (line[i] == ' ') {
      if (space_location != string::npos) {
        LOG(FATAL) << "manifest file format error, contains multiple spaces in "
                      "one line";
      }
      space_location = i;
    }
  }
  if (space_location == string::npos) {
    LOG(FATAL) << "manifest file format error, contains no spaces in this line";
  }
  const size_t key_len = space_location - line_begin;
  const size_t value_len = line_end - space_location - 1;
  out[line.substr(line_begin, key_len)] =
      line.substr(space_location + 1, value_len);
}

void RunFileRelocator::Init() {
  size_t requiredSize;
  getenv_s(&requiredSize, NULL, 0, "RUNFILES_MANIFEST_FILE");
  if (requiredSize != 0) {
    string manifest_file_path(requiredSize - 1, '\0');
    getenv_s(&requiredSize, (char*)manifest_file_path.data(), requiredSize,
             "RUNFILES_MANIFEST_FILE");
    string manifest_file_content;
    TF_CHECK_OK(ReadFileToString(Env::Default(), manifest_file_path,
                                 &manifest_file_content));
    size_t line_begin = 0;
    size_t line_end;
    while ((line_end = manifest_file_content.find('\n', line_begin)) !=
           string::npos) {
      fill_manifest(manifest, manifest_file_content, line_begin, line_end);
      line_begin = line_end + 1;
    }
    // Now line_end == string::npos
    if (line_begin < manifest_file_content.size()) {
      line_end = manifest_file_content.size();
      fill_manifest(manifest, manifest_file_content, line_begin, line_end);
    }
  }
  src_root = "/tensorflow";
  getenv_s(&requiredSize, NULL, 0, "TEST_WORKSPACE");
  if (requiredSize == 0) {
    return;
  }
  string workspace_path(requiredSize - 1, '\0');
  getenv_s(&requiredSize, (char*)workspace_path.data(), requiredSize,
           "TEST_WORKSPACE");
  src_root = workspace_path + src_root;
}
std::unique_ptr<RunFileRelocator> RunFileRelocator::m_instance;
std::once_flag RunFileRelocator::once;
std::string RunFileRelocator::Relocate(const string& runfile) const {
  // TODO: normalize path
  auto iter = manifest.find(strings::StrCat(src_root, "/", runfile));
  if (iter != manifest.end()) {
    return iter->second;
  }
  //"using $PWD/tensorflow as TensorFlowSrcRoot() for tests."
  return strings::StrCat("tensorflow/", runfile);
}

string TensorFlowSrcRoot() {
  // 'bazel test' and cmake set TEST_SRCDIR.
  // New versions of bazel also set TEST_WORKSPACE.
  const char* env = getenv("TEST_SRCDIR");
  const char* workspace = getenv("TEST_WORKSPACE");
  if (env && env[0] != '\0') {
    if (workspace && workspace[0] != '\0') {
      return strings::StrCat(env, "/", workspace, "/tensorflow");
    } else {
      return strings::StrCat(env, "/tensorflow");
    }
  } else {
    LOG(WARNING) << "TEST_SRCDIR environment variable not set: "
                 << "using $PWD/tensorflow as TensorFlowSrcRoot() for tests.";
    return "tensorflow";
  }
}

}  // namespace testing
}  // namespace tensorflow
