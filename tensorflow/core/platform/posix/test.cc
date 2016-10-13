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

#include <signal.h>

#include "tensorflow/core/platform/net.h"
#include "tensorflow/core/platform/test.h"

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace testing {

namespace {
class PosixSubProcess : public SubProcess {
 public:
  PosixSubProcess(const std::vector<string>& argv) : argv_(argv), pid_(0) {}

  ~PosixSubProcess() override {}

  bool Start() override {
    if (pid_ != 0) {
      LOG(ERROR) << "Tried to start process multiple times.";
      return false;
    }
    pid_ = fork();
    if (pid_ == 0) {
      // We are in the child process.
      const char* path = argv_[0].c_str();
      const char** argv = new const char*[argv_.size() + 1];
      int i = 0;
      for (const string& arg : argv_) {
        argv[i++] = arg.c_str();
      }
      argv[argv_.size()] = nullptr;
      execv(path, (char* const*)argv);
      // Never executes.
      return true;
    } else if (pid_ < 0) {
      LOG(ERROR) << "Failed to fork process.";
      return false;
    } else {
      // We are in the parent process and fork() was successful.
      // TODO(mrry): Consider collecting stderr from the child.
      return true;
    }
  }

  bool Kill(int signal) override {
    if (pid_ == 0) {
      LOG(ERROR) << "Tried to kill process before starting it.";
      return false;
    }
    return kill(pid_, signal) == 0;
  }

 private:
  const std::vector<string> argv_;
  pid_t pid_;
  TF_DISALLOW_COPY_AND_ASSIGN(PosixSubProcess);
};
}  // namespace

std::unique_ptr<SubProcess> CreateSubProcess(const std::vector<string>& argv) {
  return std::unique_ptr<SubProcess>(new PosixSubProcess(argv));
}

int PickUnusedPortOrDie() { return internal::PickUnusedPortOrDie(); }

string TensorFlowSrcRoot() {
  // 'bazel test' sets TEST_SRCDIR, and also TEST_WORKSPACE if a new
  // enough version of bazel is used.
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
