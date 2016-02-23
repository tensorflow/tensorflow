/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

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
      const char** argv = new const char*[argv_.size()];
      for (int i = 1; i < argv_.size(); ++i) {
        argv[i - 1] = argv_[i].c_str();
      }
      argv[argv_.size() - 1] = nullptr;
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

}  // namespace testing
}  // namespace tensorflow
