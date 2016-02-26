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

#include <cstdlib>
#include <unordered_set>

#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

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

namespace {
bool IsPortAvailable(int* port, bool is_tcp) {
  const int protocol = is_tcp ? IPPROTO_TCP : 0;
  const int fd = socket(AF_INET, is_tcp ? SOCK_STREAM : SOCK_DGRAM, protocol);

  struct sockaddr_in addr;
  socklen_t addr_len = sizeof(addr);
  int actual_port;

  CHECK_GE(*port, 0);
  CHECK_LE(*port, 65535);
  if (fd < 0) {
    LOG(ERROR) << "socket() failed: " << strerror(errno);
    return false;
  }

  // SO_REUSEADDR lets us start up a server immediately after it exists.
  int one = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) < 0) {
    LOG(ERROR) << "setsockopt() failed: " << strerror(errno);
    close(fd);
    return false;
  }

  // Try binding to port.
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons((uint16_t)*port);
  if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    LOG(WARNING) << "bind(port=" << *port << ") failed: " << strerror(errno);
    close(fd);
    return false;
  }

  // Get the bound port number.
  if (getsockname(fd, (struct sockaddr*)&addr, &addr_len) < 0) {
    LOG(WARNING) << "getsockname() failed: " << strerror(errno);
    close(fd);
    return false;
  }
  CHECK_LE(addr_len, sizeof(addr));
  actual_port = ntohs(addr.sin_port);
  CHECK_GT(actual_port, 0);
  if (*port == 0) {
    *port = actual_port;
  } else {
    CHECK_EQ(*port, actual_port);
  }
  close(fd);
  return true;
}

const int kNumRandomPortsToPick = 100;
const int kMaximumTrials = 1000;

}  // namespace

int PickUnusedPortOrDie() {
  static std::unordered_set<int> chosen_ports;

  // Type of port to first pick in the next iteration.
  bool is_tcp = true;
  int trial = 0;
  while (true) {
    int port;
    trial++;
    CHECK_LE(trial, kMaximumTrials)
        << "Failed to pick an unused port for testing.";
    if (trial == 1) {
      port = getpid() % (65536 - 30000) + 30000;
    } else if (trial <= kNumRandomPortsToPick) {
      port = rand() % (65536 - 30000) + 30000;
    } else {
      port = 0;
    }

    if (chosen_ports.find(port) != chosen_ports.end()) {
      continue;
    }
    if (!IsPortAvailable(&port, is_tcp)) {
      continue;
    }

    CHECK_GT(port, 0);
    if (!IsPortAvailable(&port, !is_tcp)) {
      is_tcp = !is_tcp;
      continue;
    }

    chosen_ports.insert(port);
    return port;
  }

  return 0;
}

string TensorFlowSrcRoot() {
  // 'bazel test' sets TEST_SRCDIR
  const char* env = getenv("TEST_SRCDIR");
  if (env && env[0] != '\0') {
    return strings::StrCat(env, "/tensorflow");
  } else {
    LOG(WARNING) << "TEST_SRCDIR environment variable not set: "
                 << "using $PWD/tensorflow as TensorFlowSrcRoot() for tests.";
    return "tensorflow";
  }
}

}  // namespace testing
}  // namespace tensorflow
