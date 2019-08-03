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
// Testing proper operation of the stacktrace handler.

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <string>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

#define READ_BUFFER_SIZE 1024

TEST(StacktraceHandlerTest, GeneratesStacktrace) {
  // Create a pipe to write/read the child stdout.
  int test_pipe[2];
  EXPECT_EQ(pipe(test_pipe), 0);

  // Fork the process.
  int test_pid = fork();

  if (test_pid == 0) {
    // Child process.
    // Close the read end of the pipe, redirect stdout and sleep.
    close(test_pipe[0]);
    dup2(test_pipe[1], STDOUT_FILENO);
    dup2(test_pipe[1], STDERR_FILENO);
    sleep(10);
  } else {
    // Parent process.
    // Close the write end of the pipe, wait a little and send SIGABRT to the
    // child process. Then watch the pipe.
    close(test_pipe[1]);
    sleep(1);

    // Send the signal.
    kill(test_pid, SIGABRT);

    // Read from the pipe.
    char buffer[READ_BUFFER_SIZE];
    std::string child_output = "";
    while (true) {
      int read_length = read(test_pipe[0], buffer, READ_BUFFER_SIZE);
      if (read_length > 0) {
        child_output += std::string(buffer, read_length);
      } else {
        break;
      }
    }
    close(test_pipe[0]);

    // Just make sure we can detect one of the calls in testing stack.
    string test_stack_frame = "testing::internal::UnitTestImpl::RunAllTests()";

    // Print the stack trace detected for information.
    LOG(INFO) << "Output from the child process:";
    LOG(INFO) << child_output;

    EXPECT_NE(child_output.find(test_stack_frame), std::string::npos);
  }
}

}  // namespace
}  // namespace tensorflow
