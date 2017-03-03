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

#include <sys/wait.h>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class SubProcessTest : public ::testing::Test {};

TEST_F(SubProcessTest, NoOutputNoComm) {
  tensorflow::SubProcess proc;
  proc.SetProgram("/bin/cat", {"cat", "/dev/null"});
  EXPECT_TRUE(proc.Start());
  EXPECT_TRUE(proc.Wait());
}

TEST_F(SubProcessTest, NoOutput) {
  tensorflow::SubProcess proc;
  proc.SetProgram("/bin/cat", {"cat", "/dev/null"});
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDERR, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  string out, err;
  int status = proc.Communicate(nullptr, &out, &err);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
  EXPECT_EQ("", out);
  EXPECT_EQ("", err);
}

TEST_F(SubProcessTest, Stdout) {
  tensorflow::SubProcess proc;
  proc.SetProgram("/bin/echo", {"echo", "-n", "hello world"});
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDERR, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  string out, err;
  int status = proc.Communicate(nullptr, &out, &err);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
  EXPECT_EQ("hello world", out);
  EXPECT_EQ("", err);
}

TEST_F(SubProcessTest, StdoutIgnored) {
  tensorflow::SubProcess proc;
  proc.SetProgram("/bin/echo", {"echo", "-n", "hello world"});
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDERR, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  int status = proc.Communicate(nullptr, nullptr, nullptr);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
}

TEST_F(SubProcessTest, Stderr) {
  tensorflow::SubProcess proc;
  proc.SetProgram("/bin/cat", {"cat", "/file_does_not_exist"});
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDERR, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  string out, err;
  int status = proc.Communicate(nullptr, &out, &err);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(1, WEXITSTATUS(status));
  EXPECT_EQ("", out);
  EXPECT_NE(string::npos, err.find("/file_does_not_exist"));
}

TEST_F(SubProcessTest, StderrIgnored) {
  tensorflow::SubProcess proc;
  proc.SetProgram("/bin/cat", {"cat", "/file_does_not_exist"});
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDERR, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  int status = proc.Communicate(nullptr, nullptr, nullptr);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(1, WEXITSTATUS(status));
}

TEST_F(SubProcessTest, Stdin) {
  tensorflow::SubProcess proc;
  proc.SetProgram("/usr/bin/wc", {"wc", "-l"});
  proc.SetChannelAction(CHAN_STDIN, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  string in = "foobar\nbarfoo\nhaha\n";
  int status = proc.Communicate(&in, nullptr, nullptr);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
}

TEST_F(SubProcessTest, StdinStdout) {
  tensorflow::SubProcess proc;
  proc.SetProgram("/usr/bin/wc", {"wc", "-l"});
  proc.SetChannelAction(CHAN_STDIN, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  string in = "foobar\nbarfoo\nhaha\n";
  string out;
  int status = proc.Communicate(&in, &out, nullptr);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
  int count = stoi(out);
  EXPECT_EQ(3, count);
}

TEST_F(SubProcessTest, StdinChildExit) {
  tensorflow::SubProcess proc;
  proc.SetProgram("/bin/sleep", {"sleep", "0"});
  proc.SetChannelAction(CHAN_STDIN, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  // Verify that the parent handles the child exiting immediately as the
  // parent is trying to write a large string to the child's stdin.
  string in;
  in.reserve(1000000);
  for (int i = 0; i < 100000; i++) {
    in += "hello xyz\n";
  }

  int status = proc.Communicate(&in, nullptr, nullptr);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
}

TEST_F(SubProcessTest, StdinStdoutOverlap) {
  tensorflow::SubProcess proc;
  proc.SetProgram("/bin/cat", {"cat"});
  proc.SetChannelAction(CHAN_STDIN, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  // Verify that the parent handles multiplexed reading/writing to the child
  // process.  The string is large enough to exceed the buffering of the pipes.
  string in;
  in.reserve(1000000);
  for (int i = 0; i < 100000; i++) {
    in += "hello xyz\n";
  }

  string out;
  int status = proc.Communicate(&in, &out, nullptr);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
  EXPECT_EQ(in, out);
}

TEST_F(SubProcessTest, KillProc) {
  tensorflow::SubProcess proc;
  proc.SetProgram("/bin/cat", {"cat"});
  proc.SetChannelAction(CHAN_STDIN, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  EXPECT_TRUE(proc.Kill(SIGKILL));
  EXPECT_TRUE(proc.Wait());

  EXPECT_FALSE(proc.Kill(SIGKILL));
}

}  // namespace tensorflow
