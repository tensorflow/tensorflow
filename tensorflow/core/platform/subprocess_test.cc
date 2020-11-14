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

#include "tensorflow/core/platform/subprocess.h"

#include <stdlib.h>
#include <string.h>

#include <algorithm>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/test.h"

#ifdef PLATFORM_WINDOWS
#define WIFEXITED(code) ((code) != 3)
#define WEXITSTATUS(code) (code)
#define SIGKILL 9
#else
#include <sys/wait.h>
#endif

namespace tensorflow {
namespace {

static string GetDataFilePath(const string& relative_path) {
#ifdef PLATFORM_WINDOWS
  // While CreateProcess on windows is resilient to not having ".exe" suffix,
  // Bazel_tools has to have the exact file path to return the resource.
  return GetDataDependencyFilepath(strings::StrCat(relative_path, ".exe"));
#else
  return GetDataDependencyFilepath(relative_path);
#endif
}

string EchoProgram() {
  return io::JoinPath("tensorflow", "core", "platform", "testdata",
                      "test_echo");
}

string EchoArgv1Program() {
  return io::JoinPath("tensorflow", "core", "platform", "testdata",
                      "test_echo_argv_1");
}

string NoopProgram() {
  return io::JoinPath("tensorflow", "core", "platform", "testdata",
                      "test_noop");
}

string StdErrProgram() {
  return io::JoinPath("tensorflow", "core", "platform", "testdata",
                      "test_stderr");
}

class SubProcessTest : public ::testing::Test {};

TEST_F(SubProcessTest, NoOutputNoComm) {
  tensorflow::SubProcess proc;
  proc.SetProgram(GetDataFilePath(NoopProgram()).c_str(), {NoopProgram()});
  EXPECT_TRUE(proc.Start());
  EXPECT_TRUE(proc.Wait());
}

TEST_F(SubProcessTest, NoOutput) {
  tensorflow::SubProcess proc;
  proc.SetProgram(GetDataFilePath(NoopProgram()).c_str(), {NoopProgram()});
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
  const char test_string[] = "hello_world";
  proc.SetProgram(GetDataFilePath(EchoArgv1Program()).c_str(),
                  {EchoArgv1Program(), test_string});
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDERR, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  string out, err;
  int status = proc.Communicate(nullptr, &out, &err);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
  EXPECT_EQ(test_string, out);
  EXPECT_EQ("", err);
}

TEST_F(SubProcessTest, StdoutIgnored) {
  tensorflow::SubProcess proc;
  const char test_string[] = "hello_world";
  proc.SetProgram(GetDataFilePath(EchoArgv1Program()).c_str(),
                  {EchoArgv1Program(), test_string});
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDERR, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  int status = proc.Communicate(nullptr, nullptr, nullptr);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
}

TEST_F(SubProcessTest, Stderr) {
  tensorflow::SubProcess proc;
  const char test_string[] = "muh_failure!";
  proc.SetProgram(GetDataFilePath(StdErrProgram()).c_str(),
                  {StdErrProgram(), test_string});
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDERR, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  string out, err;
  int status = proc.Communicate(nullptr, &out, &err);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_NE(0, WEXITSTATUS(status));
  EXPECT_EQ("", out);
  EXPECT_EQ(test_string, err);
}

TEST_F(SubProcessTest, StderrIgnored) {
  tensorflow::SubProcess proc;
  const char test_string[] = "muh_failure!";
  proc.SetProgram(GetDataFilePath(StdErrProgram()).c_str(),
                  {StdErrProgram(), test_string});
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDERR, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  int status = proc.Communicate(nullptr, nullptr, nullptr);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_NE(0, WEXITSTATUS(status));
}

TEST_F(SubProcessTest, Stdin) {
  tensorflow::SubProcess proc;
  proc.SetProgram(GetDataFilePath(EchoProgram()).c_str(), {EchoProgram()});
  proc.SetChannelAction(CHAN_STDIN, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  string in = "foobar\nbarfoo\nhaha\n";
  int status = proc.Communicate(&in, nullptr, nullptr);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
}

TEST_F(SubProcessTest, StdinStdout) {
  tensorflow::SubProcess proc;
  proc.SetProgram(GetDataFilePath(EchoProgram()).c_str(), {EchoProgram()});
  proc.SetChannelAction(CHAN_STDIN, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  string in = "foobar\nbarfoo\nhaha\n";
  string out;
  int status = proc.Communicate(&in, &out, nullptr);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
  // Sanitize out of carriage returns, because windows...
  out.erase(std::remove(out.begin(), out.end(), '\r'), out.end());
  EXPECT_EQ(in, out);
}

TEST_F(SubProcessTest, StdinChildExit) {
  tensorflow::SubProcess proc;
  proc.SetProgram(GetDataFilePath(NoopProgram()).c_str(), {NoopProgram()});
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
  proc.SetProgram(GetDataFilePath(EchoProgram()).c_str(), {EchoProgram()});
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
  // Sanitize out of carriage returns, because windows...
  out.erase(std::remove(out.begin(), out.end(), '\r'), out.end());
  EXPECT_EQ(in, out);
}

TEST_F(SubProcessTest, KillProc) {
  tensorflow::SubProcess proc;
  proc.SetProgram(GetDataFilePath(EchoProgram()).c_str(), {EchoProgram()});
  proc.SetChannelAction(CHAN_STDIN, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  EXPECT_TRUE(proc.Kill(SIGKILL));
  EXPECT_TRUE(proc.Wait());

  EXPECT_FALSE(proc.Kill(SIGKILL));
}

}  // namespace
}  // namespace tensorflow
