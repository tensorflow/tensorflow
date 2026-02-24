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

#include "xla/tsl/platform/subprocess.h"

#include <stdlib.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"
#include "tsl/platform/strcat.h"

#ifndef _WIN32
#include <limits.h>
#endif

#ifdef PLATFORM_WINDOWS
#define WIFEXITED(code) ((code) != 3)
#define WEXITSTATUS(code) (code)
#define SIGKILL 9
#else
#include <sys/wait.h>
#endif

namespace tsl {
namespace {

std::string EchoProgram() {
  std::string path = io::JoinPath(testing::XlaSrcRoot(), "tsl", "platform",
                                  "testdata", "test_echo");
  return tsl::io::AppendDotExeIfWindows(path);
}

std::string EchoArgv1Program() {
  std::string path = io::JoinPath(testing::XlaSrcRoot(), "tsl", "platform",
                                  "testdata", "test_echo_argv_1");
  return tsl::io::AppendDotExeIfWindows(path);
}

std::string NoopProgram() {
  std::string path = io::JoinPath(testing::XlaSrcRoot(), "tsl", "platform",
                                  "testdata", "test_noop");
  return tsl::io::AppendDotExeIfWindows(path);
}

std::string PwdProgram() {
  std::string path = io::JoinPath(testing::XlaSrcRoot(), "tsl", "platform",
                                  "testdata", "test_pwd");
  return tsl::io::AppendDotExeIfWindows(path);
}

std::string StdErrProgram() {
  std::string path = io::JoinPath(testing::XlaSrcRoot(), "tsl", "platform",
                                  "testdata", "test_stderr");
  return tsl::io::AppendDotExeIfWindows(path);
}

std::string CrashProgram() {
  std::string path = io::JoinPath(testing::XlaSrcRoot(), "tsl", "platform",
                                  "testdata", "test_crash");
  return tsl::io::AppendDotExeIfWindows(path);
}

class SubProcessTest : public ::testing::Test {};

TEST_F(SubProcessTest, NoOutputNoComm) {
  tsl::SubProcess proc;
  proc.SetProgram(NoopProgram().c_str(), {NoopProgram()});
  EXPECT_TRUE(proc.Start());
  EXPECT_TRUE(proc.Wait());
}

TEST_F(SubProcessTest, NoOutput) {
  tsl::SubProcess proc;
  proc.SetProgram(NoopProgram().c_str(), {NoopProgram()});
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDERR, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  std::string out, err;
  int status = proc.Communicate(nullptr, &out, &err);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
  EXPECT_EQ("", out);
  EXPECT_EQ("", err);
}

TEST_F(SubProcessTest, Stdout) {
  tsl::SubProcess proc;
  const char test_string[] = "hello_world";
  proc.SetProgram(EchoArgv1Program().c_str(),
                  {EchoArgv1Program(), test_string});
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDERR, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  std::string out, err;
  int status = proc.Communicate(nullptr, &out, &err);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
  EXPECT_EQ(test_string, out);
  EXPECT_EQ("", err);
}

TEST_F(SubProcessTest, StdoutIgnored) {
  tsl::SubProcess proc;
  const char test_string[] = "hello_world";
  proc.SetProgram(EchoArgv1Program().c_str(),
                  {EchoArgv1Program(), test_string});
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDERR, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  int status = proc.Communicate(nullptr, nullptr, nullptr);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
}

TEST_F(SubProcessTest, Stderr) {
  tsl::SubProcess proc;
  const char test_string[] = "muh_failure!";
  proc.SetProgram(StdErrProgram().c_str(), {StdErrProgram(), test_string});
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDERR, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  std::string out, err;
  int status = proc.Communicate(nullptr, &out, &err);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_NE(0, WEXITSTATUS(status));
  EXPECT_EQ("", out);
  EXPECT_EQ(test_string, err);
}

TEST_F(SubProcessTest, StderrIgnored) {
  tsl::SubProcess proc;
  const char test_string[] = "muh_failure!";
  proc.SetProgram(StdErrProgram().c_str(), {StdErrProgram(), test_string});
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDERR, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  int status = proc.Communicate(nullptr, nullptr, nullptr);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_NE(0, WEXITSTATUS(status));
}

TEST_F(SubProcessTest, Stdin) {
  tsl::SubProcess proc;
  proc.SetProgram(EchoProgram().c_str(), {EchoProgram()});
  proc.SetChannelAction(CHAN_STDIN, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  std::string in = "foobar\nbarfoo\nhaha\n";
  int status = proc.Communicate(&in, nullptr, nullptr);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
}

TEST_F(SubProcessTest, StdinStdout) {
  tsl::SubProcess proc;
  proc.SetProgram(EchoProgram().c_str(), {EchoProgram()});
  proc.SetChannelAction(CHAN_STDIN, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  std::string in = "foobar\nbarfoo\nhaha\n";
  std::string out;
  int status = proc.Communicate(&in, &out, nullptr);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
  // Sanitize out of carriage returns, because windows...
  out.erase(std::remove(out.begin(), out.end(), '\r'), out.end());
  EXPECT_EQ(in, out);
}

TEST_F(SubProcessTest, StdinChildExit) {
  tsl::SubProcess proc;
  proc.SetProgram(NoopProgram().c_str(), {NoopProgram()});
  proc.SetChannelAction(CHAN_STDIN, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  // Verify that the parent handles the child exiting immediately as the
  // parent is trying to write a large string to the child's stdin.
  std::string in;
  in.reserve(1000000);
  for (int i = 0; i < 100000; i++) {
    in += "hello xyz\n";
  }

  int status = proc.Communicate(&in, nullptr, nullptr);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
}

TEST_F(SubProcessTest, StdinStdoutOverlap) {
  tsl::SubProcess proc;
  proc.SetProgram(EchoProgram().c_str(), {EchoProgram()});
  proc.SetChannelAction(CHAN_STDIN, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  // Verify that the parent handles multiplexed reading/writing to the child
  // process.  The string is large enough to exceed the buffering of the pipes.
  std::string in;
  in.reserve(1000000);
  for (int i = 0; i < 100000; i++) {
    in += "hello xyz\n";
  }

  std::string out;
  int status = proc.Communicate(&in, &out, nullptr);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
  // Sanitize out of carriage returns, because windows...
  out.erase(std::remove(out.begin(), out.end(), '\r'), out.end());
  EXPECT_EQ(in, out);
}

TEST_F(SubProcessTest, KillProc) {
  tsl::SubProcess proc;
  proc.SetProgram(EchoProgram().c_str(), {EchoProgram()});
  proc.SetChannelAction(CHAN_STDIN, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  EXPECT_TRUE(proc.Kill(SIGKILL));
  EXPECT_TRUE(proc.Wait());

  EXPECT_FALSE(proc.Kill(SIGKILL));
}

TEST_F(SubProcessTest, ExitCallbackNormal) {
  tsl::SubProcess proc;
  proc.SetProgram(NoopProgram(), {NoopProgram()});
  absl::Notification notification;
  proc.SetExitCallback([&](SubProcess* p) { notification.Notify(); });

  EXPECT_TRUE(proc.Start());
  EXPECT_TRUE(proc.Wait());
  EXPECT_TRUE(notification.HasBeenNotified());
}

TEST_F(SubProcessTest, ExitCallbackChildKilled) {
  tsl::SubProcess proc;
  proc.SetProgram(EchoProgram(), {EchoProgram()});
  proc.SetChannelAction(CHAN_STDIN, ACTION_PIPE);
  absl::Notification notification;
  proc.SetExitCallback([&](SubProcess* p) { notification.Notify(); });

  EXPECT_TRUE(proc.Start());
  EXPECT_TRUE(proc.Kill(SIGKILL));
  EXPECT_TRUE(proc.Wait());
  EXPECT_TRUE(notification.HasBeenNotified());
}

TEST_F(SubProcessTest, ExitCallbackChildCrash) {
  tsl::SubProcess proc;
  proc.SetProgram(CrashProgram(), {CrashProgram()});
  proc.SetChannelAction(CHAN_STDIN, ACTION_PIPE);
  absl::Notification notification;
  proc.SetExitCallback([&](SubProcess* p) { notification.Notify(); });

  EXPECT_TRUE(proc.Start());
  EXPECT_TRUE(proc.Wait());
  EXPECT_TRUE(notification.HasBeenNotified());
}

TEST_F(SubProcessTest, CheckRunningBeforeStart) {
  tsl::SubProcess proc;
  proc.SetProgram(NoopProgram(), {NoopProgram()});
  EXPECT_FALSE(proc.CheckRunning());
}

TEST_F(SubProcessTest, CheckRunningWhileRunning) {
  tsl::SubProcess proc;
  proc.SetProgram(EchoProgram(), {EchoProgram()});
  proc.SetChannelAction(CHAN_STDIN, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());
  EXPECT_TRUE(proc.CheckRunning());  // Should be running, waiting for stdin
  EXPECT_TRUE(proc.Kill(SIGKILL));
  EXPECT_TRUE(proc.Wait());
  EXPECT_FALSE(proc.CheckRunning());  // Should be false now
}

TEST_F(SubProcessTest, CheckRunningReapsProcess) {
  tsl::SubProcess proc;
  proc.SetProgram(NoopProgram(), {NoopProgram()});
  absl::Notification notification;
  proc.SetExitCallback([&](SubProcess* p) { notification.Notify(); });

  EXPECT_TRUE(proc.Start());
  // Poll until process exits and CheckRunning returns false.
  while (proc.CheckRunning()) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  EXPECT_TRUE(notification.HasBeenNotified());
  // Since CheckRunning reaped it, Wait should return false.
  EXPECT_FALSE(proc.Wait());
}

TEST_F(SubProcessTest, ConcurrentWaitAndCheckRunning) {
  tsl::SubProcess proc;
  proc.SetProgram(EchoProgram(), {EchoProgram()});
  proc.SetChannelAction(CHAN_STDIN, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  absl::Notification start_signal;
  absl::Notification wait_finished;

  std::unique_ptr<tsl::Thread> wait_thread(tsl::Env::Default()->StartThread(
      tsl::ThreadOptions(), "WaitThread", [&]() {
        start_signal.WaitForNotification();
        proc.Wait();
        wait_finished.Notify();
      }));

  std::unique_ptr<tsl::Thread> check_thread(tsl::Env::Default()->StartThread(
      tsl::ThreadOptions(), "CheckThread", [&]() {
        start_signal.WaitForNotification();
        // Keep checking until Wait() finishes, even if CheckRunning returns
        // false to test race condition.
        while (!wait_finished.HasBeenNotified()) {
          proc.CheckRunning();
          absl::SleepFor(absl::Milliseconds(1));
        }
        while (proc.CheckRunning()) {
          absl::SleepFor(absl::Milliseconds(10));
        }
      }));

  start_signal.Notify();
  absl::SleepFor(absl::Milliseconds(100));
  EXPECT_TRUE(proc.Kill(SIGKILL));
  wait_thread.reset();  // Join threads
  check_thread.reset();
  EXPECT_FALSE(proc.CheckRunning());
}

TEST_F(SubProcessTest, ConcurrentCheckRunning) {
  tsl::SubProcess proc;
  proc.SetProgram(EchoProgram(), {EchoProgram()});
  proc.SetChannelAction(CHAN_STDIN, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  absl::Notification start_signal;
  absl::Notification kill_signal;
  std::vector<std::unique_ptr<tsl::Thread>> threads;

  for (int i = 0; i < 2; ++i) {
    threads.emplace_back(tsl::Env::Default()->StartThread(
        tsl::ThreadOptions(), "CheckThread", [&]() {
          start_signal.WaitForNotification();
          // Loop until kill signal even if CheckRunning returns false to
          // test race condition.
          while (!kill_signal.HasBeenNotified()) {
            proc.CheckRunning();
            absl::SleepFor(absl::Milliseconds(1));
          }
          // After kill signal, loop until CheckRunning returns false
          while (proc.CheckRunning()) {
            absl::SleepFor(absl::Milliseconds(1));
          }
        }));
  }

  start_signal.Notify();
  absl::SleepFor(absl::Milliseconds(100));
  kill_signal.Notify();
  EXPECT_TRUE(proc.Kill(SIGKILL));

  for (auto& thread : threads) {
    thread.reset();  // Join threads
  }
  EXPECT_FALSE(proc.CheckRunning());
}

TEST_F(SubProcessTest, SetDirectory) {
  tsl::SubProcess proc;
  proc.SetProgram(PwdProgram(), {PwdProgram()});

  std::string dir = io::JoinPath(::testing::TempDir(), "setdir_test");
  TF_ASSERT_OK(Env::Default()->CreateDir(dir));

  if (!proc.SetDirectory(dir)) {
    GTEST_SKIP() << "SetDirectory not supported on this platform.";
  }

  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDERR, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  std::string out, err;
  int status = proc.Communicate(nullptr, &out, &err);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
  EXPECT_EQ(dir, out);
  EXPECT_EQ("", err);
}

TEST_F(SubProcessTest, ExitStatusNormal) {
  tsl::SubProcess proc;
  proc.SetProgram(NoopProgram(), {NoopProgram()});
  EXPECT_FALSE(proc.CheckRunning());
  EXPECT_TRUE(proc.exit_normal());
  EXPECT_EQ(proc.exit_status(), 0);
}

TEST_F(SubProcessTest, KillStatus) {
  tsl::SubProcess proc;
  proc.SetProgram(EchoProgram(), {EchoProgram()});
  proc.SetChannelAction(CHAN_STDIN, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());
  EXPECT_TRUE(proc.CheckRunning());  // Should be running, waiting for stdin
  EXPECT_TRUE(proc.Kill(SIGKILL));
  EXPECT_TRUE(proc.Wait());
  EXPECT_FALSE(proc.exit_normal());
  EXPECT_EQ(proc.exit_status(), SIGKILL);
}

TEST_F(SubProcessTest, ExitStderr) {
  tsl::SubProcess proc;
  const char test_string[] = "failure!";
  proc.SetProgram(StdErrProgram(), {StdErrProgram(), test_string});
  proc.SetChannelAction(CHAN_STDOUT, ACTION_PIPE);
  proc.SetChannelAction(CHAN_STDERR, ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  proc.Communicate(nullptr, nullptr, nullptr);
  EXPECT_NE(proc.exit_status(), 0);
  EXPECT_FALSE(proc.exit_normal());
}
}  // namespace
}  // namespace tsl
