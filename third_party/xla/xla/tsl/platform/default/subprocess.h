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

#ifndef XLA_TSL_PLATFORM_DEFAULT_SUBPROCESS_H_
#define XLA_TSL_PLATFORM_DEFAULT_SUBPROCESS_H_

#include <errno.h>
#include <sys/wait.h>
#include <unistd.h>

#include <functional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/types.h"
#include "tsl/platform/thread_annotations.h"

namespace tsl {

class SubProcess {
 public:
  using EnvMap = absl::flat_hash_map<std::string, std::string>;

  // SubProcess()
  //    nfds: The number of file descriptors to use.
  explicit SubProcess(int nfds = 3);

  // Virtual for backwards compatibility; do not create new subclasses.
  // It is illegal to delete the SubProcess within its exit callback.
  virtual ~SubProcess();

  // SetChannelAction()
  //    Set how to handle a channel.  The default action is ACTION_CLOSE.
  //    The action is set for all subsequent processes, until SetChannel()
  //    is called again.
  //
  //    SetChannel may not be called while the process is running.
  //
  //    chan: Which channel this applies to.
  //    action: What to do with the channel.
  // Virtual for backwards compatibility; do not create new subclasses.
  virtual void SetChannelAction(Channel chan, ChannelAction action);

  // GetFD()
  //    Get the actual file descriptor for the given channel.
  //    All of the fds will be -1 if the process isn't running
  //    or if ChannelAction for channel is not ACTION_PIPE.
  //
  //    Fatal error conditions:
  //      Invalid channel number;
  //
  //    chan: Which channel?
  //    Return file descriptor.
  virtual inline int GetFD(Channel chan) const {
    if (!chan_valid(chan)) {
      LOG(FATAL) << "GetFD called with invalid channel: " << chan;
    }
    absl::MutexLock dataLock(&data_mu_);
    return parent_pipe_[chan];
  }

  // SetProgram()
  //    Set up a program and argument list for execution, with the full
  //    "raw" argument list passed as a vector of strings.  argv[0]
  //    should be the program name, just as in execv().
  //
  //    file: The file containing the program.  This must be an absolute path
  //          name - $PATH is not searched.
  //    argv: The argument list.
  virtual void SetProgram(const string& file, const std::vector<string>& argv);

  // SetEnviron()
  //    set the environment that the child process will exec in.
  virtual void SetEnviron(const EnvMap& environ);

  // SetDirectory()
  //    In the child process, chdir() to this directory before
  //    exec-ing.
  //    Returns false if this is not supported on the current platform.
  ABSL_MUST_USE_RESULT virtual bool SetDirectory(const string& dir);

  // SetExitCallback()
  //    Set a callback to be run when the process exits.
  //    It is illegal to delete the SubProcess within its exit callback.
  virtual void SetExitCallback(std::function<void(SubProcess*)> cb);

  // Start()
  //    Run the command that was previously set up with SetProgram().
  //    The following are fatal programming errors:
  //       * Attempting to start when a process is already running.
  //       * Attempting to start without first setting the command.
  //    Note, however, that Start() does not try to validate that the binary
  //    does anything reasonable (e.g. exists or can execute); as such, you can
  //    specify a non-existent binary and Start() will still return true.  You
  //    will get a failure from the process, but only after Start() returns.
  //
  //    Return true normally, or false if the program couldn't be started
  //    because of some error.
  // Virtual for backwards compatibility; do not create new subclasses.
  virtual bool Start();

  // Kill()
  //    Send the given signal to the process.
  //    Return true normally, or false if we couldn't send the signal - likely
  //    because the process doesn't exist.
  virtual bool Kill(int signal);

  // running()
  //    Return true if the process is currently running. This just checks the
  //    most-recently-known status.
  virtual bool running() const;

  // CheckRunning()
  //    Check to see if the process is still running.
  //    @return false, if the process has exited;
  //            true, if the process is still running.
  virtual bool CheckRunning();

  // Wait()
  //    Block until the process exits.
  //    Return true normally, or false if the process wasn't running or the
  //    process had already exited and this fact had been reported in the return
  //    value of another call of Wait() or CheckRunning().
  virtual bool Wait();

  // pid()
  //    Return the process ID of the child process.
  virtual inline pid_t pid() const {
    absl::MutexLock lock(&proc_mu_);
    return pid_;
  }

  //  Return the raw exit status of the process.
  virtual inline int exit_status() const {
    absl::MutexLock lock(wait_mu_);
    return exit_status_;
  }

  //  Return a useful string describing why the child failed
  virtual std::string error_text() const {
    absl::MutexLock lock(&data_mu_);
    return error_text_;
  }

  //  Return true if the process exited successfully
  //  (zero return code, no signal).
  virtual inline bool exit_normal() const { return exit_status() == 0; }

  //  Return the exit code, assuming that the process wasn't killed by
  //  a signal.
  virtual inline int exit_code() const {
    int status = exit_status();
    return WIFEXITED(status) ? WEXITSTATUS(status)
                             : static_cast<int>(WaitStatus::kWasKilled);
  }

  // Communicate()
  //    Read from stdout and stderr and writes to stdin until all pipes have
  //    closed, then waits for the process to exit.
  //    Note: Do NOT call Wait() after calling Communicate as it will always
  //     fail, since Communicate calls Wait() internally.
  //    'stdin_input', 'stdout_output', and 'stderr_output' may be NULL.
  //    If this process is not configured to send stdout or stderr to pipes,
  //     the output strings will not be modified.
  //    If this process is not configured to take stdin from a pipe, stdin_input
  //     will be ignored.
  //    Returns the command's exit status.
  virtual int Communicate(const string* stdin_input, string* stdout_output,
                          string* stderr_output);

  // GetArgv()
  //    Return the argv passed to SetProgram().
  virtual const char* const* GetArgv() const {
    absl::MutexLock lock(&data_mu_);
    return exec_argv_;
  }

 private:
  static constexpr int kNFds = 3;
  static bool chan_valid(int chan) { return ((chan >= 0) && (chan < kNFds)); }
  static bool retry(int e) {
    return ((e == EINTR) || (e == EAGAIN) || (e == EWOULDBLOCK));
  }
  void FreeArgs() TF_EXCLUSIVE_LOCKS_REQUIRED(data_mu_);
  void FreeEnviron() TF_EXCLUSIVE_LOCKS_REQUIRED(data_mu_);
  void ClosePipes() TF_EXCLUSIVE_LOCKS_REQUIRED(data_mu_);
  bool WaitInternal(int* status);

  // Returns kStillRunning if still running, kExited if exited, kNotRunning if
  // not running. If returns kExited, *status is filled with the exit status.
  // Will not block if flags is WNOHANG.
  enum class WaitStatus {
    kStillRunning = 0,
    kExited = 1,
    kNotRunning = 2,
    // "exit code" if the process was killed.
    // This is returned if you ask for exit_code(), and the process was
    // actually killed (in which case there isn't really an exit code).
    kWasKilled = -256,
  };
  WaitStatus WaitOrCheckRunningInternal(int flags, int* status);

  // The separation between proc_mu_ and data_mu_ mutexes allows Kill() to be
  // called by a thread while another thread is inside Wait() or Communicate().
  mutable absl::Mutex proc_mu_;
  bool running_ TF_GUARDED_BY(proc_mu_);
  pid_t pid_ TF_GUARDED_BY(proc_mu_);
  std::function<void(SubProcess*)> exit_cb_ ABSL_GUARDED_BY(proc_mu_);
  int64_t exit_cb_tid_ ABSL_GUARDED_BY(proc_mu_);

  mutable absl::Mutex wait_mu_ ABSL_ACQUIRED_AFTER(proc_mu_, data_mu_);
  int exit_status_ ABSL_GUARDED_BY(wait_mu_);
  mutable absl::Mutex data_mu_ TF_ACQUIRED_AFTER(proc_mu_);
  char* exec_path_ TF_GUARDED_BY(data_mu_);
  char** exec_argv_ TF_GUARDED_BY(data_mu_);
  char** envp_ ABSL_GUARDED_BY(data_mu_);
  std::string chdir_ ABSL_GUARDED_BY(data_mu_);
  std::string error_text_ ABSL_GUARDED_BY(data_mu_);
  ChannelAction action_[kNFds] TF_GUARDED_BY(data_mu_);
  int parent_pipe_[kNFds] TF_GUARDED_BY(data_mu_);
  int child_pipe_[kNFds] TF_GUARDED_BY(data_mu_);

  SubProcess(const SubProcess&) = delete;
  void operator=(const SubProcess&) = delete;
};

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_DEFAULT_SUBPROCESS_H_
