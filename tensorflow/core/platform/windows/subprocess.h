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

#ifndef TENSORFLOW_CORE_PLATFORM_WINDOWS_SUBPROCESS_H_
#define TENSORFLOW_CORE_PLATFORM_WINDOWS_SUBPROCESS_H_

#include <string>
#include <vector>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class SubProcess {
 public:
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

  // SetProgram()
  //    Set up a program and argument list for execution, with the full
  //    "raw" argument list passed as a vector of strings.  argv[0]
  //    should be the program name, just as in execv().
  //
  //    file: The file containing the program.  This must be an absolute path
  //          name - $PATH is not searched.
  //    argv: The argument list.
  virtual void SetProgram(const string& file, const std::vector<string>& argv);

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

  // Wait()
  //    Block until the process exits.
  //    Return true normally, or false if the process wasn't running.
  virtual bool Wait();

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

 private:
  static const int kNFds = 3;
  static bool chan_valid(int chan) { return ((chan >= 0) && (chan < kNFds)); }

  void FreeArgs() EXCLUSIVE_LOCKS_REQUIRED(data_mu_);
  void ClosePipes() EXCLUSIVE_LOCKS_REQUIRED(data_mu_);
  bool WaitInternal(int* status);

  // The separation between proc_mu_ and data_mu_ mutexes allows Kill() to be
  // called by a thread while another thread is inside Wait() or Communicate().
  mutable mutex proc_mu_;
  bool running_ GUARDED_BY(proc_mu_);
  void* win_pi_ GUARDED_BY(proc_mu_);

  mutable mutex data_mu_ ACQUIRED_AFTER(proc_mu_);
  char* exec_path_ GUARDED_BY(data_mu_);
  char** exec_argv_ GUARDED_BY(data_mu_);
  ChannelAction action_[kNFds] GUARDED_BY(data_mu_);
  void* parent_pipe_[kNFds] GUARDED_BY(data_mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(SubProcess);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_WINDOWS_SUBPROCESS_H_
