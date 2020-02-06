/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <io.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <windows.h>

#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/strcat.h"

#define PIPE_BUF_SIZE 4096

namespace tensorflow {

namespace {

static bool IsProcessFinished(HANDLE h) {
  DWORD process_return_code = STILL_ACTIVE;
  // TODO handle failure
  GetExitCodeProcess(h, &process_return_code);
  return process_return_code != STILL_ACTIVE;
}

struct ThreadData {
  string* iobuf;
  HANDLE iohandle;
};

DWORD WINAPI InputThreadFunction(LPVOID param) {
  ThreadData* args = reinterpret_cast<ThreadData*>(param);
  string* input = args->iobuf;
  HANDLE in_handle = args->iohandle;
  size_t buffer_pointer = 0;

  size_t total_bytes_written = 0;
  bool ok = true;
  while (ok && total_bytes_written < input->size()) {
    DWORD bytes_written_this_time;
    ok = WriteFile(in_handle, input->data() + total_bytes_written,
                   input->size() - total_bytes_written,
                   &bytes_written_this_time, nullptr);
    total_bytes_written += bytes_written_this_time;
  }
  CloseHandle(in_handle);

  if (!ok) {
    return GetLastError();
  } else {
    return 0;
  }
}

DWORD WINAPI OutputThreadFunction(LPVOID param) {
  ThreadData* args = reinterpret_cast<ThreadData*>(param);
  string* output = args->iobuf;
  HANDLE out_handle = args->iohandle;

  char buf[PIPE_BUF_SIZE];
  DWORD bytes_read;

  bool wait_result = WaitForSingleObject(out_handle, INFINITE);
  if (wait_result != WAIT_OBJECT_0) {
    LOG(FATAL) << "WaitForSingleObject on child process output failed. "
                  "Error code: "
               << wait_result;
  }
  while (ReadFile(out_handle, buf, sizeof(buf), &bytes_read, nullptr) &&
         bytes_read > 0) {
    output->append(buf, bytes_read);
  }
  CloseHandle(out_handle);
  return 0;
}

}  // namespace

SubProcess::SubProcess(int nfds)
    : running_(false),
      exec_path_(nullptr),
      exec_argv_(nullptr),
      win_pi_(nullptr) {
  // The input 'nfds' parameter is currently ignored and the internal constant
  // 'kNFds' is used to support the 3 channels (stdin, stdout, stderr).
  for (int i = 0; i < kNFds; i++) {
    action_[i] = ACTION_CLOSE;
    parent_pipe_[i] = nullptr;
  }
}

SubProcess::~SubProcess() {
  mutex_lock procLock(proc_mu_);
  mutex_lock dataLock(data_mu_);
  if (win_pi_) {
    CloseHandle(reinterpret_cast<PROCESS_INFORMATION*>(win_pi_)->hProcess);
    CloseHandle(reinterpret_cast<PROCESS_INFORMATION*>(win_pi_)->hThread);
    delete win_pi_;
  }
  running_ = false;
  FreeArgs();
  ClosePipes();
}

void SubProcess::FreeArgs() {
  free(exec_path_);
  exec_path_ = nullptr;

  if (exec_argv_) {
    for (int i = 0; exec_argv_[i]; i++) {
      free(exec_argv_[i]);
    }
    delete[] exec_argv_;
    exec_argv_ = nullptr;
  }
}

void SubProcess::ClosePipes() {
  for (int i = 0; i < kNFds; i++) {
    if (parent_pipe_[i] != nullptr) {
      CloseHandle(parent_pipe_[i]);
      parent_pipe_[i] = nullptr;
    }
  }
}

void SubProcess::SetProgram(const string& file,
                            const std::vector<string>& argv) {
  mutex_lock procLock(proc_mu_);
  mutex_lock dataLock(data_mu_);
  if (running_) {
    LOG(FATAL) << "SetProgram called after the process was started.";
    return;
  }

  FreeArgs();
  exec_path_ = _strdup(file.c_str());
  if (exec_path_ == nullptr) {
    LOG(FATAL) << "SetProgram failed to allocate file string.";
    return;
  }

  int argc = argv.size();
  exec_argv_ = new char*[argc + 1];
  for (int i = 0; i < argc; i++) {
    exec_argv_[i] = _strdup(argv[i].c_str());
    if (exec_argv_[i] == nullptr) {
      LOG(FATAL) << "SetProgram failed to allocate command argument.";
      return;
    }
  }
  exec_argv_[argc] = nullptr;
}

void SubProcess::SetChannelAction(Channel chan, ChannelAction action) {
  mutex_lock procLock(proc_mu_);
  mutex_lock dataLock(data_mu_);
  if (running_) {
    LOG(FATAL) << "SetChannelAction called after the process was started.";
  } else if (!chan_valid(chan)) {
    LOG(FATAL) << "SetChannelAction called with invalid channel: " << chan;
  } else if ((action != ACTION_CLOSE) && (action != ACTION_PIPE) &&
             (action != ACTION_DUPPARENT)) {
    LOG(FATAL) << "SetChannelAction called with invalid action: " << action;
  } else {
    action_[chan] = action;
  }
}

bool SubProcess::Start() {
  mutex_lock procLock(proc_mu_);
  mutex_lock dataLock(data_mu_);
  if (running_) {
    LOG(ERROR) << "Start called after the process was started.";
    return false;
  }
  if ((exec_path_ == nullptr) || (exec_argv_ == nullptr)) {
    LOG(ERROR) << "Start called without setting a program.";
    return false;
  }

  // SecurityAttributes to use in winapi calls below.
  SECURITY_ATTRIBUTES attrs;
  attrs.nLength = sizeof(SECURITY_ATTRIBUTES);
  attrs.bInheritHandle = TRUE;
  attrs.lpSecurityDescriptor = nullptr;

  // No need to store subprocess end of the pipes, they will be closed before
  // this function terminates.
  HANDLE child_pipe_[kNFds] GUARDED_BY(data_mu_);

  // Create parent/child pipes for the specified channels and make the
  // parent-side of the pipes non-blocking.
  for (int i = 0; i < kNFds; i++) {
    if (action_[i] == ACTION_PIPE) {
      if (!CreatePipe(i == CHAN_STDIN ? child_pipe_ + i : parent_pipe_ + i,
                      i == CHAN_STDIN ? parent_pipe_ + i : child_pipe_ + i,
                      &attrs, PIPE_BUF_SIZE)) {
        LOG(ERROR) << "Cannot create pipe. Error code: " << GetLastError();
        ClosePipes();
        return false;
      }

      // Parent pipes should not be inherited by the child process
      if (!SetHandleInformation(parent_pipe_[i], HANDLE_FLAG_INHERIT, 0)) {
        LOG(ERROR) << "Cannot set pipe handle attributes.";
        ClosePipes();
        return false;
      }
    } else if (action_[i] == ACTION_DUPPARENT) {
      if (i == CHAN_STDIN) {
        child_pipe_[i] = GetStdHandle(STD_INPUT_HANDLE);
      } else if (i == CHAN_STDOUT) {
        child_pipe_[i] = GetStdHandle(STD_OUTPUT_HANDLE);
      } else {
        child_pipe_[i] = GetStdHandle(STD_ERROR_HANDLE);
      }
    } else {  // ACTION_CLOSE
      parent_pipe_[i] = nullptr;
      child_pipe_[i] = nullptr;
    }
  }

  // Concatanate argv, because winapi wants it so.
  string command_line = strings::StrCat("\"", exec_path_, "\"");
  for (int i = 1; exec_argv_[i]; i++) {
    command_line.append(strings::StrCat(" \"", exec_argv_[i], "\""));
  }

  // Set up the STARTUPINFO struct with information about the pipe handles.
  STARTUPINFOA si;
  ZeroMemory(&si, sizeof(STARTUPINFO));
  si.cb = sizeof(STARTUPINFO);
  si.dwFlags |= STARTF_USESTDHANDLES;

  // Handle the pipes for the child process.
  if (child_pipe_[CHAN_STDIN]) {
    si.hStdInput = child_pipe_[CHAN_STDIN];
  }
  if (child_pipe_[CHAN_STDOUT]) {
    si.hStdOutput = child_pipe_[CHAN_STDOUT];
  }
  if (child_pipe_[CHAN_STDERR]) {
    si.hStdError = child_pipe_[CHAN_STDERR];
  }

  // Allocate the POROCESS_INFORMATION struct.
  win_pi_ = new PROCESS_INFORMATION;

  // Execute the child program.
  bool bSuccess =
      CreateProcessA(nullptr, const_cast<char*>(command_line.c_str()), nullptr,
                     nullptr, TRUE, 0, nullptr, nullptr, &si,
                     reinterpret_cast<PROCESS_INFORMATION*>(win_pi_));

  if (bSuccess) {
    for (int i = 0; i < kNFds; i++) {
      if (child_pipe_[i] != nullptr) {
        CloseHandle(child_pipe_[i]);
        child_pipe_[i] = nullptr;
      }
    }
    running_ = true;
    return true;
  } else {
    LOG(ERROR) << "Call to CreateProcess failed. Error code: "
               << GetLastError();
    ClosePipes();
    return false;
  }
}

bool SubProcess::Wait() {
  int status;
  return WaitInternal(&status);
}

bool SubProcess::WaitInternal(int* status) {
  // The waiter must release proc_mu_ while waiting in order for Kill() to work.
  proc_mu_.lock();
  bool running = running_;
  PROCESS_INFORMATION pi_ = *reinterpret_cast<PROCESS_INFORMATION*>(win_pi_);
  proc_mu_.unlock();

  bool ret = false;
  if (running && pi_.hProcess) {
    DWORD wait_status = WaitForSingleObject(pi_.hProcess, INFINITE);
    if (wait_status == WAIT_OBJECT_0) {
      DWORD process_exit_code = 0;
      if (GetExitCodeProcess(pi_.hProcess, &process_exit_code)) {
        LOG(INFO) << "SubProcess ended with return code: " << process_exit_code
                  << std::endl;
        *status = static_cast<int>(process_exit_code);
      } else {
        LOG(FATAL) << "Wait failed with code: " << GetLastError();
      }
    } else {
      LOG(FATAL) << "WaitForSingleObject call on the process handle failed. "
                    "Error code: "
                 << wait_status;
    }
  }

  proc_mu_.lock();
  if ((running_ == running) &&
      (pi_.hProcess ==
       reinterpret_cast<PROCESS_INFORMATION*>(win_pi_)->hProcess)) {
    running_ = false;
    CloseHandle(reinterpret_cast<PROCESS_INFORMATION*>(win_pi_)->hProcess);
    CloseHandle(reinterpret_cast<PROCESS_INFORMATION*>(win_pi_)->hThread);
    reinterpret_cast<PROCESS_INFORMATION*>(win_pi_)->hProcess = nullptr;
    reinterpret_cast<PROCESS_INFORMATION*>(win_pi_)->hThread = nullptr;
  }
  proc_mu_.unlock();
  return *status == 0;
}

bool SubProcess::Kill(int unused_signal) {
  proc_mu_.lock();
  bool running = running_;
  PROCESS_INFORMATION pi_ = *reinterpret_cast<PROCESS_INFORMATION*>(win_pi_);
  proc_mu_.unlock();

  bool ret = false;
  if (running && pi_.hProcess) {
    ret = TerminateProcess(pi_.hProcess, 0);
  }
  return ret;
}

int SubProcess::Communicate(const string* stdin_input, string* stdout_output,
                            string* stderr_output) {
  proc_mu_.lock();
  bool running = running_;
  proc_mu_.unlock();
  if (!running) {
    LOG(ERROR) << "Communicate called without a running process.";
    return 1;
  }

  HANDLE thread_handles[kNFds];
  int thread_count = 0;
  ThreadData thread_params[kNFds];

  // Lock data_mu_ but not proc_mu_ while communicating with the child process
  // in order for Kill() to be able to terminate the child from another thread.
  data_mu_.lock();
  if (!IsProcessFinished(
          reinterpret_cast<PROCESS_INFORMATION*>(win_pi_)->hProcess) ||
      (parent_pipe_[CHAN_STDOUT] != nullptr) ||
      (parent_pipe_[CHAN_STDERR] != nullptr)) {
    if (parent_pipe_[CHAN_STDIN] != nullptr) {
      if (stdin_input) {
        thread_params[thread_count].iobuf = const_cast<string*>(stdin_input);
        thread_params[thread_count].iohandle = parent_pipe_[CHAN_STDIN];
        parent_pipe_[CHAN_STDIN] = nullptr;
        thread_handles[thread_count] =
            CreateThread(NULL, 0, InputThreadFunction,
                         thread_params + thread_count, 0, NULL);
        thread_count++;
      }
    } else {
      CloseHandle(parent_pipe_[CHAN_STDIN]);
      parent_pipe_[CHAN_STDIN] == NULL;
    }

    if (parent_pipe_[CHAN_STDOUT] != nullptr) {
      if (stdout_output != nullptr) {
        thread_params[thread_count].iobuf = stdout_output;
        thread_params[thread_count].iohandle = parent_pipe_[CHAN_STDOUT];
        parent_pipe_[CHAN_STDOUT] = NULL;
        thread_handles[thread_count] =
            CreateThread(NULL, 0, OutputThreadFunction,
                         thread_params + thread_count, 0, NULL);
        thread_count++;
      } else {
        CloseHandle(parent_pipe_[CHAN_STDOUT]);
        parent_pipe_[CHAN_STDOUT] = nullptr;
      }
    }

    if (parent_pipe_[CHAN_STDERR] != nullptr) {
      if (stderr_output != nullptr) {
        thread_params[thread_count].iobuf = stderr_output;
        thread_params[thread_count].iohandle = parent_pipe_[CHAN_STDERR];
        parent_pipe_[CHAN_STDERR] = NULL;
        thread_handles[thread_count] =
            CreateThread(NULL, 0, OutputThreadFunction,
                         thread_params + thread_count, 0, NULL);
        thread_count++;
      } else {
        CloseHandle(parent_pipe_[CHAN_STDERR]);
        parent_pipe_[CHAN_STDERR] = nullptr;
      }
    }
  }

  // Wait for all IO threads to exit.
  if (thread_count > 0) {
    DWORD wait_result = WaitForMultipleObjects(thread_count, thread_handles,
                                               true,  // wait all threads
                                               INFINITE);
    if (wait_result != WAIT_OBJECT_0) {
      LOG(ERROR) << "Waiting on the io threads failed! result: " << wait_result
                 << std::endl;
      return -1;
    }

    for (int i = 0; i < thread_count; i++) {
      DWORD exit_code;
      if (GetExitCodeThread(thread_handles[i], &exit_code)) {
        if (exit_code) {
          LOG(ERROR) << "One of the IO threads failed with code: " << exit_code;
        }
      } else {
        LOG(ERROR) << "Error checking io thread exit statuses. Error Code: "
                   << GetLastError();
      }
    }
  }

  data_mu_.unlock();

  // Wait for the child process to exit and return its status.
  int status;
  return WaitInternal(&status) ? status : -1;
}

std::unique_ptr<SubProcess> CreateSubProcess(const std::vector<string>& argv) {
  std::unique_ptr<SubProcess> proc(new SubProcess());
  proc->SetProgram(argv[0], argv);
  proc->SetChannelAction(CHAN_STDERR, ACTION_DUPPARENT);
  proc->SetChannelAction(CHAN_STDOUT, ACTION_DUPPARENT);
  return proc;
}

}  // namespace tensorflow
