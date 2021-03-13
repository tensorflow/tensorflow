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

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <memory>
#include <vector>

#include "tensorflow/core/platform/logging.h"

// Android versions older than 28 do not have posix_spawn().
#define USE_POSIX_SPAWN !defined(__ANDROID_API__) || __ANDROID_API__ >= 28

// 1) FYI from m3b@ about fork():
// A danger of calling fork() (as opposed to clone() or vfork()) is that if
// many people have used pthread_atfork() to acquire locks, fork() can deadlock,
// because it's unlikely that the locking order will be correct in a large
// program where different layers are unaware of one another and using
// pthread_atfork() independently.
//
// The danger of not calling fork() is that if libc managed to use
// pthread_atfork() correctly (for example, to lock the environment), you'd
// miss out on that protection.  (But as far as I can see most libc's don't get
// that right; certainly glibc doesn't seem to.)
//
// clone() or vfork() are also frustrating because clone() exists only on Linux,
// and both clone(...CLONE_VM...) and vfork() have interesting issues around
// signals being delivered after the fork and before the exec.  It may be
// possible to work around the latter by blocking all signals before the fork
// and unblocking them afterwards.
//
// Fortunately, most people haven't heard of pthread_atfork().
//
//
// 2) FYI from m3b@ about execvp():
// The execvp() call implicitly uses the libc global variable environ, which was
// copied by fork(), and that copy could have raced with a setenv() call in
// another thread, since libc implementations are usually not very careful about
// this. (glibc isn't careful, for example.)
//
// If this were inside libc, we could use locks or memory barriers to avoid the
// race, but as it is, I see nothing you can do.  Even if you tried to copy the
// environment before the fork(), the copying could race with other threads
// calling setenv().  The good news is that few people call setenv().
//
// Amusingly, the standard says of fork(): "...to avoid errors, the child
// process may only execute async-signal-safe operations until such time as one
// of the exec functions is called."  Notice that execve() is listed as
// async-signal-safe, but execvp() is not, and the difference is just the
// handling of the environment.

extern "C" {

extern char** environ;

}  // extern "C"

namespace tensorflow {

SubProcess::SubProcess(int nfds)
    : running_(false), pid_(-1), exec_path_(nullptr), exec_argv_(nullptr) {
  // The input 'nfds' parameter is currently ignored and the internal constant
  // 'kNFds' is used to support the 3 channels (stdin, stdout, stderr).
  for (int i = 0; i < kNFds; i++) {
    action_[i] = ACTION_CLOSE;
    parent_pipe_[i] = -1;
    child_pipe_[i] = -1;
  }
}

SubProcess::~SubProcess() {
  mutex_lock procLock(proc_mu_);
  mutex_lock dataLock(data_mu_);
  pid_ = -1;
  running_ = false;
  FreeArgs();
  ClosePipes();
}

void SubProcess::FreeArgs() {
  free(exec_path_);
  exec_path_ = nullptr;

  if (exec_argv_) {
    for (char** p = exec_argv_; *p != nullptr; p++) {
      free(*p);
    }
    delete[] exec_argv_;
    exec_argv_ = nullptr;
  }
}

void SubProcess::ClosePipes() {
  for (int i = 0; i < kNFds; i++) {
    if (parent_pipe_[i] >= 0) {
      if (close(parent_pipe_[i]) < 0) {
        LOG(ERROR) << "close() failed: " << strerror(errno);
      }
      parent_pipe_[i] = -1;
    }
    if (child_pipe_[i] >= 0) {
      if (close(child_pipe_[i]) < 0) {
        LOG(ERROR) << "close() failed: " << strerror(errno);
      }
      child_pipe_[i] = -1;
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
  exec_path_ = strdup(file.c_str());
  if (exec_path_ == nullptr) {
    LOG(FATAL) << "SetProgram failed to allocate file string.";
    return;
  }

  int argc = argv.size();
  exec_argv_ = new char*[argc + 1];
  for (int i = 0; i < argc; i++) {
    exec_argv_[i] = strdup(argv[i].c_str());
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

#if USE_POSIX_SPAWN

// Implementation based on posix_spawn().
// We prefer to use posix_spawn() where possible to avoid calling
// pthread_atfork() handlers. POSIX does not guarantee this, but for example,
// glibc 2.24 or newer do so, as does the FreeBSD libc.
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

  // Create parent/child pipes for the specified channels and make the
  // parent-side of the pipes non-blocking.
  for (int i = 0; i < kNFds; i++) {
    if (action_[i] == ACTION_PIPE) {
      int pipe_fds[2];
      if (pipe(pipe_fds) < 0) {
        LOG(ERROR) << "Start cannot create pipe: " << strerror(errno);
        ClosePipes();
        return false;
      }
      // Handle the direction of the pipe (stdin vs stdout/err).
      if (i == 0) {
        parent_pipe_[i] = pipe_fds[1];
        child_pipe_[i] = pipe_fds[0];
      } else {
        parent_pipe_[i] = pipe_fds[0];
        child_pipe_[i] = pipe_fds[1];
      }

      if (fcntl(parent_pipe_[i], F_SETFL, O_NONBLOCK) < 0) {
        LOG(ERROR) << "Start cannot make pipe non-blocking: "
                   << strerror(errno);
        ClosePipes();
        return false;
      }
      if (fcntl(parent_pipe_[i], F_SETFD, FD_CLOEXEC) < 0) {
        LOG(ERROR) << "Start cannot make pipe close-on-exec: "
                   << strerror(errno);
        ClosePipes();
        return false;
      }
    }
  }

  posix_spawn_file_actions_t file_actions;
  int ret;
  ret = posix_spawn_file_actions_init(&file_actions);
  if (ret != 0) {
    LOG(ERROR) << "Start cannot initialize POSIX file actions: "
               << strerror(ret);
    ClosePipes();
    return false;
  }

  // In the child process: close parent-side pipes and channels marked for
  // closing.
  // For pipe channels, replace their file descriptors with the pipes.
  int devnull_fd = -1;
  for (int i = 0; i < kNFds; i++) {
    if (parent_pipe_[i] >= 0) {
      ret = posix_spawn_file_actions_addclose(&file_actions, parent_pipe_[i]);
      if (ret != 0) {
        LOG(ERROR) << "posix_spawn_file_actions_addclose() failed: "
                   << strerror(ret);
      }
    }

    switch (action_[i]) {
      case ACTION_DUPPARENT:
        // Nothing to do, posix_spawnp() took care of it.
        break;

      case ACTION_PIPE:
        ret =
            posix_spawn_file_actions_adddup2(&file_actions, child_pipe_[i], i);
        if (ret != 0) {
          LOG(ERROR) << "posix_spawn_file_actions_adddup2() failed: "
                     << strerror(ret);
        }
        ret = posix_spawn_file_actions_addclose(&file_actions, child_pipe_[i]);
        if (ret != 0) {
          LOG(ERROR) << "posix_spawn_file_actions_addclose() failed: "
                     << strerror(ret);
        }
        break;

      case ACTION_CLOSE:
      default:
        // Do not close stdin/out/err, instead redirect them to /dev/null so
        // their file descriptors remain unavailable for reuse by open(), etc.
        if (i <= CHAN_STDERR) {
          if (devnull_fd < 0) {
            ret = posix_spawn_file_actions_addopen(&file_actions, i,
                                                   "/dev/null", O_RDWR, 0);
            if (ret != 0) {
              LOG(ERROR) << "posix_spawn_file_actions_addopen() failed: "
                         << strerror(ret);
            }
            devnull_fd = i;
          } else {
            ret =
                posix_spawn_file_actions_adddup2(&file_actions, devnull_fd, i);
            if (ret != 0) {
              LOG(ERROR) << "posix_spawn_file_actions_adddup2() failed: "
                         << strerror(ret);
            }
          }
        } else {
          ret = posix_spawn_file_actions_addclose(&file_actions, i);
          if (ret != 0) {
            LOG(ERROR) << "posix_spawn_file_actions_addclose() failed: "
                       << strerror(ret);
          }
        }
        break;
    }
  }

  // Start the child process and setup the file descriptors of both processes.
  ret = posix_spawnp(&pid_, exec_path_, &file_actions, nullptr, exec_argv_,
                     environ);
  if (ret != 0) {
    LOG(ERROR) << "Start cannot spawn child process: " << strerror(ret);
    ClosePipes();
    return false;
  }

  ret = posix_spawn_file_actions_destroy(&file_actions);
  if (ret != 0) {
    LOG(WARNING) << "Start cannot destroy POSIX file actions: "
                 << strerror(ret);
  }

  // Parent process: close the child-side pipes and return.
  running_ = true;
  for (int i = 0; i < kNFds; i++) {
    if (child_pipe_[i] >= 0) {
      if (close(child_pipe_[i]) < 0) {
        LOG(ERROR) << "close() failed: " << strerror(errno);
      }
      child_pipe_[i] = -1;
    }
  }
  return true;
}

#else  // !USE_POSIX_SPAWN

// Implementation based on fork() and exec(); used when posix_spawn() is not
// available.
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

  // Create parent/child pipes for the specified channels and make the
  // parent-side of the pipes non-blocking.
  for (int i = 0; i < kNFds; i++) {
    if (action_[i] == ACTION_PIPE) {
      int pipe_fds[2];
      if (pipe(pipe_fds) < 0) {
        LOG(ERROR) << "Start cannot create pipe: " << strerror(errno);
        ClosePipes();
        return false;
      }
      // Handle the direction of the pipe (stdin vs stdout/err).
      if (i == 0) {
        parent_pipe_[i] = pipe_fds[1];
        child_pipe_[i] = pipe_fds[0];
      } else {
        parent_pipe_[i] = pipe_fds[0];
        child_pipe_[i] = pipe_fds[1];
      }

      if (fcntl(parent_pipe_[i], F_SETFL, O_NONBLOCK) < 0) {
        LOG(ERROR) << "Start cannot make pipe non-blocking: "
                   << strerror(errno);
        ClosePipes();
        return false;
      }
      if (fcntl(parent_pipe_[i], F_SETFD, FD_CLOEXEC) < 0) {
        LOG(ERROR) << "Start cannot make pipe close-on-exec: "
                   << strerror(errno);
        ClosePipes();
        return false;
      }
    }
  }

  // Start the child process and setup the file descriptors of both processes.
  // See comment (1) in the header about issues with the use of fork().
  pid_ = fork();
  if (pid_ < 0) {
    LOG(ERROR) << "Start cannot fork() child process: " << strerror(errno);
    ClosePipes();
    return false;
  }

  if (pid_ > 0) {
    // Parent process: close the child-side pipes and return.
    running_ = true;
    for (int i = 0; i < kNFds; i++) {
      if (child_pipe_[i] >= 0) {
        if (close(child_pipe_[i]) < 0) {
          LOG(ERROR) << "close() failed: " << strerror(errno);
        }
        child_pipe_[i] = -1;
      }
    }
    return true;
  }

  // Child process: close parent-side pipes and channels marked for closing.
  // For pipe channels, replace their file descriptors with the pipes.
  int devnull_fd = -1;
  for (int i = 0; i < kNFds; i++) {
    if (parent_pipe_[i] >= 0) {
      if (close(parent_pipe_[i]) < 0) {
        LOG(ERROR) << "close() failed: " << strerror(errno);
      }
      parent_pipe_[i] = -1;
    }

    switch (action_[i]) {
      case ACTION_DUPPARENT:
        // Nothing to do, fork() took care of it.
        break;

      case ACTION_PIPE:
        while (dup2(child_pipe_[i], i) < 0) {
          if (!retry(errno)) {
            _exit(1);
          }
        }
        if (close(child_pipe_[i]) < 0) {
          LOG(ERROR) << "close() failed: " << strerror(errno);
        }
        child_pipe_[i] = -1;
        break;

      case ACTION_CLOSE:
      default:
        // Do not close stdin/out/err, instead redirect them to /dev/null so
        // their file descriptors remain unavailable for reuse by open(), etc.
        if (i <= CHAN_STDERR) {
          if (devnull_fd < 0) {
            while ((devnull_fd = open("/dev/null", O_RDWR, 0)) < 0) {
              if (!retry(errno)) {
                _exit(1);
              }
            }
          }
          while (dup2(devnull_fd, i) < 0) {
            if (!retry(errno)) {
              _exit(1);
            }
          }
        } else {
          if (close(i) < 0) {
            LOG(ERROR) << "close() failed: " << strerror(errno);
          }
        }
        break;
    }
  }

  if (devnull_fd >= 0) {
    if (close(devnull_fd) < 0) {
      LOG(ERROR) << "close() failed: " << strerror(errno);
    }
  }

  // Execute the child program.
  // See comment (2) in the header about issues with the use of execvp().
  execvp(exec_path_, exec_argv_);
  _exit(1);
}

#endif  // USE_POSIX_SPAWN

bool SubProcess::Wait() {
  int status;
  return WaitInternal(&status);
}

bool SubProcess::WaitInternal(int* status) {
  // The waiter must release proc_mu_ while waiting in order for Kill() to work.
  proc_mu_.lock();
  bool running = running_;
  pid_t pid = pid_;
  proc_mu_.unlock();

  bool ret = false;
  if (running && (pid > 1)) {
    pid_t cpid;
    int cstat;
    bool done = false;
    while (!done) {
      cpid = waitpid(pid, &cstat, 0);
      if ((cpid < 0) && !retry(errno)) {
        done = true;
      } else if ((cpid == pid) && (WIFEXITED(cstat) || WIFSIGNALED(cstat))) {
        *status = cstat;
        ret = true;
        done = true;
      }
    }
  }

  proc_mu_.lock();
  if ((running_ == running) && (pid_ == pid)) {
    running_ = false;
    pid_ = -1;
  }
  proc_mu_.unlock();
  return ret;
}

bool SubProcess::Kill(int signal) {
  proc_mu_.lock();
  bool running = running_;
  pid_t pid = pid_;
  proc_mu_.unlock();

  bool ret = false;
  if (running && (pid > 1)) {
    ret = (kill(pid, signal) == 0);
  }
  return ret;
}

int SubProcess::Communicate(const string* stdin_input, string* stdout_output,
                            string* stderr_output) {
  struct pollfd fds[kNFds];
  size_t nbytes[kNFds];
  string* iobufs[kNFds];
  int fd_count = 0;

  proc_mu_.lock();
  bool running = running_;
  proc_mu_.unlock();
  if (!running) {
    LOG(ERROR) << "Communicate called without a running process.";
    return 1;
  }

  // If SIGPIPE handling is the default action, change it to ignore SIGPIPE and
  // keep it ignored, don't change it back.  This is needed while communicating
  // with the child process so the parent process can survive the death of the
  // child process while it is writing to its stdin.  If the application has
  // registered a SIGPIPE handler, then let it deal with any signals generated
  // by the premature death of the child process, don't overwrite its handler.
  struct sigaction act;
  if (sigaction(SIGPIPE, nullptr, &act) < 0) {
    LOG(ERROR) << "Communicate cannot get SIGPIPE handler: " << strerror(errno);
    return 1;
  }
  if (act.sa_handler == SIG_DFL) {
    memset(&act, 0, sizeof(act));
    act.sa_handler = SIG_IGN;
    sigemptyset(&act.sa_mask);
    if (sigaction(SIGPIPE, &act, nullptr) < 0) {
      LOG(ERROR) << "Communicate cannot ignore SIGPIPE: " << strerror(errno);
      return 1;
    }
  }

  // Lock data_mu_ but not proc_mu_ while communicating with the child process
  // in order for Kill() to be able to terminate the child from another thread.
  data_mu_.lock();

  // Initialize the poll() structures and buffer tracking.
  for (int i = 0; i < kNFds; i++) {
    if (action_[i] == ACTION_PIPE) {
      switch (i) {
        case CHAN_STDIN:
          // Special case: if no data is given to send to the child process,
          // close the pipe to unblock the child, and skip the file descriptor.
          if (stdin_input == nullptr) {
            if (close(parent_pipe_[i]) < 0) {
              LOG(ERROR) << "close() failed: " << strerror(errno);
            }
            parent_pipe_[i] = -1;
            continue;
          }
          iobufs[fd_count] = const_cast<string*>(stdin_input);
          break;
        case CHAN_STDOUT:
          iobufs[fd_count] = stdout_output;
          break;
        case CHAN_STDERR:
          iobufs[fd_count] = stderr_output;
          break;
        default:
          iobufs[fd_count] = nullptr;
          break;
      }
      nbytes[fd_count] = 0;
      fds[fd_count].fd = parent_pipe_[i];
      fds[fd_count].events = (i > 0) ? POLLIN : POLLOUT;
      fds[fd_count].revents = 0;
      fd_count++;
    }
  }

  // Loop communicating with the child process.
  int fd_remain = fd_count;
  char buf[4096];
  while (fd_remain > 0) {
    int n = poll(fds, fd_count, -1);
    if ((n < 0) && !retry(errno)) {
      LOG(ERROR) << "Communicate cannot poll(): " << strerror(errno);
      fd_remain = 0;
    } else if (n == 0) {
      LOG(ERROR) << "Communicate cannot poll(): timeout not possible";
      fd_remain = 0;
    } else if (n > 0) {
      // Handle the pipes ready for I/O.
      for (int i = 0; i < fd_count; i++) {
        if ((fds[i].revents & (POLLIN | POLLHUP)) != 0) {
          // Read from one of the child's outputs.
          ssize_t n = read(fds[i].fd, buf, sizeof(buf));
          if (n > 0) {
            if (iobufs[i] != nullptr) {
              iobufs[i]->append(buf, n);
              nbytes[i] += n;
            }
          } else if ((n == 0) || !retry(errno)) {
            fds[i].fd = -1;
            fd_remain--;
          }
        } else if ((fds[i].revents & POLLOUT) != 0) {
          // Write to the child's stdin.
          ssize_t n = iobufs[i]->size() - nbytes[i];
          if (n > 0) {
            n = write(fds[i].fd, iobufs[i]->c_str() + nbytes[i], n);
          }
          if (n >= 0) {
            nbytes[i] += n;
            if (nbytes[i] >= iobufs[i]->size()) {
              fds[i].fd = -1;
              fd_remain--;
              // Close the child's stdin pipe to unblock the process.
              if (close(parent_pipe_[CHAN_STDIN]) < 0) {
                LOG(ERROR) << "close() failed: " << strerror(errno);
              }
              parent_pipe_[CHAN_STDIN] = -1;
            }
          } else if (!retry(errno)) {
            fds[i].fd = -1;
            fd_remain--;
          }
        } else if ((fds[i].revents & (POLLERR | POLLNVAL)) != 0) {
          fds[i].fd = -1;
          fd_remain--;
        }
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
