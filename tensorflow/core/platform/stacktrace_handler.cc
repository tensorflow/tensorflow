/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/platform.h"

#if !defined(PLATFORM_GOOGLE) && !defined(IS_MOBILE_PLATFORM) && \
    defined(PLATFORM_POSIX) && (defined(__clang__) || defined(__GNUC__))
#define TF_GENERATE_STACKTRACE
#endif

#if defined(TF_GENERATE_STACKTRACE)
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <string>

#include "tensorflow/core/platform/abi.h"
#include "tensorflow/core/platform/stacktrace.h"

#endif  // defined(TF_GENERATE_STACKTRACE)

namespace tensorflow {
namespace testing {

#if defined(TF_GENERATE_STACKTRACE)
// This function will print stacktrace to STDERR.
// It avoids using malloc, so it makes sure to dump the stack even when the heap
// is corrupted. However, it can dump mangled symbols.
inline void SafePrintStackTrace() {
  static const char begin_msg[] = "*** BEGIN MANGLED STACK TRACE ***\n";
  (void)write(STDERR_FILENO, begin_msg, strlen(begin_msg));

  int buffer_size = 128;
  void *trace[128];
  // Run backtrace to get the size of the stacktrace
  buffer_size = backtrace(trace, buffer_size);

  // Print a mangled stacktrace to STDERR as safely as possible.
  backtrace_symbols_fd(trace, buffer_size, STDERR_FILENO);

  static const char end_msg[] = "*** END MANGLED STACK TRACE ***\n\n";
  (void)write(STDERR_FILENO, end_msg, strlen(end_msg));
}

static void StacktraceHandler(int sig, siginfo_t *si, void *v) {
  // Make sure our handler does not deadlock. And this should be the last thing
  // our program does. Therefore, set a timer to kill the program in 60
  // seconds.
  struct itimerval timer;
  timer.it_value.tv_sec = 60;
  timer.it_value.tv_usec = 0;
  timer.it_interval.tv_sec = 0;
  timer.it_interval.tv_usec = 0;
  setitimer(ITIMER_REAL, &timer, 0);

  struct sigaction sa_timeout;
  memset(&sa_timeout, 0, sizeof(sa_timeout));
  sa_timeout.sa_handler = SIG_DFL;
  sigaction(SIGALRM, &sa_timeout, 0);

  char buf[128];

  snprintf(buf, sizeof(buf), "*** Received signal %d ***\n", sig);
  (void)write(STDERR_FILENO, buf, strlen(buf));

  // Print "a" stack trace, as safely as possible.
  SafePrintStackTrace();

  // Up until this line, we made sure not to allocate memory, to be able to dump
  // a stack trace even in the event of heap corruption. After this line, we
  // will try to print more human readable things to the terminal.
  // But these have a higher probability to fail.
  std::string stacktrace = CurrentStackTrace();
  (void)write(STDERR_FILENO, stacktrace.c_str(), stacktrace.length());

  // Abort the program.
  struct sigaction sa;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sa.sa_handler = SIG_DFL;
  sigaction(SIGABRT, &sa, NULL);
  abort();
}

void InstallStacktraceHandler() {
  int handled_signals[] = {SIGSEGV, SIGABRT, SIGBUS, SIGILL, SIGFPE};

  for (int i = 0; i < sizeof(handled_signals) / sizeof(int); i++) {
    int sig = handled_signals[i];
    struct sigaction sa;
    struct sigaction osa;

    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
    sa.sa_sigaction = &StacktraceHandler;
    if (sigaction(sig, &sa, &osa) != 0) {
      char buf[128];
      snprintf(buf, sizeof(buf),
               "Warning, can't install backtrace signal handler for signal %d, "
               "errno:%d \n",
               sig, errno);
      (void)write(STDERR_FILENO, buf, strlen(buf));
    } else if (osa.sa_handler != SIG_DFL) {
      char buf[128];
      snprintf(buf, sizeof(buf),
               "Warning, backtrace signal handler for signal %d overwrote "
               "previous handler.\n",
               sig);
      (void)write(STDERR_FILENO, buf, strlen(buf));
    }
  }
}

#else
void InstallStacktraceHandler() {}
#endif  // defined(TF_GENERATE_STACKTRACE)

}  // namespace testing
}  // namespace tensorflow
