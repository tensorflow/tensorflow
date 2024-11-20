/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tsl/platform/stacktrace_handler.h"

// clang-format off
#include <windows.h>  // Windows.h must be declared above dgbhelp.
#include <dbghelp.h>
// clang-format on

#include <errno.h>
#include <io.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <thread>  // NOLINT(build/c++11)

#include "tsl/platform/mutex.h"
#include "tsl/platform/stacktrace.h"
#include "tsl/platform/types.h"

namespace tsl {

// This mutex allows us to unblock an alarm thread.
static mutex alarm_mu(LINKER_INITIALIZED);
static bool alarm_activated = false;

static void AlarmThreadBody() {
  // Wait until the alarm_activated bool is true, sleep for 60 seconds,
  // then kill the program.
  alarm_mu.lock();
  alarm_mu.Await(Condition(&alarm_activated));
  alarm_mu.unlock();
  Sleep(60000);

  // Reinstall the standard signal handler, so that we actually abort the
  // program, instead of just re-triggering the signal handler.
  signal(SIGABRT, SIG_DFL);
  abort();
}

// There is no generally available async-signal safe function for converting an
// integer (or pointer) to ASCII (see:
// http://man7.org/linux/man-pages/man7/signal-safety.7.html). This function
// attempts to convert a ptr to a hexadecimal string stored in buffer `buf` of
// size `size`. The string has a '0x' prefix, and a '\n\0' (newline + null
// terminator) suffix. If the returned value is true, buf contains the string
// described above. If the returned value is false, buf is unchanged. Ideally,
// clients should provide a buffer with size >= 2 * sizeof(uintptr_t) + 4.
static bool PtrToString(uintptr_t ptr, char* buf, size_t size) {
  static constexpr char kHexCharacters[] = "0123456789abcdef";
  static constexpr int kHexBase = 16;

  // The addressible space of an N byte pointer is at most 2^(8N).
  // Since we are printing an address in hexadecimal, the number of hex
  // characters we must print (lets call this H) satisfies 2^(8N) = 16^H.
  // Therefore H = 2N.
  size_t num_hex_chars = 2 * sizeof(uintptr_t);

  // The buffer size also needs 4 extra bytes:
  // 2 bytes for 0x prefix,
  // 1 byte for a '\n' newline suffix, and
  // 1 byte for a '\0' null terminator.
  if (size < (num_hex_chars + 4)) {
    return false;
  }

  buf[0] = '0';
  buf[1] = 'x';

  // Convert the entire number to hex, going backwards.
  int start_index = 2;
  for (int i = num_hex_chars - 1 + start_index; i >= start_index; --i) {
    buf[i] = kHexCharacters[ptr % kHexBase];
    ptr /= kHexBase;
  }

  // Terminate the output with a newline and NULL terminator.
  int current_index = start_index + num_hex_chars;
  buf[current_index] = '\n';
  buf[current_index + 1] = '\0';

  return true;
}

// This function will print a stacktrace of pointers to STDERR.
// It avoids using malloc, so it makes sure to dump the stack even when the heap
// is corrupted. It also does not call Window's symbolization function (which
// requires acquiring a mutex), which is not safe in a signal handler.
static inline void SafePrintStackTracePointers() {
  static constexpr char begin_msg[] = "*** BEGIN STACK TRACE POINTERS ***\n";
  (void)_write(_fileno(stderr), begin_msg, strlen(begin_msg));

  static constexpr int kMaxStackFrames = 64;
  void* trace[kMaxStackFrames];
  int num_frames = CaptureStackBackTrace(0, kMaxStackFrames, trace, NULL);

  for (int i = 0; i < num_frames; ++i) {
    char buffer[32] = "unsuccessful ptr conversion";
    PtrToString(reinterpret_cast<uintptr_t>(trace[i]), buffer, sizeof(buffer));
    (void)_write(_fileno(stderr), buffer, strlen(buffer));
  }

  static constexpr char end_msg[] = "*** END STACK TRACE POINTERS ***\n\n";
  (void)_write(_fileno(stderr), end_msg, strlen(end_msg));
}

static void StacktraceHandler(int sig) {
  // We want to make sure our handler does not deadlock; this should be the last
  // thing our program does. In unix systems, we can use setitimer and SIGALRM,
  // to send an alarm signal killing our process after a set amount of time.
  // Since Windows does not support this, we unblock a sleeping thread meant
  // to abort the program ~60 seconds after waking.
  alarm_mu.lock();
  alarm_activated = true;
  alarm_mu.unlock();

  char buf[128];
  snprintf(buf, sizeof(buf), "*** Received signal %d ***\n", sig);
  (void)write(_fileno(stderr), buf, strlen(buf));

  // Print "a" stack trace, as safely as possible.
  SafePrintStackTracePointers();

  // Up until this line, we made sure not to allocate memory, to be able to dump
  // a stack trace even in the event of heap corruption. After this line, we
  // will try to print more human readable things to the terminal.
  // But these have a higher probability to fail.
  std::string stacktrace = CurrentStackTrace();
  (void)write(_fileno(stderr), stacktrace.c_str(), stacktrace.length());

  // Reinstall the standard signal handler, so that we actually abort the
  // program, instead of just re-triggering the signal handler.
  signal(SIGABRT, SIG_DFL);
  abort();
}

namespace testing {

// Windows documentation on signal handling:
// https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/signal?view=vs-2019
void InstallStacktraceHandler() {
  int handled_signals[] = {SIGSEGV, SIGABRT, SIGILL, SIGFPE};

  std::thread alarm_thread(AlarmThreadBody);
  alarm_thread.detach();

  typedef void (*SignalHandlerPointer)(int);

  for (int sig : handled_signals) {
    SignalHandlerPointer previousHandler = signal(sig, StacktraceHandler);
    if (previousHandler == SIG_ERR) {
      char buf[128];
      snprintf(buf, sizeof(buf),
               "tensorflow::InstallStackTraceHandler: Warning, can't install "
               "backtrace signal handler for signal %d, errno:%d \n",
               sig, errno);
      (void)write(_fileno(stderr), buf, strlen(buf));
    } else if (previousHandler != SIG_DFL) {
      char buf[128];
      snprintf(buf, sizeof(buf),
               "tensorflow::InstallStackTraceHandler: Warning, backtrace "
               "signal handler for signal %d overwrote previous handler.\n",
               sig);
      (void)write(_fileno(stderr), buf, strlen(buf));
    }
  }
}

}  // namespace testing
}  // namespace tsl
