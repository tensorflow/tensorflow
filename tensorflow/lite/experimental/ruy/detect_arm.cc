/* Copyright 2019 Google LLC. All Rights Reserved.

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

/* Temporary dotprod-detection until we can rely on proper feature-detection
such as getauxval on Linux (requires a newer Linux kernel than we can
currently rely on on Android).

There are two main ways that this could be implemented: using a signal
handler or a fork. The current implementation uses a signal handler.
This is because on current Android, an uncaught signal gives a latency
of over 100 ms. In order for the fork approach to be worthwhile, it would
have to save us the hassle of handling signals, and such an approach thus
has an unavoidable 100ms latency. By contrast, the present signal-handling
approach has low latency.

Downsides of the current signal-handling approach include:
 1. Setting and restoring signal handlers is not thread-safe: we can't
    prevent another thread from interfering with us. We at least prevent
    other threads from calling our present code concurrently by using a lock,
    but we can't do anything about other threads using their own code to
    set signal handlers.
 2. Signal handlers are not entirely portable, e.g. b/132973173 showed that
    on Apple platform the EXC_BAD_INSTRUCTION signal is not always caught
    by a SIGILL handler (difference between Release and Debug builds).
 3. The signal handler approach looks confusing in a debugger (has to
    tell the debugger to 'continue' past the signal every time). Fix:
    ```
    (gdb) handle SIGILL nostop noprint pass
    ```

Here is what the nicer fork-based alternative would look like.
Its only downside, as discussed above, is high latency, 100 ms on Android.

```
bool try_asm_snippet(bool (*asm_snippet)()) {
  int child_pid = fork();
  if (child_pid == -1) {
    // Fork failed.
    return false;
  }
  if (child_pid == 0) {
    // Child process code path. Pass the raw boolean return value of
    // asm_snippet as exit code (unconventional: 1 means true == success).
    _exit(asm_snippet());
  }

  int child_status;
  waitpid(child_pid, &child_status, 0);
  if (WIFSIGNALED(child_status)) {
    // Child process terminated by signal, meaning the instruction was
    // not supported.
    return false;
  }
  // Return the exit code of the child, which per child code above was
  // the return value of asm_snippet().
  return WEXITSTATUS(child_status);
}
```
*/

#include "tensorflow/lite/experimental/ruy/detect_arm.h"

#if defined __aarch64__ && defined __linux__
#define RUY_IMPLEMENT_DETECT_DOTPROD
#endif

#ifdef RUY_IMPLEMENT_DETECT_DOTPROD

#include <setjmp.h>
#include <signal.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>  // NOLINT(build/c++11)

// Intentionally keep checking for __linux__ here in case we want to
// extend RUY_IMPLEMENT_DETECT_DOTPROD outside of linux in the future.
#ifdef __linux__
#include <sys/auxv.h>
#include <sys/utsname.h>
#endif

#endif

namespace ruy {

#ifdef RUY_IMPLEMENT_DETECT_DOTPROD

namespace {

// Waits until there are no pending SIGILL's.
void wait_until_no_pending_sigill() {
  sigset_t pending;
  do {
    sigemptyset(&pending);
    sigpending(&pending);
  } while (sigismember(&pending, SIGILL));
}

// long-jump buffer used to continue execution after a caught SIGILL.
sigjmp_buf& global_sigjmp_buf_just_before_trying_snippet() {
  static sigjmp_buf g;
  return g;
}

// SIGILL signal handler. Long-jumps to just before
// we ran the snippet that we know is the only thing that could have generated
// the SIGILL.
void sigill_handler(int) {
  siglongjmp(global_sigjmp_buf_just_before_trying_snippet(), 1);
}

// Try an asm snippet. Returns true if it passed i.e. ran without generating
// a SIGILL and returned true. Returns false if a SIGILL was generated, or
// if it returned false.
// Other signals are not handled.
bool try_asm_snippet(bool (*asm_snippet)()) {
  // This function installs and restores signal handlers. The only way it's ever
  // going to be reentrant is with a big lock around it.
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);

  // Install the SIGILL signal handler. Save any existing signal handler for
  // restoring later.
  struct sigaction sigill_action;
  memset(&sigill_action, 0, sizeof(sigill_action));
  sigill_action.sa_handler = sigill_handler;
  sigemptyset(&sigill_action.sa_mask);
  struct sigaction old_action;
  sigaction(SIGILL, &sigill_action, &old_action);

  // Try the snippet.
  bool got_sigill =
      sigsetjmp(global_sigjmp_buf_just_before_trying_snippet(), true);
  bool snippet_retval = false;
  if (!got_sigill) {
    snippet_retval = asm_snippet();
    wait_until_no_pending_sigill();
  }

  // Restore the old signal handler.
  sigaction(SIGILL, &old_action, nullptr);

  return snippet_retval && !got_sigill;
}

bool dotprod_asm_snippet() {
  // maratek@ mentioned that for some other ISA extensions (fp16)
  // there have been implementations that failed to generate SIGILL even
  // though they did not correctly implement the instruction. Just in case
  // a similar situation might exist here, we do a simple correctness test.
  int result = 0;
  asm volatile(
      "mov w0, #100\n"
      "dup v0.16b, w0\n"
      "dup v1.4s, w0\n"
      ".word 0x6e809401  // udot v1.4s, v0.16b, v0.16b\n"
      "mov %w[result], v1.s[0]\n"
      : [ result ] "=r"(result)
      :
      : "x0", "v0", "v1");
  // Expecting 100 (input accumulator value) + 100 * 100 + ... (repeat 4 times)
  return result == 40100;
}

bool DetectDotprodBySigIllMethod() {
  return try_asm_snippet(dotprod_asm_snippet);
}

// Intentionally keep checking for __linux__ here in case we want to
// extend RUY_IMPLEMENT_DETECT_DOTPROD outside of linux in the future.
#ifdef __linux__
bool IsLinuxAuxvMethodAvailable() {
  struct utsname utsbuf;
  uname(&utsbuf);
  int major, minor, patch;
  if (3 != sscanf(utsbuf.release, "%d.%d.%d", &major, &minor, &patch)) {
    return false;
  }
  // This is implemented in linux 4.14.111, not in 4.14.105.
  return major > 4 ||
         (major == 4 && (minor > 14 || (minor == 14 && patch >= 111)));
}

bool DetectDotprodByLinuxAuxvMethod() {
  // This is the value of HWCAP_ASIMDDP in sufficiently recent Linux headers,
  // however we need to support building against older headers for the time
  // being.
  const int kLocalHwcapAsimddp = 1 << 20;
  return getauxval(AT_HWCAP) & kLocalHwcapAsimddp;
}
#endif

}  // namespace

bool DetectDotprod() {
#ifdef __linux__
  if (IsLinuxAuxvMethodAvailable()) {
    return DetectDotprodByLinuxAuxvMethod();
  }
#endif

  return DetectDotprodBySigIllMethod();
}

#else   // RUY_IMPLEMENT_DETECT_DOTPROD
bool DetectDotprod() { return false; }
#endif  // RUY_IMPLEMENT_DETECT_DOTPROD

}  // namespace ruy
