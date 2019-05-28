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

// b/132973173: this signal-handling code does not work as intended on iOS,
// resulting in 'EXC_BAD_INSTRUCTION' signals killing the process.
// Is it because this code uses a signal handler for SIGILL, and Apple's
// EXC_BAD_INSTRUCTION is actually a different signal?
// Anyway, we don't need this code on Apple devices at the moment, as none of
// them supports dot-product instructions at the moment.
// In fact, for the moment, we only care about Linux, so restricting to it
// limits our risk.
#if defined __aarch64__ && defined __linux__
#define RUY_IMPLEMENT_DETECT_DOTPROD
#endif

#ifdef RUY_IMPLEMENT_DETECT_DOTPROD

#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <mutex>

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
};

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
};

}  // namespace

bool DetectDotprod() { return try_asm_snippet(dotprod_asm_snippet); }

#else
bool DetectDotprod() { return false; }
#endif

}  // namespace ruy
