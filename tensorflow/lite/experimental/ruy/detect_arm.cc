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

/* Detection of dotprod instructions on ARM.
 * The current Linux-specific code relies on sufficiently new Linux kernels:
 * At least Linux 4.15 in general; on Android, at least Linux 4.14.111 thanks to
 * a late backport. This was backported just before the Android 10 release, so
 * this is leaving out pre-release Android 10 builds as well as earlier Android
 * versions.
 *
 * It is possible to detect instructions in other ways that don't rely on
 * an OS-provided feature identification mechanism:
 *
 *   (A) We used to have a SIGILL-handler-based method that worked at least
 *       on Linux. Its downsides were (1) crashes on a few devices where
 *       signal handler installation didn't work as intended; (2) additional
 *       complexity to generalize to other Unix-ish operating systems including
 *       iOS; (3) source code complexity and fragility of anything installing
 *       and restoring signal handlers; (4) confusing behavior under a debugger.
 *
 *   (B) We also experimented with a fork-ing approach where a subprocess
 *       tries the instruction. Compared to (A), this is much simpler and more
 *       reliable and portable, but also much higher latency on Android where
 *       an uncaught signal typically causes a 100 ms latency.
 *
 * Should there be interest in either technique again in the future,
 * code implementing both (A) and (B) can be found in earlier revisions of this
 * file - in actual code for (A) and in a comment for (B).
 */

#include "tensorflow/lite/experimental/ruy/detect_arm.h"

#ifdef __linux__
#include <sys/auxv.h>
#endif

namespace ruy {

namespace {

#if defined __linux__ && defined __aarch64__
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
#if defined __linux__ && defined __aarch64__
  return DetectDotprodByLinuxAuxvMethod();
#endif

  return false;
}

}  // namespace ruy
