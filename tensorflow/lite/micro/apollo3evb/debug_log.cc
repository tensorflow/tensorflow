/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// Reference implementation of the DebugLog() function that's required for a
// platform to support the TensorFlow Lite for Microcontrollers library. This is
// the only function that's absolutely required to be available on a target
// device, since it's used for communicating test results back to the host so
// that we can verify the implementation is working correctly.
// It's designed to be as easy as possible to supply an implementation though.
// On platforms that have a POSIX stack or C library, it can be written as a
// single call to `fprintf(stderr, "%s", s)` to output a string to the error
// stream of the console, but if there's no OS or C library available, there's
// almost always an equivalent way to write out a string to some serial
// interface that can be used instead. For example on Arm M-series MCUs, calling
// the `bkpt #0xAB` assembler instruction will output the string in r1 to
// whatever debug serial connection is available. If you're running mbed, you
// can do the same by creating `Serial pc(USBTX, USBRX)` and then calling
// `pc.printf("%s", s)`.
// To add an equivalent function for your own platform, create your own
// implementation file, and place it in a subfolder with named after the OS
// you're targeting. For example, see the Cortex M bare metal version in
// tensorflow/lite/micro/bluepill/debug_log.cc or the mbed one on
// tensorflow/lite/micro/mbed/debug_log.cc.

#include "tensorflow/lite/micro/debug_log.h"

// These are headers from Ambiq's Apollo3 SDK.
#include "am_bsp.h"   // NOLINT
#include "am_util.h"  // NOLINT

extern "C" void DebugLog(const char* s) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
  static bool is_initialized = false;
  if (!is_initialized) {
    am_bsp_itm_printf_enable();
    is_initialized = true;
  }

  am_util_stdio_printf("%s", s);
#endif
}
