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
// tensorflow/lite/experimental/micro/bluepill/debug_log.cc or the mbed one on
// tensorflow/lite/experimental/micro/mbed/debug_log.cc.

#include "../tensorflow/tensorflow/lite/experimental/micro/debug_log.h"
#include <cstdio>

extern "C"{
// Copyright (c) 2014-2016, XMOS Ltd, All rights reserved
#include <stdarg.h>
#include <syscall.h>
#include <limits.h>
#include <string.h>
#include <ctype.h>

#undef debug_printf


#ifndef DEBUG_PRINTF_BUFSIZE
#define DEBUG_PRINTF_BUFSIZE 130
#endif


static void debug_printf(char * fmt, ...)
{
  char * marker;
  int intArg;
  unsigned int uintArg;
  char * strArg;

  char buf[DEBUG_PRINTF_BUFSIZE];
  char *end = &buf[DEBUG_PRINTF_BUFSIZE - 1];

  va_list args;

  va_start(args,fmt);
  marker = fmt;
  char *p = buf;
  while (*fmt) {
    if (p > end) {
      // flush
      _write(FD_STDOUT, buf, p - buf);
      p = buf;
    }
    switch (*fmt) {
    case '%':
      fmt++;
      if (*(fmt) == '-' || *(fmt) == '+' || *(fmt) == '#' || *(fmt) == ' ') {
        // Ignore flags
        fmt++;
      }
      while (*(fmt) && *(fmt) >= '0' && *(fmt) <= '9') {
        // Ignore width
        fmt++;
      }
      // Use 'tolower' to ensure both %x/%X do something sensible
      switch (tolower(*(fmt))) {

      case 's':
        strArg = va_arg(args, char *);
        int len = strlen(strArg);
        if (len > (end - buf)) {
                // flush
          _write(FD_STDOUT, buf, p - buf);
          p = buf;
        }
        if (len > (end - buf))
          len = end - buf;
        memcpy(p, strArg, len);
        p += len;
        break;
      }
      break;

    default:
      *p++ = *fmt;
    }
    fmt++;
  }
  _write(FD_STDOUT, buf, p - buf);
  va_end(args);

  return;
}


void DebugLog(const char* s) 
{ 
    debug_printf("%s", s); 
}

}
