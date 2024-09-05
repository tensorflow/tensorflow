/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/runner.h"

#ifndef TFLITE_ACCELERATION_BENCHMARK_IN_PROCESS
#include <dlfcn.h>
#endif  // !TFLITE_ACCELERATION_BENCHMARK_IN_PROCESS
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <poll.h>
#include <signal.h>
#include <unistd.h>
#endif

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/constants.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"

#if defined(__ANDROID__) && !defined(TFLITE_ACCELERATION_BENCHMARK_IN_PROCESS)
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_runner_executable.h"
#endif  // __ANDROID__ && !TFLITE_ACCELERATION_BENCHMARK_IN_PROCESS

// Implementation notes and rationale:
//
// This class's primary client is the mini-benchmark. The mini-benchmark tries
// out different acceleration configurations for TFLite. The acceleration may
// hang or crash due to driver bugs. By Default, the mini-benchmark on Android
// runs in a separate process that is forked from the host application. This is
// done to prevent the benchmark from crashing or hanging the host application.
// All other platforms run the benchmark in the same process. If
// TFLITE_ACCELERATION_BENCHMARK_IN_PROCESS is defined, the mini-benchmark is
// forced to run in process..
//
// The separate process is implemented in native code. The main alternative
// would be to use a separate service process at the Android application
// level. The most important drawbacks of that approach would have been
// manifest merge issues in tests (see for example b/158142805) and the need for
// all applications to explicitly integrate the mini-benchmark at the app level.
//
// The native code uses popen(3) for the separate process. This is one of the
// few officially supported ways of safely using fork(2) on Android (See
// b/129156634 for a discussion on popen use in Android studio device-side, see
// https://groups.google.com/g/android-platform/c/80jr-_A-9bU for discussion on
// fork in general).
//
// The native process executes a small helper binary that then dynamically
// loads the shared library that tflite code lives in. This allows for minimal
// code size as only one copy of the tflite code is needed.
//
// The 8kb helper binary is extracted into an app data folder On P- we execute
// it directly.  On Q+ this is no longer allowed (see b/112357170) but we can
// use /system/bin/linker(64) as a trampoline.  (See also Chrome's documentation
// for a similar problem:
// https://chromium.googlesource.com/chromium/src/+/master/docs/android_native_libraries.md#crashpad-packaging).
// Using /system/bin/linker(64) does not work before Android Q (See
// b/112050209).
//
// The shared library where the code to be called lives is a JNI library that
// contains tflite. We detect the path to that library with dladdr(3). This way
// differences in the way the JNI libraries are bundled, named or delivered is
// automatically handled. The downside is that dladdr is broken on Android 23
// for JNI libraries that are not extracted from the apk (See b/37069572). If
// this becomes a serious issue we can try to parse /proc/self/maps and the apk
// zip directory to figure out the name. The alternative where the caller needs
// to pass in the library name requires more integration work at the packaging
// (app or SDK) level.
//
// The methods in this class return detailed error codes, so that telemetry can
// be used to detect issues in production without using strings which can be
// hard to make privacy-compliant.

namespace tflite {
namespace acceleration {
namespace {
std::string ShellEscape(const std::string& src);
}  // namespace

MinibenchmarkStatus ProcessRunner::Init() {
  if (!function_pointer_) {
    return kMinibenchmarkPreconditionNotMet;
  }
#if !defined(__ANDROID__) || defined(TFLITE_ACCELERATION_BENCHMARK_IN_PROCESS)
  return kMinibenchmarkSuccess;
#else  // __ANDROID__ && !TFLITE_ACCELERATION_BENCHMARK_IN_PROCESS
  tflite::acceleration::AndroidInfo android_info;
  if (!tflite::acceleration::RequestAndroidInfo(&android_info).ok()) {
    return kMinibenchmarkRequestAndroidInfoFailed;
  }
  if (android_info.android_sdk_version.length() < 2 ||
      android_info.android_sdk_version < "23") {
    // The codepaths have only been tested on 23+.
    return kMinibenchmarkUnsupportedPlatform;
  }

  // Find name of this shared object.
  std::string soname;
  Dl_info dl_info;
  int status = dladdr(function_pointer_, &dl_info);
  if (status != 0) {
    if (dl_info.dli_fname) {
      soname = dl_info.dli_fname;
    } else {
      return kMinibenchmarkDliFnameWasNull;
    }
  } else {
    return kMinibenchmarkDladdrReturnedZero;
  }
  // Check for b/37069572 - on SDK level 23 dli_fname may not contain the
  // library name, only the apk. (See comment at top of file).
  if (soname.size() >= 4 && soname.substr(soname.size() - 4) == ".apk") {
    return kMinibenchmarkDliFnameHasApkNameOnly;
  }

  // Construct path to runner, extracting the helper binary if needed.
  std::string runner_path;
  // TODO(b/172541832): handle multiple concurrent callers.
  runner_path = temporary_path_ + "/runner";
  (void)unlink(runner_path.c_str());
  std::string runner_contents(
      reinterpret_cast<const char*>(g_tflite_acceleration_embedded_runner),
      g_tflite_acceleration_embedded_runner_len);
  std::ofstream f(runner_path, std::ios::binary);
  if (!f.is_open()) {
    return kMinibenchmarkCouldntOpenTemporaryFileForBinary;
  }
  f << runner_contents;
  f.close();
  if (chmod(runner_path.c_str(), 0500) != 0) {
    return kMinibenchmarkCouldntChmodTemporaryFile;
  }
  runner_path = ShellEscape(runner_path);
  if (android_info.android_sdk_version >= "29") {
    // On 29+ we need to use /system/bin/linker to load the binary from the app,
    // as exec()ing writable files was blocked for security. (See comment at top
    // of file).
#if defined(__arm__) || defined(__i386__)
    std::string linker_path = "/system/bin/linker";
#else
    std::string linker_path = "/system/bin/linker64";
#endif
    runner_path = linker_path + " " + runner_path;
  }

  // Commit.
  runner_path_ = runner_path;
  soname_ = soname;
  return kMinibenchmarkSuccess;
#endif  // !__ANDROID__ || TFLITE_ACCELERATION_BENCHMARK_IN_PROCESS
}

// TODO(b/245901066): Refactor the runner to separate Multi-process
// implementation and in process implementors, and remove the ifdef guards.
#ifndef _WIN32
bool ProcessRunner::KillProcessWhenTimedOut(FILE* fstream) {
  // The first fread() should get subprocess id. It's important to
  // read the same number of bytes as on the write side, so that this fread()
  // does not block.
  const int array_length = 1 + kPidBufferLength;
  char buffer[array_length];
  memset(buffer, '\0', array_length);
  ssize_t length = fread(buffer, 1, kPidBufferLength, fstream);
  int pid;
  if (length != kPidBufferLength || !absl::SimpleAtoi(buffer, &pid)) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Failed to get Validator subprocess id: %s", buffer);
    return false;
  }
  struct pollfd pfd[1];
  pfd[0].fd = fileno(fstream);
  // Wait for the fstream to be closed.
  pfd[0].events = POLLHUP;
  int poll_ret = poll(pfd, 1, timeout_millisec_);
  // Kill the subprocess if timed out.
  if (poll_ret == 0) {
    kill(pid, SIGKILL);
    return true;
  } else if (poll_ret < 0) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Validator timer failed: %s",
                         strerror(errno));
  }
  return false;
}
#endif  // _WIN32

MinibenchmarkStatus ProcessRunner::Run(const Allocation* model_allocation,
                                       const std::vector<std::string>& args,
                                       std::string* output, int* exitcode,
                                       int* signal) {
#ifdef _WIN32
  return kMinibenchmarkUnsupportedPlatform;
#else  // !_WIN32
  if (!output || !exitcode) {
    return kMinibenchmarkPreconditionNotMet;
  }
  int benchmark_process_status = 0;
  MinibenchmarkStatus status = kMinibenchmarkCommandFailed;
#ifdef TFLITE_ACCELERATION_BENCHMARK_IN_PROCESS
  if (function_pointer_) {
    benchmark_process_status = RunInprocess(model_allocation, args);
  } else {
    return kMinibenchmarkPreconditionNotMet;
  }
#else   // !TFLITE_ACCELERATION_BENCHMARK_IN_PROCESS
  if (runner_path_.empty()) {
    return kMinibenchmarkPreconditionNotMet;
  }
  // runner_path_ components are escaped earlier.
  std::string cmd = runner_path_ + " " + ShellEscape(soname_) + " " +
                    ShellEscape(function_name_);

  // If model is not null, open a pipe() and add pipe model path as cmdline
  // argv[3]. If model is null, argv[0] should be the model path.
  int pipe_fds[2];
  if (model_allocation != nullptr) {
    if (pipe(pipe_fds) < 0) {
      *exitcode = errno;
      return kMinibenchmarkPipeFailed;
    }
    std::string pipe_model_path = absl::StrCat(
        "pipe:", pipe_fds[0], ":", pipe_fds[1], ":", model_allocation->bytes());
    cmd = cmd + " " + ShellEscape(pipe_model_path);
  }

  // Add the rest of the cmdline args.
  for (const auto& arg : args) {
    cmd = cmd + " " + ShellEscape(arg);
  }

  FILE* f = popen(cmd.c_str(), "r");
  if (!f) {
    *exitcode = errno;
    return kMinibenchmarkPopenFailed;
  }

  // Write model to MiniBenchmark process.
  if (model_allocation != nullptr) {
    close(pipe_fds[0]);
    int written_bytes = 0;
    int remaining_bytes = model_allocation->bytes();
    const uint8_t* current =
        static_cast<const uint8_t*>(model_allocation->base());
    while (remaining_bytes > 0 &&
           (written_bytes = write(pipe_fds[1], current, remaining_bytes)) > 0) {
      remaining_bytes -= written_bytes;
      current += written_bytes;
    }
    close(pipe_fds[1]);
    if (written_bytes <= 0 || remaining_bytes > 0) {
      *exitcode = errno;
      return kMinibenchmarkPipeFailed;
    }
  }

  // Note: KillProcessWhenTimedOut() will block until f is closed or timeout has
  // reached. It will cause issue if subprocess is blocked on writing to f.
  if (timeout_millisec_ > 0 && KillProcessWhenTimedOut(f)) {
    status = kMinibenchmarkCommandTimedOut;
    TFLITE_LOG_PROD(
        TFLITE_LOG_INFO,
        "Validator did not finish after %dms. Tried to kill the test.",
        timeout_millisec_);
  }
  std::vector<char> buffer(4 * 1024, 0);
  ssize_t length;
  std::string ret;
  do {
    length = fread(buffer.data(), 1, buffer.size(), f);
    ret = ret + std::string(buffer.data(), length);
  } while (length == buffer.size());
  *output = ret;
  benchmark_process_status = pclose(f);
#endif  //  TFLITE_ACCELERATION_BENCHMARK_IN_PROCESS
  if (WIFEXITED(benchmark_process_status)) {
    *exitcode = WEXITSTATUS(benchmark_process_status);
    *signal = 0;
    if (*exitcode == kMinibenchmarkSuccess) {
      status = kMinibenchmarkSuccess;
    }
  } else if (WIFSIGNALED(benchmark_process_status)) {
    *exitcode = 0;
    *signal = WTERMSIG(benchmark_process_status);
  }
  return status;
#endif  // _WIN32
}

#ifdef TFLITE_ACCELERATION_BENCHMARK_IN_PROCESS
#ifndef __W_EXITCODE  // Mac
#define __W_EXITCODE(ret, sig) ((ret) << 8 | (sig))
#endif

int ProcessRunner::RunInprocess(const Allocation* model_allocation,
                                const std::vector<std::string>& user_args) {
  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "Running Validator in-process.");
  std::vector<std::string> args_string;
  args_string.push_back("inprocess");
  args_string.push_back("inprocess");
  args_string.push_back(function_name_);

  std::thread write_thread;
  if (model_allocation != nullptr) {
    int pipe_fds[2];
    if (pipe(pipe_fds) < 0) {
      return __W_EXITCODE(kMinibenchmarkPipeFailed, 0);
    }

    // Add pipe_model_path when model is not null.
    // Model loader won't close the write file descriptor when it's -1.
    args_string.push_back(
        absl::StrCat("pipe:", pipe_fds[0], ":-1:", model_allocation->bytes()));

    // When running MiniBenchmark in-process, we start a separate thread for
    // writing to pipe.
    write_thread = std::thread([pipe_fds, model_allocation,
                                error_reporter = error_reporter_]() {
      int written_bytes = 0;
      int remaining_bytes = model_allocation->bytes();
      const uint8_t* current =
          static_cast<const uint8_t*>(model_allocation->base());
      while (remaining_bytes > 0 &&
             (written_bytes = write(pipe_fds[1], current, remaining_bytes)) >
                 0) {
        remaining_bytes -= written_bytes;
        current += written_bytes;
      }
      close(pipe_fds[1]);
      if (written_bytes < 0 || remaining_bytes > 0) {
        TF_LITE_REPORT_ERROR(
            error_reporter,
            "Failed to write Model to pipe: %s. Expect to write %d "
            "bytes, %d bytes written.",
            strerror(errno), remaining_bytes, written_bytes);
      }
    });
  }

  for (int i = 0; i < user_args.size(); i++) {
    args_string.push_back(user_args[i]);
  }
  std::vector<std::vector<char>> args_char(args_string.size());
  std::vector<char*> argv(args_string.size());
  for (int i = 0; i < args_string.size(); i++) {
    args_char[i] = {args_string[i].begin(), args_string[i].end()};
    // Compiler adds '\0' for std::string to indicate the end of string
    // automatically. For char* string, '\0' needs to be add at the end of
    // string manually.
    args_char[i].push_back('\0');
    argv[i] = args_char[i].data();
  }

  int (*function_pointer)(int, char**) =
      reinterpret_cast<int (*)(int, char**)>(function_pointer_);
  int exit_code = __W_EXITCODE(function_pointer(argv.size(), argv.data()), 0);
  if (write_thread.joinable()) {
    write_thread.join();
  }
  return exit_code;
}
#endif  // TFLITE_ACCELERATION_BENCHMARK_IN_PROCESS

namespace {

// kDontNeedShellEscapeChars and ShellEscape are copied from absl, which
// copied them from Python. Copied here because tflite core libraries should
// not depend on absl (both for size reasons and for possible version skew):

static const char kDontNeedShellEscapeChars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+-_.=/:,@";

std::string ShellEscape(const std::string& src) {
  if (!src.empty() &&  // empty string needs quotes
      src.find_first_not_of(kDontNeedShellEscapeChars) == std::string::npos) {
    // only contains chars that don't need quotes; it's fine
    return src;
  } else if (src.find('\'') == std::string::npos) {  // NOLINT
    // no single quotes; just wrap it in single quotes
    return "'" + src + "'";
  } else {
    // needs double quote escaping
    std::string result = "\"";
    for (const char c : src) {
      switch (c) {
        case '\\':
        case '$':
        case '"':
        case '`':
          result.push_back('\\');
      }
      result.push_back(c);
    }
    result.push_back('"');
    return result;
  }
}
}  // namespace

}  // namespace acceleration
}  // namespace tflite
