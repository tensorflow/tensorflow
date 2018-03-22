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

#include "tensorflow/core/platform/cloud/gce_env_utils.h"

#if defined(PLATFORM_WINDOWS)
#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

// The order if these includes is important, windows.h has to come first.
// clang-format off
#include <windows.h>   // NOLINT
#include <tchar.h>     // NOLINT
#include <shellapi.h>  // NOLINT
// clang-format on
#else
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#endif

namespace tensorflow {

constexpr char kExpectedGoogleProductName[] = "Google";
constexpr char kExpectedGceProductName[] = "Google Compute Engine";

constexpr char kWinCheckCommand[] = "powershell.exe";
constexpr char kWinCheckCommandArgs[] =
    "(Get-WmiObject -Class Win32_BIOS).Manufacturer";

constexpr char kLinuxProductNameFile[] = "/sys/class/dmi/id/product_name";

const size_t kBiosDataBufferSize = 256;

namespace {

#if defined(PLATFORM_WINDOWS)

Status IsRunningOnWinGce(bool* is_running_under_gce) {
  *is_running_under_gce = FALSE;
  SECURITY_ATTRIBUTES sa;
  sa.nLength = sizeof(sa);
  sa.lpSecurityDescriptor = NULL;
  sa.bInheritHandle = TRUE;

  // Handles to input and output of the pipe connecting us
  // to the child process running powershell(). The output of this
  // child process will be written to 'process_output_in' and read from
  // 'process_output_in'.
  HANDLE process_output_out = NULL;
  HANDLE process_output_in = NULL;

  // Create the actually pipe connecting us to the child process.
  if (!CreatePipe(&process_output_out, &process_output_in, &sa, 0)) {
    return errors::Internal("CreatePipe() failed");
  }
  if (!SetHandleInformation(process_output_out, HANDLE_FLAG_INHERIT, 0)) {
    return errors::Internal("SetHandleInformation() failed");
  }

  PROCESS_INFORMATION pi;
  STARTUPINFO si;
  DWORD flags = CREATE_NO_WINDOW;
  ZeroMemory(&pi, sizeof(pi));
  ZeroMemory(&si, sizeof(si));
  si.cb = sizeof(si);
  si.dwFlags |= STARTF_USESTDHANDLES;
  si.hStdInput = NULL;

  // Connect the process to pipe's input.
  si.hStdError = process_output_in;
  si.hStdOutput = process_output_in;
  // Execute (and wait for) powershell command to read the product information
  // out of the registry.
  TCHAR cmd[kBiosDataBufferSize];
  snprintf(cmd, kBiosDataBufferSize, "%s %s", _T(kWinCheckCommand),
           _T(kWinCheckCommandArgs));

  if (!CreateProcess(NULL, cmd, NULL, NULL, TRUE, flags, NULL, NULL, &si,
                     &pi)) {
    return errors::Internal("CreateProcess() failed");
  }

  WaitForSingleObject(pi.hProcess, INFINITE);
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);

  // Read data from the pipe. Note that we are reading only kBiosDataBufferSize
  // chars. There might be technically more data than that but we are looking
  // for Google product identifiers that are much shorter than
  // kBiosDataBufferSize.
  DWORD dwread = 0;
  CHAR buffer[kBiosDataBufferSize];
  if (!ReadFile(process_output_out, buffer, kBiosDataBufferSize, &dwread,
                NULL)) {
    return errors::Internal("Failed reading from the pipe.");
  }
  std::string output(buffer, 0, dwread);
  // Trim whitespaces
  output.erase(output.begin(),
               std::find_if(output.begin(), output.end(),
                            [](int ch) { return !std::isspace(ch); }));
  output.erase(std::find_if(output.rbegin(), output.rend(),
                            [](int ch) { return !std::isspace(ch); })
                   .base(),
               output.end());
  *is_running_under_gce =
      output == kExpectedGceProductName || output == kExpectedGoogleProductName;
  return Status::OK();
}

#else

Status IsRunningOnLinuxGce(Env* env, bool* is_running_under_gce) {
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(kLinuxProductNameFile, &file));
  char buf[kBiosDataBufferSize + 1];
  std::fill(buf, buf + kBiosDataBufferSize + 1, '\0');
  StringPiece product_name;
  const Status s = file->Read(0, kBiosDataBufferSize, &product_name, buf);
  if (!s.ok() && !errors::IsOutOfRange(s)) {
    // We expect OutOfRange error because bios file doesn't correspond to its
    // state size,
    return s;
  }
  str_util::RemoveLeadingWhitespace(&product_name);
  str_util::RemoveTrailingWhitespace(&product_name);
  *is_running_under_gce = (product_name == kExpectedGceProductName ||
                           product_name == kExpectedGoogleProductName);
  return Status::OK();
}

#endif

}  // namespace

Status IsRunningOnGce(Env* env, bool* is_running_under_gce) {
  *is_running_under_gce = false;
#if defined(PLATFORM_WINDOWS)
  return IsRunningOnWinGce(is_running_under_gce);
#else
  return IsRunningOnLinuxGce(env, is_running_under_gce);
#endif
}

}  // namespace tensorflow
