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

// Test that popens a child process with the VLOG-ing environment variable set
// for the logging framework, and observes VLOG_IS_ON and VLOG macro output.

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/test.h"

#include <string.h>

namespace tensorflow {
namespace {

int RealMain(const char* argv0, bool do_vlog) {
  if (do_vlog) {
#if !defined(PLATFORM_GOOGLE)
    // Note, we only test this when !defined(PLATFORM_GOOGLE) because
    // VmoduleActivated doesn't exist in that implementation.
    //
    // Also, we call this internal API to simulate what would happen if
    // differently-named translation units attempted to VLOG, so we don't need
    // to create dummy translation unit files.
    bool ok = internal::LogMessage::VmoduleActivated("vmodule_test.cc", 7) &&
              internal::LogMessage::VmoduleActivated("shoobadooba.h", 3);
    if (!ok) {
      fprintf(stderr, "vmodule activated levels not as expected.\n");
      return EXIT_FAILURE;
    }
#endif

    // Print info on which VLOG levels are activated.
    fprintf(stderr, "VLOG_IS_ON(8)? %d\n", VLOG_IS_ON(8));
    fprintf(stderr, "VLOG_IS_ON(7)? %d\n", VLOG_IS_ON(7));
    fprintf(stderr, "VLOG_IS_ON(6)? %d\n", VLOG_IS_ON(6));
    // Do some VLOG-ing.
    VLOG(8) << "VLOG(8)";
    VLOG(7) << "VLOG(7)";
    VLOG(6) << "VLOG(6)";
    LOG(INFO) << "INFO";
    return EXIT_SUCCESS;
  }

  // Popen the child process.
  std::string command = std::string(argv0);
#if defined(PLATFORM_GOOGLE)
  command = command + " do_vlog --vmodule=vmodule_test=7 --alsologtostderr";
#else
  command =
      "TF_CPP_VMODULE=vmodule_test=7,shoobadooba=3 " + command + " do_vlog";
#endif
  command += " 2>&1";
  fprintf(stderr, "Running: \"%s\"\n", command.c_str());
  FILE* f = popen(command.c_str(), "r");
  if (f == nullptr) {
    fprintf(stderr, "Failed to popen child: %s\n", strerror(errno));
    return EXIT_FAILURE;
  }

  // Read data from the child's stdout.
  constexpr int kBufferSizeBytes = 4096;
  char buffer[kBufferSizeBytes];
  size_t result = fread(buffer, sizeof(buffer[0]), kBufferSizeBytes - 1, f);
  if (result == 0) {
    fprintf(stderr, "Failed to read from child stdout: %zu %s\n", result,
            strerror(errno));
    return EXIT_FAILURE;
  }
  buffer[result] = '\0';
  int status = pclose(f);
  if (status == -1) {
    fprintf(stderr, "Failed to close popen child: %s\n", strerror(errno));
    return EXIT_FAILURE;
  }

  // Check output is as expected.
  const char kExpected[] =
      "VLOG_IS_ON(8)? 0\nVLOG_IS_ON(7)? 1\nVLOG_IS_ON(6)? 1\n";
  if (strstr(buffer, kExpected) == nullptr) {
    fprintf(stderr, "error: unexpected output from child: \"%.*s\"\n",
            kBufferSizeBytes, buffer);
    return EXIT_FAILURE;
  }
  bool ok = strstr(buffer, "VLOG(7)\n") != nullptr &&
            strstr(buffer, "VLOG(6)\n") != nullptr &&
            strstr(buffer, "VLOG(8)\n") == nullptr;
  if (!ok) {
    fprintf(stderr, "error: VLOG output not as expected: \"%.*s\"\n",
            kBufferSizeBytes, buffer);
    return EXIT_FAILURE;
  }

  // Success!
  return EXIT_SUCCESS;
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  bool do_vlog = argc >= 2 && strcmp(argv[1], "do_vlog") == 0;
  return tensorflow::RealMain(argv[0], do_vlog);
}
