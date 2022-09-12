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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_RUNNER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_RUNNER_H_

#include <string>
#include <vector>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"

namespace tflite {
namespace acceleration {

// Class that runs a C-main-function -compatible exported symbol in a separate
// process on Android. Support all Android 24+ devices and some 23 devices. May
// also work on 22- but has not been tested.
//
// Requirements on the caller:
// - The function to be called must reside in the same shared library as this
// code and it must be exported (typically by being extern "C" and having a name
// that starts with Java).
// - The function to be called must have the same signature as C main:
//     extern "C" int Java_foo(int argc, char** argv)
// - The librunner_main.so shared object from this directory must
// reside in the same location as the shared object above (typically by
// depending on the runner_main_library_for_deps target from this
// directory in a java_library or java_binary)
//
// The return values are meant to be detailed enough for telemetry.
//
// For reasons behind the above restrictions and requirements, see
// implementation notes in runner.cc
//
// Warning: this class will just run the provided code in-process when compiled
// for non-Android.
class ProcessRunner {
 public:
  // Construct ProcessRunner. 'temporary_path' should be a suitable subdirectory
  // of the app data path for extracting the helper binary on Android P-.
  //
  // Since the function will be called through popen() only return values
  // between 0 and 127 are well-defined.
  ProcessRunner(const std::string& temporary_path,
                const std::string& function_name,
                int (*function_pointer)(int argc, char** argv));

  // Initialize runner.
  MinibenchmarkStatus Init();

  // Run function in separate process. Returns function's output to stdout and
  // the shell exitcode. Stderr is discarded.
  //
  // The function will be called with argc and argv corresponding to a command
  // line like:
  //     helper_binary function_name (optional: model path) args
  // If model is not null, runner will use pipe() to pass the model
  // to subprocess. Otherwise, args[0] should be a model path.
  // The args are escaped for running through the shell.
  //
  // The 'output' and 'exitcode' and `signal` are set as follows based on the
  // return value:
  //   kMinibenchmarkUnknownStatus, kMinibenchmarkPreconditionNotMet: undefined
  //   kMinibenchmarkPopenFailed:
  //       *output is an empty string
  //       *exitcode is errno after popen()
  //       *signal is 0
  //   kMinibenchmarkCommandFailed, kMinibenchmarkSuccess:
  //       *output is stdout produced from function
  //       *exitcode is:
  //        - if the process terminated normally:
  //          the return value of the benchmark function or, if function
  //          loading fails one the MinibenchmarkStatus values listed under
  //          'Runner main status codes' to describe the failure.
  //        - if the process has been terminated by a signal: 0
  //       *signal is:
  //        - if the process has been terminated by a signal: the signal number
  //        - 0 otherwise
  //
  // To be considered successful, the function must return
  // kMinibenchmarkSuccess. This is because some GPU drivers call exit(0) as a
  // bailout and we don't want to confuse that with a successful run.
  MinibenchmarkStatus Run(flatbuffers::FlatBufferBuilder* model,
                          const std::vector<std::string>& args,
                          std::string* output, int* exitcode, int* signal);

  ProcessRunner(ProcessRunner&) = delete;
  ProcessRunner& operator=(const ProcessRunner&) = delete;

 private:
#ifndef __ANDROID__
  int RunInprocess(flatbuffers::FlatBufferBuilder* model,
                   const std::vector<std::string>& args);
#endif  // !__ANDROID__
  std::string temporary_path_;
  std::string function_name_;
  void* function_pointer_;
  std::string runner_path_;
  std::string soname_;
};

}  // namespace acceleration
}  // namespace tflite
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_RUNNER_H_
