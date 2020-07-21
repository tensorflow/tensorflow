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

#include "tensorflow/lite/micro/micro_error_reporter.h"

int main(int argc, char** argv) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;
  TF_LITE_REPORT_ERROR(error_reporter, "Number: %d", 42);
  TF_LITE_REPORT_ERROR(error_reporter, "Badly-formed format string %");
  TF_LITE_REPORT_ERROR(error_reporter,
                       "Another % badly-formed %% format string");

  // Workaround gcc C/C++ horror-show.  For 32-bit targets  va_list is simply an alias for
  // char *, gcc-7 (at least) converts the second string constant to const char * 
  // only a warning that C++ actually forbids this with the result that
  // MicroErrorReporter::Report(const char* format, va_list args) 
  // is called rather than
  // MicroErrorReporter::Report(const char* format, ...) 
  // with predictably painful results.  
  // TODO The clean solution for this would to remove the 
  // overload of ErrorReporter::Report. However, this would be a breaking API change...
  //
#if __GNUG__ 
  TF_LITE_REPORT_ERROR(error_reporter, "~~~%s~~~", "ALL TESTS PASSED", 0/*dummy*/);
#else
  TF_LITE_REPORT_ERROR(error_reporter, "~~~%s~~~", "ALL TESTS PASSED");
#endif
}
