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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MINI_BENCHMARK_TEST_HELPER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MINI_BENCHMARK_TEST_HELPER_H_

#include <string>

namespace tflite {
namespace acceleration {

class MiniBenchmarkTestHelper {
 public:
  // Dump the in-memory binary data stream to the testing temporary directory w/
  // a file name as 'filename'.
  // It retruns the full file path of the dumped file.
  static std::string DumpToTempFile(const std::string& filename,
                                    const unsigned char* data, size_t length);

  // The constructor will check whether the testing environment supports to run
  // the mini benchmark. If yes, it will do additional testing setup
  // accordingly.
  MiniBenchmarkTestHelper();
  ~MiniBenchmarkTestHelper() {}
  bool should_perform_test() const { return should_perform_test_; }

 private:
  bool should_perform_test_;
};

}  // namespace acceleration
}  // namespace tflite
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MINI_BENCHMARK_TEST_HELPER_H_
