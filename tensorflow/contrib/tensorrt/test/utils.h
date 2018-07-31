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

#ifndef TENSORFLOW_CONTRIB_TENSORRT_TEST_UTILS_H_
#define TENSORFLOW_CONTRIB_TENSORRT_TEST_UTILS_H_

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tensorrt {
namespace test {

// Helper methods to inject values used by testing tools.
void EnableTestValue();
void ClearTestValues(const string& pattern);
void AddTestValue(const string& label, const string& value);
string GetTestValue(const string& label);

#define TRT_RETURN_IF_TEST_VALUE(label, value_to_return)     \
  do {                                                       \
    if (::tensorflow::tensorrt::test::GetTestValue(label) == \
        value_to_return) {                                   \
      return errors::Internal("Injected manually");          \
    }                                                        \
  } while (0)

}  // namespace test
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_TENSORRT_TEST_UTILS_H_
