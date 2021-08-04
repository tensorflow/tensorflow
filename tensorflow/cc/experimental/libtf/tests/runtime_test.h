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
#ifndef TENSORFLOW_CC_EXPERIMENTAL_LIBTF_TESTS_RUNTIME_TEST_H_
#define TENSORFLOW_CC_EXPERIMENTAL_LIBTF_TESTS_RUNTIME_TEST_H_

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "tensorflow/c/eager/unified_api_testutil.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/cc/experimental/libtf/object.h"
#include "tensorflow/cc/experimental/libtf/runtime/runtime.h"
#include "tensorflow/cc/experimental/libtf/value.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"

namespace tf {
namespace libtf {
namespace runtime {

typedef Runtime (*RuntimeFn)();

class RuntimeTest : public ::testing::TestWithParam<RuntimeFn> {};

}  // namespace runtime
}  // namespace libtf
}  // namespace tf

#endif  // TENSORFLOW_CC_EXPERIMENTAL_LIBTF_TESTS_RUNTIME_TEST_H_
