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

#include "tensorflow/compiler/xla/tests/literal_test_util.h"

#include "tensorflow/compiler/xla/literal_comparison.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/test.h"

namespace xla {

namespace {

// Writes the given literal to a file in the test temporary directory.
void WriteLiteralToTempFile(const LiteralSlice& literal, const string& name) {
  auto get_hostname = [] {
    char hostname[1024];
    gethostname(hostname, sizeof hostname);
    hostname[sizeof hostname - 1] = 0;
    return string(hostname);
  };
  int64 now_usec = tensorflow::Env::Default()->NowMicros();
  string filename = tensorflow::io::JoinPath(
      tensorflow::testing::TmpDir(),
      tensorflow::strings::Printf("tempfile-%s-%llx-%s", get_hostname().c_str(),
                                  now_usec, name.c_str()));
  TF_CHECK_OK(tensorflow::WriteBinaryProto(tensorflow::Env::Default(), filename,
                                           literal.ToProto()));
  LOG(ERROR) << "wrote to " << name << " file: " << filename;
}

// Callback helper that dumps literals to temporary files in the event of a
// miscomparison.
void OnMiscompare(const LiteralSlice& expected, const LiteralSlice& actual,
                  const LiteralSlice& mismatches) {
  LOG(INFO) << "expected: " << ShapeUtil::HumanString(expected.shape()) << " "
            << literal_comparison::ToStringTruncated(expected);
  LOG(INFO) << "actual:   " << ShapeUtil::HumanString(actual.shape()) << " "
            << literal_comparison::ToStringTruncated(actual);
  LOG(INFO) << "Dumping literals to temp files...";
  WriteLiteralToTempFile(expected, "expected");
  WriteLiteralToTempFile(actual, "actual");
  WriteLiteralToTempFile(mismatches, "mismatches");
}

::testing::AssertionResult StatusToAssertion(const Status& s) {
  if (s.ok()) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << s.error_message();
}

}  // namespace

/* static */ ::testing::AssertionResult LiteralTestUtil::EqualShapes(
    const Shape& expected, const Shape& actual) {
  return StatusToAssertion(literal_comparison::EqualShapes(expected, actual));
}

/* static */ ::testing::AssertionResult LiteralTestUtil::EqualShapesAndLayouts(
    const Shape& expected, const Shape& actual) {
  if (expected.ShortDebugString() != actual.ShortDebugString()) {
    return ::testing::AssertionFailure()
           << "want: " << expected.ShortDebugString()
           << " got: " << actual.ShortDebugString();
  }
  return ::testing::AssertionSuccess();
}

/* static */ ::testing::AssertionResult LiteralTestUtil::Equal(
    const LiteralSlice& expected, const LiteralSlice& actual) {
  return StatusToAssertion(literal_comparison::Equal(expected, actual));
}

/* static */ ::testing::AssertionResult LiteralTestUtil::Near(
    const LiteralSlice& expected, const LiteralSlice& actual,
    const ErrorSpec& error_spec, bool detailed_message) {
  return StatusToAssertion(literal_comparison::Near(
      expected, actual, error_spec, detailed_message, &OnMiscompare));
}

/* static */ ::testing::AssertionResult LiteralTestUtil::NearOrEqual(
    const LiteralSlice& expected, const LiteralSlice& actual,
    const tensorflow::gtl::optional<ErrorSpec>& error) {
  if (error.has_value()) {
    VLOG(1) << "Expects near";
    return StatusToAssertion(literal_comparison::Near(
        expected, actual, *error, /*detailed_message=*/false, &OnMiscompare));
  }
  VLOG(1) << "Expects equal";
  return StatusToAssertion(literal_comparison::Equal(expected, actual));
}

}  // namespace xla
