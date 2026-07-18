/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_FINGERPRINT_TEST_HELPERS_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_FINGERPRINT_TEST_HELPERS_H_

#include <iostream>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "experimental.h"  // from @XNNPACK
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/weight_cache.h"
#include "tensorflow/lite/delegates/xnnpack/weight_cache_test_helpers.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite::xnnpack {

struct TfLiteDelegateDeleter {
  void operator()(TfLiteDelegate* delegate) {
    TfLiteXNNPackDelegateDelete(delegate);
  }
};

using TfLiteDelegatePtr =
    std::unique_ptr<TfLiteDelegate, TfLiteDelegateDeleter>;

struct DelegateTest : public virtual testing::Test {
  void SetUp() override {
    TfLiteXNNPackDelegateOptions delegate_options =
        TfLiteXNNPackDelegateOptionsDefault();

    // By default, we try to setup a file weight cache to also check fingerprint
    // generation. If the test system doesn't support a file system, then the
    // cache file will be invalid.
    if (cache_file.IsValid()) {
      xnn_clear_fingerprints();
      delegate_options.weight_cache_file_path = cache_file.GetCPath();
      delegate_options.weight_cache_file_descriptor =
          cache_file.Duplicate().Release();
      delegate_options.flags |=
          TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
      check_for_cache_fingerprints = true;
    }

    xnnpack_delegate =
        TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&delegate_options));
    ASSERT_THAT(xnnpack_delegate, testing::NotNull());
  }

  void TearDown() override {
    if (check_for_cache_fingerprints) {
      ASSERT_TRUE(cache_file.IsValid());
      EXPECT_TRUE(IsCompatibleCacheFile(cache_file));
      if (AlterXNNPackFingerprints()) {
        EXPECT_FALSE(IsCompatibleCacheFile(cache_file));
      }
    }
  }

  // Artificially change fingerprint values.
  //
  // This allows us to check that changing a fingerprint value will make the
  // cache file incompatible.
  //
  // Returns the current number of fingerprints.
  int AlterXNNPackFingerprints() {
    int i = 0;
    int modified = 0;
    for (const xnn_fingerprint* fingerprint = xnn_get_fingerprint_by_idx(i);
         fingerprint != nullptr;
         fingerprint = xnn_get_fingerprint_by_idx(++i)) {
      xnn_fingerprint new_fingerprint = *fingerprint;
      ++new_fingerprint.value;
      xnn_set_fingerprint(new_fingerprint);
      ++modified;
    }
    std::cerr << "Fingerprint modified. The next call to IsCompatibleCacheFile "
                 "should fail.\n";
    return modified;
  }

  // Replaces the xnnpack delegate with a custom one.
  void UseCustomDelegate(const TfLiteXNNPackDelegateOptions& delegate_options) {
    check_for_cache_fingerprints = false;
    xnnpack_delegate =
        TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&delegate_options));
    ASSERT_THAT(xnnpack_delegate, testing::NotNull());
  }

  // Replaces the xnnpack delegate with one that sets up a file backed weight
  // cache.
  void UseDelegateWithFileWeightCache() {}

  // The default delegate is created in a generic way.
  TfLiteDelegatePtr xnnpack_delegate;
  tflite::xnnpack::TempFileDesc cache_file;
  bool check_for_cache_fingerprints = false;
};

}  // namespace tflite::xnnpack

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_FINGERPRINT_TEST_HELPERS_H_
