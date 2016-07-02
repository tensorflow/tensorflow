/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/cloud/base64.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(Base64, EncodeDecode) {
  const string original = "a simple test message!";
  string encoded;
  TF_EXPECT_OK(Base64Encode(original, &encoded));
  EXPECT_EQ("YSBzaW1wbGUgdGVzdCBtZXNzYWdlIQ", encoded);

  string decoded;
  TF_EXPECT_OK(Base64Decode(encoded, &decoded));
  EXPECT_EQ(original, decoded);
}

}  // namespace tensorflow
