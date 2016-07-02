/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/saved_tensor_slice_util.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace checkpoint {

namespace {

// Testing serialization of tensor name and tensor slice in the ordered code
// format.
TEST(TensorShapeUtilTest, TensorNameSliceToOrderedCode) {
  {
    TensorSlice s = TensorSlice::ParseOrDie("-:-:1,3:4,5");
    string buffer = EncodeTensorNameSlice("foo", s);
    string name;
    s.Clear();
    TF_CHECK_OK(DecodeTensorNameSlice(buffer, &name, &s));
    EXPECT_EQ("foo", name);
    EXPECT_EQ("-:-:1,3:4,5", s.DebugString());
  }
}

}  // namespace

}  // namespace checkpoint

}  // namespace tensorflow
