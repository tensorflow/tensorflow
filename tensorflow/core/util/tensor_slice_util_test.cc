/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/util/tensor_slice_util.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// Testing copying data from one tensor slice to another tensor slice
TEST(TensorSliceUtilTest, CopyTensorSliceToTensorSlice) {
  // We map out a 2-d tensor of size 4 X 5 and we want the final results look
  // like this:
  //
  //   0   1   2   3   4
  //   5   6   7   8   9
  //  10  11  12  13  14
  //  15  16  17  18  19
  //
  // We assume this is a row-major matrix
  //
  TensorShape shape({4, 5});

  // We will try to do a couple of slice to slice copies.

  // Case 1: simple identity copy
  // The slice is the "interior" of the matrix
  //   .   .   .   .   .
  //   .   6   7   8   .
  //   ,  11  12  13   .
  //   .   .   .   .   .
  {
    TensorSlice slice_s = TensorSlice::ParseOrDie("1,2:1,3");
    TensorSlice slice_d = TensorSlice::ParseOrDie("1,2:1,3");
    const float ptr_s[] = {6, 7, 8, 11, 12, 13};
    float ptr_d[6];
    for (int i = 0; i < 6; ++i) {
      ptr_d[i] = 0;
    }
    EXPECT_TRUE(CopyDataFromTensorSliceToTensorSlice(shape, slice_s, slice_d,
                                                     ptr_s, ptr_d));
    for (int i = 0; i < 6; ++i) {
      EXPECT_EQ(ptr_s[i], ptr_d[i]);
    }
  }

  // Case 2: no intersection
  {
    TensorSlice slice_s = TensorSlice::ParseOrDie("1,2:1,3");
    TensorSlice slice_d = TensorSlice::ParseOrDie("3,1:2,3");
    const float ptr_s[] = {6, 7, 8, 11, 12, 13};
    float ptr_d[6];
    EXPECT_FALSE(CopyDataFromTensorSliceToTensorSlice(shape, slice_s, slice_d,
                                                      ptr_s, ptr_d));
  }

  // Case 3: a trickier case
  // The source slice is on the upper left corner:
  //   0   1   2   .   .
  //   5   6   7   .   .
  //  10  11  12   .   .
  //   .   .   .   .   .
  //
  // The destination slice is the right part of the middle stripe:
  //   .   .   .   .   .
  //   .   X   X   X   X
  //   .   X   X   X   X
  //   .   .   .   .   .
  //
  // So we expect to copy over the 2X2 block:
  //   .   .   .   .   .
  //   .   6   7   .   .
  //   .  11  12   .   .
  //   .   .   .   .   .
  {
    TensorSlice slice_s = TensorSlice::ParseOrDie("0,3:0,3");
    TensorSlice slice_d = TensorSlice::ParseOrDie("1,2:1,4");
    const float ptr_s[] = {0, 1, 2, 5, 6, 7, 10, 11, 12};
    float ptr_d[8];
    for (int i = 0; i < 8; ++i) {
      ptr_d[i] = 0;
    }
    EXPECT_TRUE(CopyDataFromTensorSliceToTensorSlice(shape, slice_s, slice_d,
                                                     ptr_s, ptr_d));
    const float expected[] = {6, 7, 0, 0, 11, 12, 0, 0};
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(expected[i], ptr_d[i]);
    }
  }
}

}  // namespace
}  // namespace tensorflow
