// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/tools/android/test/jni/object_tracking/frame_pair.h"
#include "tensorflow/tools/android/test/jni/object_tracking/keypoint_detector.h"
#include "tensorflow/tools/android/test/jni/object_tracking/keypoint.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include <cstring>

namespace tf_tracking {

// Tests that WeightedDeltaCompare returns 0 when deltas are equal.
TEST(WeightedDeltaCompareTest, ReturnsZeroWhenEqual) {
    WeightedDelta a{1.0f, 5.0f};
    WeightedDelta b{2.0f, 5.0f};
    int cmp = tf_tracking::WeightedDeltaCompare(&a, &b);
    EXPECT_EQ(cmp, 0) << "Comparison should return 0 for equal deltas";
}

// Tests that WeightedDeltaCompare returns positive when a < b.
TEST(WeightedDeltaCompareTest, ReturnsPositiveWhenALessThanB) {
    WeightedDelta a{1.0f, 3.0f};
    WeightedDelta b{2.0f, 5.0f};
    int cmp = tf_tracking::WeightedDeltaCompare(&a, &b);
    EXPECT_GT(cmp, 0) << "Comparison should return positive when a < b";
}

// Tests that WeightedDeltaCompare returns negative when a > b.
TEST(WeightedDeltaCompareTest, ReturnsNegativeWhenAGreaterThanB) {
    WeightedDelta a{1.0f, 7.0f};
    WeightedDelta b{2.0f, 5.0f};
    int cmp = tf_tracking::WeightedDeltaCompare(&a, &b);
    EXPECT_LT(cmp, 0) << "Comparison should return negative when a > b";
}

// Tests that qsort with WeightedDeltaCompare produces stable ordering.
TEST(WeightedDeltaCompareTest, QSortStableOrdering) {
    WeightedDelta arr[3] = {
        {1.0f, 1.0f},
        {2.0f, 2.0f},
        {3.0f, 2.0f}
    };
    qsort(arr, 3, sizeof(WeightedDelta), tf_tracking::WeightedDeltaCompare);
    EXPECT_FLOAT_EQ(arr[0].delta, 2.0f);
    EXPECT_FLOAT_EQ(arr[1].delta, 2.0f);
    EXPECT_FLOAT_EQ(arr[2].delta, 1.0f);
}

// Tests that KeypointCompare returns 0 when scores are equal.
TEST(KeypointCompareTest, ReturnsZeroWhenEqual) {
    Keypoint a(0, 0);
    a.score_ = 4.2f;
    a.type_ = 0;
    Keypoint b(1, 1);
    b.score_ = 4.2f;
    b.type_ = 1;
    int cmp = KeypointCompare(&a, &b);
    EXPECT_EQ(cmp, 0);
}

// Tests that KeypointCompare returns positive when a < b.
TEST(KeypointCompareTest, ReturnsPositiveWhenALessThanB) {
    Keypoint a(0, 0);
    a.score_ = 3.0f;
    a.type_ = 0;
    Keypoint b(1, 1);
    b.score_ = 5.0f;
    b.type_ = 1;
    int cmp = KeypointCompare(&a, &b);
    EXPECT_GT(cmp, 0);
}

// Tests that KeypointCompare returns negative when a > b.
TEST(KeypointCompareTest, ReturnsNegativeWhenAGreaterThanB) {
    Keypoint a(0, 0);
    a.score_ = 5.0f;
    a.type_ = 0;
    Keypoint b(1, 1);
    b.score_ = 3.0f;
    b.type_ = 1;
    int cmp = KeypointCompare(&a, &b);
    EXPECT_LT(cmp, 0);
}

// Tests that qsort with KeypointCompare produces stable ordering.
TEST(KeypointCompareTest, QSortStableOrdering) {
    Keypoint arr[3] = {
        Keypoint(0, 0),
        Keypoint(1, 1),
        Keypoint(2, 2)
    };
    arr[0].score_ = 1.0f; arr[0].type_ = 0;
    arr[1].score_ = 2.0f; arr[1].type_ = 1;
    arr[2].score_ = 2.0f; arr[2].type_ = 2;

    qsort(arr, 3, sizeof(Keypoint), KeypointCompare);
    EXPECT_FLOAT_EQ(arr[0].score_, 2.0f);
    EXPECT_FLOAT_EQ(arr[1].score_, 2.0f);
    EXPECT_FLOAT_EQ(arr[2].score_, 1.0f);
}

}   // namespace tf_tracking
