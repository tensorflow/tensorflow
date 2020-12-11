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

namespace {

TEST_P(FillOpTest, FillFloat) {
  FillOpModel<int64_t, float> m(TensorType_INT64, {3}, {2, 2, 2}, 4.0,
                                GetParam());
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2}));
}

TEST_P(FillOpTest, FillFloatInt32Dims) {
  FillOpModel<int32_t, float> m(TensorType_INT32, {3}, {2, 2, 2}, 4.0,
                                GetParam());
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2}));
}

TEST_P(FillOpTest, FillOutputScalar) {
  FillOpModel<int64_t, float> m(TensorType_INT64, {0}, {}, 4.0, GetParam());
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({4.0}));
  EXPECT_THAT(m.GetOutputShape(), IsEmpty());
}

}  // namespace
