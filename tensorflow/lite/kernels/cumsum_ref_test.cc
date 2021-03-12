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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <vector>

#include "tensorflow/lite/kernels/internal/reference/cumsum.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace ops {
namespace builtin {

namespace {

template <typename T>
class CumsumOpModel {
 public:
  CumsumOpModel(const TensorData& input, const TensorData& output,
                bool exclusive, bool reverse)
      : exclusive_(exclusive), reverse_(reverse) {
    input_shape_.assign(input.shape.begin(), input.shape.end());
    output_.resize(GetInputShape().FlatSize());
  }

  std::vector<T>& GetOutput() { return output_; }
  std::vector<int32_t>* GetAxis() { return &axis_; }
  std::vector<T>* GetInput() { return &input_; }

  const RuntimeShape GetInputShape() {
    return RuntimeShape(input_shape_.size(), input_shape_.data());
  }

  void ShowTensor(int index, char* name) {}

  template <typename Ttype>
  void PopulateTensor(std::vector<Ttype>* v,
                      const std::initializer_list<Ttype>& il) {
    v->assign(il);
  }

  void Invoke() {
    tflite::reference_ops::Cumsum(GetInput()->data(), GetInputShape(),
                                  GetAxis()->at(0), exclusive(), reverse(),
                                  GetOutput().data());
  }

  bool reverse() { return reverse_; }
  bool exclusive() { return exclusive_; }

 private:
  std::vector<T> input_;
  std::vector<int32_t> input_shape_;
  std::vector<int32_t> axis_;
  std::vector<T> output_;
  bool reverse_;
  bool exclusive_;
};

TEST(CumsumOpTest, SimpleIntTest) {
  CumsumOpModel<int32_t> m({TensorType_INT32, {2, 4}}, {TensorType_INT32, {}},
                           false, false);

  m.PopulateTensor<int32_t>(m.GetInput(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int32_t>(m.GetAxis(), {1});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({1, 3, 6, 10, 5, 11, 18, 26}));
}

TEST(CumsumOpTest, SimpleInt64Test) {
  CumsumOpModel<int64_t> m({TensorType_INT64, {2, 4}}, {TensorType_INT64, {}},
                           false, false);

  m.PopulateTensor<int64_t>(
      m.GetInput(),
      {100000000001l, 100000000002l, 100000000003l, 100000000004l,
       100000000005l, 100000000006l, 100000000007l, 100000000008l});
  m.PopulateTensor<int32_t>(m.GetAxis(), {1});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), testing::ElementsAreArray(
                                 {100000000001l, 200000000003l, 300000000006l,
                                  400000000010l, 100000000005l, 200000000011l,
                                  300000000018l, 400000000026l}));
}

TEST(CumsumOpTest, SimpleIntAxis0Test) {
  CumsumOpModel<int32_t> m({TensorType_INT32, {2, 4}}, {TensorType_INT32, {}},
                           false, false);

  m.PopulateTensor<int32_t>(m.GetInput(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int32_t>(m.GetAxis(), {0});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({1, 2, 3, 4, 6, 8, 10, 12}));
}

TEST(CumsumOpTest, Simple1DIntTest) {
  CumsumOpModel<int32_t> m({TensorType_INT32, {8}}, {TensorType_INT32, {}},
                           false, false);

  m.PopulateTensor<int32_t>(m.GetInput(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int32_t>(m.GetAxis(), {0});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({1, 3, 6, 10, 15, 21, 28, 36}));
}

TEST(CumsumOpTest, SimpleIntReverseTest) {
  CumsumOpModel<int32_t> m({TensorType_INT32, {2, 4}}, {TensorType_INT32, {}},
                           false, true);

  m.PopulateTensor<int32_t>(m.GetInput(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int32_t>(m.GetAxis(), {1});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({10, 9, 7, 4, 26, 21, 15, 8}));
}

TEST(CumsumOpTest, SimpleIntExclusiveTest) {
  CumsumOpModel<int32_t> m({TensorType_INT32, {2, 4}}, {TensorType_INT32, {}},
                           true, false);

  m.PopulateTensor<int32_t>(m.GetInput(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int32_t>(m.GetAxis(), {1});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({0, 1, 3, 6, 0, 5, 11, 18}));
}

TEST(CumsumOpTest, SimpleFloatTest) {
  CumsumOpModel<float> m({TensorType_FLOAT32, {2, 4}}, {TensorType_FLOAT32, {}},
                         false, false);

  m.PopulateTensor<float>(m.GetInput(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int32_t>(m.GetAxis(), {1});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), testing::ElementsAreArray(
                                 ArrayFloatNear({1, 3, 6, 10, 5, 11, 18, 26})));
}

}  // namespace
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
