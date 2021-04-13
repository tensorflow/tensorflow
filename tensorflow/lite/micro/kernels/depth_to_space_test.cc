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
#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

#ifdef notdef
class DepthToSpaceOpModel : public SingleOpModel {
 public:
  DepthToSpaceOpModel(const TensorData& tensor_data, int block_size) {}
};
#endif  // notdef

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(DepthToSpaceOpModelFloat32_1114_2) {
#ifdef notdef
  DepthToSpaceOpModel m({TensorType_FLOAT32, {1, 1, 1, 4}}, 2);
  m.SetInput<float>({1.4, 2.3, 3.2, 4.1});
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({1.4, 2.3, 3.2, 4.1}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 2, 2, 1));
#endif  // notdef
}

TF_LITE_MICRO_TEST(DepthToSpaceOpModelFloat32_1124_2) {
#ifdef notdef
  DepthToSpaceOpModel m({TensorType_UINT8, {1, 1, 2, 4}}, 2);
  m.SetInput<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8});
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({1, 2, 5, 6, 3, 4, 7, 8}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 2, 4, 1));
#endif  // notdef
}

TF_LITE_MICRO_TEST(DepthToSpaceOpModelFloat32_1214_2) {
#ifdef notdef
  DepthToSpaceOpModel m({TensorType_INT8, {1, 2, 1, 4}}, 2);
  m.SetInput<int8_t>({1, 2, 3, 4, 5, 6, 7, 8});
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 4, 2, 1));
#endif  // notdef
}

TF_LITE_MICRO_TEST(DepthToSpaceOpModelFloat32_1224_2) {
#ifdef notdef
  DepthToSpaceOpModel m({TensorType_INT32, {1, 2, 2, 4}}, 2);
  m.SetInput<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  EXPECT_THAT(m.GetOutput<int32_t>(),
              ElementsAreArray(
                  {1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 4, 4, 1));
#endif  // notdef
}

TF_LITE_MICRO_TEST(DepthToSpaceOpModelFloat32_1111_1) {
#ifdef notdef
  DepthToSpaceOpModel m({TensorType_INT64, {1, 1, 1, 1}}, 1);
  m.SetInput<int64_t>({4});
  EXPECT_THAT(m.GetOutput<int64_t>(), ElementsAreArray({4}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 1, 1, 1));
#endif  // notdef
}

TF_LITE_MICRO_TEST(DepthToSpaceOpModelInt8_1114_2) {
#ifdef notdef
  DepthToSpaceOpModel m({TensorType_FLOAT32, {1, 1, 1, 4}}, 2);
  m.SetInput<float>({1.4, 2.3, 3.2, 4.1});
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({1.4, 2.3, 3.2, 4.1}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 2, 2, 1));
#endif  // notdef
}

TF_LITE_MICRO_TEST(DepthToSpaceOpModelInt8_1124_2) {
#ifdef notdef
  DepthToSpaceOpModel m({TensorType_UINT8, {1, 1, 2, 4}}, 2);
  m.SetInput<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8});
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({1, 2, 5, 6, 3, 4, 7, 8}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 2, 4, 1));
#endif  // notdef
}

TF_LITE_MICRO_TEST(DepthToSpaceOpModelInt8_1214_2) {
#ifdef notdef
  DepthToSpaceOpModel m({TensorType_INT8, {1, 2, 1, 4}}, 2);
  m.SetInput<int8_t>({1, 2, 3, 4, 5, 6, 7, 8});
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 4, 2, 1));
#endif  // notdef
}

TF_LITE_MICRO_TEST(DepthToSpaceOpModelInt8_1224_2) {
#ifdef notdef
  DepthToSpaceOpModel m({TensorType_INT32, {1, 2, 2, 4}}, 2);
  m.SetInput<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  EXPECT_THAT(m.GetOutput<int32_t>(),
              ElementsAreArray(
                  {1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 4, 4, 1));
#endif  // notdef
}

TF_LITE_MICRO_TEST(DepthToSpaceOpModelInt8_1111_1) {
#ifdef notdef
  DepthToSpaceOpModel m({TensorType_INT64, {1, 1, 1, 1}}, 1);
  m.SetInput<int64_t>({4});
  EXPECT_THAT(m.GetOutput<int64_t>(), ElementsAreArray({4}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 1, 1, 1));
#endif  // notdef
}

TF_LITE_MICRO_TESTS_END
