/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

// std::bit_cast is only available since c++20, provide a handrolled version
// of bit_cast here. Implementation modified from abseil bit_cast.
template <
    typename Dest, typename Source,
    typename std::enable_if<sizeof(Dest) == sizeof(Source) &&
                                std::is_trivially_copyable<Source>::value &&
                                std::is_trivially_copyable<Dest>::value &&
                                std::is_default_constructible<Dest>::value,
                            int>::type = 0>
inline Dest bit_cast(const Source& source) {
  Dest dest;
  memcpy(static_cast<void*>(std::addressof(dest)),
         static_cast<const void*>(std::addressof(source)), sizeof(dest));
  return dest;
}

class BitcastOpModel : public SingleOpModel {
 public:
  BitcastOpModel(const TensorData& input, const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_BITCAST, BuiltinOptions_BitcastOptions,
                 CreateBitcastOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  int input() const { return input_; }
  int output() const { return output_; }

 protected:
  int input_;
  int output_;
};

TEST(BitcastOpModel, BitastInt32ToUint32) {
  BitcastOpModel m({TensorType_INT32, {2, 3}}, {TensorType_UINT32, {2, 3}});
  std::vector<int32_t> input = {INT32_MIN, -100, -1, 0, 100, INT32_MAX};
  m.PopulateTensor<int32_t>(m.input(), input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  std::vector<uint32_t> output;

  std::transform(input.cbegin(), input.cend(), std::back_inserter(output),
                 [](int32_t a) { return bit_cast<std::uint32_t>(a); });
  EXPECT_THAT(m.ExtractVector<uint32_t>(m.output()), ElementsAreArray(output));
}

TEST(BitcastOpModel, BitastUInt32Toint32) {
  BitcastOpModel m({TensorType_UINT32, {2, 3}}, {TensorType_INT32, {2, 3}});
  std::vector<uint32_t> input = {0,
                                 1,
                                 100,
                                 bit_cast<uint32_t>(INT32_MAX),
                                 bit_cast<uint32_t>(INT32_MIN),
                                 UINT32_MAX};
  m.PopulateTensor<uint32_t>(m.input(), input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  std::vector<int32_t> output;

  std::transform(input.cbegin(), input.cend(), std::back_inserter(output),
                 [](uint32_t a) { return bit_cast<std::uint32_t>(a); });
  EXPECT_THAT(m.ExtractVector<int32_t>(m.output()), ElementsAreArray(output));
}

TEST(BitcastOpModel, BitcastUInt32Toint16) {
  BitcastOpModel m({TensorType_UINT32, {2, 1}}, {TensorType_INT16, {2, 1, 2}});
  std::vector<uint32_t> input = {(uint32_t)UINT16_MAX + 1,
                                 (uint32_t)UINT16_MAX};
  m.PopulateTensor<uint32_t>(m.input(), input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  // 00..01 00..00
  // 00..00 11..11
  std::vector<int16_t> output = {1, 0, 0, -1};
#else
  // 00..00 00..01
  // 11..11 00..00
  std::vector<int16_t> output = {0, 1, -1, 0};
#endif
  EXPECT_THAT(m.ExtractVector<int16_t>(m.output()), ElementsAreArray(output));
}

TEST(BitcastOpModel, BitcastInt16ToUint32) {
  BitcastOpModel m({TensorType_INT16, {2, 1, 2}}, {TensorType_UINT32, {2, 1}});
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  std::vector<int16_t> input = {1, 0, 0, -1};
#else
  std::vector<int16_t> input = {0, 1, -1, 0};
#endif
  m.PopulateTensor<int16_t>(m.input(), input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  std::vector<uint32_t> output = {(uint32_t)UINT16_MAX + 1,
                                  (uint32_t)UINT16_MAX};
  EXPECT_THAT(m.ExtractVector<uint32_t>(m.output()), ElementsAreArray(output));
}

TEST(BitcastOpModel, BitcastInt16ToUint32WrongShape) {
#if GTEST_HAS_DEATH_TEST
  EXPECT_DEATH(BitcastOpModel m({TensorType_INT16, {2, 2, 7}},
                                {TensorType_UINT32, {2, 7}}),
               "7 != 2");
#endif
}

}  // namespace
}  // namespace tflite
