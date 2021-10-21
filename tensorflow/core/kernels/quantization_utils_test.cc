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

#include <cmath>
#define EIGEN_USE_THREADS

#include <limits>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

void TestRequantizeMany(Eigen::ThreadPoolDevice* eigen_device, float input_min,
                        float input_max, float output_min, float output_max,
                        const std::vector<qint32>& values_quantized,
                        int tolerance = 1) {
  const int values_count = values_quantized.size();
  std::vector<quint8> expected_values;
  expected_values.reserve(values_count);
  for (int value_index = 0; value_index < values_count; ++value_index) {
    expected_values.push_back(FloatToQuantized<quint8>(
        QuantizedToFloat(values_quantized[value_index], input_min, input_max),
        output_min, output_max));
  }

  Tensor i_tensor =
      tensorflow::test::AsTensor(gtl::ArraySlice<qint32>(values_quantized));
  Tensor o_tensor(DT_QUINT8, TensorShape{values_count});
  auto output_values = o_tensor.flat<quint8>();

  if (eigen_device == nullptr) {
    auto input_array = i_tensor.flat<qint32>();
    RequantizeManyInNewRange(input_array.data(), input_array.size(), input_min,
                             input_max, output_min, output_max,
                             output_values.data());
  } else {
    RequantizeManyInNewRangeUsingEigen<qint32, quint8>(
        *eigen_device, i_tensor, input_min, input_max, output_min, output_max,
        &o_tensor);
  }

  const string tolerance_str = strings::StrCat("+-", tolerance);
  for (size_t value_index = 0; value_index < values_count; ++value_index) {
    int e = expected_values[value_index];
    int v = output_values(value_index);
    ASSERT_TRUE(std::abs(e - v) <= tolerance)
        << "actual=" << v << ", expected=" << e << tolerance_str
        << ", values_quantized[" << value_index
        << "]=" << values_quantized[value_index] << ", input_min=" << input_min
        << ", input_max=" << input_max << ", output_min=" << output_min
        << ", output_max=" << output_max << ", value_index=" << value_index;
  }
}

void TestRequantizeMany8To32Bit(float input_min, float input_max,
                                float output_min, float output_max,
                                const std::vector<quint8>& values_quantized,
                                int tolerance = 256) {
  const int values_count = values_quantized.size();
  std::vector<qint32> expected_values;
  expected_values.reserve(values_count);
  for (int value_index = 0; value_index < values_count; ++value_index) {
    expected_values.push_back(FloatToQuantized<qint32>(
        QuantizedToFloat(values_quantized[value_index], input_min, input_max),
        output_min, output_max));
  }

  const Tensor i_tensor =
      tensorflow::test::AsTensor(gtl::ArraySlice<quint8>(values_quantized));
  Tensor o_tensor(DT_QINT32, TensorShape{values_count});
  auto output_values = o_tensor.flat<qint32>();

  const auto input_array = i_tensor.flat<quint8>();
  RequantizeManyInNewRange(input_array.data(), input_array.size(), input_min,
                           input_max, output_min, output_max,
                           output_values.data());

  const string tolerance_str = strings::StrCat("+-", tolerance);
  for (int value_index = 0; value_index < values_count; ++value_index) {
    const qint32 e = expected_values[value_index];
    const qint32 v = output_values(value_index);
    ASSERT_TRUE(std::abs(e - v) <= tolerance)
        << "actual=" << v << ", expected=" << e << tolerance_str
        << ", values_quantized[" << value_index
        << "]=" << values_quantized[value_index] << ", input_min=" << input_min
        << ", input_max=" << input_max << ", output_min=" << output_min
        << ", output_max=" << output_max << ", value_index=" << value_index;
  }
}

// If eigen_device is NULL, then the reference implementation is tested.
void TestRequantizeManyInNewRange32To8Bit(
    Eigen::ThreadPoolDevice* eigen_device) {
  if (true) {
    // These are the float values we're going to test the conversions on.
    const size_t values_count = 6;
    const float values[values_count] = {0.0f,  0.45f,  1.0f,
                                        -1.0f, 127.0f, 255.0f};
    // These are the input and output ranges we'll test.
    const size_t ranges_count = 6;
    const float ranges[ranges_count][4] = {
        {0.0f, 255.0f, 0.0f, 255.0f},    //
        {0.0f, 1.0f, 0.0f, 1.0f},        //
        {-1.0f, 1.0f, -1.0f, 1.0f},      //
        {-1.0f, 1.0f, -255.0f, 255.0f},  //
        {3.0f, 3.0f, 0.0f, 255.0f},      // input min == max
        {0.0f, 255.0f, 5.0f, 5.0f},      // output min == max
    };
    for (int i = 0; i < ranges_count; ++i) {
      const auto& r = ranges[i];
      std::vector<qint32> values_quantized;
      for (int value_index = 0; value_index < values_count; ++value_index) {
        const float v = values[value_index];
        values_quantized.push_back(FloatToQuantized<qint32>(v, r[0], r[1]));
      }
      TestRequantizeMany(eigen_device, r[0], r[1], r[2], r[3],
                         values_quantized);
    }

    // Test with many different values in the input quantized range.
    qint32 low = Eigen::NumTraits<qint32>::lowest();
    qint32 high = Eigen::NumTraits<qint32>::highest();
    std::vector<qint32> vals{low, high};
    int num_steps = 14419;
    qint32 step = static_cast<int32>((1LL << 32) / num_steps);
    qint32 v = low + static_cast<qint32>(1);
    for (int i = 0; i < num_steps; ++i) {
      vals.push_back(v);
      v += step;
    }
    TestRequantizeMany(eigen_device, -1.0f, 1.0f, -1.0f, 1.0f, vals);
    TestRequantizeMany(eigen_device, -255.0f, 255.0f, -255.0f, 255.0f, vals);
    TestRequantizeMany(eigen_device, -1.0f, 1.0f, -12345678.0f, 12345678.0f,
                       vals);
    TestRequantizeMany(eigen_device, -1.0f, 12345678.0f, -12345678.0f,
                       12345678.0f, vals);
  }
  // Test when the input range is large and output range is small.
  // Use all quantized values where the float is in the output range.
  const float out_min = -29.1234;
  const float out_max = 23.1234;
  const float in_min = -1e6;
  const float in_max = 1e6;

  qint32 low = FloatToQuantized<qint32>(out_min, in_min, in_max);
  qint32 high = FloatToQuantized<qint32>(out_max, in_min, in_max);
  std::vector<qint32> vals;
  vals.clear();
  for (int32_t i = low; i <= high; ++i) vals.push_back(i);
  TestRequantizeMany(eigen_device, in_min, in_max, out_min, out_max, vals);
}

void TestRequantizeManyInNewRange8To32Bit() {
  // These are the float values we're going to test the conversions on.
  const size_t values_count = 6;
  const float values[values_count] = {0.0f, 0.45f, 1.0f, -1.0f, 127.0f, 255.0f};
  // These are the input and output ranges we'll test.
  const size_t ranges_count = 6;
  const float ranges[ranges_count][4] = {
      {0.0f, 255.0f, 0.0f, 255.0f},    //
      {0.0f, 1.0f, 0.0f, 1.0f},        //
      {-1.0f, 1.0f, -1.0f, 1.0f},      //
      {-1.0f, 1.0f, -255.0f, 255.0f},  //
      {3.0f, 3.0f, 0.0f, 255.0f},      // input min == max
      {0.0f, 255.0f, 5.0f, 5.0f},      // output min == max
  };
  for (int i = 0; i < ranges_count; ++i) {
    const auto& r = ranges[i];
    std::vector<quint8> values_quantized;
    for (int value_index = 0; value_index < values_count; ++value_index) {
      const float v = values[value_index];
      values_quantized.push_back(FloatToQuantized<quint8>(v, r[0], r[1]));
    }
    TestRequantizeMany8To32Bit(r[0], r[1], r[2], r[3], values_quantized);
  }

  // Test with many different values in the input quantized range.
  int low = Eigen::NumTraits<quint8>::lowest();
  int high = Eigen::NumTraits<quint8>::highest();
  std::vector<quint8> vals;
  for (int val = low; val <= high; ++val) {
    vals.push_back(val);
  }
  TestRequantizeMany8To32Bit(-1.0f, 1.0f, -1.0f, 1.0f, vals);
  TestRequantizeMany8To32Bit(-255.0f, 255.0f, -255.0f, 255.0f, vals);
  TestRequantizeMany8To32Bit(-1.0f, 1.0f, -12345678.0f, 12345678.0f, vals);
  TestRequantizeMany8To32Bit(-1.0f, 12345678.0f, -12345678.0f, 12345678.0f,
                             vals);
}

template <typename InputType, typename OutputType>
void TestRequantizeManyInNewRangeEigenVsNonEigen() {
  thread::ThreadPool threadpool(Env::Default(), "test", 2 /* num_threads */);
  Eigen::ThreadPoolDevice eigen_device(threadpool.AsEigenThreadPool(),
                                       2 /* num_threads */);

  const size_t ranges_count = 6;
  const float ranges[ranges_count][4] = {
      {0.0f, 255.0f, 0.0f, 255.0f},    //
      {0.0f, 1.0f, 0.0f, 1.0f},        //
      {-1.0f, 1.0f, -1.0f, 1.0f},      //
      {-1.0f, 1.0f, -255.0f, 255.0f},  //
      {3.0f, 3.0f, 0.0f, 255.0f},      // input min == max
      {0.0f, 255.0f, 5.0f, 5.0f},      // output min == max
  };

  // Random values.
  for (size_t range_index = 0; range_index < ranges_count; ++range_index) {
    const float input_min = ranges[range_index][0];
    const float input_max = ranges[range_index][1];
    const float output_min = ranges[range_index][2];
    const float output_max = ranges[range_index][3];
    const int values_count = 10000;
    random::PhiloxRandom philox(testing::RandomSeed(), 17);
    random::SimplePhilox rnd(&philox);
    std::vector<InputType> values_quantized;
    for (int i = 0; i < values_count; ++i) {
      float v = (rnd.RandFloat() * (input_max - input_min)) + input_min;
      values_quantized.push_back(
          FloatToQuantized<InputType>(v, input_min, input_max));
    }

    Tensor i_tensor = tensorflow::test::AsTensor(
        gtl::ArraySlice<InputType>(values_quantized));
    const auto i_array = i_tensor.flat<InputType>();
    Tensor o_tensor_eigen(DataTypeToEnum<OutputType>::v(),
                          TensorShape{values_count});
    auto output_values_eigen = o_tensor_eigen.flat<OutputType>();
    Tensor o_tensor_ref(DataTypeToEnum<OutputType>::v(),
                        TensorShape{values_count});
    auto output_values_ref = o_tensor_ref.flat<OutputType>();

    RequantizeManyInNewRange(i_array.data(), i_array.size(), input_min,
                             input_max, output_min, output_max,
                             output_values_ref.data());
    RequantizeManyInNewRangeUsingEigen<InputType, OutputType>(
        eigen_device, i_tensor, input_min, input_max, output_min, output_max,
        &o_tensor_eigen);

    const int tolerance = 1;
    for (int i = 0; i < values_quantized.size(); ++i) {
      auto expected = output_values_ref(i);
      auto actual = output_values_eigen(i);
      // The eigen computation uses float for constants and computation
      // instead of doubles, so can be different by 1 or 2 in some cases
      // (e.g., input value 144.062744140625, min -1, max 255, type quint8).
      ASSERT_TRUE(std::abs(expected - actual) <= tolerance)
          << "expected=" << expected << " actual=" << actual
          << " tolerance=" << tolerance << " v=" << values_quantized[i]
          << " i=" << i << " input_min=" << input_min
          << " input_max=" << input_max
          << " input_type=" << DataTypeString(DataTypeToEnum<InputType>::v())
          << " output_type=" << DataTypeString(DataTypeToEnum<OutputType>::v());
    }
  }
}

template <typename InputType, typename OutputType>
void TimeRequantizeManyInNewRange(int64_t num_elements, int64_t iterations,
                                  bool use_eigen) {
  const float input_min = -100.0f;
  const float input_max = 100.0f;
  const float output_min = -1000000.0f;
  const float output_max = 1000000.0f;

  random::PhiloxRandom philox(testing::RandomSeed(), 17);
  random::SimplePhilox rnd(&philox);
  std::vector<InputType> values_quantized;
  for (int i = 0; i < num_elements; ++i) {
    float v = (rnd.RandFloat() * (input_max - input_min)) + input_min;
    values_quantized.push_back(
        FloatToQuantized<InputType>(v, input_min, input_max));
  }

  thread::ThreadPool threadpool(Env::Default(), "test", 4 /* num_threads */);
  Eigen::ThreadPoolDevice eigen_device(threadpool.AsEigenThreadPool(),
                                       4 /* num_threads */);

  Tensor i_tensor =
      tensorflow::test::AsTensor(gtl::ArraySlice<InputType>(values_quantized));
  const auto i_array = i_tensor.flat<InputType>();
  Tensor o_tensor_eigen(DataTypeToEnum<OutputType>::v(),
                        TensorShape{num_elements});
  Tensor o_tensor_ref(DataTypeToEnum<OutputType>::v(),
                      TensorShape{num_elements});
  auto output_values_ref = o_tensor_ref.flat<OutputType>();

  int64_t total_duration = 0;
  for (int i = 0; i < iterations; ++i) {
    const int64_t start_time = Env::Default()->NowMicros();
    if (use_eigen) {
      RequantizeManyInNewRangeUsingEigen<InputType, OutputType>(
          eigen_device, i_tensor, input_min, input_max, output_min, output_max,
          &o_tensor_eigen);
    } else {
      RequantizeManyInNewRange<InputType, OutputType>(
          i_array.data(), i_array.size(), input_min, input_max, output_min,
          output_max, output_values_ref.data());
    }
    const int64_t end_time = Env::Default()->NowMicros();
    total_duration += end_time - start_time;
  }
  const int64_t one_run_duration = total_duration / iterations;

  const int64_t num_ops = num_elements;

  const double million_ops_per_second =
      (iterations * num_ops) / static_cast<double>(total_duration);

  LOG(INFO) << "TimeRequantizeManyInNewRange: " << num_elements
            << (use_eigen ? " eigen" : " ref") << ": iterations=" << iterations
            << ", MOps/s=" << million_ops_per_second
            << ", one_run_duration=" << one_run_duration
            << ", total_duration=" << total_duration;
}

template <typename T>
void TestFloatToQuantizedInPlaceUsingEigen(
    Eigen::ThreadPoolDevice* eigen_device) {
  // These are the float values we're going to test the conversions on.
  typedef std::pair<float, float> FPair;
  for (FPair min_and_max : std::vector<FPair>{FPair(-255.0f, 255.0f),  //
                                              FPair(-1.0f, 1.0f),      //
                                              FPair(-1.0f, 255.0f),    //
                                              FPair(0.0f, 1e6),        //
                                              FPair(0.0f, 1.0f),       //
                                              FPair(-31.0f, 13.0f)}) {
    const float f_min = min_and_max.first;
    const float f_max = min_and_max.second;
    const float f_range = f_max - f_min;
    const int values_count = 50000;
    Tensor input(DT_FLOAT, TensorShape{values_count});
    auto input_array = input.flat<float>();
    for (int i = 0; i < values_count; ++i) {
      input_array(i) = f_min + f_range * i / (values_count - 1);
    }

    Tensor output(DataTypeToEnum<T>::v(), TensorShape{values_count});
    FloatTensorToQuantizedInPlaceUsingEigen<T>(*eigen_device, input, f_min,
                                               f_max, &output);
    auto output_array = output.flat<T>();

    const int tolerance = 1;
    for (int i = 0; i < values_count; ++i) {
      int32_t expected = FloatToQuantized<T>(input_array(i), f_min, f_max);
      int32_t actual = output_array(i);

      // The eigen computation uses float for constants and computation
      // instead
      // of doubles, so can be different by 1 or 2 in some cases (e.g., input
      // value 144.062744140625, min -1, max 255, type quint8).
      ASSERT_TRUE(std::abs(expected - actual) <= tolerance)
          << "expected=" << expected << " actual=" << actual
          << " tolerance=" << tolerance << " v=" << input_array(i) << " i=" << i
          << " f_min=" << f_min << " f_max=" << f_max
          << " type=" << DataTypeString(DataTypeToEnum<T>::v());
    }
  }
}

template <typename T>
void TestQuantizedToFloatInPlaceUsingEigen(
    Eigen::ThreadPoolDevice* eigen_device) {
  // These are the float values we're going to test the conversions on.
  typedef std::pair<float, float> FPair;
  for (FPair min_and_max : std::vector<FPair>{
           FPair(-255.0f, 255.0f),
           FPair(-1.0f, 1.0f),
           FPair(-1.0f, 255.0f),
           FPair(0.0f, 1e6),
           FPair(0.0f, 1.0f),
           FPair(-31.0f, 13.0f),
           FPair(-5.89505e+08, 5.89505e+08),
       }) {
    const float f_min = min_and_max.first;
    const float f_max = min_and_max.second;
    const int values_count = sizeof(T) == 1 ? 256 : 50000;
    Tensor input(DataTypeToEnum<T>::v(), TensorShape{values_count});
    auto input_array = input.flat<T>();
    const double q_range = static_cast<double>(Eigen::NumTraits<T>::highest()) -
                           Eigen::NumTraits<T>::lowest();
    for (int i = 0; i < values_count; ++i) {
      if (sizeof(T) == 1) {
        input_array(i) = Eigen::NumTraits<T>::lowest() + i;
      } else {
        int64_t offset = static_cast<int64_t>(q_range / values_count * i);
        input_array(i) = static_cast<int32>(
            std::min<int64_t>(Eigen::NumTraits<T>::lowest() + offset,
                              Eigen::NumTraits<T>::highest()));
      }
    }

    Tensor output(DT_FLOAT, TensorShape{values_count});
    QuantizedTensorToFloatInPlaceUsingEigen<T>(*eigen_device, input, f_min,
                                               f_max, &output);
    auto output_array = output.flat<float>();
    const double range = static_cast<double>(f_max) - f_min;
    for (int i = 0; i < values_count; ++i) {
      float expected = QuantizedToFloat<T>(input_array(i), f_min, f_max);
      float actual = output_array(i);
      ASSERT_NEAR(expected, actual, range * 1.1e-7)
          << "expected=" << expected << " actual=" << actual
          << " v=" << input_array(i) << " i=" << i << " f_min=" << f_min
          << " f_max=" << f_max
          << " type=" << DataTypeString(DataTypeToEnum<T>::v());
    }
  }
}

}  // namespace

void TestFloatToQuantized() {
  EXPECT_EQ(quint8(0), FloatToQuantized<quint8>(0.0f, 0.0f, 1.0f));
  EXPECT_EQ(quint8(0), FloatToQuantized<quint8>(0.0f, 0.0f, 2.0f));
  EXPECT_EQ(quint8(128), FloatToQuantized<quint8>(0.5f, 0.0f, 1.0f));
  EXPECT_EQ(quint8(128), FloatToQuantized<quint8>(1.0f, 0.0f, 2.0f));
  EXPECT_EQ(quint8(255), FloatToQuantized<quint8>(1.0f, 0.0f, 1.0f));
  EXPECT_EQ(quint8(255), FloatToQuantized<quint8>(2.0f, 0.0f, 2.0f));
  EXPECT_EQ(quint8(0), FloatToQuantized<quint8>(-128.0f, -128.0f, 127.0f));
  EXPECT_EQ(quint8(128), FloatToQuantized<quint8>(0.0f, -128.0f, 127.0f));
  EXPECT_EQ(quint8(255), FloatToQuantized<quint8>(127.0f, -128.0f, 127.0f));
  EXPECT_EQ(quint8(0), FloatToQuantized<quint8>(1.0f, 1.0f, 256.0f));
  EXPECT_EQ(quint8(127), FloatToQuantized<quint8>(128.0f, 1.0f, 256.0f));
  EXPECT_EQ(quint8(255), FloatToQuantized<quint8>(256.0f, 1.0f, 256.0f));

  const int int32_min = std::numeric_limits<int>::min();
  const int int32_max = std::numeric_limits<int>::max();

  EXPECT_EQ(qint32(int32_min),
            FloatToQuantized<qint32>(-128.0f, -128.0f, 128.0f));
  EXPECT_EQ(qint32(0), FloatToQuantized<qint32>(0.0f, -128.0f, 128.0f));
  EXPECT_EQ(qint32(int32_max),
            FloatToQuantized<qint32>(128.0f, -128.0f, 128.0f));
}

void TestQuantizedToFloat() {
  EXPECT_LT(fabsf(0.0f - QuantizedToFloat<quint8>(0, 0.0f, 1.0f)), 1 / 255.0f);
  EXPECT_LT(fabsf(0.0f - QuantizedToFloat<quint8>(0, 0.0f, 2.0f)), 1 / 255.0f);
  EXPECT_LT(fabsf(0.5f - QuantizedToFloat<quint8>(127, 0.0f, 1.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(1.0f - QuantizedToFloat<quint8>(127, 0.0f, 2.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(1.0f - QuantizedToFloat<quint8>(255, 0.0f, 1.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(2.0f - QuantizedToFloat<quint8>(255, 0.0f, 2.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(1.0f - QuantizedToFloat<quint8>(0, 1.0f, 256.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(128.0f - QuantizedToFloat<quint8>(127, 1.0f, 256.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(256.0f - QuantizedToFloat<quint8>(255, 1.0f, 256.0f)),
            1 / 255.0f);

  const int int32_min = std::numeric_limits<int>::min();
  const int int32_max = std::numeric_limits<int>::max();

  EXPECT_NEAR(-1.0f, QuantizedToFloat<qint32>(qint32(int32_min), -1.0f, 1.0f),
              1e-5f);
  EXPECT_NEAR(0.0f, QuantizedToFloat<qint32>(qint32(0), -1.0f, 1.0f), 1e-5f);
  EXPECT_NEAR(1.0f, QuantizedToFloat<qint32>(qint32(int32_max), -1.0f, 1.0f),
              1e-5f);

  EXPECT_NEAR(32.0f, QuantizedToFloat<qint32>(qint32(32), int32_min, int32_max),
              1.0);
}

void TestAvoidBias() {
  for (int i = 0; i < 256; ++i) {
    const float as_float = QuantizedToFloat<quint8>(i, 0.0f, 2.0f);
    const int back_to_int = FloatToQuantized<quint8>(as_float, 0.0f, 2.0f);
    EXPECT_EQ(i, back_to_int);
  }

  // All perfectly representable floats should survive quantization, even
  // if we pick a range where min is not itself perfectly representable.
  const float min = -0.1375f;
  const float max = 1.1385f;
  const float step_size = (max - min) / 255.0f;
  const float tolerance = step_size / 1000.0f;
  // This is the smallest perfectly representable float in the range.
  float first_float = std::ceil(min / step_size) * step_size;
  for (float f = first_float; f <= max; f += step_size) {
    const int as_int = FloatToQuantized<quint8>(f, min, max);
    const float back_to_float = QuantizedToFloat<quint8>(as_int, min, max);
    EXPECT_NEAR(f, back_to_float, tolerance);
  }
}

void TestRequantizeInNewRange() {
  // These are the float values we're going to test the conversions on.
  const size_t values_count = 6;
  const float values[values_count] = {0.0f, 0.5f, 1.0f, -1.0f, 127.0f, 255.0f};
  // These are the input and output ranges we'll test.
  const size_t ranges_count = 4;
  const float ranges[ranges_count][4] = {
      {0.0f, 255.0f, 0.0f, 255.0f},
      {0.0f, 1.0f, 0.0f, 1.0f},
      {-1.0f, 1.0f, -1.0f, 1.0f},
      {-1.0f, 1.0f, -255.0f, 255.0f},
  };
  for (size_t value_index = 0; value_index < values_count; ++value_index) {
    const float value_float = values[value_index];
    for (size_t range_index = 0; range_index < ranges_count; ++range_index) {
      const float input_min = ranges[range_index][0];
      const float input_max = ranges[range_index][1];
      const float output_min = ranges[range_index][2];
      const float output_max = ranges[range_index][3];
      const quint8 input_value =
          FloatToQuantized<quint8>(value_float, input_min, input_max);
      // Here we convert the quantized input value to what we expect
      // to get in the output range.
      const qint32 expected_value = FloatToQuantized<qint32>(
          QuantizedToFloat(input_value, input_min, input_max), output_min,
          output_max);
      EXPECT_EQ(expected_value,
                (RequantizeInNewRange<quint8, qint32>(
                    input_value, input_min, input_max, output_min, output_max)))
          << "value_float=" << value_float << ", input_min=" << input_min
          << ", input_max=" << input_max << ", output_min=" << output_min
          << ", output_max=" << output_max;
    }
  }
}

void TestRequantizeInNewRangeRealData() {
  const float input_min = -0.739539f;
  const float input_max = 0.641057f;
  const float output_min = -2381.49f;
  const float output_max = 2207.6f;

  // Start with a value that can be perfectly represented in 8 bits. This
  // ensures minimal quantization error, and allows us to use EXPECT_LT below.
  const float value_as_float =
      QuantizedToFloat<quint8>(83, input_min, input_max);

  const quint8 value_as_quint8 =
      FloatToQuantized<quint8>(value_as_float, input_min, input_max);
  EXPECT_EQ(quint8(83), value_as_quint8);
  const qint32 actual_output = RequantizeInNewRange<quint8, qint32>(
      value_as_quint8, input_min, input_max, output_min, output_max);
  const qint32 value_as_qint32 =
      FloatToQuantized<qint32>(value_as_float, output_min, output_max);
  EXPECT_LT(std::abs(value_as_qint32 - actual_output), 10);
}

void TestRequantizeInNewRange32To8Bit() {
  // These are the float values we're going to test the conversions on.
  const size_t values_count = 6;
  const float values[values_count] = {0.0f, 0.45f, 1.0f, -1.0f, 127.0f, 255.0f};
  // These are the input and output ranges we'll test.
  const size_t ranges_count = 4;
  const float ranges[ranges_count][4] = {
      {0.0f, 255.0f, 0.0f, 255.0f},
      {0.0f, 1.0f, 0.0f, 1.0f},
      {-1.0f, 1.0f, -1.0f, 1.0f},
      {-1.0f, 1.0f, -255.0f, 255.0f},
  };
  for (size_t value_index = 0; value_index < values_count; ++value_index) {
    const float value_float = values[value_index];
    for (size_t range_index = 0; range_index < ranges_count; ++range_index) {
      const float input_min = ranges[range_index][0];
      const float input_max = ranges[range_index][1];
      const float output_min = ranges[range_index][2];
      const float output_max = ranges[range_index][3];
      const qint32 input_value =
          FloatToQuantized<qint32>(value_float, input_min, input_max);
      // Here we convert the quantized input value to what we expect
      // to get in the output range.
      const quint8 expected_value = FloatToQuantized<quint8>(
          QuantizedToFloat(input_value, input_min, input_max), output_min,
          output_max);
      EXPECT_EQ(expected_value,
                (RequantizeInNewRange<qint32, quint8>(
                    input_value, input_min, input_max, output_min, output_max)))
          << "input_value=" << input_value << ", value_float=" << value_float
          << ", input_min=" << input_min << ", input_max=" << input_max
          << ", output_min=" << output_min << ", output_max=" << output_max;
    }
  }
}

void TestRequantizeManyInNewRange32To8Bit() {
  TestRequantizeManyInNewRange32To8Bit(nullptr /* eigen_device */);
}

void TestRequantizeManyInNewRange32To8BitUsingEigen() {
  thread::ThreadPool threadpool(Env::Default(), "test", 2 /* num_threads */);
  Eigen::ThreadPoolDevice eigen_device(threadpool.AsEigenThreadPool(),
                                       2 /* num_threads */);
  TestRequantizeManyInNewRange32To8Bit(&eigen_device);
}

void TestRequantizeManyInNewRange32To8BitEigenVsNonEigen() {
  TestRequantizeManyInNewRangeEigenVsNonEigen<qint32, quint8>();
}

void TestRequantizeManyInNewRange32To8BitSignedEigenVsNonEigen() {
  TestRequantizeManyInNewRangeEigenVsNonEigen<qint32, qint8>();
}

void TestFloatTensorToQuantized() {
  const int input_width = 3;
  const int input_height = 3;
  const float input_min = 0.0f;
  const float input_max = 255.0f;
  Tensor input(DT_FLOAT, TensorShape({input_height, input_width}));
  test::FillValues<float>(&input, {1.0f, -1.0f, 10.0f, 10.25f, 127.0f, 255.0f,
                                   512.0f, 0.0f, 23.0f});
  Tensor expected(DT_QUINT8, TensorShape({input_height, input_width}));
  test::FillValues<quint8>(&expected, {1, 0, 10, 10, 127, 255, 255, 0, 23});
  Tensor output = FloatTensorToQuantized<quint8>(input, input_min, input_max);
  test::ExpectTensorEqual<quint8>(expected, output);
}

// Verify that FloatToQuantizedInPlaceUsingEigen is same result as
// FloatToQuantized.
void TestFloatToQuantizedInPlaceUsingEigen() {
  thread::ThreadPool threadpool(Env::Default(), "test", 2 /* num_threads */);
  Eigen::ThreadPoolDevice eigen_device(threadpool.AsEigenThreadPool(),
                                       2 /* num_threads */);

  TestFloatToQuantizedInPlaceUsingEigen<quint8>(&eigen_device);
  TestFloatToQuantizedInPlaceUsingEigen<qint8>(&eigen_device);
  TestFloatToQuantizedInPlaceUsingEigen<quint16>(&eigen_device);
  TestFloatToQuantizedInPlaceUsingEigen<qint16>(&eigen_device);
}

void TestOverflowWithEigen() {
  thread::ThreadPool threadpool(Env::Default(), "test", 2 /* num_threads */);
  Eigen::ThreadPoolDevice eigen_device(threadpool.AsEigenThreadPool(),
                                       2 /* num_threads */);

  const int num_vals = 4;
  const float input_min = 0.0f;
  const float input_max = 2400.0f;
  TensorShape shape({num_vals});
  Tensor input(DT_FLOAT, shape);
  test::FillValues<float>(&input, {-100.f, 0.f, 2400.0f, 2400.0f});
  Tensor expected(DT_QINT32, shape);
  // Note that the positive expected values are not the highest int32 value,
  // because the implementation does a bounds check using float, not int32.
  test::FillValues<qint32>(
      &expected,
      {static_cast<int32>(-2147483648), static_cast<int32>(-2147483648),
       static_cast<int32>(2147483520), static_cast<int32>(2147483520)});

  FloatToQuantizedStruct<qint32> f2q(input_min, input_max);
  Tensor output(DT_QINT32, shape);
  auto input_array = input.flat<float>();
  output.flat<qint32>() = QUANTIZE_WITH_EIGEN(input_array, f2q, qint32);
  test::ExpectTensorEqual<qint32>(expected, output);
}

void TestQuantizedTensorToFloat() {
  const int input_width = 3;
  const int input_height = 3;
  const float input_min = -128.0f;
  const float input_max = 127.0f;
  Tensor input(DT_QUINT8, TensorShape({input_height, input_width}));
  test::FillValues<quint8>(&input, {0, 128, 255, 23, 24, 25, 243, 244, 245});
  Tensor expected(DT_FLOAT, TensorShape({input_height, input_width}));
  test::FillValues<float>(&expected, {-128.0f, 0.0f, 127.0f, -105.0f, -104.0f,
                                      -103.0f, 115.0f, 116.0f, 117.0f});
  Tensor output = QuantizedTensorToFloat<quint8>(input, input_min, input_max);
  test::ExpectTensorEqual<float>(expected, output);

  // Test for signed 32 bit.
  // Note that we cannot use input mins and maxes that match the range because
  // there are 7 too few bits of mantissa accuracy in floats to represent
  // 2**31-1 accurately.  Also there is no good fraction to use because 2**31-1
  // is a mersenne prime.
  Tensor input32(DT_QINT32, TensorShape({input_height, input_width}));

  // Use a quantizer centered at 0.
  float input_range = 1LL << 25;
  int64_t num_levels = (1LL << 32) - 1;
  float step_size =
      static_cast<float>(static_cast<double>(input_range) / num_levels);
  float q_compatible_min_value =
      roundf(-(input_range / 2.0) / step_size) * step_size;
  float q_compatible_max_value = q_compatible_min_value + input_range;
  test::FillValues<qint32>(&input32, {-16384, 0, 16256, -13440, -13312, -13184,
                                      14720, 14848, 14976});

  Tensor output32 = QuantizedTensorToFloat<qint32>(
      input32, q_compatible_min_value, q_compatible_max_value);
  test::FillValues<float>(&expected, {-128.0f, 0.0f, 127.0f, -105.0f, -104.0f,
                                      -103.0f, 115.0f, 116.0f, 117.0f});
  // The quantization error in going between 1<<25 and 1<<32 levels.
  const double kTolerance = .5 / 128.0;
  test::ExpectTensorNear<float>(expected, output32, kTolerance);
}

// Verify that QuantizedToFloatInPlaceUsingEigen is same result as
// QuantizedToFloat.
void TestQuantizedToFloatInPlaceUsingEigen() {
  thread::ThreadPool threadpool(Env::Default(), "test", 2 /* num_threads */);
  Eigen::ThreadPoolDevice eigen_device(threadpool.AsEigenThreadPool(),
                                       2 /* num_threads */);

  TestQuantizedToFloatInPlaceUsingEigen<quint8>(&eigen_device);
  TestQuantizedToFloatInPlaceUsingEigen<qint8>(&eigen_device);
  TestQuantizedToFloatInPlaceUsingEigen<quint16>(&eigen_device);
  TestQuantizedToFloatInPlaceUsingEigen<qint16>(&eigen_device);
  TestQuantizedToFloatInPlaceUsingEigen<qint32>(&eigen_device);
}

void BenchmarkRequantizeManyInNewRange() {
  TimeRequantizeManyInNewRange<qint32, quint8>(1000, 1000, false);
  TimeRequantizeManyInNewRange<qint32, quint8>(1000, 1000, true);
  TimeRequantizeManyInNewRange<qint32, quint8>(100000, 100, false);
  TimeRequantizeManyInNewRange<qint32, quint8>(100000, 100, true);
  TimeRequantizeManyInNewRange<qint32, quint8>(1000000, 10, false);
  TimeRequantizeManyInNewRange<qint32, quint8>(1000000, 10, true);

  TimeRequantizeManyInNewRange<quint8, qint32>(1000, 1000, false);
  TimeRequantizeManyInNewRange<quint8, qint32>(1000, 1000, true);
  TimeRequantizeManyInNewRange<quint8, qint32>(100000, 100, false);
  TimeRequantizeManyInNewRange<quint8, qint32>(100000, 100, true);
  TimeRequantizeManyInNewRange<quint8, qint32>(1000000, 10, false);
  TimeRequantizeManyInNewRange<quint8, qint32>(1000000, 10, true);
}

#ifdef QUANTIZATION_UTILS_USE_NEON
template <int POW>
void TestDivide64x2Pow(int64 val, int64 ref) {
  const int64x2_t val_64x2 = vmovq_n_s64(val);
  const int64x2_t ret = Divide64x2Pow<POW>(val_64x2);
  // TODO(b/70947959) Change back to int64 when possible
  int64_t rets[2];
  vst1q_s64(rets, ret);
  EXPECT_EQ(rets[0], ref);
  EXPECT_EQ(rets[1], ref);
  VLOG(1) << "div: val " << val << ", " << ref;
}

template <int POW>
void TestDivide64x2PowRound(int64 val, int64 ref) {
  const int64x2_t val_64x2 = vmovq_n_s64(val);
  const int64x2_t shifted = Divide64x2PowRound<POW>(val_64x2);
  // TODO(b/70947959) Change back to int64 when possible
  int64_t rets[2];
  vst1q_s64(rets, shifted);
  EXPECT_EQ(rets[0], ref) << "in = " << val << ", " << POW
                          << ", act = " << rets[0] << ", ref = " << ref;
  EXPECT_EQ(rets[1], ref);
  VLOG(1) << "div round: " << val << ", " << rets[0];
}

void TestDivide64x2PowAll() {
  for (int64 i = 0; i < 1000; ++i) {
    TestDivide64x2PowRound<1>(
        i, static_cast<int64_t>(static_cast<float>(i) / 2.0f + 0.5f));
    TestDivide64x2PowRound<1>(
        -i, static_cast<int64_t>(static_cast<float>(-i) / 2.0f - 0.5f));
    TestDivide64x2PowRound<2>(
        i, static_cast<int64_t>(static_cast<float>(i) / 4.0f + 0.5f));
    TestDivide64x2PowRound<2>(
        -i, static_cast<int64_t>(static_cast<float>(-i) / 4.0f - 0.5f));
    TestDivide64x2PowRound<4>(
        i, static_cast<int64_t>(static_cast<float>(i) / 16.0f + 0.5f));
    TestDivide64x2PowRound<4>(
        -i, static_cast<int64_t>(static_cast<float>(-i) / 16.0f - 0.5f));
    TestDivide64x2PowRound<8>(
        i, static_cast<int64_t>(static_cast<float>(i) / 256.0f + 0.5f));
    TestDivide64x2PowRound<8>(
        -i, static_cast<int64_t>(static_cast<float>(-i) / 256.0f - 0.5f));
    TestDivide64x2PowRound<16>(
        i, static_cast<int64_t>(static_cast<float>(i) / 65536.0f + 0.5f));
    TestDivide64x2PowRound<16>(
        -i, static_cast<int64_t>(static_cast<float>(-i) / 65536.0f - 0.5f));
  }

  TestDivide64x2Pow<2>(100, 25);
  TestDivide64x2Pow<2>(-100, -25);
  TestDivide64x2Pow<4>(100, 6);
  TestDivide64x2Pow<4>(-100, -6);

  for (int64 i = 0; i < 1000; ++i) {
    TestDivide64x2Pow<1>(i, i / 2);
    TestDivide64x2Pow<1>(-i, -i / 2);
    TestDivide64x2Pow<2>(i, i / 4);
    TestDivide64x2Pow<2>(-i, -i / 4);
    TestDivide64x2Pow<4>(i, i / 16);
    TestDivide64x2Pow<4>(-i, -i / 16);
    TestDivide64x2Pow<8>(i, i / 256);
    TestDivide64x2Pow<8>(-i, -i / 256);
    TestDivide64x2Pow<16>(i, i / 65536);
    TestDivide64x2Pow<16>(-i, -i / 65536);
  }
}

uint8x8_t To8x8(uint8 val) { return vmov_n_u8(val); }

int16x8_t To16x8(int16 val) { return vmovq_n_s16(val); }

int32x2_t To32x2(int32 val) {
  int32 vals[2];
  vals[0] = val;
  vals[1] = val;
  return vld1_s32(vals);
}

template <int RESOLUTION, typename T_CALC>
T_CALC ComputeRefLerp(T_CALC top_left, T_CALC top_right, T_CALC bottom_left,
                      T_CALC bottom_right, T_CALC x_lerp, T_CALC y_lerp) {
  constexpr T_CALC RESOLUTION_POW = (1 << RESOLUTION);
  const T_CALC top =
      top_left * RESOLUTION_POW + (top_right - top_left) * x_lerp;
  const T_CALC bottom =
      bottom_left * RESOLUTION_POW + (bottom_right - bottom_left) * x_lerp;
  const T_CALC out = top + (bottom - top) / RESOLUTION_POW * y_lerp;
  return (out + RESOLUTION_POW / 2) / RESOLUTION_POW;
}

template <int RESOLUTION>
void TestComputeLerp8x8(uint8 top_left, uint8 top_right, uint8 bottom_left,
                        uint8 bottom_right, int16 x_lerp, int16 y_lerp) {
  uint8x8_t top_left8x8 = To8x8(top_left);
  uint8x8_t top_right8x8 = To8x8(top_right);
  uint8x8_t bottom_left8x8 = To8x8(bottom_left);
  uint8x8_t bottom_right8x8 = To8x8(bottom_right);
  int16x8_t x_lerp16x8 = To16x8(x_lerp);
  int16x8_t y_lerp16x8 = To16x8(y_lerp);
  const uint8x8_t ret =
      ComputeLerp8x8<RESOLUTION>(top_left8x8, top_right8x8, bottom_left8x8,
                                 bottom_right8x8, x_lerp16x8, y_lerp16x8);

  uint8 rets[8];
  vst1_u8(rets, ret);

  const int16 ref = ComputeRefLerp<RESOLUTION, int16>(
      static_cast<int16>(top_left), static_cast<int16>(top_right),
      static_cast<int16>(bottom_left), static_cast<int16>(bottom_right), x_lerp,
      y_lerp);

  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(ref, static_cast<int16>(rets[i]));
  }

  VLOG(1) << "Lerp(8): " << static_cast<int>(top_left) << ", "
          << static_cast<int>(top_right) << ", "
          << static_cast<int>(bottom_left) << ", "
          << static_cast<int>(bottom_right) << ", " << x_lerp << ", " << y_lerp
          << ", " << static_cast<int>(rets[0]) << ", " << ref;
}

template <int RESOLUTION>
void TestComputeLerp32x2(int32 top_left, int32 top_right, int32 bottom_left,
                         int32 bottom_right, int32 x_lerp, int32 y_lerp) {
  int32x2_t top_left32x2 = To32x2(top_left);
  int32x2_t top_right32x2 = To32x2(top_right);
  int32x2_t bottom_left32x2 = To32x2(bottom_left);
  int32x2_t bottom_right32x2 = To32x2(bottom_right);
  int32x2_t x_lerp32x2 = To32x2(x_lerp);
  int32x2_t y_lerp32x2 = To32x2(y_lerp);
  const int32x2_t ret =
      ComputeLerp32x2<RESOLUTION>(top_left32x2, top_right32x2, bottom_left32x2,
                                  bottom_right32x2, x_lerp32x2, y_lerp32x2);
  int32 rets[2];
  vst1_s32(rets, ret);
  const int64 ref = ComputeRefLerp<RESOLUTION, int64>(
      static_cast<int64_t>(top_left), static_cast<int64_t>(top_right),
      static_cast<int64_t>(bottom_left), static_cast<int64_t>(bottom_right),
      static_cast<int64_t>(x_lerp), static_cast<int64_t>(y_lerp));
  EXPECT_EQ(static_cast<int64_t>(rets[0]), ref);
  VLOG(1) << "Lerp(32): " << top_left << ", " << top_right << ", "
          << bottom_left << ", " << bottom_right << ", " << x_lerp << ", "
          << y_lerp << ", " << rets[0] << ", " << ref;
}

void TestComputeLerp4xAll() {
  constexpr int32 RESOLUTION_32 = 30;
  constexpr int32 RESOLUTION_MULT_32 = (1 << RESOLUTION_32);
  constexpr int32 HALF_32 = RESOLUTION_MULT_32 / 2;
  TestComputeLerp32x2<RESOLUTION_32>(100, 200, 300, 400, HALF_32, HALF_32);
  TestComputeLerp32x2<RESOLUTION_32>(100, 100, 200, 200, HALF_32, HALF_32);
  TestComputeLerp32x2<RESOLUTION_32>(200, 200, 100, 100, HALF_32, HALF_32);
  TestComputeLerp32x2<RESOLUTION_32>(100, 200, 100, 200, HALF_32, HALF_32);
  TestComputeLerp32x2<RESOLUTION_32>(200, 100, 200, 100, HALF_32, HALF_32);
  TestComputeLerp32x2<RESOLUTION_32>(200, 200, 200, 200, HALF_32, HALF_32);

  constexpr int32 RESOLUTION_8 = 7;
  constexpr int32 RESOLUTION_MULT_8 = (1 << RESOLUTION_8);
  constexpr int32 HALF_8 = RESOLUTION_MULT_8 / 2;
  TestComputeLerp8x8<RESOLUTION_8>(10, 20, 30, 40, HALF_8, HALF_8);
  TestComputeLerp8x8<RESOLUTION_8>(100, 100, 200, 200, HALF_8, HALF_8);
  TestComputeLerp8x8<RESOLUTION_8>(200, 200, 100, 100, HALF_8, HALF_8);
  TestComputeLerp8x8<RESOLUTION_8>(100, 200, 100, 200, HALF_8, HALF_8);
  TestComputeLerp8x8<RESOLUTION_8>(200, 100, 200, 100, HALF_8, HALF_8);
  TestComputeLerp8x8<RESOLUTION_8>(200, 200, 200, 200, HALF_8, HALF_8);
}

#endif

}  // namespace tensorflow

#define RUN_TEST(t) \
  TEST(QuantizationUtilsTest, t) { tensorflow::t(); }

RUN_TEST(TestFloatToQuantized);
RUN_TEST(TestQuantizedToFloat);
RUN_TEST(TestAvoidBias);
RUN_TEST(TestRequantizeInNewRange);
RUN_TEST(TestRequantizeInNewRangeRealData);
RUN_TEST(TestRequantizeInNewRange32To8Bit);
RUN_TEST(TestRequantizeManyInNewRange32To8Bit);
RUN_TEST(TestRequantizeManyInNewRange32To8BitUsingEigen);
RUN_TEST(TestRequantizeManyInNewRange32To8BitEigenVsNonEigen);
RUN_TEST(TestRequantizeManyInNewRange32To8BitSignedEigenVsNonEigen);
RUN_TEST(TestFloatTensorToQuantized);
RUN_TEST(TestRequantizeManyInNewRange8To32Bit);
RUN_TEST(TestFloatToQuantizedInPlaceUsingEigen);
RUN_TEST(TestOverflowWithEigen);
RUN_TEST(TestQuantizedTensorToFloat);
RUN_TEST(TestQuantizedToFloatInPlaceUsingEigen);

#if defined(__ANDROID__)

RUN_TEST(BenchmarkRequantizeManyInNewRange);

#ifdef QUANTIZATION_UTILS_USE_NEON

RUN_TEST(TestDivide64x2PowAll);
RUN_TEST(TestComputeLerp4xAll);

#endif  // QUANTIZATION_UTILS_USE_NEON

#endif  // __ANDROID__

int main(int argc, char** argv) {
  // On Linux, add: absl::SetFlag(&FLAGS_logtostderr, true);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
