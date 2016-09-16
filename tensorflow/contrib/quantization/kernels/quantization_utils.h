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

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_QUANTIZATION_KERNELS_QUANTIZATION_UTILS_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_QUANTIZATION_KERNELS_QUANTIZATION_UTILS_H_

#define EIGEN_USE_THREADS

// This is a set of functions that standardizes how quantized values are
// interpreted as float numbers.
// All of the current implementations are for reference and have not been
// optimized. They should be implementable using fixed point representations
// to avoid a dependency on floating-point hardware.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "public/gemmlowp.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {

// We have to be able to detect and handle overflows in int32, so this function
// uses doubles and int64's to make sure we have enough room.
template <class T>
int64 FloatToQuantizedUnclamped(float input, float range_min, float range_max) {
  const int64 lowest_quantized =
      static_cast<double>(Eigen::NumTraits<T>::lowest());
  if (range_min == range_max) {
    return lowest_quantized;
  }
  const int number_of_bits = sizeof(T) * 8;
  const int64 number_of_steps = static_cast<int64>(1) << number_of_bits;
  const double range_adjust = (number_of_steps / (number_of_steps - 1.0));
  const double range = ((range_max - range_min) * range_adjust);
  const double range_scale = (number_of_steps / range);
  int64 quantized =
      (round(input * range_scale) - round(range_min * range_scale));
  quantized += lowest_quantized;
  return quantized;
}

// This converts the float into the final quantized type, clamping/saturating
// any over or underflows.
template <class T>
T FloatToQuantized(float input, float range_min, float range_max) {
  int64 quantized = FloatToQuantizedUnclamped<T>(input, range_min, range_max);
  const int64 lowest_quantized =
      static_cast<int64>(Eigen::NumTraits<T>::lowest());
  const int64 highest_quantized =
      static_cast<int64>(Eigen::NumTraits<T>::highest());
  quantized = std::max(quantized, lowest_quantized);
  quantized = std::min(quantized, highest_quantized);
  return static_cast<T>(static_cast<int32>(quantized));
}

template <class T>
float QuantizedToFloat(T input, float range_min, float range_max) {
  if (range_min == range_max) {
    return range_min;
  }
  const int number_of_bits = sizeof(T) * 8;
  const int64 number_of_steps = static_cast<int64>(1) << number_of_bits;
  const double range_adjust = (number_of_steps / (number_of_steps - 1.0));
  const double range = ((range_max - range_min) * range_adjust);
  const double range_scale = (range / number_of_steps);
  const int64 lowest_quantized =
      static_cast<int64>(Eigen::NumTraits<T>::lowest());
  const double offset_input = static_cast<double>(input) - lowest_quantized;
  const double result = range_min + (offset_input * range_scale);
  return static_cast<float>(result);
}

template <class T>
float FloatForOneQuantizedLevel(float range_min, float range_max) {
  const int64 highest = static_cast<int64>(Eigen::NumTraits<T>::highest());
  const int64 lowest = static_cast<int64>(Eigen::NumTraits<T>::lowest());
  const float float_for_one_quantized_level =
      (range_max - range_min) / (highest - lowest);
  return float_for_one_quantized_level;
}

template <class T1, class T2, class T3>
void QuantizationRangeForMultiplication(float min_a, float max_a, float min_b,
                                        float max_b, float* min_c,
                                        float* max_c) {
  const float a_float_for_one_quant_level =
      FloatForOneQuantizedLevel<T1>(min_a, max_a);
  const float b_float_for_one_quant_level =
      FloatForOneQuantizedLevel<T2>(min_b, max_b);

  const int64 c_highest = static_cast<int64>(Eigen::NumTraits<T3>::highest());
  const int64 c_lowest = static_cast<int64>(Eigen::NumTraits<T3>::lowest());
  const float c_float_for_one_quant_level =
      a_float_for_one_quant_level * b_float_for_one_quant_level;

  *min_c = c_float_for_one_quant_level * c_lowest;
  *max_c = c_float_for_one_quant_level * c_highest;
}

// input_array is an eigen Tensor.  q2f is a QuantizedToFloatStruct.
// This evaluates to an eigen tensor expression, to be used like:
// auto tensor = DEQUANTIZE_WITH_EIGEN(input_tensor, q2f);
#define DEQUANTIZE_WITH_EIGEN(input_array, q2f)                       \
  (q2f.range_min +                                                    \
   (((input_array.template cast<float>() - q2f.lowest_quantized())) * \
    q2f.range_scale));

// input_array is an eigen Tensor.  f2q is a FloatToQuantizedStruct.
// OutputType is the type of output (e.g. quint8).
// This evaluates to an eigen tensor expression, to be used like:
// auto tensor = QUANTIZE_WITH_EIGEN(input_tensor, f2q, T);
#define QUANTIZE_WITH_EIGEN(input_array, f2q, OutputType) \
  ((input_array * f2q.range_scale).round() -              \
   (f2q.range_min_scaled - f2q.lowest_quantized()))       \
      .cwiseMax(f2q.lower_bound_float())                  \
      .cwiseMin(f2q.upper_bound_float())                  \
      .template cast<int32>()                             \
      .template cast<OutputType>()

// For use with DEQUANTIZE_WITH_EIGEN.
template <typename T>
struct QuantizedToFloatStruct {
  static constexpr int number_of_bits = sizeof(T) * 8;
  static constexpr int64 number_of_steps = static_cast<int64>(1)
                                           << number_of_bits;

  static float lowest_quantized() {
    return static_cast<float>(Eigen::NumTraits<T>::lowest());
  }

  QuantizedToFloatStruct(float range_min, float range_max)
      : range_min(range_min),
        range_scale((range_max - range_min) / (number_of_steps - 1.0)) {}

  const float range_min;
  const float range_scale;
};

// For use with QUANTIZE_WITH_EIGEN.
template <typename T>
struct FloatToQuantizedStruct {
  static constexpr int number_of_bits = sizeof(T) * 8;
  static constexpr int64 number_of_steps = static_cast<int64>(1)
                                           << number_of_bits;
  static constexpr double range_adjust =
      (number_of_steps / (number_of_steps - 1.0));

  // Casting QInt32's lowest or highest to a float gives a float that can't be
  // cast back to int32 or QInt32.  Instead, use bounds that can be converted
  // back to int32 without going outside the range of an int32.
  static float lower_bound_float() {
    return Eigen::numext::maxi(
        static_cast<float>(Eigen::NumTraits<T>::lowest()), -2.147483648e+09f);
  }
  static float upper_bound_float() {
    return Eigen::numext::mini(
        static_cast<float>(Eigen::NumTraits<T>::highest()), +2.147483520e+09f);
  }

  static float lowest_quantized() {
    return static_cast<float>(Eigen::NumTraits<T>::lowest());
  }

  FloatToQuantizedStruct(float range_min, float range_max)
      : range_min(range_min),
        range_scale(range_max == range_min
                        ? 0.0
                        : (number_of_steps - 1.0) / (range_max - range_min)),
        range_min_scaled(round(range_min * range_scale)) {}

  const float range_min;
  const float range_scale;
  const float range_min_scaled;
};

template <class T1, class T2>
inline T2 RequantizeInNewRange(T1 input, float min_input, float max_input,
                               float min_new, float max_new) {
  const float input_float = QuantizedToFloat<T1>(input, min_input, max_input);
  return FloatToQuantized<T2>(input_float, min_new, max_new);
}

template <class T1, class T2>
inline void RequantizeManyInNewRange(const T1* input, size_t count,
                                     float min_input, float max_input,
                                     float min_output, float max_output,
                                     T2* output) {
  for (size_t index = 0; index < count; ++index) {
    const float input_float =
        QuantizedToFloat<T1>(input[index], min_input, max_input);
    output[index] = FloatToQuantized<T2>(input_float, min_output, max_output);
  }
}

// Because converting 32-bit accumulated results down to eight bit is a common
// case, we have a specialized code path to handle it as efficiently as
// possible using only fixed-point math for the inner loop.
template <>
inline void RequantizeManyInNewRange<qint32, quint8>(
    const qint32* input, size_t count, float min_input, float max_input,
    float min_output, float max_output, quint8* output) {
  // Initially we calculate all the constants we need once, before we go into
  // the inner loop.  If this is updated, also update the Eigen version.
  const int fp_shift = 16;
  const float input_range = max_input - min_input;
  const float output_range = max_output - min_output;
  const float recip_output_range =
      output_range == 0.0 ? 0.0 : (255.0 / output_range);
  const float input_rezero = (min_input + max_input) / 2.0;
  const int64 range_scale_fp =
      output_range == 0.0 ? 0.0
                          : static_cast<int64>(255.0 * (1 << fp_shift) *
                                               input_range / output_range);
  const int64 input_offset_fp =
      static_cast<int64>(input_rezero * recip_output_range * (1 << fp_shift));
  const int64 output_offset_fp =
      output_range == 0.0 ? 0 : static_cast<int64>((1 << fp_shift) *
                                                   (min_output * 255.0) /
                                                   output_range);
  const int64 rounding_delta = 1 << (fp_shift - 1);

  // Inside this loop we just do minimal adds, multiplies, and shifts, in a way
  // that could be easily adapted for a SIMD implementation. It should also be
  // possible to perform all the calculations in 32-bit rather than 64, but
  // that's not been implemented yet.
  for (size_t index = 0; index < count; ++index) {
    const int64 input_value = static_cast<int64>(input[index]);
    const int64 fp_value =
        ((input_value * range_scale_fp) >> 32) + input_offset_fp;
    const int64 offset_intermediate = fp_value - output_offset_fp;
    const int64 round_intermediate = offset_intermediate + rounding_delta;
    int64 quantized_int64 = round_intermediate >> fp_shift;
    quantized_int64 = std::max(quantized_int64, 0LL);
    quantized_int64 = std::min(quantized_int64, 255LL);
    output[index] = static_cast<quint8>(static_cast<int32>(quantized_int64));
  }
}

template <int shift>
struct int64_right_shift_op {
  EIGEN_EMPTY_STRUCT_CTOR(int64_right_shift_op)
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const int64 operator()(const int64& a) const {
    return a >> shift;
  }
};

// See RequantizeManyInNewRange() for a non-eigen reference implementation.
template <class T1, class T2>
inline void RequantizeManyInNewRangeUsingEigen(
    const Eigen::ThreadPoolDevice& device, const Tensor& input, float min_input,
    float max_input, float min_output, float max_output, Tensor* output) {
  auto input_array = input.flat<T1>();
  QuantizedToFloatStruct<T1> q2f(min_input, max_input);
  auto input_float = DEQUANTIZE_WITH_EIGEN(input_array, q2f);
  FloatToQuantizedStruct<T2> f2q(min_output, max_output);
  auto input_requantized = QUANTIZE_WITH_EIGEN(input_float, f2q, T2);

  output->flat<T2>().device(device) = input_requantized;
}

// See RequantizeManyInNewRange() for a non-eigen reference implementation.
//
// Because converting 32-bit accumulated results down to eight bit is a common
// case, we have a specialized code path to handle it as efficiently as
// possible using only fixed-point math for the inner loop.
template <>
inline void RequantizeManyInNewRangeUsingEigen<qint32, quint8>(
    const Eigen::ThreadPoolDevice& device, const Tensor& input, float min_input,
    float max_input, float min_output, float max_output, Tensor* output) {
  // Initially we calculate all the constants we need once, before we go into
  // the inner loop.  If this is updated, also update the non-Eigen version.
  const int fp_shift = 16;
  const float input_range = max_input - min_input;
  const float output_range = max_output - min_output;
  const float recip_output_range =
      output_range == 0.0 ? 0.0 : (255.0 / output_range);
  const float input_rezero = (min_input + max_input) / 2.0;
  const int64 range_scale_fp =
      output_range == 0.0 ? 0.0
                          : static_cast<int64>(255.0 * (1 << fp_shift) *
                                               input_range / output_range);
  const int64 input_offset_fp =
      static_cast<int64>(input_rezero * recip_output_range * (1 << fp_shift));
  const int64 output_offset_fp =
      output_range == 0.0 ? 0 : static_cast<int64>((1 << fp_shift) *
                                                   (min_output * 255.0) /
                                                   output_range);
  const int64 rounding_delta = 1 << (fp_shift - 1);

  // Inside this eigen expression we just do minimal adds, multiplies, and
  // shifts. It should be possible to perform all the calculations in 32-bit
  // rather than 64, but that's not been implemented yet.
  auto input_array = input.flat<qint32>();
  auto fp_value = ((input_array.template cast<int64>() * range_scale_fp)
                       .unaryExpr(int64_right_shift_op<32>())) +
                  (input_offset_fp - output_offset_fp + rounding_delta);
  auto intermediate = fp_value.unaryExpr(int64_right_shift_op<fp_shift>());
  auto input_requantized = intermediate.cwiseMax(0LL)
                               .cwiseMin(255LL)
                               .template cast<int32>()
                               .template cast<quint8>();
  output->flat<quint8>().device(device) = input_requantized;
}

// REQUIRES: 'result->NumElements() == input.NumElements()'
template <class T>
void FloatTensorToQuantizedInPlaceUsingEigen(
    const Eigen::ThreadPoolDevice& device, const Tensor& input, float min,
    float max, Tensor* result) {
  DCHECK_EQ(DataTypeToEnum<T>::v(), result->dtype());
  auto flat_input = input.flat<float>();
  auto flat_result = result->flat<T>();
  DCHECK_EQ(flat_input.size(), flat_result.size());

  FloatToQuantizedStruct<T> f2q(min, max);
  flat_result.device(device) = QUANTIZE_WITH_EIGEN(flat_input, f2q, T);
}

template <class T>
void FloatTensorToQuantizedInPlace(const Tensor& input, float min, float max,
                                   Tensor* result) {
  DCHECK_EQ(DataTypeToEnum<T>::v(), result->dtype());
  auto flat_input = input.flat<float>();
  auto flat_result = result->flat<T>();
  const int data_size = flat_input.size();
  DCHECK(data_size == flat_result.size());
  for (int i = 0; i < data_size; ++i) {
    flat_result(i) = FloatToQuantized<T>(flat_input(i), min, max);
  }
}

template <class T>
Tensor FloatTensorToQuantized(const Tensor& input, float min, float max) {
  Tensor result(DataTypeToEnum<T>::v(), input.shape());
  FloatTensorToQuantizedInPlace<T>(input, min, max, &result);
  return result;
}

// REQUIRES: 'result->NumElements() == input.NumElements()'
template <class T>
void QuantizedTensorToFloatInPlaceUsingEigen(
    const Eigen::ThreadPoolDevice& device, const Tensor& input, float min,
    float max, Tensor* result) {
  DCHECK_EQ(DataTypeToEnum<T>::v(), input.dtype());
  auto flat_input = input.flat<T>();
  auto flat_result = result->flat<float>();
  const int data_size = flat_input.size();
  DCHECK(data_size == flat_result.size());

  QuantizedToFloatStruct<T> q2f(min, max);
  flat_result.device(device) = DEQUANTIZE_WITH_EIGEN(flat_input, q2f);
}

// REQUIRES: 'result->NumElements() == input.NumElements()'
template <class T>
void QuantizedTensorToFloatInPlace(const Tensor& input, float min, float max,
                                   Tensor* result) {
  DCHECK_EQ(DataTypeToEnum<T>::v(), input.dtype());
  auto flat_input = input.flat<T>();
  auto flat_result = result->flat<float>();
  const int data_size = flat_input.size();
  DCHECK(data_size == flat_result.size());
  for (int i = 0; i < data_size; ++i) {
    flat_result(i) = QuantizedToFloat<T>(flat_input(i), min, max);
  }
}

template <class T>
Tensor QuantizedTensorToFloat(const Tensor& input, float min, float max) {
  Tensor result(DT_FLOAT, input.shape());
  QuantizedTensorToFloatInPlace<T>(input, min, max, &result);
  return result;
}

void GetOutputMinAndMaxForQuantizedAdd(float input_min, float input_max,
                                       float smaller_input_min,
                                       float smaller_input_max,
                                       float* output_min, float* output_max);

// Add <input> and <smaller_input>.  If <smaller_input> has fewer elements than
// <input>, then it is broadcast onto <input>.
template <typename T1, typename T2, typename T3>
void QuantizedAddUsingEigen(const Eigen::ThreadPoolDevice& device,
                            const Tensor& input, float input_min,
                            float input_max, const Tensor& smaller_input,
                            float smaller_input_min, float smaller_input_max,
                            Tensor* output, float* output_min,
                            float* output_max) {
  const auto& input_flat = input.flat<T1>();
  const auto& smaller_input_flat = smaller_input.flat<T2>();
  auto output_flat = output->flat<T3>();

  GetOutputMinAndMaxForQuantizedAdd(input_min, input_max, smaller_input_min,
                                    smaller_input_max, output_min, output_max);
  // To do addition properly, we need to compensate for a possibly unbalanced
  // zero point in the total representation. The quantized value that
  // represents the real number zero needs to be subtracted before addition to
  // make sure that the identity of zero + zero = zero holds.
  const T3 zero_in_total_space =
      FloatToQuantized<T3>(0.0f, *output_min, *output_max);

  const int64 input_element_count = input.NumElements();
  const int64 smaller_input_element_count = smaller_input.NumElements();

  QuantizedToFloatStruct<T1> smaller_input_q2f(smaller_input_min,
                                               smaller_input_max);
  QuantizedToFloatStruct<T2> input_q2f(input_min, input_max);
  FloatToQuantizedStruct<T3> f2q(*output_min, *output_max);

  auto smaller_input_float =
      DEQUANTIZE_WITH_EIGEN(smaller_input_flat, smaller_input_q2f);
  auto smaller_input_in_total_space =
      QUANTIZE_WITH_EIGEN(smaller_input_float, f2q, T3);

  auto input_float = DEQUANTIZE_WITH_EIGEN(input_flat, input_q2f);
  auto input_in_total_space = QUANTIZE_WITH_EIGEN(input_float, f2q, T3);

  Eigen::array<Eigen::DenseIndex, 1> bcast;
  bcast[0] = input_element_count / smaller_input_element_count;
  output_flat.device(device) =
      input_in_total_space +
      (smaller_input_in_total_space.broadcast(bcast) + zero_in_total_space);
}

// This is a reference implementation of the bias addition for quantized
// buffers, designed to provide a clear specification for the result we
// want. We'll want to specialize this for particular hardware, and
// probably even fuse it with matrix multiplications in a lot of cases. It's
// important to show the clamping behavior we want in particular.
template <typename T1, typename T2, typename T3>
void QuantizedAdd(const Eigen::ThreadPoolDevice& device, const Tensor& input,
                  float input_min, float input_max, const Tensor& smaller_input,
                  float smaller_input_min, float smaller_input_max,
                  Tensor* output, float* output_min, float* output_max) {
  const auto& input_flat = input.flat<T1>();
  const auto& smaller_input_flat = smaller_input.flat<T2>();
  auto output_flat = output->flat<T3>();

  GetOutputMinAndMaxForQuantizedAdd(input_min, input_max, smaller_input_min,
                                    smaller_input_max, output_min, output_max);
  // To do addition properly, we need to compensate for a possibly unbalanced
  // zero point in the total representation. The quantized value that
  // represents the real number zero needs to be subtracted before addition to
  // make sure that the identity of zero + zero = zero holds.
  const T3 zero_in_total_space =
      FloatToQuantized<T3>(0.0f, *output_min, *output_max);

  const int64 input_element_count = input.NumElements();
  const int64 smaller_input_element_count = smaller_input.NumElements();

  float total_min = *output_min;
  float total_max = *output_max;
  const size_t how_many_iterations =
      (input_element_count / smaller_input_element_count);
  for (size_t iteration = 0; iteration < how_many_iterations; ++iteration) {
    const size_t offset = iteration * smaller_input_element_count;
    for (int c = 0; c < smaller_input_element_count; ++c) {
      const int index = (offset + c);
      // The two numbers we're going to add can each be in very different
      // ranges (e.g. the quantized value '127' may represent very different
      // real numbers in both) so we need to convert them to a common range
      // before we sum them.
      const T1 input_value = input_flat(index);
      const T3 input_in_total_space = RequantizeInNewRange<T1, T3>(
          input_value, input_min, input_max, total_min, total_max);
      const T2 smaller_input_value = smaller_input_flat(c);
      const T3 smaller_input_in_total_space =
          RequantizeInNewRange<T2, T3>(smaller_input_value, smaller_input_min,
                                       smaller_input_max, total_min, total_max);
      const T3 total_pre = input_in_total_space + smaller_input_in_total_space;
      // As noted above, we need to compensate for the offset of the actual
      // zero point in the space we're operating in.
      const T3 total = total_pre + zero_in_total_space;
      output_flat(index) = total;
    }
  }
}

// See gemmlowp/internal/multi_thread_gemm.h for definitions of
// Prepare, Wait, StartWorker, and CreateWorkers.
class TensorflowGemmlowpWorkersPool {
 public:
  TensorflowGemmlowpWorkersPool(thread::ThreadPool* workers)
      : workers_(workers) {}

  ~TensorflowGemmlowpWorkersPool() {
    // This workaround ensures that all worker tasks have exited methods in the
    // BlockingCounter. Without this, there is a race where the context is torn
    // down while the counter is in use.
    counter_to_decrement_when_ready_.Reset(0);
  }

  void Prepare(int workers_count) {
    counter_to_decrement_when_ready_.Reset(workers_count);
  }

  void Wait() { counter_to_decrement_when_ready_.Wait(); }

  void StartWorker(int index, gemmlowp::Task* task) {
    CHECK(workers_ != nullptr);
    // <index> is ignored - the tensorflow threadpool does not support assigning
    // to a specific thread.
    workers_->Schedule([this, task]() {
      // TODO(cwhipkey): get a local_allocator from a thread local.
      gemmlowp::Allocator local_allocator;
      CHECK(task != nullptr);
      task->local_allocator = &local_allocator;
      task->Run();
      delete task;
      counter_to_decrement_when_ready_.DecrementCount();
    });
  }

  void CreateWorkers(std::size_t workers_count) {}

 private:
  thread::ThreadPool* const workers_;

  // The BlockingCounter used to wait for the workers.
  gemmlowp::BlockingCounter counter_to_decrement_when_ready_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorflowGemmlowpWorkersPool);
};

class TensorflowGemmContext : public gemmlowp::MultiThreadGemmContextBase {
 public:
  TensorflowGemmContext(int num_threads, thread::ThreadPool* workers)
      : workers_pool_(workers) {
    set_max_num_threads(num_threads);
  }

  TensorflowGemmlowpWorkersPool* workers_pool() { return &workers_pool_; }

 private:
  TensorflowGemmlowpWorkersPool workers_pool_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorflowGemmContext);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_QUANTIZATION_KERNELS_QUANTIZATION_UTILS_H_
