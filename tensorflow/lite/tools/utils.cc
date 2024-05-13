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

#include "tensorflow/lite/tools/utils.h"

#include <algorithm>
#include <complex>
#include <random>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace utils {

namespace {
std::mt19937* get_random_engine() {
  static std::mt19937* engine = []() -> std::mt19937* {
    return new std::mt19937();
  }();
  return engine;
}

template <typename T, typename Distribution>
inline InputTensorData CreateInputTensorData(int num_elements,
                                             Distribution distribution) {
  InputTensorData tmp;
  auto* random_engine = get_random_engine();
  tmp.bytes = sizeof(T) * num_elements;
  T* raw = new T[num_elements];
  std::generate_n(raw, num_elements, [&]() {
    if (std::is_same<T, std::complex<float>>::value) {
      return static_cast<T>(distribution(*random_engine),
                            distribution(*random_engine));
    } else {
      return static_cast<T>(distribution(*random_engine));
    }
  });
  tmp.data = VoidUniquePtr(static_cast<void*>(raw),
                           [](void* ptr) { delete[] static_cast<T*>(ptr); });
  return tmp;
}

}  // namespace

InputTensorData CreateRandomTensorData(const TfLiteTensor& tensor,
                                       float low_range, float high_range) {
  int num_elements = NumElements(tensor.dims);
  switch (tensor.type) {
    case kTfLiteComplex64: {
      return CreateInputTensorData<std::complex<float>>(
          num_elements,
          std::uniform_real_distribution<float>(low_range, high_range));
    }
    case kTfLiteFloat32: {
      return CreateInputTensorData<float>(
          num_elements,
          std::uniform_real_distribution<float>(low_range, high_range));
    }
    case kTfLiteFloat16: {
      // TODO(b/138843274): Remove this preprocessor guard when bug is fixed.
#if TFLITE_ENABLE_FP16_CPU_BENCHMARKS
#if __GNUC__ && \
    (__clang__ || __ARM_FP16_FORMAT_IEEE || __ARM_FP16_FORMAT_ALTERNATIVE)
      // __fp16 is available on Clang or when __ARM_FP16_FORMAT_* is defined.
      return CreateInputTensorData<__fp16>(
          num_elements, std::uniform_real_distribution<float>(-0.5f, 0.5f));
#else
      TFLITE_LOG(FATAL) << "Don't know how to populate tensor " << t->name
                        << " of type FLOAT16 on this platform.";
#endif
#else
      // You need to build with -DTFLITE_ENABLE_FP16_CPU_BENCHMARKS=1 using a
      // compiler that supports __fp16 type. Note: when using Clang and *not*
      // linking with compiler-rt, a definition of __gnu_h2f_ieee and
      // __gnu_f2h_ieee must be supplied.
      TFLITE_LOG(FATAL) << "Populating the tensor " << tensor.name
                        << " of type FLOAT16 is disabled.";
#endif  // TFLITE_ENABLE_FP16_CPU_BENCHMARKS
      break;
    }
    case kTfLiteFloat64: {
      return CreateInputTensorData<double>(
          num_elements,
          std::uniform_real_distribution<double>(low_range, high_range));
    }
    case kTfLiteInt64: {
      return CreateInputTensorData<int64_t>(
          num_elements,
          std::uniform_int_distribution<int64_t>(low_range, high_range));
    }
    case kTfLiteInt32: {
      return CreateInputTensorData<int32_t>(
          num_elements,
          std::uniform_int_distribution<int32_t>(low_range, high_range));
    }
    case kTfLiteUInt32: {
      return CreateInputTensorData<uint32_t>(
          num_elements,
          std::uniform_int_distribution<uint32_t>(low_range, high_range));
    }
    case kTfLiteInt16: {
      return CreateInputTensorData<int16_t>(
          num_elements,
          std::uniform_int_distribution<int16_t>(low_range, high_range));
    }
    case kTfLiteUInt8: {
      // std::uniform_int_distribution is specified not to support char types.
      return CreateInputTensorData<uint8_t>(
          num_elements,
          std::uniform_int_distribution<uint32_t>(low_range, high_range));
    }
    case kTfLiteInt8: {
      // std::uniform_int_distribution is specified not to support char types.
      return CreateInputTensorData<int8_t>(
          num_elements,
          std::uniform_int_distribution<int32_t>(low_range, high_range));
    }
    case kTfLiteString: {
      // Don't populate input for string. Instead, return a default-initialized
      // `InputTensorData` object directly.
      break;
    }
    case kTfLiteBool: {
      // According to std::uniform_int_distribution specification, non-int type
      // is not supported.
      return CreateInputTensorData<bool>(
          num_elements, std::uniform_int_distribution<uint32_t>(0, 1));
    }
    default: {
      TFLITE_LOG(FATAL) << "Don't know how to populate tensor " << tensor.name
                        << " of type " << tensor.type;
    }
  }
  return InputTensorData();
}

void GetDataRangesForType(TfLiteType type, float* low_range,
                          float* high_range) {
  if (type == kTfLiteComplex64 || type == kTfLiteFloat32 ||
      type == kTfLiteFloat64) {
    *low_range = -0.5f;
    *high_range = 0.5f;
  } else if (type == kTfLiteInt64 || type == kTfLiteUInt64 ||
             type == kTfLiteInt32 || type == kTfLiteUInt32) {
    *low_range = 0;
    *high_range = 99;
  } else if (type == kTfLiteUInt8) {
    *low_range = 0;
    *high_range = 254;
  } else if (type == kTfLiteInt8) {
    *low_range = -127;
    *high_range = 127;
  }
}

}  // namespace utils
}  // namespace tflite
