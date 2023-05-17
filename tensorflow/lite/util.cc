/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/util.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <complex>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/macros.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

TfLiteStatus UnresolvedOpInvoke(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_KERNEL_LOG(context,
                     "Encountered an unresolved custom op. Did you miss "
                     "a custom op or delegate?");
  return kTfLiteError;
}

}  // namespace

bool IsFlexOp(const char* custom_name) {
  return custom_name && strncmp(custom_name, kFlexCustomCodePrefix,
                                strlen(kFlexCustomCodePrefix)) == 0;
}

std::unique_ptr<TfLiteIntArray, TfLiteIntArrayDeleter> BuildTfLiteIntArray(
    const std::vector<int>& data) {
  std::unique_ptr<TfLiteIntArray, TfLiteIntArrayDeleter> result(
      TfLiteIntArrayCreate(data.size()));
  std::copy(data.begin(), data.end(), result->data);
  return result;
}

TfLiteIntArray* ConvertVectorToTfLiteIntArray(const std::vector<int>& input) {
  return ConvertArrayToTfLiteIntArray(static_cast<int>(input.size()),
                                      input.data());
}

TfLiteIntArray* ConvertArrayToTfLiteIntArray(const int ndims, const int* dims) {
  TfLiteIntArray* output = TfLiteIntArrayCreate(ndims);
  for (size_t i = 0; i < ndims; i++) {
    output->data[i] = dims[i];
  }
  return output;
}

bool EqualArrayAndTfLiteIntArray(const TfLiteIntArray* a, const int b_size,
                                 const int* b) {
  if (!a) return false;
  if (a->size != b_size) return false;
  for (int i = 0; i < a->size; ++i) {
    if (a->data[i] != b[i]) return false;
  }
  return true;
}

size_t CombineHashes(std::initializer_list<size_t> hashes) {
  size_t result = 0;
  // Hash combiner used by TensorFlow core.
  for (size_t hash : hashes) {
    result = result ^
             (hash + 0x9e3779b97f4a7800ULL + (result << 10) + (result >> 4));
  }
  return result;
}

TfLiteStatus GetSizeOfType(TfLiteContext* context, const TfLiteType type,
                           size_t* bytes) {
  // TODO(levp): remove the default case so that new types produce compilation
  // error.
  switch (type) {
    case kTfLiteFloat32:
      *bytes = sizeof(float);
      break;
    case kTfLiteInt32:
      *bytes = sizeof(int32_t);
      break;
    case kTfLiteUInt32:
      *bytes = sizeof(uint32_t);
      break;
    case kTfLiteUInt8:
      *bytes = sizeof(uint8_t);
      break;
    case kTfLiteInt64:
      *bytes = sizeof(int64_t);
      break;
    case kTfLiteUInt64:
      *bytes = sizeof(uint64_t);
      break;
    case kTfLiteBool:
      *bytes = sizeof(bool);
      break;
    case kTfLiteComplex64:
      *bytes = sizeof(std::complex<float>);
      break;
    case kTfLiteComplex128:
      *bytes = sizeof(std::complex<double>);
      break;
    case kTfLiteUInt16:
      *bytes = sizeof(uint16_t);
      break;
    case kTfLiteInt16:
      *bytes = sizeof(int16_t);
      break;
    case kTfLiteInt8:
      *bytes = sizeof(int8_t);
      break;
    case kTfLiteFloat16:
      *bytes = sizeof(TfLiteFloat16);
      break;
    case kTfLiteFloat64:
      *bytes = sizeof(double);
      break;
    case kTfLiteInt4:
      // TODO(b/246647008): Multiplying this value by the number of elements
      // does not yield the size of a tensor when 4-bit values are packed
      // 2 to a byte.
      *bytes = sizeof(int8_t);
      break;
    default:
      if (context) {
        TF_LITE_KERNEL_LOG(
            context,
            "Type %d is unsupported. Only float16, float32, float64, int8, "
            "int16, int32, int64, uint8, uint64, bool, complex64 and "
            "complex128 supported currently.",
            type);
      }
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteRegistration CreateUnresolvedCustomOp(const char* custom_op_name) {
  return TfLiteRegistration{nullptr,
                            nullptr,
                            nullptr,
                            /*invoke*/ &UnresolvedOpInvoke,
                            nullptr,
                            BuiltinOperator_CUSTOM,
                            custom_op_name,
                            1};
}

bool IsUnresolvedCustomOp(const TfLiteRegistration& registration) {
  return registration.builtin_code == tflite::BuiltinOperator_CUSTOM &&
         registration.invoke == &UnresolvedOpInvoke;
}

std::string GetOpNameByRegistration(const TfLiteRegistration& registration) {
  auto op = registration.builtin_code;
  std::string result =
      EnumNameBuiltinOperator(static_cast<BuiltinOperator>(op));
  if ((op == kTfLiteBuiltinCustom || op == kTfLiteBuiltinDelegate) &&
      registration.custom_name) {
    result += " " + std::string(registration.custom_name);
  }
  return result;
}

bool IsValidationSubgraph(const char* name) {
  // NOLINTNEXTLINE: can't use absl::StartsWith as absl is not allowed.
  return name && std::string(name).find(kValidationSubgraphNamePrefix) == 0;
}

TfLiteStatus MultiplyAndCheckOverflow(size_t a, size_t b, size_t* product) {
  // Multiplying a * b where a and b are size_t cannot result in overflow in a
  // size_t accumulator if both numbers have no non-zero bits in their upper
  // half.
  constexpr size_t size_t_bits = 8 * sizeof(size_t);
  constexpr size_t overflow_upper_half_bit_position = size_t_bits / 2;
  *product = a * b;
  // If neither integers have non-zero bits past 32 bits can't overflow.
  // Otherwise check using slow devision.
  if (TFLITE_EXPECT_FALSE((a | b) >> overflow_upper_half_bit_position != 0)) {
    if (a != 0 && *product / a != b) return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus BytesRequired(TfLiteType type, const int* dims, size_t dims_size,
                           size_t* bytes, TfLiteContext* context_) {
  TF_LITE_ENSURE(context_, bytes != nullptr);
  // When 'dims_size' is 0, we simply assume it's a scalar. Therefore, we start
  // 'count' as 1.
  size_t count = 1;
  for (int k = 0; k < dims_size; k++) {
    size_t old_count = count;
    TF_LITE_ENSURE_MSG(
        context_,
        MultiplyAndCheckOverflow(old_count, dims[k], &count) == kTfLiteOk,
        "BytesRequired number of elements overflowed.\n");
  }
  size_t type_size = 0;
  TF_LITE_ENSURE_OK(context_, GetSizeOfType(context_, type, &type_size));
  TF_LITE_ENSURE_MSG(
      context_, MultiplyAndCheckOverflow(type_size, count, bytes) == kTfLiteOk,
      "BytesRequired number of bytes overflowed.\n");

  // GetSizeOfType doesn't work for kTfLiteInt4 due to it having 2 values packed
  // into 1 byte so the output of GetSizeOfType is the same as int8 aka 1 byte.
  // Thus the required bytes must be divided by half after everything for int4.
  if (type == kTfLiteInt4) {
    *bytes = (*bytes + 1) / 2;
  }

  return kTfLiteOk;
}
}  // namespace tflite
