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

#include "tensorflow/contrib/ignite/kernels/dataset/ignite_binary_object_parser.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

BinaryObjectParser::BinaryObjectParser() : byte_swapper_(ByteSwapper(false)) {}

Status BinaryObjectParser::Parse(uint8_t** ptr,
                                 std::vector<Tensor>* out_tensors,
                                 std::vector<int32_t>* types) const {
  uint8_t object_type_id = ParseByte(ptr);

  // Skip non-leaf nodes.
  if (object_type_id != WRAPPED_OBJ && object_type_id != COMPLEX_OBJ)
    types->push_back(object_type_id);

  switch (object_type_id) {
    case BYTE: {
      out_tensors->emplace_back(cpu_allocator(), DT_UINT8, TensorShape({}));
      out_tensors->back().scalar<uint8>()() = ParseByte(ptr);
      break;
    }
    case SHORT: {
      out_tensors->emplace_back(cpu_allocator(), DT_INT16, TensorShape({}));
      out_tensors->back().scalar<int16>()() = ParseShort(ptr);
      break;
    }
    case USHORT: {
      out_tensors->emplace_back(cpu_allocator(), DT_UINT16, TensorShape({}));
      out_tensors->back().scalar<uint16>()() = ParseUnsignedShort(ptr);
      break;
    }
    case INT: {
      out_tensors->emplace_back(cpu_allocator(), DT_INT32, TensorShape({}));
      out_tensors->back().scalar<int32>()() = ParseInt(ptr);
      break;
    }
    case LONG: {
      out_tensors->emplace_back(cpu_allocator(), DT_INT64, TensorShape({}));
      out_tensors->back().scalar<int64>()() = ParseLong(ptr);
      break;
    }
    case FLOAT: {
      out_tensors->emplace_back(cpu_allocator(), DT_FLOAT, TensorShape({}));
      out_tensors->back().scalar<float>()() = ParseFloat(ptr);
      break;
    }
    case DOUBLE: {
      out_tensors->emplace_back(cpu_allocator(), DT_DOUBLE, TensorShape({}));
      out_tensors->back().scalar<double>()() = ParseDouble(ptr);
      break;
    }
    case BOOL: {
      out_tensors->emplace_back(cpu_allocator(), DT_BOOL, TensorShape({}));
      out_tensors->back().scalar<bool>()() = ParseBool(ptr);
      break;
    }
    case STRING: {
      out_tensors->emplace_back(cpu_allocator(), DT_STRING, TensorShape({}));
      out_tensors->back().scalar<string>()() = ParseString(ptr);
      break;
    }
    case DATE: {
      out_tensors->emplace_back(cpu_allocator(), DT_INT64, TensorShape({}));
      out_tensors->back().scalar<int64>()() = ParseLong(ptr);
      break;
    }
    case BYTE_ARR: {
      int32_t length = ParseInt(ptr);
      uint8_t* arr = ParseByteArr(ptr, length);
      out_tensors->emplace_back(cpu_allocator(), DT_UINT8,
                                TensorShape({length}));
      std::copy_n(arr, length, out_tensors->back().flat<uint8>().data());
      break;
    }
    case SHORT_ARR: {
      int32_t length = ParseInt(ptr);
      int16_t* arr = ParseShortArr(ptr, length);
      out_tensors->emplace_back(cpu_allocator(), DT_INT16,
                                TensorShape({length}));
      std::copy_n(arr, length, out_tensors->back().flat<int16>().data());
      break;
    }
    case USHORT_ARR: {
      int32_t length = ParseInt(ptr);
      uint16_t* arr = ParseUnsignedShortArr(ptr, length);
      out_tensors->emplace_back(cpu_allocator(), DT_UINT16,
                                TensorShape({length}));
      std::copy_n(arr, length, out_tensors->back().flat<uint16>().data());
      break;
    }
    case INT_ARR: {
      int32_t length = ParseInt(ptr);
      int32_t* arr = ParseIntArr(ptr, length);
      out_tensors->emplace_back(cpu_allocator(), DT_INT32,
                                TensorShape({length}));
      std::copy_n(arr, length, out_tensors->back().flat<int32>().data());
      break;
    }
    case LONG_ARR: {
      int32_t length = ParseInt(ptr);
      int64_t* arr = ParseLongArr(ptr, length);
      out_tensors->emplace_back(cpu_allocator(), DT_INT64,
                                TensorShape({length}));
      std::copy_n(arr, length, out_tensors->back().flat<int64>().data());
      break;
    }
    case FLOAT_ARR: {
      int32_t length = ParseInt(ptr);
      float* arr = ParseFloatArr(ptr, length);
      out_tensors->emplace_back(cpu_allocator(), DT_FLOAT,
                                TensorShape({length}));
      std::copy_n(arr, length, out_tensors->back().flat<float>().data());
      break;
    }
    case DOUBLE_ARR: {
      int32_t length = ParseInt(ptr);
      double* arr = ParseDoubleArr(ptr, length);
      out_tensors->emplace_back(cpu_allocator(), DT_DOUBLE,
                                TensorShape({length}));
      std::copy_n(arr, length, out_tensors->back().flat<double>().data());
      break;
    }
    case BOOL_ARR: {
      int32_t length = ParseInt(ptr);
      bool* arr = ParseBoolArr(ptr, length);
      out_tensors->emplace_back(cpu_allocator(), DT_BOOL,
                                TensorShape({length}));
      std::copy_n(arr, length, out_tensors->back().flat<bool>().data());
      break;
    }
    case STRING_ARR: {
      int32_t length = ParseInt(ptr);
      out_tensors->emplace_back(cpu_allocator(), DT_STRING,
                                TensorShape({length}));
      for (int32_t i = 0; i < length; i++)
        out_tensors->back().vec<string>()(i) = ParseString(ptr);
      break;
    }
    case DATE_ARR: {
      int32_t length = ParseInt(ptr);
      int64_t* arr = ParseLongArr(ptr, length);
      out_tensors->emplace_back(cpu_allocator(), DT_INT64,
                                TensorShape({length}));
      std::copy_n(arr, length, out_tensors->back().flat<int64>().data());
      break;
    }
    case WRAPPED_OBJ: {
      int32_t byte_arr_size = ParseInt(ptr);
      TF_RETURN_IF_ERROR(Parse(ptr, out_tensors, types));
      int32_t offset = ParseInt(ptr);

      break;
    }
    case COMPLEX_OBJ: {
      uint8_t version = ParseByte(ptr);
      int16_t flags = ParseShort(ptr);
      int32_t type_id = ParseInt(ptr);
      int32_t hash_code = ParseInt(ptr);
      int32_t length = ParseInt(ptr);
      int32_t schema_id = ParseInt(ptr);
      int32_t schema_offset = ParseInt(ptr);

      // 24 is size of header just read.
      uint8_t* end = *ptr + schema_offset - 24;
      int32_t i = 0;
      while (*ptr < end) {
        i++;
        TF_RETURN_IF_ERROR(Parse(ptr, out_tensors, types));
      }

      *ptr += (length - schema_offset);

      break;
    }
    default: {
      return errors::Unknown("Unknowd binary type (type id ",
                             (int)object_type_id, ")");
    }
  }

  return Status::OK();
}

uint8_t BinaryObjectParser::ParseByte(uint8_t** ptr) const {
  uint8_t res = **ptr;
  *ptr += 1;

  return res;
}

int16_t BinaryObjectParser::ParseShort(uint8_t** ptr) const {
  int16_t* res = *reinterpret_cast<int16_t**>(ptr);
  byte_swapper_.SwapIfRequiredInt16(res);
  *ptr += 2;

  return *res;
}

uint16_t BinaryObjectParser::ParseUnsignedShort(uint8_t** ptr) const {
  uint16_t* res = *reinterpret_cast<uint16_t**>(ptr);
  byte_swapper_.SwapIfRequiredUnsignedInt16(res);
  *ptr += 2;

  return *res;
}

int32_t BinaryObjectParser::ParseInt(uint8_t** ptr) const {
  int32_t* res = *reinterpret_cast<int32_t**>(ptr);
  byte_swapper_.SwapIfRequiredInt32(res);
  *ptr += 4;

  return *res;
}

int64_t BinaryObjectParser::ParseLong(uint8_t** ptr) const {
  int64_t* res = *reinterpret_cast<int64_t**>(ptr);
  byte_swapper_.SwapIfRequiredInt64(res);
  *ptr += 8;

  return *res;
}

float BinaryObjectParser::ParseFloat(uint8_t** ptr) const {
  float* res = *reinterpret_cast<float**>(ptr);
  byte_swapper_.SwapIfRequiredFloat(res);
  *ptr += 4;

  return *res;
}

double BinaryObjectParser::ParseDouble(uint8_t** ptr) const {
  double* res = *reinterpret_cast<double**>(ptr);
  byte_swapper_.SwapIfRequiredDouble(res);
  *ptr += 8;

  return *res;
}

bool BinaryObjectParser::ParseBool(uint8_t** ptr) const {
  bool res = **reinterpret_cast<bool**>(ptr);
  *ptr += 1;

  return res;
}

string BinaryObjectParser::ParseString(uint8_t** ptr) const {
  int32_t length = ParseInt(ptr);
  string res(*reinterpret_cast<char**>(ptr), length);
  *ptr += length;

  return res;
}

uint8_t* BinaryObjectParser::ParseByteArr(uint8_t** ptr, int length) const {
  uint8_t* res = *reinterpret_cast<uint8_t**>(ptr);
  *ptr += length;

  return res;
}

int16_t* BinaryObjectParser::ParseShortArr(uint8_t** ptr, int length) const {
  int16_t* res = *reinterpret_cast<int16_t**>(ptr);
  byte_swapper_.SwapIfRequiredInt16Arr(res, length);
  *ptr += length * 2;

  return res;
}

uint16_t* BinaryObjectParser::ParseUnsignedShortArr(uint8_t** ptr,
                                                    int length) const {
  uint16_t* res = *reinterpret_cast<uint16_t**>(ptr);
  byte_swapper_.SwapIfRequiredUnsignedInt16Arr(res, length);
  *ptr += length * 2;

  return res;
}

int32_t* BinaryObjectParser::ParseIntArr(uint8_t** ptr, int length) const {
  int32_t* res = *reinterpret_cast<int32_t**>(ptr);
  byte_swapper_.SwapIfRequiredInt32Arr(res, length);
  *ptr += length * 4;

  return res;
}

int64_t* BinaryObjectParser::ParseLongArr(uint8_t** ptr, int length) const {
  int64_t* res = *reinterpret_cast<int64_t**>(ptr);
  byte_swapper_.SwapIfRequiredInt64Arr(res, length);
  *ptr += length * 8;

  return res;
}

float* BinaryObjectParser::ParseFloatArr(uint8_t** ptr, int length) const {
  float* res = *reinterpret_cast<float**>(ptr);
  byte_swapper_.SwapIfRequiredFloatArr(res, length);
  *ptr += length * 4;

  return res;
}

double* BinaryObjectParser::ParseDoubleArr(uint8_t** ptr, int length) const {
  double* res = *reinterpret_cast<double**>(ptr);
  byte_swapper_.SwapIfRequiredDoubleArr(res, length);
  *ptr += length * 8;

  return res;
}

bool* BinaryObjectParser::ParseBoolArr(uint8_t** ptr, int length) const {
  bool* res = *reinterpret_cast<bool**>(ptr);
  *ptr += length;

  return res;
}

}  // namespace tensorflow
