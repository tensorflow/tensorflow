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

#include "ignite_binary_object_parser.h"

namespace tensorflow {

Status BinaryObjectParser::Parse(uint8_t** ptr,
                                 std::vector<Tensor>* out_tensors,
                                 std::vector<int32_t>* types) {
  uint8_t object_type_id = **ptr;
  *ptr += 1;

  switch (object_type_id) {
    case BYTE: {
      Tensor tensor(cpu_allocator(), DT_UINT8, {});
      tensor.scalar<uint8>()() = *((uint8_t*)*ptr);
      *ptr += 1;
      out_tensors->push_back(std::move(tensor));
      break;
    }
    case SHORT: {
      Tensor tensor(cpu_allocator(), DT_INT16, {});
      tensor.scalar<int16>()() = *((int16_t*)*ptr);
      *ptr += 2;
      out_tensors->push_back(std::move(tensor));
      break;
    }
    case INT: {
      Tensor tensor(cpu_allocator(), DT_INT32, {});
      tensor.scalar<int32>()() = *((int32_t*)*ptr);
      *ptr += 4;
      out_tensors->push_back(std::move(tensor));
      break;
    }
    case LONG: {
      Tensor tensor(cpu_allocator(), DT_INT64, {});
      tensor.scalar<int64>()() = *((int64_t*)*ptr);
      *ptr += 8;
      out_tensors->push_back(std::move(tensor));
      break;
    }
    case FLOAT: {
      Tensor tensor(cpu_allocator(), DT_FLOAT, {});
      tensor.scalar<float>()() = *((float*)*ptr);
      *ptr += 4;
      out_tensors->push_back(std::move(tensor));
      break;
    }
    case DOUBLE: {
      Tensor tensor(cpu_allocator(), DT_DOUBLE, {});
      tensor.scalar<double>()() = *((double*)*ptr);
      *ptr += 8;
      out_tensors->push_back(std::move(tensor));
      break;
    }
    case UCHAR: {
      Tensor tensor(cpu_allocator(), DT_UINT16, {});
      tensor.scalar<uint16>()() = *((uint16_t*)*ptr);
      *ptr += 2;
      out_tensors->push_back(std::move(tensor));
      break;
    }
    case BOOL: {
      Tensor tensor(cpu_allocator(), DT_BOOL, {});
      tensor.scalar<bool>()() = *((bool*)*ptr);
      *ptr += 1;
      out_tensors->push_back(std::move(tensor));

      break;
    }
    case STRING: {
      int32_t length = *((int32_t*)*ptr);
      *ptr += 4;
      Tensor tensor(cpu_allocator(), DT_STRING, {});
      tensor.scalar<std::string>()() = std::string((char*)*ptr, length);
      *ptr += length;
      out_tensors->push_back(std::move(tensor));

      break;
    }
    case DATE: {
      Tensor tensor(cpu_allocator(), DT_INT64, {});
      tensor.scalar<int64>()() = *((int64_t*)*ptr);
      *ptr += 8;
      out_tensors->push_back(std::move(tensor));

      break;
    }
    case BYTE_ARR: {
      int32_t length = *((int32_t*)*ptr);
      *ptr += 4;
      Tensor tensor(cpu_allocator(), DT_UINT8, TensorShape({length}));

      uint8_t* arr = (uint8_t*)*ptr;
      *ptr += length;

      std::copy_n(arr, length, tensor.flat<uint8>().data());
      out_tensors->push_back(std::move(tensor));
      break;
    }
    case SHORT_ARR: {
      int32_t length = *((int32_t*)*ptr);
      *ptr += 4;
      Tensor tensor(cpu_allocator(), DT_INT16, TensorShape({length}));

      int16_t* arr = (int16_t*)*ptr;
      *ptr += length * 2;

      std::copy_n(arr, length, tensor.flat<int16>().data());
      out_tensors->push_back(std::move(tensor));
      break;
    }
    case INT_ARR: {
      int32_t length = *((int32_t*)*ptr);
      *ptr += 4;
      Tensor tensor(cpu_allocator(), DT_INT32, TensorShape({length}));

      int32_t* arr = (int32_t*)*ptr;
      *ptr += length * 4;

      std::copy_n(arr, length, tensor.flat<int32>().data());
      out_tensors->push_back(std::move(tensor));
      break;
    }
    case LONG_ARR: {
      int32_t length = *((int32_t*)*ptr);
      *ptr += 4;
      Tensor tensor(cpu_allocator(), DT_INT64, TensorShape({length}));

      int64_t* arr = (int64_t*)*ptr;
      *ptr += length * 8;

      std::copy_n(arr, length, tensor.flat<int64>().data());
      out_tensors->push_back(std::move(tensor));
      break;
    }
    case FLOAT_ARR: {
      int32_t length = *((int32_t*)*ptr);
      *ptr += 4;
      Tensor tensor(cpu_allocator(), DT_FLOAT, TensorShape({length}));

      float* arr = (float*)*ptr;
      *ptr += 4 * length;

      std::copy_n(arr, length, tensor.flat<float>().data());
      out_tensors->push_back(std::move(tensor));
      break;
    }
    case DOUBLE_ARR: {
      int32_t length = *((int32_t*)*ptr);
      *ptr += 4;
      Tensor tensor(cpu_allocator(), DT_DOUBLE, TensorShape({length}));

      double* arr = (double*)*ptr;
      *ptr += 8 * length;

      std::copy_n(arr, length, tensor.flat<double>().data());
      out_tensors->push_back(std::move(tensor));
      break;
    }
    case UCHAR_ARR: {
      int32_t length = *((int32_t*)*ptr);
      *ptr += 4;
      Tensor tensor(cpu_allocator(), DT_UINT16, TensorShape({length}));

      uint16_t* arr = (uint16_t*)*ptr;
      *ptr += length * 2;

      std::copy_n(arr, length, tensor.flat<uint16>().data());
      out_tensors->push_back(std::move(tensor));
      break;
    }
    case BOOL_ARR: {
      int32_t length = *((int32_t*)*ptr);
      *ptr += 4;
      Tensor tensor(cpu_allocator(), DT_BOOL, TensorShape({length}));

      bool* arr = (bool*)*ptr;
      *ptr += length;

      std::copy_n(arr, length, tensor.flat<bool>().data());
      out_tensors->push_back(std::move(tensor));
      break;
    }
    case STRING_ARR: {
      int32_t length = *((int32_t*)*ptr);
      *ptr += 4;
      Tensor tensor(cpu_allocator(), DT_STRING, TensorShape({length}));

      for (int32_t i = 0; i < length; i++) {
        int32_t str_length = *((int32_t*)*ptr);
        *ptr += 4;
        const int8_t* str = (const int8_t*)*ptr;
        *ptr += str_length;
        tensor.vec<std::string>()(i) = std::string((char*)str, str_length);
      }

      out_tensors->push_back(std::move(tensor));
      break;
    }
    case DATE_ARR: {
      int32_t length = *((int32_t*)*ptr);
      *ptr += 4;
      Tensor tensor(cpu_allocator(), DT_INT64, TensorShape({length}));
      int64_t* arr = (int64_t*)*ptr;
      *ptr += length * 8;

      std::copy_n(arr, length, tensor.flat<int64>().data());
      out_tensors->push_back(std::move(tensor));
      break;
    }
    case WRAPPED_OBJ: {
      int32_t byte_arr_size = *((int32_t*)*ptr);
      *ptr += 4;

      TF_RETURN_IF_ERROR(Parse(ptr, out_tensors, types));

      int32_t offset = *((int32_t*)*ptr);
      *ptr += 4;

      break;
    }
    case COMPLEX_OBJ: {
      uint8_t version = **ptr;
      *ptr += 1;
      int16_t flags = *((int16_t*)*ptr);  // USER_TYPE = 1, HAS_SCHEMA = 2
      *ptr += 2;
      int32_t type_id = *((int32_t*)*ptr);
      *ptr += 4;
      int32_t hash_code = *((int32_t*)*ptr);
      *ptr += 4;
      int32_t length = *((int32_t*)*ptr);
      *ptr += 4;
      int32_t schema_id = *((int32_t*)*ptr);
      *ptr += 4;
      int32_t schema_offset = *((int32_t*)*ptr);
      *ptr += 4;

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
      return errors::Internal("Unknowd binary type (type id ",
                              (int)object_type_id, ")");
    }
  }

  return Status::OK();
}

}  // namespace tensorflow
