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

namespace ignite {

tensorflow::Status BinaryObjectParser::Parse(
    uint8_t*& ptr, std::vector<tensorflow::Tensor>& out_tensors,
    std::vector<int32_t>& types) {
  uint8_t object_type_id = *ptr;
  ptr += 1;

  switch (object_type_id) {
    case BYTE: {
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(),
                                tensorflow::DT_UINT8, {});
      tensor.scalar<tensorflow::uint8>()() = *((uint8_t*)ptr);
      ptr += 1;
      out_tensors.emplace_back(std::move(tensor));
      break;
    }
    case SHORT: {
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(),
                                tensorflow::DT_INT16, {});
      tensor.scalar<tensorflow::int16>()() = *((int16_t*)ptr);
      ptr += 2;
      out_tensors.emplace_back(std::move(tensor));
      break;
    }
    case INT: {
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(),
                                tensorflow::DT_INT32, {});
      tensor.scalar<tensorflow::int32>()() = *((int32_t*)ptr);
      ptr += 4;
      out_tensors.emplace_back(std::move(tensor));
      break;
    }
    case LONG: {
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(),
                                tensorflow::DT_INT64, {});
      tensor.scalar<tensorflow::int64>()() = *((int64_t*)ptr);
      ptr += 8;
      out_tensors.emplace_back(std::move(tensor));
      break;
    }
    case FLOAT: {
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(),
                                tensorflow::DT_FLOAT, {});
      tensor.scalar<float>()() = *((float*)ptr);
      ptr += 4;
      out_tensors.emplace_back(std::move(tensor));
      break;
    }
    case DOUBLE: {
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(),
                                tensorflow::DT_DOUBLE, {});
      tensor.scalar<double>()() = *((double*)ptr);
      ptr += 8;
      out_tensors.emplace_back(std::move(tensor));
      break;
    }
    case UCHAR: {
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(),
                                tensorflow::DT_UINT16, {});
      tensor.scalar<tensorflow::uint16>()() = *((uint16_t*)ptr);
      ptr += 2;
      out_tensors.emplace_back(std::move(tensor));
      break;
    }
    case BOOL: {
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(),
                                tensorflow::DT_BOOL, {});
      tensor.scalar<bool>()() = *((bool*)ptr);
      ptr += 1;
      out_tensors.emplace_back(std::move(tensor));

      break;
    }
    case STRING: {
      int32_t length = *((int32_t*)ptr);
      ptr += 4;
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(),
                                tensorflow::DT_STRING, {});
      tensor.scalar<std::string>()() = std::string((char*)ptr, length);
      ptr += length;
      out_tensors.emplace_back(std::move(tensor));

      break;
    }
    case DATE: {
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(),
                                tensorflow::DT_INT64, {});
      tensor.scalar<tensorflow::int64>()() = *((int64_t*)ptr);
      ptr += 8;
      out_tensors.emplace_back(std::move(tensor));

      break;
    }
    case BYTE_ARR: {
      int32_t length = *((int32_t*)ptr);
      ptr += 4;
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(),
                                tensorflow::DT_UINT8,
                                tensorflow::TensorShape({length}));

      uint8_t* arr = (uint8_t*)ptr;
      ptr += length;

      std::copy_n(arr, length, tensor.flat<tensorflow::uint8>().data());
      out_tensors.emplace_back(std::move(tensor));
      break;
    }
    case SHORT_ARR: {
      int32_t length = *((int32_t*)ptr);
      ptr += 4;
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(),
                                tensorflow::DT_INT16,
                                tensorflow::TensorShape({length}));

      int16_t* arr = (int16_t*)ptr;
      ptr += length * 2;

      std::copy_n(arr, length, tensor.flat<tensorflow::int16>().data());
      out_tensors.emplace_back(std::move(tensor));
      break;
    }
    case INT_ARR: {
      int32_t length = *((int32_t*)ptr);
      ptr += 4;
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(),
                                tensorflow::DT_INT32,
                                tensorflow::TensorShape({length}));

      int32_t* arr = (int32_t*)ptr;
      ptr += length * 4;

      std::copy_n(arr, length, tensor.flat<tensorflow::int32>().data());
      out_tensors.emplace_back(std::move(tensor));
      break;
    }
    case LONG_ARR: {
      int32_t length = *((int32_t*)ptr);
      ptr += 4;
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(),
                                tensorflow::DT_INT64,
                                tensorflow::TensorShape({length}));

      int64_t* arr = (int64_t*)ptr;
      ptr += length * 8;

      std::copy_n(arr, length, tensor.flat<tensorflow::int64>().data());
      out_tensors.emplace_back(std::move(tensor));
      break;
    }
    case FLOAT_ARR: {
      int32_t length = *((int32_t*)ptr);
      ptr += 4;
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(),
                                tensorflow::DT_FLOAT,
                                tensorflow::TensorShape({length}));

      float* arr = (float*)ptr;
      ptr += 4 * length;

      std::copy_n(arr, length, tensor.flat<float>().data());
      out_tensors.emplace_back(std::move(tensor));
      break;
    }
    case DOUBLE_ARR: {
      int32_t length = *((int32_t*)ptr);
      ptr += 4;
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(),
                                tensorflow::DT_DOUBLE,
                                tensorflow::TensorShape({length}));

      double* arr = (double*)ptr;
      ptr += 8 * length;

      std::copy_n(arr, length, tensor.flat<double>().data());
      out_tensors.emplace_back(std::move(tensor));
      break;
    }
    case UCHAR_ARR: {
      int32_t length = *((int32_t*)ptr);
      ptr += 4;
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(),
                                tensorflow::DT_UINT16,
                                tensorflow::TensorShape({length}));

      uint16_t* arr = (uint16_t*)ptr;
      ptr += length * 2;

      std::copy_n(arr, length, tensor.flat<tensorflow::uint16>().data());
      out_tensors.emplace_back(std::move(tensor));
      break;
    }
    case BOOL_ARR: {
      int32_t length = *((int32_t*)ptr);
      ptr += 4;
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(),
                                tensorflow::DT_BOOL,
                                tensorflow::TensorShape({length}));

      bool* arr = (bool*)ptr;
      ptr += length;

      std::copy_n(arr, length, tensor.flat<bool>().data());
      out_tensors.emplace_back(std::move(tensor));
      break;
    }
    case STRING_ARR: {
      int32_t length = *((int32_t*)ptr);
      ptr += 4;
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(),
                                tensorflow::DT_STRING,
                                tensorflow::TensorShape({length}));

      for (int32_t i = 0; i < length; i++) {
        int32_t str_length = *((int32_t*)ptr);
        ptr += 4;
        const int8_t* str = (const int8_t*)ptr;
        ptr += str_length;
        tensor.vec<std::string>()(i) = std::string((char*)str, str_length);
      }

      out_tensors.emplace_back(std::move(tensor));
      break;
    }
    case DATE_ARR: {
      int32_t length = *((int32_t*)ptr);
      ptr += 4;
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(),
                                tensorflow::DT_INT64,
                                tensorflow::TensorShape({length}));
      int64_t* arr = (int64_t*)ptr;
      ptr += length * 8;

      std::copy_n(arr, length, tensor.flat<tensorflow::int64>().data());
      out_tensors.emplace_back(std::move(tensor));
      break;
    }
    case WRAPPED_OBJ: {
      int32_t byte_arr_size = *((int32_t*)ptr);
      ptr += 4;

      tensorflow::Status status = Parse(ptr, out_tensors, types);
      if (!status.ok()) return status;

      int32_t offset = *((int32_t*)ptr);
      ptr += 4;

      break;
    }
    case COMPLEX_OBJ: {
      uint8_t version = *ptr;
      ptr += 1;
      int16_t flags = *((int16_t*)ptr);  // USER_TYPE = 1, HAS_SCHEMA = 2
      ptr += 2;
      int32_t type_id = *((int32_t*)ptr);
      ptr += 4;
      int32_t hash_code = *((int32_t*)ptr);
      ptr += 4;
      int32_t length = *((int32_t*)ptr);
      ptr += 4;
      int32_t schema_id = *((int32_t*)ptr);
      ptr += 4;
      int32_t schema_offset = *((int32_t*)ptr);
      ptr += 4;

      uint8_t* end = ptr + schema_offset - 24;
      int32_t i = 0;
      while (ptr < end) {
        i++;
        tensorflow::Status status = Parse(ptr, out_tensors, types);
        if (!status.ok()) return status;
      }

      ptr += (length - schema_offset);

      break;
    }
    default: {
      return tensorflow::errors::Internal("Unknowd binary type (type id ",
                                          (int)object_type_id, ")");
    }
  }

  return tensorflow::Status::OK();
}

}  // namespace ignite
