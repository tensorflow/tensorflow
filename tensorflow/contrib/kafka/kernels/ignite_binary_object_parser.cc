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

#include "ignite_binary_object_parser.h"
#include <iostream> 

namespace ignite {

char* BinaryObjectParser::Parse(char *ptr, std::vector<tensorflow::Tensor>* out_tensors, std::vector<int>* types) {
  char object_type_id = *ptr;
  ptr += 1;

  switch(object_type_id) {
    case 1: {
      // byte
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_INT8, {});
      tensor.scalar<tensorflow::int8>()() = *ptr;
      ptr += 1;
      out_tensors->emplace_back(std::move(tensor));
      break;
    }
    case 2: {
      // short
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_INT16, {});
      tensor.scalar<tensorflow::int16>()() = *((short*)ptr);
      ptr += 2;
      out_tensors->emplace_back(std::move(tensor));
      break;
    }
    case 3: {
      // int
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_INT32, {});
      tensor.scalar<tensorflow::int32>()() = *((int*)ptr);
      ptr += 4;
      out_tensors->emplace_back(std::move(tensor));
      break;
    }
    case 4: {
      // long
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_INT64, {});
      tensor.scalar<tensorflow::int64>()() = *((long*)ptr);
      ptr += 8;
      out_tensors->emplace_back(std::move(tensor));
      break;
    }
    case 5: {
      // float
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_FLOAT, {});
      tensor.scalar<float>()() = *((float*)ptr);
      ptr += 4;
      out_tensors->emplace_back(std::move(tensor));
      break;
    }
    case 6: {
      // double
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_DOUBLE, {});
      tensor.scalar<double>()() = *((double*)ptr);
      ptr += 8;
      out_tensors->emplace_back(std::move(tensor));
      break;
    }
    case 7: {
      // uchar
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_UINT16, {});
      tensor.scalar<tensorflow::uint16>()() = *((unsigned short*)ptr);
      ptr += 2;
      out_tensors->emplace_back(std::move(tensor));
      break;
    }
    case 8: {
      // bool
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_BOOL, {});
      tensor.scalar<bool>()() = *((bool*)ptr);
      ptr += 1;
      out_tensors->emplace_back(std::move(tensor));

      break;
    }
    case 9: {
      // string
      int length = *((int*)ptr);
      ptr += 4;
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_STRING, {});
      tensor.scalar<std::string>()() = std::string(ptr, length);
      ptr += length;
      out_tensors->emplace_back(std::move(tensor));

      break;
    }
    case 12: {
      // byte arr
      int length = *((int*)ptr);
      ptr += 4;
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_INT8, tensorflow::TensorShape({length}));

      char* arr = ptr;
      ptr += length;
      for (int i = 0; i < length; i++)
        tensor.vec<tensorflow::int8>()(i) = arr[i];

      break;
    }
    case 13: {
      // short arr
      int length = *((int*)ptr);
      ptr += 4;
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_INT16, tensorflow::TensorShape({length}));

      short* arr = (short*)ptr;
      ptr += length * 2;
      for (int i = 0; i < length; i++)
        tensor.vec<tensorflow::int16>()(i) = arr[i];

      break;
    }
    case 14: {
      int length = *((int*)ptr);
      ptr += 4;
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_INT32, tensorflow::TensorShape({length}));

      int* arr = (int*)ptr;
      ptr += length * 4;
      for (int i = 0; i < length; i++)
        tensor.vec<tensorflow::int32>()(i) = arr[i];

      break;
    }
    case 15: {
      int length = *((int*)ptr);
      ptr += 4;
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_INT64, tensorflow::TensorShape({length}));

      long* arr = (long*)ptr;
      ptr += length * 8;
      for (int i = 0; i < length; i++)
        tensor.vec<tensorflow::int64>()(i) = arr[i];

      // long arr
      break;
    }
    case 16: {
      int length = *((int*)ptr);
      ptr += 4;
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_FLOAT, tensorflow::TensorShape({length}));

      float* arr = (float*)ptr;
      ptr += length * 4;
      for (int i = 0; i < length; i++)
        tensor.vec<float>()(i) = arr[i];

      // float arr
      break;
    }
    case 17: {
      // double arr
      int length = *((int*)ptr);
      ptr += 4;
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_DOUBLE, tensorflow::TensorShape({length}));

      double* arr = (double*)ptr;
      ptr += 8 * length;
      //for (int i = 0; i < length; i++) 
      //  tensor.vec<double>()(i) = arr[i];

      out_tensors->emplace_back(std::move(tensor));
      break;
    }
    case 18: {
      // uchar arr
      int length = *((int*)ptr);
      ptr += 4;
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_UINT16, tensorflow::TensorShape({length}));

      unsigned char* arr = (unsigned char*)ptr;
      ptr += length * 2;
      for (int i = 0; i < length; i++)
        tensor.vec<tensorflow::uint16>()(i) = arr[i];

      break;
    }
    case 19: {
      // bool arr
      int length = *((int*)ptr);
      ptr += 4;
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_BOOL, tensorflow::TensorShape({length}));

      bool* arr = (bool*)ptr;
      ptr += length;
      for (int i = 0; i < length; i++)
        tensor.vec<bool>()(i) = arr[i];

      break;
    }
    case 20: {
      // string arr
      int length = *((int*)ptr);
      ptr += 4;
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_STRING, tensorflow::TensorShape({length}));

      for (int i = 0; i < length; i++) {
        int str_length = *((int*)ptr);
        ptr += 4;
        char* str = ptr;
        ptr+= str_length;
        tensor.vec<std::string>()(i) = std::string(str, str_length);
      }

      // TODO!
      break;
    }
    case 27: {
      int byte_arr_size = *((int*)ptr);
      ptr += 4;
      // payload
      ptr = Parse(ptr, out_tensors, types);

      int offset = *((int*)ptr);
      ptr += 4;

      //std::cout << "Wrapped object " << byte_arr_size << ", offset " << offset << "\n";

      break;
    }
  	case 103: {
      //std::cout << "Complex object..." << std::endl;
  	  char version = *ptr;
      ptr += 1;
  	  short flags = *((short*)ptr); // USER_TYPE = 1, HAS_SCHEMA = 2
      ptr += 2;
  	  int type_id = *((int*)ptr);
      ptr += 4;
  	  int hash_code = *((int*)ptr);
      ptr += 4;
  	  int length = *((int*)ptr);
      ptr += 4;
  	  int schema_id = *((int*)ptr);
      ptr += 4;
  	  int schema_offset = *((int*)ptr);
      ptr += 4;
	    int field_cnt = (length - schema_offset) / 8;
	    //std::cout << "LENGTH : " << length << std::endl;
      //    std::cout << "SCHMEA ID : " << schema_id << std::endl;
	    //std::cout << "SCHEMA OFFSET : " << schema_offset << std::endl;
	    //std::cout << "FIELD COUNT : " << field_cnt << std::endl;

      char* end = ptr + schema_offset - 24;//26;
      int i = 0;
      while (ptr < end) {
        //std::cout << "Parse field " << i << ", ptr = " << (long) ptr << ", end = " << (long) end << std::endl;
        i++;
        ptr = Parse(ptr, out_tensors, types);
      }

      ptr += (length - schema_offset);

  	  break;
  	}
  	default: {
      // TODO: Error
      std::cout << "Unknown type " << (int)object_type_id << std::endl;
  	}
  }

  return ptr;
}

} // namespace ignite
