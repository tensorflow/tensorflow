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
    case 3: {
      // int
      std::cout << "Add integer\n";
      int val = ReadInt(ptr);
    	
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_INT32, {});
      tensor.scalar<tensorflow::int32>()() = val;
      out_tensors->emplace_back(std::move(tensor));

      break;
    }
    case 17: {
      // double arr
      std::cout << "Add array\n";

      int length = ReadInt(ptr);

	    double* arr = (double*)ptr;
      ptr += 8 * length;
				
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_DOUBLE, tensorflow::TensorShape({length}));

      std::cout << "Array length: " << length << "\n";
      for (int i = 0; i < length; i++) {
        tensor.vec<tensorflow::double>()(i) = arr[i];
      }

      out_tensors->emplace_back(std::move(tensor));

      break;
    }
    case 27: {
      std::cout << "Wrapped object..." << std::endl;
      int byte_arr_size = ReadInt(ptr);
      // payload
      ptr = Parse(ptr);

      int offset = ReadInt(ptr);

      std::cout << "Byte array size " << byte_arr_size << ", offset " << offset << "\n";

      break;
    }
  	case 103: {
      std::cout << "Complex object..." << std::endl;
  	  char version = ReadByte(ptr);
  	  short flags = ReadShort(ptr); // USER_TYPE = 1, HAS_SCHEMA = 2
  	  int type_id = ReadInt(ptr);
  	  int hash_code = ReadInt(ptr);
  	  int length = ReadInt(ptr);
  	  int schema_id = ReadInt(ptr);
  	  int schema_offset = ReadInt(ptr);

      char* end = ptr + length - 26;
      int i = 0;
      while (ptr < end) {
        std::cout << "Parse field " << i << ", ptr = " << (long) ptr << ", end = " << (long) end << std::endl;
        i++;
        ptr = Parse(ptr);
      }

      ptr += 2; // TODO: WHY?

  	  break;
  	}
  	default: {
      // TODO: Error
      std::cout << "Unknown type " << object_type_id << "\n";
  	}
  }

  return ptr;
}

char BinaryObjectParser::ReadByte(char*& ptr) {
  char res = *ptr;
  ptr += 1;
  return res;
}

short BinaryObjectParser::ReadShort(char*& ptr) {
  short res = *(short*)ptr;
  ptr += 2;
  return res;
}

int BinaryObjectParser::ReadInt(char*& ptr) {
  int res = *(int*)ptr;
  ptr += 4;
  return res;
}

long BinaryObjectParser::ReadLong(char*& ptr) {
  long res = *(long*)ptr;
  ptr += 8;
  return res;
}

} // namespace ignite
