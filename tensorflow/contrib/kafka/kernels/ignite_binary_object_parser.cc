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

#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>

#include <sys/socket.h>    //socket
#include <arpa/inet.h> //inet_addr
#include <unistd.h>
#include <netinet/in.h>
#include <stdio.h> 

#include "ignite_client.h"

namespace ignite {

BinaryObjectParser::BinaryObjectParser(char *ptr, std::vector<tensorflow::Tensor>* out_tensors, Client* client) {
  this->ptr = ptr;
  this->out_tensors = out_tensors;
  this->client = client;
};

void BinaryObjectParser::Parse() {
  char object_type_id = *ptr;
  ptr += 1;

  switch(object_type_id) {
    case 3: {
      // int
      printf("Add integer\n");
      int val = ReadInt();
    	
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_INT32, {});
      tensor.scalar<tensorflow::int32>()() = val;
      out_tensors->emplace_back(std::move(tensor));

      break;
    }
    case 17: {
      // double arr
      printf("Add array\n");
      int length = ReadInt();
	    double* arr = (double*)ptr;

      ptr += 8 * length;
				
      tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_DOUBLE, tensorflow::TensorShape({length}));

      printf("length = %d\n", length);
      for (int i = 0; i < length; i++) {
        tensor.vec<double>()(i) = arr[i];
      }

      out_tensors->emplace_back(std::move(tensor));

      break;
    }
    case 27: {
      int byte_arr_size = ReadInt();
      // payload
      Parse();
      int offset = ReadInt();
      printf("byte_arr_size = %d, offset = %d\n", byte_arr_size, offset);
      break;
    }
	case 103: {
	  char version = ReadByte();
	  short flags = ReadShort(); // USER_TYPE = 1, HAS_SCHEMA = 2
	  int type_id = ReadInt();
	  int hash_code = ReadInt();
	  int length = ReadInt();
	  int schema_id = ReadInt();
	  int schema_offset = ReadInt();

	  BinaryType* type = client->GetType(type_id);

	  // printf("Version: %d, TypeId: %d, offset: %d, TypeName: %s\n", version, type_id, schema_offset, type->type_name.c_str());

	  // printf("Type : %s, fields: %d\n", type->type_name.c_str(), type->field_cnt);
	  for (int i = 0; i < type->field_cnt; i++) {
	    BinaryField* field = type->fields[i];
	    printf("%s : ", field->field_name.c_str());
	    // printf("Processing field %s\n", field->field_name.c_str());
	    // printf("Field: %s, type = %d\n", field->field_name.c_str(), field->type_id);
	    Parse();
	    // printf("Field %s successfully processed\n", field->field_name.c_str());
	  }

	  ReadShort();

	  break;
	}
	default: {
		printf("Unknown type: %d\n", object_type_id);
	}
  }
}

char BinaryObjectParser::ReadByte() {
  char res = *ptr;
  ptr += 1;
  return res;
}

short BinaryObjectParser::ReadShort() {
  short res = *(short*)ptr;
  ptr += 2;
  return res;
}

int BinaryObjectParser::ReadInt() {
  int res = *(int*)ptr;
  ptr += 4;
  return res;
}

long BinaryObjectParser::ReadLong() {
  int res = *(long*)ptr;
  ptr += 8;
  return res;
}

} // namespace ignite
