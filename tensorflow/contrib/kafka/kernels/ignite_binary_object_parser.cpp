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

	binary_object_parser::binary_object_parser(char *ptr, std::vector<tensorflow::Tensor>* out_tensors) {
		this->ptr = ptr;
		this->out_tensors = out_tensors;
	};

	void binary_object_parser::parse() {
		char object_type_id = *ptr;
		ptr += 1;
		// printf("Type : %d\n", type);

		switch(object_type_id) {
			case 3: {
				// int
				printf("Add integer\n");
				int val = read_int();
				
				tensorflow::Tensor tensor(tensorflow::cpu_allocator(), tensorflow::DT_INT32, {});
				tensor.scalar<tensorflow::int32>()() = val;
				out_tensors->emplace_back(std::move(tensor));

				break;
			}
			case 17: {
				// double arr
				printf("Add array\n");
				int length = read_int();
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
				int byte_arr_size = read_int();
				// payload
				parse();
				int offset = read_int();
				printf("byte_arr_size = %d, offset = %d\n", byte_arr_size, offset);
				break;
			}
			case 103: {
				char version = read_byte();
	            short flags = read_short(); // USER_TYPE = 1, HAS_SCHEMA = 2
	            int type_id = read_int();
	            int hash_code = read_int();
	            int length = read_int();
	            int schema_id = read_int();
	            int schema_offset = read_int();

	            client cl;
	            binary_type* type = cl.get_type(type_id);

	            // printf("Version: %d, TypeId: %d, offset: %d, TypeName: %s\n", version, type_id, schema_offset, type->type_name.c_str());

	            // printf("Type : %s, fields: %d\n", type->type_name.c_str(), type->field_cnt);
	            for (int i = 0; i < type->field_cnt; i++) {
	            	binary_field* field = type->fields[i];
	            	printf("%s : ", field->field_name.c_str());
	            	// printf("Processing field %s\n", field->field_name.c_str());
	            	// printf("Field: %s, type = %d\n", field->field_name.c_str(), field->type_id);
	            	parse();
	            	// printf("Field %s successfully processed\n", field->field_name.c_str());
	            }

	            read_short();

	            break;
			}
			default: {
				printf("Unknown type: %d\n", object_type_id);
			}
		}
	}

	char binary_object_parser::read_byte() {
		char res = *ptr;
		ptr += 1;
		return res;
	}

	short binary_object_parser::read_short() {
		short res = *(short*)ptr;
		ptr += 2;
		return res;
	}

	int binary_object_parser::read_int() {
		int res = *(int*)ptr;
		ptr += 4;
		return res;
	}

	long binary_object_parser::read_long() {
		int res = *(long*)ptr;
		ptr += 8;
		return res;
	}
}