#include <string>
#include <map>
#include <netinet/in.h>

#include "tensorflow/core/framework/dataset.h"

namespace ignite {

	struct binary_field {
	    std::string field_name;
	    int type_id;
	    int field_id;
	};

	struct binary_type {
	    int type_id;
	    std::string type_name;
	    int field_cnt;
	    binary_field** fields;
	};

	class client {
	public:
		client();
		binary_type* get_type(int type_id);
		void scan_query(std::string cache_name, std::vector<tensorflow::Tensor>* out_tensors);
		void test();
	private:
		int sock;
		struct sockaddr_in server;
		// Read data
		char read_byte();
		short read_short();
		int read_int();
		long read_long();

		void parse_binary_object(char* arr, int offset);
		// Write data
		void write_byte(char data);
		void write_short(short data);
		void write_int(int data);
		void write_long(long data);
		// Network
		void conn(std::string address, int port);
		int javaHashCode(std::string str);
	};
}
