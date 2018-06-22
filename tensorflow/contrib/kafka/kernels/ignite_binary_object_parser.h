#include "tensorflow/core/framework/dataset.h"

namespace ignite {

	class binary_object_parser {
	private:
		char* ptr;
		std::vector<tensorflow::Tensor>* out_tensors;
		char read_byte();
		short read_short();
		int read_int();
		long read_long();
	public:
		binary_object_parser(char *ptr, std::vector<tensorflow::Tensor>* out_tensors);
		void parse();
	};
}