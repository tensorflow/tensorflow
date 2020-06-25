//#include "tensorflow/lite/experimental/micro/examples/hello_world/gcc_fun.h"
#include "tensorflow/lite/micro/examples/hello_world/main_functions.h"
void gcc_fun_cpp()
{
return;
}

extern "C" struct model_info * setup_NN_gcc(const unsigned char *model_data , bool debug){ return setup_NN(model_data,debug);}
extern "C" float * loop_NN_float_gcc(float * input_data){return loop_NN_float(input_data);}
extern "C" void gcc_fun(){gcc_fun_cpp();}
extern "C" void print_string_gcc(const char * str){print_string( str);}


