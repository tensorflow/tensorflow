//#include "tensorflow/lite/micro/examples/hello_world/gcc_fun.h"
#include "tensorflow/lite/micro/examples/hello_world/main_functions.h"
//#include "tensorflow/lite/micro/examples/hello_world/PDM_.h"
void gcc_fun_cpp()
{
return;
}

extern "C" struct model_info * setup_NN(const unsigned char *model_data ){ return setup_NN_gcc(model_data);}
extern "C" float * loop_NN(float * input_data){return loop_NN_gcc(input_data);}

extern "C" void print_string(const char * str){print_string_gcc( str);}

/*
extern "C" void * getHandle(){return getHandle_gcc();}
extern "C" void setup_PDM(){setup_PDM_gcc();}
extern "C" void pdm_desable(){pdm_desable_gcc();}


extern "C" void pdm_config_print(void){pdm_config_print_gcc();}
extern "C" void pdm_data_get(void){pdm_data_get_gcc();}
extern "C" bool is_data_ready(){return is_data_ready_gcc();}
extern "C" void clear_data_ready(){clear_data_ready_gcc();}
extern "C" int16_t * getBuffer(){return getBuffer_gcc();}

extern "C" void getBuffer_ext(int16_t ** buf){buf[0]= getBuffer_gcc();}
*/
