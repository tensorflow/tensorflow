
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_TF_MICRO_SIMPLE_API_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_TF_MICRO_SIMPLE_API_H_

   
//initialize model with simple api 
void tf_micro_simple_setup(const void * model_data, unsigned char * tensor_arena, int kTensorArenaSize);

//invoke model with simple api
void tf_micro_simple_invoke(float* input_data, int num_inputs, float* results,
                     int num_outputs);


#endif //TENSORFLOW_LITE_EXPERIMENTAL_MICRO_TF_MICRO                     