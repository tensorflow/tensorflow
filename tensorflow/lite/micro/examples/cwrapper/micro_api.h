
#ifndef TENSORFLOW_LITE_MICRO_C_API_H_
#define TENSORFLOW_LITE_MICRO_C_API_H_


#ifdef __cplusplus
extern "C" {
#endif


void tf_micro_model_setup(const void * model_data, unsigned char * tensor_arena, int kTensorArenaSize);

void tf_micro_model_invoke(float* input_data, int num_inputs, float* results, int num_outputs);


#ifdef __cplusplus
}
#endif


#endif //TENSORFLOW_LITE_MICRO_C_API_H_             
