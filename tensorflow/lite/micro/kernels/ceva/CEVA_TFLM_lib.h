

#ifndef CEVA_TFLM_LIB_H_
#define CEVA_TFLM_LIB_H_

#include "types.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */


void CEVA_TFLM_AffineQuantize_Int8(const float_32* input_data,
                                   int8_t* output_data, int flat_size,
                                   float_32 scale, int zero_point);



void CEVA_TFLM_FullyConnected_Float32(
    const void* params_inp, const int input_shape, const float* input_data,
    const int weights_shape_DimensionsCount, const int* weights_shape_DimsData,
    const float* weights_data, const int bias_shape, const float* bias_data,
    const int output_shape_DimensionsCount, const int* output_shape_DimsData,
    float* output_data);
void CEVA_TFLM_FullyConnected_int8(
    const void* params_inp, const int input_shape, const int8_t* input_data,
    const int filter_shape_DimensionsCount, const int* filter_shape_DimsData,
    const int8_t* filter_data, const int bias_shape, const int32_t* bias_data,
    const int output_shape_DimensionsCount, const int* output_shape_DimsData,
    int8_t* output_data);



#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif  // CEVA_TFLM_LIB_H_
