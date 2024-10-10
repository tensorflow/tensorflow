#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <initializer_list>
#include <numeric>
#include <vector>
#include <iostream>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {

namespace GroupNormalization {

constexpr int kDataInputTensor = 0;
constexpr int kDataGammaTensor = 5;
constexpr int kDataBetaTensor = 6;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode*node){
    // group norm prepare;
    using namespace tflite;
    TF_LITE_ENSURE_EQ(context, NumInputs(node), 8);
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

    const TfLiteTensor* input = GetInput(context, node, 0);
    TfLiteTensor* output = GetOutput(context, node, 0);

    int num_dims = NumDimensions(input);

    TfLiteIntArray* output_size = TfLiteIntArrayCreate(num_dims);
    for(int i=0; i<num_dims; i++){
        output_size->data[i] = input->dims->data[i];
    }

    return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node){
    //group normalization invoke;
    using namespace tflite;

    const TfLiteTensor* input = GetInput(context, node, kDataInputTensor);
    const TfLiteTensor* gamma = GetInput(context, node,  kDataGammaTensor);
    const TfLiteTensor* beta = GetInput(context, node, kDataBetaTensor);
    const TfLiteTensor* groups_ = GetInput(context, node, 2);
    const TfLiteTensor* epsilon_ = GetInput(context, node, 7);
    TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

    float* input_data = input->data.f;
    float* output_data = output->data.f;
    const int* groups_data = GetTensorData<int32_t>(groups_);
    float* epsilon_data = epsilon_->data.f;

    const RuntimeShape input_shape = GetTensorShape(input);

    const int batches = input->dims->data[0];
    const int height = input->dims->data[1];
    const int width = input->dims->data[2];
    const int channels = input->dims->data[3];

    const int groups = groups_data[0];
    const float epsilon = epsilon_data[0];

    float* gamma_data = gamma->data.f;
    float* beta_data = beta->data.f;

    const int group_size = channels/groups;

    for(int b = 0; b<batches; ++b){
       for(int g=0; g<groups; ++g){
            float sum = 0.f;
            float sum_squared = 0.f;
            for(int c=0; c<group_size; ++c){
                for(int h = 0; h<height; ++h){
                    for(int w=0; w<width; ++w){
                        const auto input_offset = 
                            Offset(input_shape, b, h, w, g*group_size + c);
                        sum+= input_data[input_offset];
                        sum_squared += 
                            input_data[input_offset]*input_data[input_offset];
                    }
                }
            }
            float mean = sum / (1.0f*(height*width*group_size));
            float square_mean = sum_squared / (1.0f*(height*width*group_size));
            float stddev = sqrt(square_mean - mean*mean + epsilon);

            for(int c=0; c<group_size; ++c){
                for(int h = 0; h<height; ++h){
                    for(int w=0; w<width; ++w){
                        const auto input_offset = 
                            Offset(input_shape, b, h, w, g*group_size + c);
                        output_data[input_offset]= 
                            ((input_data[input_offset] - mean)/stddev)*
                            gamma_data[group_size*g+c] + beta_data[group_size*g+c];
                    }
                }
            }
        }
    }

    return kTfLiteOk;
}

}

TfLiteRegistration* Register_GROUP_NORMALIZATION(){
    static TfLiteRegistration r = { nullptr, nullptr,
        GroupNormalization::Prepare, GroupNormalization::Invoke};
    return &r;
}

}
} 
}   