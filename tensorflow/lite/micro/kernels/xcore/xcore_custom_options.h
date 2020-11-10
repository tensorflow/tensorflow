#ifndef XCORE_CUSTOM_OPTIONS_H_
#define XCORE_CUSTOM_OPTIONS_H_

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_planning.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

void parse_custom_options(TfLiteContext *context, const char *buffer,
                          size_t length, ExecutionPlan *plan);

void parse_custom_options(TfLiteContext *context, const char *buffer,
                          size_t length, PoolingParams &pooling_params,
                          ExecutionPlan *plan = nullptr);

void parse_custom_options(TfLiteContext *context, const char *buffer,
                          size_t length, Conv2DParams &conv2d_params,
                          ExecutionPlan *plan = nullptr);

void parse_custom_options(TfLiteContext *context, const char *buffer,
                          size_t length, int32_t *stride_h = nullptr,
                          int32_t *stride_w = nullptr,
                          int32_t *pool_h = nullptr, int32_t *pool_w = nullptr,
                          int32_t *K_w = nullptr, Conv2DPadding *pad = nullptr,
                          ExecutionPlan *plan = nullptr);

uint32_t get_named_uint32_custom_option(TfLiteContext *context,
                                        const char *buffer, const size_t length,
                                        const std::string named_key);

flexbuffers::Vector get_named_uint32_custom_option_vector(
    TfLiteContext *context, const char *buffer, const size_t length,
    const std::string named_key);

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_CUSTOM_OPTIONS_H_
