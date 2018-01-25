#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_SSD_OPS_H
#define TENSORFLOW_CONTRIB_LITE_KERNELS_SSD_OPS_H

#include <vector>

#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"

namespace tflite {
namespace ops {
namespace custom {

namespace ssd {

// Copied over from tensorflow/contrib/lite/kernels/svdf.cc
void* Init(TfLiteContext* context,
           const char* buffer, size_t length);

void Free(TfLiteContext* context, void* buffer);

// Bounding Box Postprocessing
TfLiteStatus PostprocessBoxesPrepare(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus PostprocessBoxesEval(TfLiteContext* context, TfLiteNode* node);


// Probability Postprocessing
TfLiteStatus PostprocessProbsPrepare(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus PostprocessProbsEval(TfLiteContext* context, TfLiteNode* node);

// Non-Max Suppression
TfLiteStatus NonMaxSuppressionPrepare(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus NonMaxSuppressionEval(TfLiteContext* context, TfLiteNode* node);

} // namespace ssd

TfLiteRegistration* Register_SSDPostprocessBoxes();

TfLiteRegistration* Register_SSDPostprocessProbs();

TfLiteRegistration* Register_SSDNonMaxSuppression();

} // namespace custom
} // namespace ops
} // namespace tflite

#endif //TENSORFLOW_CONTRIB_LITE_KERNELS_SSD_OPS_H
