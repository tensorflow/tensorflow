#ifndef TENSORFLOW_CORE_KERNELS_BATCHED_NON_MAX_SUPPRESSION_H_
#define TENSORFLOW_CORE_KERNELS_BATCHED_NON_MAX_SUPPRESSION_H_
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
Status DoNMSBatched(OpKernelContext* context, const Tensor& boxes,
                    const Tensor& scores, const Tensor& box_counts_tensor,
                    const int max_output_size, const float iou_threshold_val,
                    const float score_threshold, bool pad_to_max_output,
                    int* num_saved_outputs, int kernel);
}  // namespace tensorflow
#endif
#endif