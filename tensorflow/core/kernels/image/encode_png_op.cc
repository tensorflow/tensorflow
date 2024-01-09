/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/image_ops.cc

#include <memory>
#include <vector>

#define EIGEN_USE_THREADS

#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/png/png_io.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/overflow.h"
#include "tsl/platform/mutex.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

// Encode an image to a PNG stream
class EncodePngOp : public OpKernel {
 public:
  explicit EncodePngOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("compression", &compression_));
    OP_REQUIRES(context, -1 <= compression_ && compression_ <= 9,
                errors::InvalidArgument("compression should be in [-1,9], got ",
                                        compression_));

    DataType dt = context->input_type(0);
    OP_REQUIRES(context, dt == DataType::DT_UINT8 || dt == DataType::DT_UINT16,
                errors::InvalidArgument(
                    "image must have type uint8 or uint16, got ", dt));

    if (dt == DataType::DT_UINT8) {
      desired_channel_bits_ = 8;
    } else {
      desired_channel_bits_ = 16;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& image = context->input(0);
    OP_REQUIRES(context, image.dims() >= 3,
                errors::InvalidArgument("images must be at least rank 3",
                                        image.shape().DebugString()));
    OP_REQUIRES(context, image.NumElements() >= 0,
                errors::Internal("Invalid image provided."));
    OP_REQUIRES(
        context,
        FastBoundsCheck(image.NumElements(), std::numeric_limits<int32>::max()),
        errors::InvalidArgument("image cannot have >= int32 max elements"));

    const int batch_dims = image.dims() - 3;
    const int32_t height = static_cast<int32>(image.dim_size(batch_dims));
    const int32_t width = static_cast<int32>(image.dim_size(batch_dims + 1));
    const int32_t channels = static_cast<int32>(image.dim_size(batch_dims + 2));

    // In some cases, we pass width*channels*2 to png.
    const int32_t max_row_width = std::numeric_limits<int32>::max() / 2;

    OP_REQUIRES(context, FastBoundsCheck(width * channels, max_row_width),
                errors::InvalidArgument("image too wide to encode"));

    OP_REQUIRES(context, channels >= 1 && channels <= 4,
                errors::InvalidArgument(
                    "image must have 1, 2, 3, or 4 channels, got ", channels));

    // Encode image to png string
    Tensor* output = nullptr;
    TensorShape out_shape;
    int64_t num_batches = 1;
    for (int i = 0; i < batch_dims; ++i) {
      OP_REQUIRES_OK(context, out_shape.AddDimWithStatus(image.dim_size(i)));
      num_batches = MultiplyWithoutOverflow(num_batches, image.dim_size(i));
    }
    OP_REQUIRES(context, num_batches >= 0,
                errors::InvalidArgument(
                    "Invalid number of batches: ", num_batches,
                    ", input image shape: ", image.shape().DebugString()));

    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    const CPUDevice& device = context->template eigen_device<CPUDevice>();

    tsl::mutex bad_image_mu;
    std::vector<int64_t> bad_image_indices;
    tstring* output_data = output->flat<tstring>().data();

    const uint8_t* image_data = static_cast<uint8_t*>(image.data());
    const int64_t row_bytes = width * channels * desired_channel_bits_ / 8;
    const int64_t image_bytes = height * row_bytes;

    // The following cost is a rough estimate per image encoded.
    auto cost =
        Eigen::TensorOpCost(image_bytes,  // Bytes to load.
                            image_bytes,  // Dummy number of bytes to store.
                            image_bytes   // Dummy number of cycles.
        );
    device.parallelFor(num_batches, cost,
                       [image_data, row_bytes, image_bytes, height, width,
                        channels, desired_channel_bits = desired_channel_bits_,
                        compression = compression_, output_data, &bad_image_mu,
                        &bad_image_indices](int64_t start, int64_t end) {
                         for (int64_t i = start; i < end; ++i) {
                           bool success = png::WriteImageToBuffer(
                               image_data + i * image_bytes, width, height,
                               row_bytes, channels, desired_channel_bits,
                               compression, output_data + i, nullptr);
                           if (TF_PREDICT_FALSE(!success)) {
                             tsl::mutex_lock lock(bad_image_mu);
                             bad_image_indices.push_back(i);
                           }
                         }
                       });

    OP_REQUIRES(
        context, bad_image_indices.empty(),
        errors::Internal(
            "PNG encoding failed at the following flattened batch indices: ",
            absl::StrJoin(bad_image_indices, ", ")));
  }

 private:
  int compression_;
  int desired_channel_bits_;
};
REGISTER_KERNEL_BUILDER(Name("EncodePng").Device(DEVICE_CPU), EncodePngOp);

}  // namespace tensorflow
