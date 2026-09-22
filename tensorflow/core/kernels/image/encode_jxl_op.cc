/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <limits>
#include <vector>

#define EIGEN_USE_THREADS

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/tsl/platform/macros.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/jxl/jxl_io.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/util/overflow.h"
#include "tsl/platform/mutex.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

// Encode an image to a JPEG XL stream
class EncodeJxlOp : public OpKernel {
 public:
  explicit EncodeJxlOp(OpKernelConstruction* context) : OpKernel(context) {
    float quality = 95.0f;
    OP_REQUIRES_OK(context, context->GetAttr("quality", &quality));
    OP_REQUIRES(context, quality >= 0.0f && quality <= 100.0f,
                absl::InvalidArgumentError(absl::StrCat(
                    "quality should be in [0.0, 100.0], got ", quality)));
    // libjxl's own JPEG-style quality mapping: 100 -> 0.0 (lossless),
    // 95 -> 0.5 (default lossy), 90 -> 1.0 (visually lossless), 0 -> 25.0.
    distance_ = jxl::DistanceFromQuality(quality);

    OP_REQUIRES_OK(context, context->GetAttr("effort", &effort_));
    OP_REQUIRES(context, effort_ >= 1 && effort_ <= 9,
                absl::InvalidArgumentError(
                    absl::StrCat("effort should be in [1, 9], got ", effort_)));

    DataType dt = context->input_type(0);
    OP_REQUIRES(
        context,
        dt == DataType::DT_UINT8 || dt == DataType::DT_UINT16 ||
            dt == DataType::DT_HALF || dt == DataType::DT_FLOAT,
        absl::InvalidArgumentError(absl::StrCat(
            "image must have type uint8, uint16, half, or float, got ", dt)));

    if (dt == DataType::DT_UINT8) {
      desired_channel_bits_ = 8;
      is_float_ = false;
    } else if (dt == DataType::DT_UINT16) {
      desired_channel_bits_ = 16;
      is_float_ = false;
    } else if (dt == DataType::DT_HALF) {
      desired_channel_bits_ = 16;
      is_float_ = true;
    } else {
      desired_channel_bits_ = 32;
      is_float_ = true;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& image = context->input(0);
    OP_REQUIRES(context, image.dims() >= 3,
                absl::InvalidArgumentError(
                    absl::StrCat("images must be at least rank 3, got shape: ",
                                 image.shape().DebugString())));
    OP_REQUIRES(context, image.NumElements() >= 0,
                absl::InternalError("Invalid image provided."));

    const int batch_dims = image.dims() - 3;
    const int64_t height = image.dim_size(batch_dims);
    const int64_t width = image.dim_size(batch_dims + 1);
    const int64_t channels = image.dim_size(batch_dims + 2);

    OP_REQUIRES(context, height > 0 && width > 0,
                absl::InvalidArgumentError(absl::StrCat(
                    "image height and width must be > 0, got shape ",
                    image.shape().DebugString())));

    OP_REQUIRES(context, channels == 1 || channels == 3 || channels == 4,
                absl::InvalidArgumentError(absl::StrCat(
                    "image must have 1, 3, or 4 channels, got ", channels)));

    // libjxl addresses a single image using 32-bit sizes, so it is the
    // per-image element count that must fit in an int32, not the size of the
    // whole batch. Each image in the batch is located with 64-bit arithmetic
    // below, so a batch may exceed int32 max elements in total.
    const int64_t image_pixels = MultiplyWithoutOverflow(height, width);
    const int64_t elements_per_image =
        MultiplyWithoutOverflow(image_pixels, channels);
    OP_REQUIRES(
        context,
        elements_per_image > 0 &&
            FastBoundsCheck(elements_per_image,
                            std::numeric_limits<int32_t>::max()),
        absl::InvalidArgumentError(absl::StrCat(
            "a single image cannot have >= int32 max elements, got ",
            elements_per_image, " for shape ", image.shape().DebugString())));

    const int64_t row_width = MultiplyWithoutOverflow(width, channels);
    const int64_t max_row_width = std::numeric_limits<int32_t>::max() / 2;
    OP_REQUIRES(context,
                row_width > 0 && FastBoundsCheck(row_width, max_row_width),
                absl::InvalidArgumentError("image too wide to encode"));

    // Encode image to JPEG XL string
    Tensor* output = nullptr;
    TensorShape out_shape;
    int64_t num_batches = 1;
    for (int i = 0; i < batch_dims; ++i) {
      OP_REQUIRES_OK(context, out_shape.AddDimWithStatus(image.dim_size(i)));
      num_batches = MultiplyWithoutOverflow(num_batches, image.dim_size(i));
    }
    OP_REQUIRES(context, num_batches >= 0,
                absl::InvalidArgumentError(absl::StrCat(
                    "Invalid number of batches: ", num_batches,
                    ", input image shape: ", image.shape().DebugString())));

    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    const CPUDevice& device = context->template eigen_device<CPUDevice>();

    tsl::mutex bad_image_mu;
    std::vector<int64_t> bad_image_indices;
    tstring* output_data = output->flat<tstring>().data();

    const uint8_t* image_data = static_cast<const uint8_t*>(image.data());
    const int64_t bytes_per_sample = desired_channel_bits_ / 8;
    const int64_t image_bytes = elements_per_image * bytes_per_sample;

    auto cost = Eigen::TensorOpCost(image_bytes, image_bytes, image_bytes * 10);
    // Safe to narrow: `elements_per_image` was checked to fit in an int32, so
    // each individual dimension does too.
    const int height_i = static_cast<int>(height);
    const int width_i = static_cast<int>(width);
    const int channels_i = static_cast<int>(channels);
    device.parallelFor(
        num_batches, cost,
        [image_data, image_bytes, height_i, width_i, channels_i,
         desired_channel_bits = desired_channel_bits_, is_float = is_float_,
         distance = distance_, effort = effort_, output_data, &bad_image_mu,
         &bad_image_indices](int64_t start, int64_t end) {
          for (int64_t i = start; i < end; ++i) {
            bool success = jxl::WriteImageToBuffer(
                image_data + i * image_bytes, width_i, height_i, channels_i,
                desired_channel_bits, distance, effort, output_data + i,
                is_float);
            if (TF_PREDICT_FALSE(!success)) {
              tsl::mutex_lock lock(bad_image_mu);
              bad_image_indices.push_back(i);
            }
          }
        });

    OP_REQUIRES(context, bad_image_indices.empty(),
                absl::InternalError(absl::StrCat(
                    "JPEG XL encoding failed at flattened batch indices: ",
                    absl::StrJoin(bad_image_indices, ", "))));
  }

 private:
  float distance_ = 0.5f;
  int effort_ = 7;
  int desired_channel_bits_ = 8;
  bool is_float_ = false;
};

REGISTER_KERNEL_BUILDER(Name("EncodeJxl").Device(DEVICE_CPU), EncodeJxlOp);

}  // namespace tensorflow
