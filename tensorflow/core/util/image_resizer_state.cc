/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/image_resizer_state.h"

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

// ImageResizerStateImpl -------------------------------------------------------
class ImageResizerStateImpl {
 public:
  ImageResizerStateImpl(bool align_corners, bool half_pixel_centers,
                        bool need_swap)
      : align_corners_(align_corners),
        half_pixel_centers_(half_pixel_centers),
        need_swap_(need_swap) {}
  ImageResizerStateImpl(const ImageResizerStateImpl&) = delete;
  ImageResizerStateImpl& operator=(const ImageResizerStateImpl&) = delete;
  virtual ~ImageResizerStateImpl() = default;

  void ValidateAndCreateOutput(OpKernelContext* context,
                               const TensorShape& input_shape,
                               const TensorShape& output_shape) {
    ValidateAndCalculateOutputSize(context, input_shape, output_shape);
    if (!context->status().ok()) return;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &data_.output));
  }

  // ValidateAndCalculateOutputSize checks the bounds on the input tensors
  // and requested size, sets up some of the resizing state such as the
  // height_scale and width_scale.
  // If any of these operations fails, it sets an error status in
  // the context, which the caller must check.
  void ValidateAndCalculateOutputSize(OpKernelContext* context,
                                      const TensorShape& input_shape,
                                      const TensorShape& output_shape) {
    ValidateAndCalculateOutputSizeImpl(context,
                                       need_swap_ ? output_shape : input_shape,
                                       need_swap_ ? input_shape : output_shape);
  }

  const ImageResizerStateData& data() const { return data_; }

 private:
  // Implementation of the date validation and calculation.
  void ValidateAndCalculateOutputSizeImpl(
      OpKernelContext* context, const TensorShape& original_shape,
      const TensorShape& after_resize_shape) {
    // Verify `half_pixel_centers_` and `align_corners_`.
    OP_REQUIRES(
        context,
        !half_pixel_centers_ || (half_pixel_centers_ && !align_corners_),
        errors::InvalidArgument("If half_pixel_centers is True, "
                                "align_corners must be False."));

    // Verify `original_shape` and `after_resize_shape`.
    OP_REQUIRES(context, original_shape.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        original_shape.DebugString()));
    OP_REQUIRES(context, after_resize_shape.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        after_resize_shape.DebugString()));

    // Verify the batch size.
    data_.batch_size = original_shape.dim_size(0);
    OP_REQUIRES(
        context, data_.batch_size == after_resize_shape.dim_size(0),
        errors::InvalidArgument(
            "input and output shapes should share the same batch size"));

    // Verify the channel size.
    data_.channels = original_shape.dim_size(3);
    OP_REQUIRES(
        context, data_.channels == after_resize_shape.dim_size(3),
        errors::InvalidArgument(
            "input and output shapes should share the same channel size"));
    OP_REQUIRES(
        context, data_.channels > 0,
        errors::InvalidArgument("image must have at least one channel"));

    // Verify the original size.
    const int64 original_height = original_shape.dim_size(1),
                original_width = original_shape.dim_size(2);
    OP_REQUIRES(
        context,
        FastBoundsCheck(original_height, std::numeric_limits<int32>::max()) &&
            FastBoundsCheck(original_width, std::numeric_limits<int32>::max()),
        errors::InvalidArgument(
            "input sizes must not be geater than max int32"));
    OP_REQUIRES(
        context, original_height > 0 && original_width > 0,
        errors::InvalidArgument("input image must be of non-zero size"));

    // Verify the size after resizing.
    const int64 after_resize_height = after_resize_shape.dim_size(1),
                after_resize_width = after_resize_shape.dim_size(2);
    OP_REQUIRES(context, after_resize_height > 0 && after_resize_width > 0,
                errors::InvalidArgument("output dimensions must be positive"));

    // Calculate resizing scales for height and width respectively.
    data_.height_scale = CalculateResizeScale(
        original_height, after_resize_height, align_corners_);
    data_.width_scale = CalculateResizeScale(original_width, after_resize_width,
                                             align_corners_);

    // Guard against overflows
    OP_REQUIRES(context,
                ceilf((after_resize_height - 1) * data_.height_scale) <=
                    static_cast<float>(std::numeric_limits<int64>::max()),
                errors::InvalidArgument(
                    "input image height scale would cause an overflow"));
    OP_REQUIRES(context,
                ceilf((after_resize_width - 1) * data_.width_scale) <=
                    static_cast<float>(INT_MAX),
                errors::InvalidArgument(
                    "input image width scale would cause an overflow"));

    if (need_swap_) {
      data_.in_height = after_resize_height;
      data_.in_width = after_resize_width;
      data_.out_height = original_height;
      data_.out_width = original_width;
    } else {
      data_.in_height = original_height;
      data_.in_width = original_width;
      data_.out_height = after_resize_height;
      data_.out_width = after_resize_width;
    }
  }

  ImageResizerStateData data_;

  const bool align_corners_;
  const bool half_pixel_centers_;

  const bool need_swap_;
};

// ImageResizerStateBase -------------------------------------------------------

ImageResizerStateBase::ImageResizerStateBase(bool align_corners,
                                             bool half_pixel_centers,
                                             bool need_swap)
    : impl_(std::make_unique<ImageResizerStateImpl>(
          align_corners, half_pixel_centers, need_swap)) {}

ImageResizerStateBase::~ImageResizerStateBase() = default;

void ImageResizerStateBase::ValidateAndCreateOutput(
    OpKernelContext* context, const TensorShape& input_shape,
    const TensorShape& output_shape) {
  return impl_->ValidateAndCreateOutput(context, input_shape, output_shape);
}

const ImageResizerStateData& ImageResizerStateBase::GetData() const {
  return impl_->data();
}

// ImageResizerState------------------------------------------------------------

ImageResizerState::ImageResizerState(bool align_corners,
                                     bool half_pixel_centers)
    : ImageResizerStateBase(align_corners, half_pixel_centers,
                            /*need_swap=*/false) {}
ImageResizerState::~ImageResizerState() = default;

// ImageResizerGradientState ---------------------------------------------------

ImageResizerGradientState::ImageResizerGradientState(bool align_corners,
                                                     bool half_pixel_centers)
    : ImageResizerStateBase(align_corners, half_pixel_centers,
                            /*need_swap=*/true) {}
ImageResizerGradientState::~ImageResizerGradientState() = default;

}  // namespace tensorflow