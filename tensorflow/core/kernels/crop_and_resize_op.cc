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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/crop_and_resize_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

static inline void ParseAndCheckBoxSizes(OpKernelContext* context,
                                         const Tensor& boxes,
                                         const Tensor& box_ind,
                                         int* num_boxes) {
  if (boxes.NumElements() == 0 && box_ind.NumElements() == 0) {
    *num_boxes = 0;
    return;
  }
  // The shape of 'boxes' is [num_boxes, 4].
  OP_REQUIRES(context, boxes.dims() == 2,
              errors::InvalidArgument("boxes must be 2-D",
                                      boxes.shape().DebugString()));
  *num_boxes = boxes.dim_size(0);
  OP_REQUIRES(context, boxes.dim_size(1) == 4,
              errors::InvalidArgument("boxes must have 4 columns"));

  // The shape of 'box_ind' is [num_boxes].
  OP_REQUIRES(context, box_ind.dims() == 1,
              errors::InvalidArgument("box_ind must be 1-D",
                                      box_ind.shape().DebugString()));
  OP_REQUIRES(context, box_ind.dim_size(0) == *num_boxes,
              errors::InvalidArgument("box_ind has incompatible shape"));
}

// Verifies that all values in box_ind are in [0, batch).
template <typename Device>
inline void CheckValidBoxInd(
    OpKernelContext* context,
    typename TTypes<int32, 1>::ConstTensor box_ind_data, int batch);

template <typename Device, typename T>
class CropAndResizeOp : public OpKernel {
 public:
  explicit CropAndResizeOp(OpKernelConstruction* context) : OpKernel(context) {
    string method;
    OP_REQUIRES_OK(context, context->GetAttr("method", &method));
    OP_REQUIRES(context, method == "bilinear",
                errors::InvalidArgument("method must be 'bilinear'", method));
    OP_REQUIRES_OK(context, context->GetAttr("extrapolation_value",
                                             &extrapolation_value_));
  }

  void Compute(OpKernelContext* context) override {
    // The shape of 'image' is [batch, image_height, image_width, channels].
    const Tensor& image = context->input(0);
    OP_REQUIRES(context, image.dims() == 4,
                errors::InvalidArgument("input image must be 4-D",
                                        image.shape().DebugString()));

    const int batch = image.dim_size(0);
    const int image_height = image.dim_size(1);
    const int image_width = image.dim_size(2);
    const int depth = image.dim_size(3);
    OP_REQUIRES(context, image_height > 0 && image_width > 0,
                errors::InvalidArgument("image dimensions must be positive"));

    // The shape of 'boxes' is [num_boxes, 4].
    const Tensor& boxes = context->input(1);

    // The shape of 'box_ind' is [num_boxes].
    const Tensor& box_ind = context->input(2);

    int num_boxes = 0;
    ParseAndCheckBoxSizes(context, boxes, box_ind, &num_boxes);

    // The shape of 'crop_size' is [2].
    const Tensor& crop_size = context->input(3);

    OP_REQUIRES(context, crop_size.dims() == 1,
                errors::InvalidArgument("crop_size must be 1-D",
                                        crop_size.shape().DebugString()));
    OP_REQUIRES(context, crop_size.dim_size(0) == 2,
                errors::InvalidArgument("crop_size must have two elements",
                                        crop_size.shape().DebugString()));

    auto crop_size_vec = crop_size.vec<int32>();
    const int crop_height = internal::SubtleMustCopy(crop_size_vec(0));
    const int crop_width = internal::SubtleMustCopy(crop_size_vec(1));
    OP_REQUIRES(context, crop_height > 0 && crop_width > 0,
                errors::InvalidArgument("crop dimensions must be positive"));

    // Allocate output tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({num_boxes, crop_height, crop_width, depth}),
            &output));

    typename TTypes<T, 4>::ConstTensor image_data = image.tensor<T, 4>();
    typename TTypes<float, 2>::ConstTensor boxes_data =
        boxes.tensor<float, 2>();
    typename TTypes<int32, 1>::ConstTensor box_ind_data =
        box_ind.tensor<int32, 1>();
    typename TTypes<float, 4>::Tensor crops_data = output->tensor<float, 4>();

    CheckValidBoxInd<Device>(context, box_ind_data, batch);

    bool status = functor::CropAndResize<Device, T>()(
        context->eigen_device<Device>(), image_data, boxes_data, box_ind_data,
        extrapolation_value_, crops_data);
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launch CropAndResizeKernel."));
    }
  }

 private:
  float extrapolation_value_;
};

// Partial specialization of CropAndResize functor for a CPUDevice.
namespace functor {
template <typename T>
struct CropAndResize<CPUDevice, T> {
  bool operator()(const CPUDevice& d, typename TTypes<T, 4>::ConstTensor image,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_ind,
                  float extrapolation_value,
                  typename TTypes<float, 4>::Tensor crops) {
    const int batch = image.dimension(0);
    const int image_height = image.dimension(1);
    const int image_width = image.dimension(2);

    const int num_boxes = crops.dimension(0);
    const int crop_height = crops.dimension(1);
    const int crop_width = crops.dimension(2);
    const int depth = crops.dimension(3);

    for (int b = 0; b < num_boxes; ++b) {
      const float y1 = boxes(b, 0);
      const float x1 = boxes(b, 1);
      const float y2 = boxes(b, 2);
      const float x2 = boxes(b, 3);

      const int32 b_in = box_ind(b);
      if (b_in < 0 || b_in >= batch) {
        continue;
      }

      const float height_scale =
          (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
                            : 0;
      const float width_scale =
          (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1)
                           : 0;

      for (int y = 0; y < crop_height; ++y) {
        const float in_y = (crop_height > 1)
                               ? y1 * (image_height - 1) + y * height_scale
                               : 0.5 * (y1 + y2) * (image_height - 1);
        if (in_y < 0 || in_y > image_height - 1) {
          for (int x = 0; x < crop_width; ++x) {
            for (int d = 0; d < depth; ++d) {
              crops(b, y, x, d) = extrapolation_value;
            }
          }
          continue;
        }
        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        for (int x = 0; x < crop_width; ++x) {
          const float in_x = (crop_width > 1)
                                 ? x1 * (image_width - 1) + x * width_scale
                                 : 0.5 * (x1 + x2) * (image_width - 1);
          if (in_x < 0 || in_x > image_width - 1) {
            for (int d = 0; d < depth; ++d) {
              crops(b, y, x, d) = extrapolation_value;
            }
            continue;
          }
          const int left_x_index = floorf(in_x);
          const int right_x_index = ceilf(in_x);
          const float x_lerp = in_x - left_x_index;

          for (int d = 0; d < depth; ++d) {
            const float top_left(
                static_cast<float>(image(b_in, top_y_index, left_x_index, d)));
            const float top_right(
                static_cast<float>(image(b_in, top_y_index, right_x_index, d)));
            const float bottom_left(static_cast<float>(
                image(b_in, bottom_y_index, left_x_index, d)));
            const float bottom_right(static_cast<float>(
                image(b_in, bottom_y_index, right_x_index, d)));
            const float top = top_left + (top_right - top_left) * x_lerp;
            const float bottom =
                bottom_left + (bottom_right - bottom_left) * x_lerp;
            crops(b, y, x, d) = top + (bottom - top) * y_lerp;
          }
        }
      }
    }
    return true;
  }
};
}  // namespace functor

template <typename Device, typename T>
class CropAndResizeGradImageOp : public OpKernel {
 public:
  explicit CropAndResizeGradImageOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string method;
    OP_REQUIRES_OK(context, context->GetAttr("method", &method));
    OP_REQUIRES(context, method == "bilinear",
                errors::InvalidArgument("method must be 'bilinear'", method));
  }

  void Compute(OpKernelContext* context) override {
    // The shape of 'grads' is [num_boxes, crop_height, crop_width, depth].
    const Tensor& grads = context->input(0);

    OP_REQUIRES(context, grads.dims() == 4,
                errors::InvalidArgument("grads image must be 4-D",
                                        grads.shape().DebugString()));
    const int crop_height = grads.dim_size(1);
    const int crop_width = grads.dim_size(2);
    OP_REQUIRES(context, crop_height > 0 && crop_width > 0,
                errors::InvalidArgument("grads dimensions must be positive"));

    // The shape of 'boxes' is [num_boxes, 4].
    const Tensor& boxes = context->input(1);

    // The shape of 'box_ind' is [num_boxes].
    const Tensor& box_ind = context->input(2);

    int num_boxes = 0;
    ParseAndCheckBoxSizes(context, boxes, box_ind, &num_boxes);

    OP_REQUIRES(
        context, grads.dim_size(0) == num_boxes,
        errors::InvalidArgument("boxes and grads have incompatible shape"));

    // The shape of 'image_size' is [4].
    const Tensor& image_size = context->input(3);
    OP_REQUIRES(context, image_size.dims() == 1,
                errors::InvalidArgument("image_size must be 1-D",
                                        image_size.shape().DebugString()));
    OP_REQUIRES(context, image_size.dim_size(0) == 4,
                errors::InvalidArgument("image_size must have 4 elements",
                                        image_size.shape().DebugString()));

    auto image_size_vec = image_size.vec<int32>();
    const int batch = internal::SubtleMustCopy(image_size_vec(0));
    const int image_height = internal::SubtleMustCopy(image_size_vec(1));
    const int image_width = internal::SubtleMustCopy(image_size_vec(2));
    const int depth = internal::SubtleMustCopy(image_size_vec(3));

    OP_REQUIRES(context, image_height > 0 && image_width > 0,
                errors::InvalidArgument("image dimensions must be positive"));
    OP_REQUIRES(
        context, grads.dim_size(3) == depth,
        errors::InvalidArgument("image_size and grads are incompatible"));

    // Allocate output tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(
                     0, TensorShape({batch, image_height, image_width, depth}),
                     &output));

    typename TTypes<float, 4>::ConstTensor grads_data =
        grads.tensor<float, 4>();
    typename TTypes<float, 2>::ConstTensor boxes_data =
        boxes.tensor<float, 2>();
    typename TTypes<int32, 1>::ConstTensor box_ind_data =
        box_ind.tensor<int32, 1>();
    typename TTypes<T, 4>::Tensor output_data = output->tensor<T, 4>();

    CheckValidBoxInd<Device>(context, box_ind_data, batch);

    bool status = functor::CropAndResizeBackpropImage<Device, T>()(
        context->eigen_device<Device>(), grads_data, boxes_data, box_ind_data,
        output_data);
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launch CropAndResizeBackpropImageKernel."));
    }
  }
};

// Partial specialization of CropAndResizeBackpropImage functor for a CPUDevice.
namespace functor {
template <typename T>
struct CropAndResizeBackpropImage<CPUDevice, T> {
  bool operator()(const CPUDevice& d,
                  typename TTypes<float, 4>::ConstTensor grads,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_ind,
                  typename TTypes<T, 4>::Tensor grads_image) {
    const int batch = grads_image.dimension(0);
    const int image_height = grads_image.dimension(1);
    const int image_width = grads_image.dimension(2);

    const int num_boxes = grads.dimension(0);
    const int crop_height = grads.dimension(1);
    const int crop_width = grads.dimension(2);
    const int depth = grads.dimension(3);

    grads_image.setZero();

    for (int b = 0; b < num_boxes; ++b) {
      const float y1 = boxes(b, 0);
      const float x1 = boxes(b, 1);
      const float y2 = boxes(b, 2);
      const float x2 = boxes(b, 3);

      const int32 b_in = box_ind(b);
      if (b_in < 0 || b_in >= batch) {
        continue;
      }

      const float height_scale =
          (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
                            : 0;
      const float width_scale =
          (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1)
                           : 0;

      for (int y = 0; y < crop_height; ++y) {
        const float in_y = (crop_height > 1)
                               ? y1 * (image_height - 1) + y * height_scale
                               : 0.5 * (y1 + y2) * (image_height - 1);
        if (in_y < 0 || in_y > image_height - 1) {
          continue;
        }
        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        for (int x = 0; x < crop_width; ++x) {
          const float in_x = (crop_width > 1)
                                 ? x1 * (image_width - 1) + x * width_scale
                                 : 0.5 * (x1 + x2) * (image_width - 1);
          if (in_x < 0 || in_x > image_width - 1) {
            continue;
          }
          const int left_x_index = floorf(in_x);
          const int right_x_index = ceilf(in_x);
          const float x_lerp = in_x - left_x_index;

          for (int d = 0; d < depth; ++d) {
            const float dtop = (1 - y_lerp) * grads(b, y, x, d);
            grads_image(b_in, top_y_index, left_x_index, d) +=
                static_cast<T>((1 - x_lerp) * dtop);
            grads_image(b_in, top_y_index, right_x_index, d) +=
                static_cast<T>(x_lerp * dtop);
            const float dbottom = y_lerp * grads(b, y, x, d);
            grads_image(b_in, bottom_y_index, left_x_index, d) +=
                static_cast<T>((1 - x_lerp) * dbottom);
            grads_image(b_in, bottom_y_index, right_x_index, d) +=
                static_cast<T>(x_lerp * dbottom);
          }
        }
      }
    }
    return true;
  }
};
}  // namespace functor

template <typename Device, typename T>
class CropAndResizeGradBoxesOp : public OpKernel {
 public:
  explicit CropAndResizeGradBoxesOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string method;
    OP_REQUIRES_OK(context, context->GetAttr("method", &method));
    OP_REQUIRES(context, method == "bilinear",
                errors::InvalidArgument("method must be 'bilinear'", method));
  }

  void Compute(OpKernelContext* context) override {
    // The shape of 'grads' is [num_boxes, crop_height, crop_width, depth].
    const Tensor& grads = context->input(0);

    OP_REQUIRES(context, grads.dims() == 4,
                errors::InvalidArgument("grads image must be 4-D",
                                        grads.shape().DebugString()));

    const int crop_height = grads.dim_size(1);
    const int crop_width = grads.dim_size(2);
    const int depth = grads.dim_size(3);
    OP_REQUIRES(context, crop_height > 0 && crop_width > 0,
                errors::InvalidArgument("grads dimensions must be positive"));

    // The shape of 'image' is [batch, image_height, image_width, depth].
    const Tensor& image = context->input(1);
    OP_REQUIRES(context, image.dims() == 4,
                errors::InvalidArgument("input image must be 4-D",
                                        image.shape().DebugString()));

    const int batch = image.dim_size(0);
    const int image_height = image.dim_size(1);
    const int image_width = image.dim_size(2);
    OP_REQUIRES(context, image_height > 0 && image_width > 0,
                errors::InvalidArgument("image dimensions must be positive"));
    OP_REQUIRES(context, image.dim_size(3) == depth,
                errors::InvalidArgument("image, grads depth differ"));

    // The shape of 'boxes' is [num_boxes, 4].
    const Tensor& boxes = context->input(2);

    // The shape of 'box_ind' is [num_boxes].
    const Tensor& box_ind = context->input(3);

    int num_boxes = 0;
    ParseAndCheckBoxSizes(context, boxes, box_ind, &num_boxes);

    OP_REQUIRES(
        context, grads.dim_size(0) == num_boxes,
        errors::InvalidArgument("boxes and grads have incompatible shape"));

    // Allocate output tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({num_boxes, 4}), &output));

    typename TTypes<float, 4>::ConstTensor grads_data =
        grads.tensor<float, 4>();
    typename TTypes<T, 4>::ConstTensor image_data = image.tensor<T, 4>();
    typename TTypes<float, 2>::ConstTensor boxes_data =
        boxes.tensor<float, 2>();
    typename TTypes<int32, 1>::ConstTensor box_ind_data =
        box_ind.tensor<int32, 1>();
    typename TTypes<float, 2>::Tensor output_data = output->tensor<float, 2>();

    CheckValidBoxInd<Device>(context, box_ind_data, batch);

    bool status = functor::CropAndResizeBackpropBoxes<Device, T>()(
        context->eigen_device<Device>(), grads_data, image_data, boxes_data,
        box_ind_data, output_data);
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launch CropAndResizeBackpropBoxesKernel."));
    }
  }
};

// Partial specialization of CropAndResizeBackpropBoxes functor for a CPUDevice.
namespace functor {
template <typename T>
struct CropAndResizeBackpropBoxes<CPUDevice, T> {
  bool operator()(const CPUDevice& d,
                  typename TTypes<float, 4>::ConstTensor grads,
                  typename TTypes<T, 4>::ConstTensor image,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_ind,
                  typename TTypes<float, 2>::Tensor grads_boxes) {
    const int batch = image.dimension(0);
    const int image_height = image.dimension(1);
    const int image_width = image.dimension(2);

    const int num_boxes = grads.dimension(0);
    const int crop_height = grads.dimension(1);
    const int crop_width = grads.dimension(2);
    const int depth = grads.dimension(3);

    grads_boxes.setZero();

    for (int b = 0; b < num_boxes; ++b) {
      const float y1 = boxes(b, 0);
      const float x1 = boxes(b, 1);
      const float y2 = boxes(b, 2);
      const float x2 = boxes(b, 3);

      const int32 b_in = box_ind(b);
      if (b_in < 0 || b_in >= batch) {
        continue;
      }

      const float height_ratio =
          (crop_height > 1)
              ? static_cast<float>(image_height - 1) / (crop_height - 1)
              : 0;
      const float width_ratio =
          (crop_width > 1)
              ? static_cast<float>(image_width - 1) / (crop_width - 1)
              : 0;

      const float height_scale =
          (crop_height > 1) ? (y2 - y1) * height_ratio : 0;
      const float width_scale = (crop_width > 1) ? (x2 - x1) * width_ratio : 0;

      for (int y = 0; y < crop_height; ++y) {
        const float in_y = (crop_height > 1)
                               ? y1 * (image_height - 1) + y * height_scale
                               : 0.5 * (y1 + y2) * (image_height - 1);
        if (in_y < 0 || in_y > image_height - 1) {
          continue;
        }
        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        for (int x = 0; x < crop_width; ++x) {
          const float in_x = (crop_width > 1)
                                 ? x1 * (image_width - 1) + x * width_scale
                                 : 0.5 * (x1 + x2) * (image_width - 1);
          if (in_x < 0 || in_x > image_width - 1) {
            continue;
          }
          const int left_x_index = floorf(in_x);
          const int right_x_index = ceilf(in_x);
          const float x_lerp = in_x - left_x_index;

          for (int d = 0; d < depth; ++d) {
            const float top_left(
                static_cast<float>(image(b_in, top_y_index, left_x_index, d)));
            const float top_right(
                static_cast<float>(image(b_in, top_y_index, right_x_index, d)));
            const float bottom_left(static_cast<float>(
                image(b_in, bottom_y_index, left_x_index, d)));
            const float bottom_right(static_cast<float>(
                image(b_in, bottom_y_index, right_x_index, d)));
            // Compute the image gradient.
            float image_grad_y = (1 - x_lerp) * (bottom_left - top_left) +
                                 x_lerp * (bottom_right - top_right);
            float image_grad_x = (1 - y_lerp) * (top_right - top_left) +
                                 y_lerp * (bottom_right - bottom_left);
            // Modulate the image gradient with the incoming gradient.
            const float top_grad = grads(b, y, x, d);
            image_grad_y *= top_grad;
            image_grad_x *= top_grad;
            // dy1, dy2
            if (crop_height > 1) {
              grads_boxes(b, 0) +=
                  image_grad_y * (image_height - 1 - y * height_ratio);
              grads_boxes(b, 2) += image_grad_y * (y * height_ratio);
            } else {
              grads_boxes(b, 0) += image_grad_y * 0.5 * (image_height - 1);
              grads_boxes(b, 2) += image_grad_y * 0.5 * (image_height - 1);
            }
            // dx1, dx2
            if (crop_width > 1) {
              grads_boxes(b, 1) +=
                  image_grad_x * (image_width - 1 - x * width_ratio);
              grads_boxes(b, 3) += image_grad_x * (x * width_ratio);
            } else {
              grads_boxes(b, 1) += image_grad_x * 0.5 * (image_width - 1);
              grads_boxes(b, 3) += image_grad_x * 0.5 * (image_width - 1);
            }
          }
        }
      }
    }
    return true;
  }
};
}  // namespace functor

// Specialization of CheckValidBoxInd for a CPUDevice.
template <>
inline void CheckValidBoxInd<CPUDevice>(
    OpKernelContext* context, typename TTypes<int32, 1>::ConstTensor box_ind,
    int batch) {
  const int num_boxes = box_ind.dimension(0);
  for (int b = 0; b < num_boxes; ++b) {
    OP_REQUIRES(context, box_ind(b) >= 0 && box_ind(b) < batch,
                errors::OutOfRange("box_ind has values outside [0, batch)"));
  }
}

#define REGISTER_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("CropAndResize")                    \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<T>("T")              \
                              .HostMemory("crop_size"),            \
                          CropAndResizeOp<CPUDevice, T>);          \
                                                                   \
  REGISTER_KERNEL_BUILDER(Name("CropAndResizeGradBoxes")           \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<T>("T"),             \
                          CropAndResizeGradBoxesOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#define REGISTER_KERNEL(T)                               \
  REGISTER_KERNEL_BUILDER(Name("CropAndResizeGradImage") \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<T>("T")    \
                              .HostMemory("image_size"), \
                          CropAndResizeGradImageOp<CPUDevice, T>);

TF_CALL_half(REGISTER_KERNEL);
TF_CALL_float(REGISTER_KERNEL);
TF_CALL_double(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#if GOOGLE_CUDA

// Forward declaration of the CheckValidBoxIndHelper specialization for GPU.
namespace functor {
template <>
void CheckValidBoxIndHelper<GPUDevice>::operator()(
    const GPUDevice& d, typename TTypes<int32, 1>::ConstTensor box_ind,
    int batch, typename TTypes<bool, 0>::Tensor isvalid);
extern template struct CheckValidBoxIndHelper<GPUDevice>;
}  // namespace functor

// Specialization of CheckValidBoxInd for a GPUDevice.
template <>
inline void CheckValidBoxInd<GPUDevice>(
    OpKernelContext* context, typename TTypes<int32, 1>::ConstTensor box_ind,
    int batch) {
  const int num_boxes = box_ind.dimension(0);
  if (num_boxes == 0) {
    return;
  }
  Tensor isvalid_tensor;
  OP_REQUIRES_OK(context,
                 context->allocate_temp(DataTypeToEnum<bool>::value,
                                        TensorShape({}), &isvalid_tensor));

  typename TTypes<bool, 0>::Tensor isvalid = isvalid_tensor.tensor<bool, 0>();

  functor::CheckValidBoxIndHelper<GPUDevice>()(
      context->eigen_device<GPUDevice>(), box_ind, batch, isvalid);

  auto* stream = context->op_device_context()->stream();
  OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

  bool isvalid_host = false;
  perftools::gputools::DeviceMemoryBase isvalid_gpu(isvalid.data(),
                                                    sizeof(bool));
  stream->ThenMemcpy(&isvalid_host, isvalid_gpu, sizeof(bool));
  stream->BlockHostUntilDone();

  OP_REQUIRES(context, stream->ok(),
              errors::Internal("cudaMemcpy from device to host failed"));

  OP_REQUIRES(context, isvalid_host,
              errors::OutOfRange("box_ind has values outside [0, batch)"));
}

#define REGISTER_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("CropAndResize")                    \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("T")              \
                              .HostMemory("crop_size"),            \
                          CropAndResizeOp<GPUDevice, T>);          \
                                                                   \
  REGISTER_KERNEL_BUILDER(Name("CropAndResizeGradImage")           \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("T")              \
                              .HostMemory("image_size"),           \
                          CropAndResizeGradImageOp<GPUDevice, T>); \
                                                                   \
  REGISTER_KERNEL_BUILDER(Name("CropAndResizeGradBoxes")           \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("T"),             \
                          CropAndResizeGradBoxesOp<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
