/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/image_ops.cc.

#include <cmath>
#include <cstdint>
#include <string>
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/image/crop_and_resize_op.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

enum InterpolationMethod {
  BILINEAR = 0,
  NEAREST = 1,
};

template <typename T>
__global__ void CropAndResizeKernel(
    const int32 nthreads, const T* __restrict__ image_ptr,
    const float* __restrict__ boxes_ptr, const int32* __restrict__ box_ind_ptr,
    int num_boxes, int batch, int image_height, int image_width,
    int crop_height, int crop_width, int depth, int method_id,
    float extrapolation_value, float* __restrict__ crops_ptr) {
  // Precompute some constants outside the loop.
  //
  // The compiler doesn't hoist them outside the loop because of the
  // `continue`'s in the loop -- it isn't sure that these are always reached.
  const float height_scale_factor =
      crop_height > 1 ? (image_height - 1) / static_cast<float>(crop_height - 1)
                      : 0;
  const float width_scale_factor =
      crop_width > 1 ? (image_width - 1) / static_cast<float>(crop_width - 1)
                     : 0;
  // These may seem trivial, but the implicit int->float conversion here is
  // actually expensive.
  float image_height_minus_one = image_height - 1;
  float image_width_minus_one = image_width - 1;

  GPU_1D_KERNEL_LOOP(out_idx, nthreads) {
    // out_idx = d + depth * (w + crop_width * (h + crop_height * b))
    uint32_t idx = out_idx;
    const uint32_t d = idx % depth;
    idx /= depth;
    const uint32_t x = idx % crop_width;
    idx /= crop_width;
    const uint32_t y = idx % crop_height;
    const uint32_t b = idx / crop_height;

    const float y1 = boxes_ptr[b * 4];
    const float x1 = boxes_ptr[b * 4 + 1];
    const float y2 = boxes_ptr[b * 4 + 2];
    const float x2 = boxes_ptr[b * 4 + 3];

    const int32 b_in = box_ind_ptr[b];
    if (b_in < 0 || b_in >= batch) {
      continue;
    }

    const float in_y =
        (crop_height > 1)
            ? y1 * image_height_minus_one + y * (y2 - y1) * height_scale_factor
            : 0.5f * (y1 + y2) * image_height_minus_one;
    if (in_y < 0 || in_y > image_height_minus_one) {
      crops_ptr[out_idx] = extrapolation_value;
      continue;
    }

    const float in_x =
        (crop_width > 1)
            ? x1 * image_width_minus_one + x * (x2 - x1) * width_scale_factor
            : 0.5f * (x1 + x2) * image_width_minus_one;
    if (in_x < 0 || in_x > image_width_minus_one) {
      crops_ptr[out_idx] = extrapolation_value;
      continue;
    }

    if (method_id == BILINEAR) {
      const int top_y_index = floorf(in_y);
      const int bottom_y_index = ceilf(in_y);
      const float y_lerp = in_y - top_y_index;

      const int left_x_index = floorf(in_x);
      const int right_x_index = ceilf(in_x);
      const float x_lerp = in_x - left_x_index;

      const float top_left(static_cast<float>(
          image_ptr[((b_in * image_height + top_y_index) * image_width +
                     left_x_index) *
                        depth +
                    d]));
      const float top_right(static_cast<float>(
          image_ptr[((b_in * image_height + top_y_index) * image_width +
                     right_x_index) *
                        depth +
                    d]));
      const float bottom_left(static_cast<float>(
          image_ptr[((b_in * image_height + bottom_y_index) * image_width +
                     left_x_index) *
                        depth +
                    d]));
      const float bottom_right(static_cast<float>(
          image_ptr[((b_in * image_height + bottom_y_index) * image_width +
                     right_x_index) *
                        depth +
                    d]));
      const float top = top_left + (top_right - top_left) * x_lerp;
      const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
      crops_ptr[out_idx] = top + (bottom - top) * y_lerp;
    } else {  // method_id == kMethodNearestId
      const int closest_x_index = roundf(in_x);
      const int closest_y_index = roundf(in_y);
      crops_ptr[out_idx] = static_cast<float>(
          image_ptr[((b_in * image_height + closest_y_index) * image_width +
                     closest_x_index) *
                        depth +
                    d]);
    }
  }
}

template <typename T>
__global__ void CropAndResizeBackpropImageKernel(
    const int32 nthreads, const float* __restrict__ grads_ptr,
    const float* __restrict__ boxes_ptr, const int32* __restrict__ box_ind_ptr,
    int num_boxes, int batch, int image_height, int image_width,
    int crop_height, int crop_width, int depth, T* __restrict__ grads_image_ptr,
    int method_id) {
  GPU_1D_KERNEL_LOOP(out_idx, nthreads) {
    // out_idx = d + depth * (w + crop_width * (h + crop_height * b))
    int idx = out_idx;
    const int d = idx % depth;
    idx /= depth;
    const int x = idx % crop_width;
    idx /= crop_width;
    const int y = idx % crop_height;
    const int b = idx / crop_height;

    const float y1 = boxes_ptr[b * 4];
    const float x1 = boxes_ptr[b * 4 + 1];
    const float y2 = boxes_ptr[b * 4 + 2];
    const float x2 = boxes_ptr[b * 4 + 3];

    const int32 b_in = box_ind_ptr[b];
    if (b_in < 0 || b_in >= batch) {
      continue;
    }

    const float height_scale =
        (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
                          : 0;
    const float width_scale =
        (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;

    const float in_y = (crop_height > 1)
                           ? y1 * (image_height - 1) + y * height_scale
                           : 0.5f * (y1 + y2) * (image_height - 1);
    if (in_y < 0 || in_y > image_height - 1) {
      continue;
    }

    const float in_x = (crop_width > 1)
                           ? x1 * (image_width - 1) + x * width_scale
                           : 0.5f * (x1 + x2) * (image_width - 1);
    if (in_x < 0 || in_x > image_width - 1) {
      continue;
    }

    if (method_id == BILINEAR) {
      const int top_y_index = floorf(in_y);
      const int bottom_y_index = ceilf(in_y);
      const float y_lerp = in_y - top_y_index;

      const int left_x_index = floorf(in_x);
      const int right_x_index = ceilf(in_x);
      const float x_lerp = in_x - left_x_index;

      const float dtop = (1 - y_lerp) * grads_ptr[out_idx];
      GpuAtomicAdd(grads_image_ptr +
                       ((b_in * image_height + top_y_index) * image_width +
                        left_x_index) *
                           depth +
                       d,
                   static_cast<T>((1 - x_lerp) * dtop));
      GpuAtomicAdd(grads_image_ptr +
                       ((b_in * image_height + top_y_index) * image_width +
                        right_x_index) *
                           depth +
                       d,
                   static_cast<T>(x_lerp * dtop));

      const float dbottom = y_lerp * grads_ptr[out_idx];
      GpuAtomicAdd(grads_image_ptr +
                       ((b_in * image_height + bottom_y_index) * image_width +
                        left_x_index) *
                           depth +
                       d,
                   static_cast<T>((1 - x_lerp) * dbottom));
      GpuAtomicAdd(grads_image_ptr +
                       ((b_in * image_height + bottom_y_index) * image_width +
                        right_x_index) *
                           depth +
                       d,
                   static_cast<T>(x_lerp * dbottom));
    } else {  // method_id == NEAREST
      const int closest_x_index = roundf(in_x);
      const int closest_y_index = roundf(in_y);
      GpuAtomicAdd(grads_image_ptr +
                       ((b_in * image_height + closest_y_index) * image_width +
                        closest_x_index) *
                           depth +
                       d,
                   static_cast<T>(grads_ptr[out_idx]));
    }
  }
}

template <typename T>
__global__ void CropAndResizeBackpropBoxesKernel(
    const int32 nthreads, const float* __restrict__ grads_ptr,
    const T* __restrict__ image_ptr, const float* __restrict__ boxes_ptr,
    const int32* __restrict__ box_ind_ptr, int num_boxes, int batch,
    int image_height, int image_width, int crop_height, int crop_width,
    int depth, float* __restrict__ grads_boxes_ptr) {
  GPU_1D_KERNEL_LOOP(out_idx, nthreads) {
    // out_idx = d + depth * (w + crop_width * (h + crop_height * b))
    int idx = out_idx;
    const int d = idx % depth;
    idx /= depth;
    const int x = idx % crop_width;
    idx /= crop_width;
    const int y = idx % crop_height;
    const int b = idx / crop_height;

    const float y1 = boxes_ptr[b * 4];
    const float x1 = boxes_ptr[b * 4 + 1];
    const float y2 = boxes_ptr[b * 4 + 2];
    const float x2 = boxes_ptr[b * 4 + 3];

    const int32 b_in = box_ind_ptr[b];
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

    const float height_scale = (crop_height > 1) ? (y2 - y1) * height_ratio : 0;
    const float width_scale = (crop_width > 1) ? (x2 - x1) * width_ratio : 0;

    const float in_y = (crop_height > 1)
                           ? y1 * (image_height - 1) + y * height_scale
                           : 0.5f * (y1 + y2) * (image_height - 1);
    if (in_y < 0 || in_y > image_height - 1) {
      continue;
    }

    const float in_x = (crop_width > 1)
                           ? x1 * (image_width - 1) + x * width_scale
                           : 0.5f * (x1 + x2) * (image_width - 1);
    if (in_x < 0 || in_x > image_width - 1) {
      continue;
    }

    const int top_y_index = floorf(in_y);
    const int bottom_y_index = ceilf(in_y);
    const float y_lerp = in_y - top_y_index;

    const int left_x_index = floorf(in_x);
    const int right_x_index = ceilf(in_x);
    const float x_lerp = in_x - left_x_index;

    const float top_left(static_cast<float>(
        image_ptr[((b_in * image_height + top_y_index) * image_width +
                   left_x_index) *
                      depth +
                  d]));
    const float top_right(static_cast<float>(
        image_ptr[((b_in * image_height + top_y_index) * image_width +
                   right_x_index) *
                      depth +
                  d]));
    const float bottom_left(static_cast<float>(
        image_ptr[((b_in * image_height + bottom_y_index) * image_width +
                   left_x_index) *
                      depth +
                  d]));
    const float bottom_right(static_cast<float>(
        image_ptr[((b_in * image_height + bottom_y_index) * image_width +
                   right_x_index) *
                      depth +
                  d]));

    // Compute the image gradient.
    float image_grad_y = (1 - x_lerp) * (bottom_left - top_left) +
                         x_lerp * (bottom_right - top_right);
    float image_grad_x = (1 - y_lerp) * (top_right - top_left) +
                         y_lerp * (bottom_right - bottom_left);
    // Modulate the image gradient with the incoming gradient.
    const float top_grad = grads_ptr[out_idx];
    image_grad_y *= top_grad;
    image_grad_x *= top_grad;

    float dy1, dy2;
    if (crop_height > 1) {
      dy1 = image_grad_y * (image_height - 1 - y * height_ratio);
      dy2 = image_grad_y * (y * height_ratio);
    } else {
      dy1 = image_grad_y * 0.5f * (image_height - 1);
      dy2 = image_grad_y * 0.5f * (image_height - 1);
    }

    float dx1, dx2;
    if (crop_width > 1) {
      dx1 = image_grad_x * (image_width - 1 - x * width_ratio);
      dx2 = image_grad_x * (x * width_ratio);
    } else {
      dx1 = image_grad_x * 0.5f * (image_width - 1);
      dx2 = image_grad_x * 0.5f * (image_width - 1);
    }

    GpuAtomicAdd(grads_boxes_ptr + b * 4 + 0, dy1);
    GpuAtomicAdd(grads_boxes_ptr + b * 4 + 1, dx1);
    GpuAtomicAdd(grads_boxes_ptr + b * 4 + 2, dy2);
    GpuAtomicAdd(grads_boxes_ptr + b * 4 + 3, dx2);
  }
}

}  // namespace

namespace functor {

template <typename T>
struct CropAndResize<GPUDevice, T> {
  bool operator()(const OpKernelContext* context,
                  typename TTypes<T, 4>::ConstTensor image,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_ind,
                  const std::string& method_name, float extrapolation_value,
                  typename TTypes<float, 4>::Tensor crops) {
    const int batch = image.dimension(0);
    const int image_height = image.dimension(1);
    const int image_width = image.dimension(2);

    const int num_boxes = crops.dimension(0);
    const int crop_height = crops.dimension(1);
    const int crop_width = crops.dimension(2);
    const int depth = crops.dimension(3);

    const int total_count = num_boxes * crop_height * crop_width * depth;
    const GPUDevice& d = context->eigen_device<GPUDevice>();

    InterpolationMethod method = BILINEAR;
    if (method_name == "nearest") {
      method = NEAREST;
    }

    if (total_count > 0) {
      GpuLaunchConfig config = GetGpuLaunchConfig(total_count, d);
      TF_CHECK_OK(GpuLaunchKernel(
          CropAndResizeKernel<T>, config.block_count, config.thread_per_block,
          0, d.stream(), config.virtual_thread_count, image.data(),
          boxes.data(), box_ind.data(), num_boxes, batch, image_height,
          image_width, crop_height, crop_width, depth, method,
          extrapolation_value, crops.data()));
    }
    return d.ok();
  }
};

template <typename T>
struct CropAndResizeBackpropImage<GPUDevice, T> {
  bool operator()(const OpKernelContext* context,
                  typename TTypes<float, 4>::ConstTensor grads,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_ind,
                  typename TTypes<T, 4>::Tensor grads_image,
                  const std::string& method_name) {
    const int batch = grads_image.dimension(0);
    const int image_height = grads_image.dimension(1);
    const int image_width = grads_image.dimension(2);

    const int num_boxes = grads.dimension(0);
    const int crop_height = grads.dimension(1);
    const int crop_width = grads.dimension(2);
    const int depth = grads.dimension(3);
    const GPUDevice& d = context->eigen_device<GPUDevice>();

    int total_count;
    GpuLaunchConfig config;

    // Initialize grads_image with all zeros.
    total_count = batch * image_height * image_width * depth;
    if (total_count > 0) {
      config = GetGpuLaunchConfig(total_count, d);
      TF_CHECK_OK(GpuLaunchKernel(
          SetZero<T>, config.block_count, config.thread_per_block, 0,
          d.stream(), config.virtual_thread_count, grads_image.data()));
    }

    // Configure interpolation method.
    InterpolationMethod method = BILINEAR;
    if (method_name == "nearest") {
      method = NEAREST;
    }

    // Accumulate.
    total_count = num_boxes * crop_height * crop_width * depth;
    if (total_count > 0) {
      config = GetGpuLaunchConfig(total_count, d);
      TF_CHECK_OK(GpuLaunchKernel(
          CropAndResizeBackpropImageKernel<T>, config.block_count,
          config.thread_per_block, 0, d.stream(), config.virtual_thread_count,
          grads.data(), boxes.data(), box_ind.data(), num_boxes, batch,
          image_height, image_width, crop_height, crop_width, depth,
          grads_image.data(), method));
    }
    return d.ok();
  }
};

template <typename T>
struct CropAndResizeBackpropBoxes<GPUDevice, T> {
  bool operator()(const GPUDevice& d,
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

    int total_count;
    GpuLaunchConfig config;

    // Initialize grads_boxes with all zeros.
    total_count = num_boxes * 4;
    if (total_count > 0) {
      config = GetGpuLaunchConfig(total_count, d);
      TF_CHECK_OK(GpuLaunchKernel(
          SetZero<float>, config.block_count, config.thread_per_block, 0,
          d.stream(), config.virtual_thread_count, grads_boxes.data()));
    }

    // Accumulate.
    total_count = num_boxes * crop_height * crop_width * depth;
    if (total_count > 0) {
      config = GetGpuLaunchConfig(total_count, d);
      TF_CHECK_OK(GpuLaunchKernel(
          CropAndResizeBackpropBoxesKernel<T>, config.block_count,
          config.thread_per_block, 0, d.stream(), config.virtual_thread_count,
          grads.data(), image.data(), boxes.data(), box_ind.data(), num_boxes,
          batch, image_height, image_width, crop_height, crop_width, depth,
          grads_boxes.data()));
    }
    return d.ok();
  }
};

#define DEFINE_GPU_SPECS(T)                                 \
  template struct CropAndResize<GPUDevice, T>;              \
  template struct CropAndResizeBackpropImage<GPUDevice, T>; \
  template struct CropAndResizeBackpropBoxes<GPUDevice, T>;

TF_CALL_half(DEFINE_GPU_SPECS);
TF_CALL_float(DEFINE_GPU_SPECS);
TF_CALL_double(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS

template struct CheckValidBoxIndexHelper<GPUDevice>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
