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

#include <functional>
#include <string>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/platform/cuda.h"
#include "tensorflow/core/platform/stream_executor.h"

using stream_executor::cuda::ScopedActivateExecutorContext;
#endif  // GOOGLE_CUDA

namespace tensorflow {
namespace {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
using Callback = std::function<void()>;

static inline Status ParseAndCheckBoxSizes(const Tensor& boxes,
                                           const Tensor& box_index,
                                           int* num_boxes) {
  if (boxes.NumElements() == 0 && box_index.NumElements() == 0) {
    *num_boxes = 0;
    return Status::OK();
  }
  // The shape of 'boxes' is [num_boxes, 4].
  if (boxes.dims() != 2) {
    return errors::InvalidArgument("boxes must be 2-D",
                                   boxes.shape().DebugString());
  }
  *num_boxes = boxes.dim_size(0);
  if (boxes.dim_size(1) != 4) {
    return errors::InvalidArgument("boxes must have 4 columns");
  }
  // The shape of 'box_index' is [num_boxes].
  if (box_index.dims() != 1) {
    return errors::InvalidArgument("box_index must be 1-D",
                                   box_index.shape().DebugString());
  }
  if (box_index.dim_size(0) != *num_boxes) {
    return errors::InvalidArgument("box_index has incompatible shape");
  }
  return Status::OK();
}

// Conditionally calls the compute callback if all values in box_index are in
// [0, batch_size) then calls done.
template <typename Device>
inline void RunIfBoxIndexIsValid(
    OpKernelContext* context, typename TTypes<int32, 1>::ConstTensor box_index,
    int batch_size, const Callback& compute, const Callback& done);

// Specialization of CheckValidBoxIndex for a CPUDevice.
template <>
inline void RunIfBoxIndexIsValid<CPUDevice>(
    OpKernelContext* context, typename TTypes<int32, 1>::ConstTensor box_index,
    int batch_size, const Callback& compute, const Callback& done) {
  const int num_boxes = box_index.dimension(0);
  for (int b = 0; b < num_boxes; ++b) {
    OP_REQUIRES_ASYNC(
        context, FastBoundsCheck(box_index(b), batch_size),
        errors::OutOfRange("box_index has values outside [0, batch_size)"),
        done);
  }
  if (compute) {
    compute();
  }
  if (done) {
    done();
  }
}

}  // namespace

template <typename Device, typename T>
class CropAndResizeOp : public AsyncOpKernel {
 public:
  explicit CropAndResizeOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("method", &method_));
    OP_REQUIRES(context, method_ == "bilinear" || method_ == "nearest",
                errors::InvalidArgument(
                    "method must be 'bilinear' or 'nearest'", method_));
    OP_REQUIRES_OK(context, context->GetAttr("extrapolation_value",
                                             &extrapolation_value_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    // The shape of 'image' is [batch_size, image_height, image_width,
    // channels].
    const Tensor& image = context->input(0);
    // The shape of 'boxes' is [num_boxes, 4].
    const Tensor& boxes = context->input(1);
    // The shape of 'box_index' is [num_boxes].
    const Tensor& box_index = context->input(2);
    // The shape of 'crop_size' is [2].
    const Tensor& crop_size = context->input(3);

    // Validate inputs dimensions.
    OP_REQUIRES_ASYNC(context, image.dims() == 4,
                      errors::InvalidArgument("input image must be 4-D",
                                              image.shape().DebugString()),
                      done);
    const int batch_size = image.dim_size(0);
    const int image_height = image.dim_size(1);
    const int image_width = image.dim_size(2);
    const int depth = image.dim_size(3);
    OP_REQUIRES_ASYNC(
        context, image_height > 0 && image_width > 0,
        errors::InvalidArgument("image dimensions must be positive"), done);
    int num_boxes = 0;
    OP_REQUIRES_OK_ASYNC(
        context, ParseAndCheckBoxSizes(boxes, box_index, &num_boxes), done);

    OP_REQUIRES_ASYNC(context, crop_size.dims() == 1,
                      errors::InvalidArgument("crop_size must be 1-D",
                                              crop_size.shape().DebugString()),
                      done);
    OP_REQUIRES_ASYNC(
        context, crop_size.dim_size(0) == 2,
        errors::InvalidArgument("crop_size must have two elements",
                                crop_size.shape().DebugString()),
        done);

    // Copy and validate crop sizes.
    auto crop_size_vec = crop_size.vec<int32>();
    const int crop_height = internal::SubtleMustCopy(crop_size_vec(0));
    const int crop_width = internal::SubtleMustCopy(crop_size_vec(1));
    OP_REQUIRES_ASYNC(
        context, crop_height > 0 && crop_width > 0,
        errors::InvalidArgument("crop dimensions must be positive"), done);

    // Allocate output tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_output(
            0, TensorShape({num_boxes, crop_height, crop_width, depth}),
            &output),
        done);

    auto compute_callback = [this, context, output]() {
      const Tensor& image = context->input(0);
      const Tensor& boxes = context->input(1);
      const Tensor& box_index = context->input(2);
      const bool status =
          (method_ == "bilinear")
              ? functor::CropAndResize<Device, T, true>()(
                    context, image.tensor<T, 4>(), boxes.tensor<float, 2>(),
                    box_index.tensor<int32, 1>(), extrapolation_value_,
                    output->tensor<float, 4>())
              : functor::CropAndResize<Device, T, false>()(
                    context, image.tensor<T, 4>(), boxes.tensor<float, 2>(),
                    box_index.tensor<int32, 1>(), extrapolation_value_,
                    output->tensor<float, 4>());

      if (!status) {
        context->SetStatus(
            errors::Internal("Failed launch CropAndResizeKernel."));
      }
    };

    RunIfBoxIndexIsValid<Device>(context, box_index.tensor<int32, 1>(),
                                 batch_size, std::move(compute_callback),
                                 std::move(done));
  }

 private:
  float extrapolation_value_;
  string method_;
};

// Partial specialization of CropAndResize functor for a CPUDevice.
namespace functor {
template <typename T, bool isBilinear>
struct CropAndResize<CPUDevice, T, isBilinear> {
  bool operator()(const OpKernelContext* context,
                  typename TTypes<T, 4>::ConstTensor image,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_index,
                  float extrapolation_value,
                  typename TTypes<float, 4>::Tensor crops) {
    const int batch_size = image.dimension(0);
    const int image_height = image.dimension(1);
    const int image_width = image.dimension(2);

    const int num_boxes = crops.dimension(0);
    const int crop_height = crops.dimension(1);
    const int crop_width = crops.dimension(2);
    const int depth = crops.dimension(3);

    // Sharding across boxes.
    auto CropAndResizePerBox = [&](int start_box, int limit_box) {
      for (int b = start_box; b < limit_box; ++b) {
        const float y1 = boxes(b, 0);
        const float x1 = boxes(b, 1);
        const float y2 = boxes(b, 2);
        const float x2 = boxes(b, 3);

        const int32 b_in = box_index(b);
        if (!FastBoundsCheck(b_in, batch_size)) {
          continue;
        }

        const float height_scale =
            (crop_height > 1)
                ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
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
          if (isBilinear) {
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
                const float top_left(static_cast<float>(
                    image(b_in, top_y_index, left_x_index, d)));
                const float top_right(static_cast<float>(
                    image(b_in, top_y_index, right_x_index, d)));
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
          } else {  // method == "nearest"
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
              const int closest_x_index = roundf(in_x);
              const int closest_y_index = roundf(in_y);
              for (int d = 0; d < depth; ++d) {
                crops(b, y, x, d) = static_cast<float>(
                    image(b_in, closest_y_index, closest_x_index, d));
              }
            }
          }
        }
      }
    };

    // A rough estimation of the cost for each cropped box.
    double cost_per_pixel =
        depth * (Eigen::TensorOpCost::AddCost<float>() * 6 +
                 Eigen::TensorOpCost::MulCost<float>() * 3 +
                 Eigen::TensorOpCost::CastCost<T, float>() * 4) +
        (Eigen::TensorOpCost::AddCost<float>() * 2 +
         Eigen::TensorOpCost::AddCost<float>() * 3);
    if (!isBilinear) {
      cost_per_pixel = depth * Eigen::TensorOpCost::CastCost<T, float>() +
                       Eigen::TensorOpCost::AddCost<float>() * 4 +
                       Eigen::TensorOpCost::MulCost<float>() * 4;
    }
    const double cost_per_box = crop_height * crop_width * cost_per_pixel;

    const DeviceBase::CpuWorkerThreads& worker_threads =
        *(context->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, num_boxes,
          cost_per_box, CropAndResizePerBox);

    return true;
  }
};

}  // namespace functor

template <typename Device, typename T>
class CropAndResizeGradImageOp : public AsyncOpKernel {
 public:
  explicit CropAndResizeGradImageOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("method", &method_));
    OP_REQUIRES(context, method_ == "bilinear" || method_ == "nearest",
                errors::InvalidArgument(
                    "method must be 'bilinear' or 'nearest'", method_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    // The shape of 'grads' is [num_boxes, crop_height, crop_width, depth].
    const Tensor& grads = context->input(0);
    // The shape of 'boxes' is [num_boxes, 4].
    const Tensor& boxes = context->input(1);
    // The shape of 'box_index' is [num_boxes].
    const Tensor& box_index = context->input(2);
    // The shape of 'image_size' is [4].
    const Tensor& image_size = context->input(3);

    // Validate input shapes.
    OP_REQUIRES_ASYNC(context, grads.dims() == 4,
                      errors::InvalidArgument("grads image must be 4-D",
                                              grads.shape().DebugString()),
                      done);
    const int crop_height = grads.dim_size(1);
    const int crop_width = grads.dim_size(2);
    OP_REQUIRES_ASYNC(
        context, crop_height > 0 && crop_width > 0,
        errors::InvalidArgument("grads dimensions must be positive"), done);
    int num_boxes = 0;
    OP_REQUIRES_OK_ASYNC(
        context, ParseAndCheckBoxSizes(boxes, box_index, &num_boxes), done);
    OP_REQUIRES_ASYNC(
        context, grads.dim_size(0) == num_boxes,
        errors::InvalidArgument("boxes and grads have incompatible shape"),
        done);

    OP_REQUIRES_ASYNC(context, image_size.dims() == 1,
                      errors::InvalidArgument("image_size must be 1-D",
                                              image_size.shape().DebugString()),
                      done);
    OP_REQUIRES_ASYNC(context, image_size.dim_size(0) == 4,
                      errors::InvalidArgument("image_size must have 4 elements",
                                              image_size.shape().DebugString()),
                      done);
    auto image_size_vec = image_size.vec<int32>();
    const int batch_size = internal::SubtleMustCopy(image_size_vec(0));
    const int image_height = internal::SubtleMustCopy(image_size_vec(1));
    const int image_width = internal::SubtleMustCopy(image_size_vec(2));
    const int depth = internal::SubtleMustCopy(image_size_vec(3));
    OP_REQUIRES_ASYNC(
        context, image_height > 0 && image_width > 0,
        errors::InvalidArgument("image dimensions must be positive"), done);
    OP_REQUIRES_ASYNC(
        context, grads.dim_size(3) == depth,
        errors::InvalidArgument("image_size and grads are incompatible"), done);

    // Allocate output tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_output(
            0, TensorShape({batch_size, image_height, image_width, depth}),
            &output),
        done);

    auto compute_callback = [this, context, output]() {
      const Tensor& grads = context->input(0);
      const Tensor& boxes = context->input(1);
      const Tensor& box_index = context->input(2);
      const bool status =
          (method_ == "bilinear")
              ? functor::CropAndResizeBackpropImage<Device, T, true>()(
                    context, grads.tensor<float, 4>(), boxes.tensor<float, 2>(),
                    box_index.tensor<int32, 1>(), output->tensor<T, 4>())
              : functor::CropAndResizeBackpropImage<Device, T, false>()(
                    context, grads.tensor<float, 4>(), boxes.tensor<float, 2>(),
                    box_index.tensor<int32, 1>(), output->tensor<T, 4>());

      if (!status) {
        context->SetStatus(errors::Internal(
            "Failed launch CropAndResizeBackpropImage kernel."));
      }
    };

    RunIfBoxIndexIsValid<Device>(context, box_index.tensor<int32, 1>(),
                                 batch_size, std::move(compute_callback),
                                 std::move(done));
  }

 private:
  string method_;
};

// Partial specialization of CropAndResizeBackpropImage functor for a CPUDevice.
namespace functor {
// Split two methods' code to avoid branch mis-prediction.
// By template isBilinear variable.
template <typename T, bool isBilinear>
struct CropAndResizeBackpropImage<CPUDevice, T, isBilinear> {
  bool operator()(const OpKernelContext* context,
                  typename TTypes<float, 4>::ConstTensor grads,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_index,
                  typename TTypes<T, 4>::Tensor grads_image) {
    const int batch_size = grads_image.dimension(0);
    const int image_height = grads_image.dimension(1);
    const int image_width = grads_image.dimension(2);

    const int num_boxes = grads.dimension(0);
    const int crop_height = grads.dimension(1);
    const int crop_width = grads.dimension(2);
    const int depth = grads.dimension(3);

    grads_image.setZero();

    // Parallelize the loop using Eigen thread pool.
    // Assign ranges of boxes into different threads.
    auto CropAndResizeBackImgPerBox = [&](int start_box, int limit_box) {
      for (int b = start_box; b < limit_box; ++b) {
        const float y1 = boxes(b, 0);
        const float x1 = boxes(b, 1);
        const float y2 = boxes(b, 2);
        const float x2 = boxes(b, 3);

        const int32 b_in = box_index(b);
        if (!FastBoundsCheck(b_in, batch_size)) {
          continue;
        }

        const float height_scale =
            (crop_height > 1)
                ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
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
          const int closest_y_index = roundf(in_y);
          const int top_y_index = floorf(in_y);
          const int bottom_y_index = ceilf(in_y);
          const float y_lerp = in_y - top_y_index;
          const float one_y_lerp = 1 - y_lerp;

          for (int x = 0; x < crop_width; ++x) {
            const float in_x = (crop_width > 1)
                                   ? x1 * (image_width - 1) + x * width_scale
                                   : 0.5 * (x1 + x2) * (image_width - 1);
            if (in_x < 0 || in_x > image_width - 1) {
              continue;
            }
            const int closest_x_index = roundf(in_x);

            if (isBilinear) {
              const int left_x_index = floorf(in_x);
              const int right_x_index = ceilf(in_x);
              const float x_lerp = in_x - left_x_index;
              const float one_x_lerp = 1 - x_lerp;

              for (int d = 0; d < depth; ++d) {
                const float dtop = one_y_lerp * grads(b, y, x, d);
                grads_image(b_in, top_y_index, left_x_index, d) +=
                    static_cast<T>(one_x_lerp * dtop);
                grads_image(b_in, top_y_index, right_x_index, d) +=
                    static_cast<T>(x_lerp * dtop);
                const float dbottom = y_lerp * grads(b, y, x, d);
                grads_image(b_in, bottom_y_index, left_x_index, d) +=
                    static_cast<T>(one_x_lerp * dbottom);
                grads_image(b_in, bottom_y_index, right_x_index, d) +=
                    static_cast<T>(x_lerp * dbottom);
              }
            } else {  // method_name == "nearest"
              for (int d = 0; d < depth; ++d) {
                grads_image(b_in, closest_y_index, closest_x_index, d) +=
                    static_cast<T>(grads(b, y, x, d));
              }
            }
          }
        }
      }
    };

    // A rough estimation of the cost for each cropped box.
    // Including calculation cost in the depth loop and pixel loop.
    const double cost_per_pixel =
        (isBilinear
             ? depth * (Eigen::TensorOpCost::AddCost<float>() * 4 +
                        Eigen::TensorOpCost::MulCost<float>() * 4 +
                        Eigen::TensorOpCost::CastCost<T, float>() * 4)
             : depth * (Eigen::TensorOpCost::AddCost<float>() +
                        Eigen::TensorOpCost::CastCost<T, float>())) +
        (Eigen::TensorOpCost::AddCost<float>() * 2 +
         Eigen::TensorOpCost::AddCost<float>() * 3);
    const double cost_per_box = crop_height * crop_width * cost_per_pixel;

    const DeviceBase::CpuWorkerThreads& worker_threads =
        *(context->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, num_boxes,
          cost_per_box, CropAndResizeBackImgPerBox);
    return true;
  }
};

}  // namespace functor

template <typename Device, typename T>
class CropAndResizeGradBoxesOp : public AsyncOpKernel {
 public:
  explicit CropAndResizeGradBoxesOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    string method;
    OP_REQUIRES_OK(context, context->GetAttr("method", &method));
    OP_REQUIRES(context, method == "bilinear",
                errors::InvalidArgument("method must be 'bilinear'", method));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    // The shape of 'grads' is [num_boxes, crop_height, crop_width, depth].
    const Tensor& grads = context->input(0);
    // The shape of 'boxes' is [num_boxes, 4].
    const Tensor& boxes = context->input(2);
    // The shape of 'box_index' is [num_boxes].
    const Tensor& box_index = context->input(3);
    // The shape of 'image' is [batch_size, image_height, image_width, depth].
    const Tensor& image = context->input(1);

    // Validate input shapes.
    OP_REQUIRES_ASYNC(context, grads.dims() == 4,
                      errors::InvalidArgument("grads image must be 4-D",
                                              grads.shape().DebugString()),
                      done);
    const int crop_height = grads.dim_size(1);
    const int crop_width = grads.dim_size(2);
    const int depth = grads.dim_size(3);
    OP_REQUIRES_ASYNC(
        context, crop_height > 0 && crop_width > 0,
        errors::InvalidArgument("grads dimensions must be positive"), done);

    OP_REQUIRES_ASYNC(context, image.dims() == 4,
                      errors::InvalidArgument("input image must be 4-D",
                                              image.shape().DebugString()),
                      done);
    const int batch_size = image.dim_size(0);
    const int image_height = image.dim_size(1);
    const int image_width = image.dim_size(2);
    OP_REQUIRES_ASYNC(
        context, image_height > 0 && image_width > 0,
        errors::InvalidArgument("image dimensions must be positive"), done);
    OP_REQUIRES_ASYNC(context, image.dim_size(3) == depth,
                      errors::InvalidArgument("image, grads depth differ"),
                      done);

    int num_boxes = 0;
    OP_REQUIRES_OK_ASYNC(
        context, ParseAndCheckBoxSizes(boxes, box_index, &num_boxes), done);

    OP_REQUIRES_ASYNC(
        context, grads.dim_size(0) == num_boxes,
        errors::InvalidArgument("boxes and grads have incompatible shape"),
        done);

    // Allocate output tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(context, context->allocate_output(
                                      0, TensorShape({num_boxes, 4}), &output),
                         done);

    auto compute_callback = [context, output]() {
      const Tensor& grads = context->input(0);
      const Tensor& image = context->input(1);
      const Tensor& boxes = context->input(2);
      const Tensor& box_index = context->input(3);
      const bool status = functor::CropAndResizeBackpropBoxes<Device, T>()(
          context->eigen_device<Device>(), grads.tensor<float, 4>(),
          image.tensor<T, 4>(), boxes.tensor<float, 2>(),
          box_index.tensor<int32, 1>(), output->tensor<float, 2>());
      if (!status) {
        context->SetStatus(errors::Internal(
            "Failed launch CropAndResizeBackpropBoxes kernel."));
      }
    };

    RunIfBoxIndexIsValid<Device>(context, box_index.tensor<int32, 1>(),
                                 batch_size, std::move(compute_callback),
                                 std::move(done));
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
                  typename TTypes<int32, 1>::ConstTensor box_index,
                  typename TTypes<float, 2>::Tensor grads_boxes) {
    const int batch_size = image.dimension(0);
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

      const int32 b_in = box_index(b);
      if (!FastBoundsCheck(b_in, batch_size)) {
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

#define REGISTER_KERNEL(T)                                \
  REGISTER_KERNEL_BUILDER(Name("CropAndResize")           \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<T>("T")     \
                              .HostMemory("crop_size"),   \
                          CropAndResizeOp<CPUDevice, T>); \
                                                          \
  REGISTER_KERNEL_BUILDER(Name("CropAndResizeGradBoxes")  \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<T>("T"),    \
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

// Forward declaration of the CheckValidBoxIndexHelper specialization for GPU.
namespace functor {
template <>
void CheckValidBoxIndexHelper<GPUDevice>::operator()(
    const GPUDevice& d, typename TTypes<int32, 1>::ConstTensor box_index,
    int batch_size, typename TTypes<bool, 0>::Tensor isvalid);
extern template struct CheckValidBoxIndexHelper<GPUDevice>;
}  // namespace functor

namespace {

// Specialization of CheckValidBoxIndex for a GPUDevice.
template <>
inline void RunIfBoxIndexIsValid<GPUDevice>(
    OpKernelContext* context, typename TTypes<int32, 1>::ConstTensor box_index,
    int batch_size, const Callback& compute, const Callback& done) {
  const int num_boxes = box_index.dimension(0);
  if (num_boxes == 0) {
    compute();
    done();
    return;
  }

  Tensor isvalid_dev_tensor;
  OP_REQUIRES_OK_ASYNC(
      context, context->allocate_temp(DataTypeToEnum<bool>::value,
                                      TensorShape({}), &isvalid_dev_tensor),
      done);
  typename TTypes<bool, 0>::Tensor isvalid_dev =
      isvalid_dev_tensor.tensor<bool, 0>();

  // Run the actual box check on the device.
  functor::CheckValidBoxIndexHelper<GPUDevice>()(
      context->eigen_device<GPUDevice>(), box_index, batch_size, isvalid_dev);

  // Copy the result back to the host.
  auto* stream = context->op_device_context()->stream();
  OP_REQUIRES_ASYNC(context, stream,
                    errors::Internal("No GPU stream available."), done);
  Tensor isvalid_host_tensor;
  // Use pinned host memory on the host to avoid unnecessary
  // synchronization.
  AllocatorAttributes alloc_attr;
  alloc_attr.set_on_host(true);
  alloc_attr.set_gpu_compatible(true);
  OP_REQUIRES_OK_ASYNC(
      context,
      context->allocate_temp(DataTypeToEnum<bool>::value, TensorShape({}),
                             &isvalid_host_tensor, alloc_attr),
      done);
  se::DeviceMemoryBase wrapped(isvalid_dev.data(), sizeof(bool));
  const bool status =
      stream
          ->ThenMemcpy(
              isvalid_host_tensor.scalar<bool>().data() /* destination */,
              wrapped /* source */, sizeof(bool))
          .ok();
  OP_REQUIRES_ASYNC(
      context, status,
      errors::Internal("Failed to launch copy of isvalid from device to host."),
      done);

  // We capture both temporary tensors to prevent them from being deallocated
  // when ComputeAsync returns and before the closure runs.
  TensorReference isvalid_dev_ref(isvalid_dev_tensor);
  auto wrapped_callback = [context, isvalid_host_tensor, isvalid_dev_ref,
                           compute, done]() {
    auto stream = context->op_device_context()->stream();
    ScopedActivateExecutorContext scoped_activation{stream->parent()};
    const bool isvalid = isvalid_host_tensor.scalar<bool>()();
    isvalid_dev_ref.Unref();
    OP_REQUIRES_ASYNC(
        context, isvalid,
        errors::OutOfRange("box_index has values outside [0, batch_size)"),
        done);
    compute();
    done();
  };

  context->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
      stream, wrapped_callback);
}

}  // namespace

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
