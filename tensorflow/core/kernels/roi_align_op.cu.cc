/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// An example Op.

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <algorithm>
#include <vector>
#include "tensorflow/core/kernels/roi_align_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "third_party/cub/device/device_radix_sort.cuh"
#include "third_party/cub/device/device_segmented_radix_sort.cuh"
#include "third_party/cub/device/device_select.cuh"

#define CUDA_CHECK(result)                                    \
  do {                                                        \
    cudaError_t error(result);                                \
    CHECK(error == cudaSuccess) << cudaGetErrorString(error); \
  } while (0)

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
namespace {

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T bilinear_interpolate(
    const T* bottom_data, const int height, const int width, T y, T x,
    const int index, /* index for debug only*/ const T* lower_bound = nullptr,
    const T* upper_bound = nullptr, int chann = -1, bool debug = false) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }
  if (y <= 0) y = 0.;
  if (x <= 0) x = 0.;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  // if (debug && chann >= 0 && chann < 4) {
  //   int diff = bottom_data - lower_bound;
  //   printf(
  //       " BI y=%f x=%f yl=%d yh=%d xl=%d xh=%d w=%d h=%d lx=%f ly=%f v1=%f
  //       v2=%f v3=%f v4=%f c=%d index=%d " "offset%d %d %d %d\n", y, x, y_low,
  //       y_high, x_low, x_high, width, height, lx,ly, v1,v2,v3,v4 ,chann,
  //       index, diff + y_low * width + x_low, diff + y_low * width + x_high,
  //       diff + y_high * width + x_low, diff + y_high * width + x_high);
  // }
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC void bilinear_interpolate_gradient(
    const int height, const int width, T y, T x, T& w1, T& w2, T& w3, T& w4,
    int& x_low, int& x_high, int& y_low, int& y_high,
    const int index /* index for debug only*/, const int level = -1,
    int chann = -1, bool debug = false) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = bottom_data[y_low * width + x_low];
  // T v2 = bottom_data[y_low * width + x_high];
  // T v3 = bottom_data[y_high * width + x_low];
  // T v4 = bottom_data[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

// scale [y1,x1,y2,x2] boxes with level calculated with eq1 in FPN paper
// arXiv:1612.03144 and return levels and scaled boxes
template <typename T>
__global__ void Boxes2ScaledBoxesAndLevels(const CudaLaunchConfig config,
                                           const T* boxes, int min_level,
                                           int max_level, float canonical_scale,
                                           int canonical_level, int* levels,
                                           T* scaled_boxes, bool is_bw = false,
                                           bool debug = false) {
  CUDA_1D_KERNEL_LOOP(i, config.virtual_thread_count) {
    const T* box = boxes + i * 4;
    T* scaled_box = scaled_boxes + i * 4;
    T y1 = box[0];
    T x1 = box[1];
    T y2 = box[2];
    T x2 = box[3];
    T height = y2 - y1;
    T width = x2 - x1;
    T box_area_sqrt = sqrtf(width * height);
    int level =
        max(min_level,
            min((int)floorf(canonical_level +
                            __log2f(box_area_sqrt / canonical_scale + 1e-6f)),
                max_level));
    levels[i] = level - min_level;
    T level_scale = 1 << level;

    scaled_box[0] = y1 / level_scale;
    scaled_box[1] = x1 / level_scale;
    scaled_box[2] = height / level_scale;
    scaled_box[3] = width / level_scale;
  }
}

template <typename T>
__global__ void RoIAlignForward(
    const Cuda2DLaunchConfig nthreads, const T* bottom_data,
    const T spatial_scale, const int num_levels, const int channels,
    const int height, const int width, const int n_rois,
    const int pooled_height, const int pooled_width, const int sampling_ratio,
    const T* scaled_roi_boxes, const int32* levels, int roi_cols, T* top_data,
    bool debug = false) {
  CUDA_AXIS_KERNEL_LOOP(image_index, nthreads.virtual_thread_count.y, Y) {
    CUDA_AXIS_KERNEL_LOOP(index, nthreads.virtual_thread_count.x, X) {
      // CUDA_1D_KERNEL_LOOP(index, nthreads.virtual_thread_count) {
      // (n, c, ph, pw) is an element in the pooled output
      //  returns (b,n,c,h,w)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      // RoI could have 4 or 5 columns
      const T* offset_bottom_rois =
          scaled_roi_boxes + image_index * n_rois * roi_cols + n * roi_cols;
      T roi_start_w = offset_bottom_rois[1] * spatial_scale;
      T roi_start_h = offset_bottom_rois[0] * spatial_scale;

      // Force malformed ROIs to be 1x1
      T roi_width = Eigen::numext::maxi(offset_bottom_rois[3], (T)1.);
      T roi_height = Eigen::numext::maxi(offset_bottom_rois[2], (T)1.);
      T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
      int level = levels[image_index * n_rois + n];
      const T* offset_bottom_data =
          bottom_data + image_index * height * width * channels * num_levels +
          height * width * channels * level + c * height * width;
      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h = (sampling_ratio > 0)
                               ? sampling_ratio
                               : ceil(roi_height / pooled_height);  // e.g., = 2
      int roi_bin_grid_w = (sampling_ratio > 0)
                               ? sampling_ratio
                               : ceil(roi_width / pooled_width);

      // We do average (integral) pooling inside a bin
      const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

      T output_val = 0.;
      int level_height = (T)height / (T)(1 << level);
      int level_width = (T)width / (T)(1 << level);
      for (int iy = 0; iy < roi_bin_grid_h; iy++)  // e.g., iy = 0, 1
      {
        const T y = roi_start_h + ph * bin_size_h +
                    static_cast<T>(iy + .5f) * bin_size_h /
                        static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          const T x = roi_start_w + pw * bin_size_w +
                      static_cast<T>(ix + .5f) * bin_size_w /
                          static_cast<T>(roi_bin_grid_w);
          T val = bilinear_interpolate(offset_bottom_data, level_height,
                                       level_width, y, x, index, bottom_data,
                                       bottom_data + (5 * 256 * 256 * 256), c);
          output_val += val;
        }
      }
      output_val /= count;

      top_data[nthreads.virtual_thread_count.x * image_index + index] =
          output_val;
    }
  }
}
// Gradient of RoIAlign wrt feature inputs
template <typename T>
__global__ void RoIAlignBackwardFeature(
    const Cuda2DLaunchConfig nthreads,
    const T* inp_grads,  // grads
    const T spatial_scale, const int num_levels, const int channels,
    const int height, const int width, const int n_rois,
    const int pooled_height, const int pooled_width, const int sampling_ratio,
    const int roi_cols, const T* input_rois,
    int32* levels,  // scaled rois,  levels
    T* output_grads /* input_grad */, bool debug = false) {
  CUDA_AXIS_KERNEL_LOOP(image_index, nthreads.virtual_thread_count.y, Y) {
    CUDA_AXIS_KERNEL_LOOP(index, nthreads.virtual_thread_count.x, X) {
      // (n, c, ph, pw) is an element in the pooled output
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;
      const T* offset_input_rois =
          input_rois + image_index * n_rois * roi_cols + n * roi_cols;
      // Do not using rounding; this implementation detail is critical
      T roi_start_w = offset_input_rois[1] * spatial_scale;
      T roi_start_h = offset_input_rois[0] * spatial_scale;

      // Force malformed ROIs to be 1x1
      T roi_width = Eigen::numext::maxi(offset_input_rois[3], (T)1.);
      T roi_height = Eigen::numext::maxi(offset_input_rois[2], (T)1.);
      T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
      int level = levels[n];
      T* offset_output_grads =
          output_grads + image_index * height * width * channels * num_levels +
          height * width * channels * level + c * height * width;
      int inp_grad_offset =
          (image_index * n_rois * channels + n * channels + c) * pooled_height *
          pooled_width;
      const T* offset_inp_grads = inp_grads + inp_grad_offset;
      const T inp_grads_this_bin = offset_inp_grads[ph * pooled_width + pw];
      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h = (sampling_ratio > 0)
                               ? sampling_ratio
                               : ceil(roi_height / pooled_height);  // e.g., = 2
      int roi_bin_grid_w = (sampling_ratio > 0)
                               ? sampling_ratio
                               : ceil(roi_width / pooled_width);

      // We do average (integral) pooling inside a bin
      const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4
      int level_height = (T)height / (T)(1 << level);
      int level_width = (T)width / (T)(1 << level);

      for (int iy = 0; iy < roi_bin_grid_h; iy++)  // e.g., iy = 0, 1
      {
        const T y = roi_start_h + ph * bin_size_h +
                    static_cast<T>(iy + .5f) * bin_size_h /
                        static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          const T x = roi_start_w + pw * bin_size_w +
                      static_cast<T>(ix + .5f) * bin_size_w /
                          static_cast<T>(roi_bin_grid_w);

          T w1, w2, w3, w4;
          int x_low, x_high, y_low, y_high;

          bilinear_interpolate_gradient(level_height, level_width, y, x, w1, w2,
                                        w3, w4, x_low, x_high, y_low, y_high,
                                        index, level, c, debug);

          T g1 = inp_grads_this_bin * w1 / count;
          T g2 = inp_grads_this_bin * w2 / count;
          T g3 = inp_grads_this_bin * w3 / count;
          T g4 = inp_grads_this_bin * w4 / count;
          if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
            CudaAtomicAdd(offset_output_grads + y_low * width + x_low,
                          static_cast<T>(g1));
            CudaAtomicAdd(offset_output_grads + y_low * width + x_high,
                          static_cast<T>(g2));
            CudaAtomicAdd(offset_output_grads + y_high * width + x_low,
                          static_cast<T>(g3));
            CudaAtomicAdd(offset_output_grads + y_high * width + x_high,
                          static_cast<T>(g4));
          }  // if
        }    // ix
      }      // iy
    }        // CUDA_1D_KERNEL_LOOP
  }          // RoIAlignBackward
}

}  // namespace

class ROIAlignOp : public tensorflow::OpKernel {
 public:
  explicit ROIAlignOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("spatial_scale", &spatial_scale_));
    OP_REQUIRES_OK(context, context->GetAttr("pooled_height", &pooled_height_));
    OP_REQUIRES_OK(context, context->GetAttr("pooled_width", &pooled_width_));
    OP_REQUIRES_OK(context, context->GetAttr("min_level", &min_level_));
    OP_REQUIRES_OK(context, context->GetAttr("max_level", &max_level_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("canonical_scale", &canonical_scale_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("canonical_level", &canonical_level_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("sampling_ratio", &sampling_ratio_));
    OP_REQUIRES_OK(context, context->GetAttr("debug", &debug_));

    is_nhwc_ = false;
    CHECK_GT(spatial_scale_, 0);
    CHECK_GT(pooled_height_, 0);
    CHECK_GT(pooled_width_, 0);
    CHECK_GE(sampling_ratio_, 0);
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    const auto X = context->input(0);
    const auto RoIs = context->input(1);
    TensorShape output_shape;
    Tensor* Y = nullptr;
    int64 RoIDim0 = RoIs.dim_size(1);
    const int64 batch = X.dim_size(0);
    const int64 num_levels = X.dim_size(1);
    const int64 channels = X.dim_size(2);
    const int64 height = X.dim_size(3);
    const int64 width = X.dim_size(4);
    const int64 roi_cols = RoIs.dim_size(2);  // should be 4
    const int64 n_rois = RoIs.dim_size(1);    // num_rois,
    std::vector<int64> shape = {batch, n_rois, channels, pooled_height_,
                                pooled_width_};  // N,K,C,H,W
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(shape, &output_shape));
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &Y));
    if (RoIs.NumElements() == 0) {
      return;
    }

    const int64 total_count = Y->NumElements();
    if (total_count == 0) return;

    const GPUDevice& d = context->eigen_device<GPUDevice>();
    Tensor levels;
    Tensor scaled_boxes;
    OP_REQUIRES_OK(context, context->allocate_temp(RoIs.dtype(), RoIs.shape(),
                                                   &scaled_boxes));
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataType::DT_INT32,
                                TensorShape({batch, n_rois, 1}), &levels));
    CudaLaunchConfig config1D = GetCudaLaunchConfig(batch * n_rois, d);
    VLOG(1) << "Before boxes cudaconfig numelts= "
            << config1D.virtual_thread_count << " " << name() << " block "
            << config1D.block_count << " threads=" << config1D.thread_per_block;
    Boxes2ScaledBoxesAndLevels<float>
        <<<config1D.block_count, config1D.thread_per_block, 0, d.stream()>>>(
            config1D, RoIs.flat<float>().data(), min_level_, max_level_,
            canonical_scale_, canonical_level_, (levels).flat<int32>().data(),
            (scaled_boxes).flat<float>().data(), false, debug_);
    // d.synchronize();
    VLOG(1) << "after boxes scaled_shape" << scaled_boxes.shape()
            << " levels.shape" << levels.shape() << " input shape "
            << X.shape();
    Cuda2DLaunchConfig config = GetCuda2DLaunchConfig(
        n_rois * channels * pooled_height_ * pooled_width_, batch, d);
    VLOG(1) << "before RoiAlign forward " << name() << " X " << X.shape()
            << " boxes= " << scaled_boxes.shape()
            << " levels=" << levels.shape() << " output shape=" << Y->shape()
            << " block ( " << config.block_count.x << ","
            << config.block_count.y << "," << config.block_count.z << " ) "
            << " thread ( " << config.thread_per_block.x << ","
            << config.thread_per_block.y << "," << config.thread_per_block.z
            << " )"
            << " virt ( " << config.virtual_thread_count.x << ","
            << config.virtual_thread_count.y << ","
            << config.virtual_thread_count.z << ")";
    RoIAlignForward<float>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            config, X.flat<float>().data(), spatial_scale_, num_levels,
            channels, height, width, n_rois, pooled_height_, pooled_width_,
            sampling_ratio_, (scaled_boxes).flat<float>().data(),
            (levels).flat<int32>().data(), roi_cols, (*Y).flat<float>().data(),
            debug_);
    // d.synchronize();
    VLOG(1) << "after RoiAlign forward, X= " << X.shape().DebugString()
            << " scaled_boxes=" << scaled_boxes.shape()
            << " pooled_width=" << pooled_width_ << " output=" << Y->shape();
    // CHECK_EQ(cudaGetLastError(), CUDA_SUCCESS);
  }

 private:
  float spatial_scale_;
  int pooled_height_;
  int pooled_width_;
  int sampling_ratio_;
  int min_level_;
  int max_level_;
  float canonical_scale_;
  int canonical_level_;
  bool is_nhwc_;
  bool debug_;
};

class ROIAlignOpGrad : public tensorflow::OpKernel {
 public:
  explicit ROIAlignOpGrad(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("spatial_scale", &spatial_scale_));
    OP_REQUIRES_OK(context, context->GetAttr("pooled_height", &pooled_height_));
    OP_REQUIRES_OK(context, context->GetAttr("pooled_width", &pooled_width_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("sampling_ratio", &sampling_ratio_));
    OP_REQUIRES_OK(context, context->GetAttr("min_level", &min_level_));
    OP_REQUIRES_OK(context, context->GetAttr("max_level", &max_level_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("canonical_scale", &canonical_scale_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("canonical_level", &canonical_level_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("sampling_ratio", &sampling_ratio_));
    OP_REQUIRES_OK(context, context->GetAttr("debug", &debug_));
    is_nhwc_ = false;
    CHECK_GT(spatial_scale_, 0);
    CHECK_GT(pooled_height_, 0);
    CHECK_GT(pooled_width_, 0);
    CHECK_GE(sampling_ratio_, 0);
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    const auto grads = context->input(0);
    const auto features = context->input(1);
    const auto RoIs = context->input(2);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, features.shape(), &output));
    const GPUDevice& d = context->eigen_device<GPUDevice>();
    const int64 batch = features.dim_size(0);
    const int64 num_levels = features.dim_size(1);
    const int64 channels = features.dim_size(2);
    const int64 height = features.dim_size(3);
    const int64 width = features.dim_size(4);
    const int64 roi_cols = RoIs.dim_size(2);
    const int64 n_rois = RoIs.dim_size(1);
    Tensor levels;
    Tensor scaled_boxes;
    OP_REQUIRES_OK(context, context->allocate_temp(RoIs.dtype(), RoIs.shape(),
                                                   &scaled_boxes));
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataType::DT_INT32,
                                TensorShape({batch, n_rois, 1}), &levels));
    CudaLaunchConfig config1D = GetCudaLaunchConfig(batch * n_rois, d);
    VLOG(1) << "Before boxes cudaconfig numelts= "
            << config1D.virtual_thread_count << " " << name();
    Boxes2ScaledBoxesAndLevels<float>
        <<<config1D.block_count, config1D.thread_per_block, 0, d.stream()>>>(
            config1D, RoIs.flat<float>().data(), min_level_, max_level_,
            canonical_scale_, canonical_level_, (levels).flat<int32>().data(),
            (scaled_boxes).flat<float>().data(), true, debug_);
    // d.synchronize();
    VLOG(1) << "after boxes scaled_shape" << scaled_boxes.shape()
            << " levels.shape" << levels.shape();
    // CHECK_EQ(cudaGetLastError(), CUDA_SUCCESS);

    Cuda2DLaunchConfig config = GetCuda2DLaunchConfig(
        n_rois * channels * pooled_height_ * pooled_width_, batch, d);
    VLOG(1) << "before RoiAlign Backward " << name()
            << " grads=" << grads.shape() << " features=" << features.shape()
            << " RoIs" << RoIs.shape() << " block ( " << config.block_count.x
            << "," << config.block_count.y << "," << config.block_count.z
            << " ) "
            << " thread ( " << config.thread_per_block.x << ","
            << config.thread_per_block.y << "," << config.thread_per_block.z
            << " )"
            << " virt ( " << config.virtual_thread_count.x << ","
            << config.virtual_thread_count.y << ","
            << config.virtual_thread_count.z << ")";
    CudaLaunchConfig zconfig = GetCudaLaunchConfig(output->NumElements(), d);
    SetZero<<<zconfig.block_count, zconfig.thread_per_block, 0, d.stream()>>>(
        zconfig.virtual_thread_count, (*output).flat<float>().data());

    RoIAlignBackwardFeature<float>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            config, grads.flat<float>().data(), spatial_scale_, num_levels,
            channels, height, width, n_rois, pooled_height_, pooled_width_,
            sampling_ratio_, roi_cols, (scaled_boxes).flat<float>().data(),
            (levels).flat<int32>().data(), (*output).flat<float>().data(),
            debug_);
    // d.synchronize();
    VLOG(1) << "after RoiAlign Backward, X.shape() "
            << features.shape().DebugString()
            << " scaled_boxes=" << scaled_boxes.shape()
            << " pooled_width=" << pooled_width_
            << "output shape=" << output->shape();
    // CHECK_EQ(cudaGetLastError(), CUDA_SUCCESS);
  }

 private:
  float spatial_scale_;
  int pooled_height_;
  int pooled_width_;
  int sampling_ratio_;
  int min_level_;
  int max_level_;
  float canonical_scale_;
  int canonical_level_;
  bool is_nhwc_;
  bool debug_;
};

REGISTER_KERNEL_BUILDER(Name("ROIAlign").Device(tensorflow::DEVICE_GPU),
                        tensorflow::ROIAlignOp);
REGISTER_KERNEL_BUILDER(Name("ROIAlignGrad").Device(tensorflow::DEVICE_GPU),
                        tensorflow::ROIAlignOpGrad);
}  // namespace tensorflow
#endif