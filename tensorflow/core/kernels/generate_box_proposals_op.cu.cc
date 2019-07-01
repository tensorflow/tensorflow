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
#include "tensorflow/core/kernels/non_max_suppression_op.h"
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

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;
#define TF_OP_REQUIRES_CUDA_SUCCESS_ASYNC(context, result, done)       \
  do {                                                                 \
    cudaError_t error(result);                                         \
    if (!SE_PREDICT_TRUE(error == cudaSuccess)) {                      \
      context->SetStatus(errors::Internal("Cuda call failed with",     \
                                          cudaGetErrorString(error))); \
      done();                                                          \
      return;                                                          \
    }                                                                  \
  } while (0)

namespace {
// Decode d_bbox_deltas with respect to anchors into absolute coordinates,
// clipping if necessary.
// H is height, W is width, A is the number of anchors.
// prenms_nboxes maximum number of boxes per image to decode.
// d_boxes_keep_flags mask for boxes to consider in NMS.
// min_size is the lower bound of the shortest edge for the boxes to consider.
// bbox_xform_clip is the upper bound of encoded width and height.
// correct_transform is a flag to apply coordinate transformation correction to
// x2 and y2.
__global__ void GeneratePreNMSUprightBoxesKernel(
    const Cuda2DLaunchConfig config, const int* d_sorted_scores_keys,
    const float4* d_bbox_deltas, const float4* d_anchors, const int H,
    const int W, const int A, const float min_size, const float* d_img_info_vec,
    const float bbox_xform_clip, const bool correct_transform,
    float4* d_out_boxes,
    const int prenms_nboxes,  // leading dimension of out_boxes
    float* d_inout_scores, char* d_boxes_keep_flags) {
  // constants to calculate offsets in to the input and output arrays.
  const int K = H * W;   // Stride of Anchor
  const int WA = W * A;  // Stride of H
  const int KA = K * A;  // Stride of image
  CUDA_AXIS_KERNEL_LOOP(image_index, config.virtual_thread_count.y, Y) {
    CUDA_AXIS_KERNEL_LOOP(ibox, config.virtual_thread_count.x, X) {
      // box_conv_index : # of the same box, but indexed in the
      // scores from the conv layer, of shape (H,W,A) the num_images dimension
      // was already removed box_conv_index = a*K + h*W + w
      const int box_conv_index = d_sorted_scores_keys[image_index * KA + ibox];

      // We want to decompose box_conv_index in (h,w,a)
      // such as box_conv_index = h*W*A + W*A + a
      // (avoiding modulos in the process)
      int remaining = box_conv_index;
      const int dH = WA;  // stride of H
      const int h = remaining / dH;
      remaining -= h * dH;
      const int dW = A;  // stride of W
      const int w = remaining / dW;
      remaining -= w * dW;
      // Loading the anchor a
      // float4 is a struct with float x,y,z,w
      const float4 anchor = d_anchors[box_conv_index];
      // x1,y1,x2,y2 :coordinates of anchor a, shifted for position (h,w)
      float x1 = anchor.y;
      float x2 = anchor.w;
      float y1 = anchor.x;
      float y2 = anchor.z;

      // TODO use fast math when possible

      // Deltas of shape (N,H,W,A4)
      int deltas_idx = box_conv_index + image_index * KA;
      float4 deltas = d_bbox_deltas[deltas_idx];
      float dx = deltas.y;
      float dy = deltas.x;
      float dw = deltas.w;
      float dh = deltas.z;
      // printf("deltas_idx=%d dx=%f, dy=%f, dw=%f,
      // dh=%f\n",deltas_idx,dx,dy,dw,dh);
      // Upper bound on dw,dh
      dw = fmin(dw, bbox_xform_clip);
      dh = fmin(dh, bbox_xform_clip);

      // Applying the deltas
      float width = x2 - x1 + 1.0f;
      const float ctr_x = x1 + 0.5f * width;
      const float pred_ctr_x = ctr_x + width * dx;  // TODO fuse madd
      const float pred_w = width * expf(dw);
      x1 = pred_ctr_x - 0.5f * pred_w;
      x2 = pred_ctr_x + 0.5f * pred_w;

      float height = y2 - y1 + 1.0f;
      const float ctr_y = y1 + 0.5f * height;
      const float pred_ctr_y = ctr_y + height * dy;
      const float pred_h = height * expf(dh);
      y1 = pred_ctr_y - 0.5f * pred_h;
      y2 = pred_ctr_y + 0.5f * pred_h;

      if (correct_transform) {
        x2 -= 1.0f;
        y2 -= 1.0f;
      }
      // const float y2_old=y2;
      // const float x2_old=x2;
      // const float x1_old=x1;
      // const float y1_old=y1;
      // Clipping box to image
      const float img_height = d_img_info_vec[5 * image_index + 0];
      const float img_width = d_img_info_vec[5 * image_index + 1];
      const float min_size_scaled =
          min_size * d_img_info_vec[5 * image_index + 2];
      // min_size * d_img_info_vec[3 * image_index + 2];
      x1 = fmax(fmin(x1, img_width - 1.0f), 0.0f);
      y1 = fmax(fmin(y1, img_height - 1.0f), 0.0f);
      x2 = fmax(fmin(x2, img_width - 1.0f), 0.0f);
      y2 = fmax(fmin(y2, img_height - 1.0f), 0.0f);

      // Filter boxes
      // Removing boxes with one dim < min_size
      // (center of box is in image, because of previous step)
      width = x2 - x1 + 1.0f;  // may have changed
      height = y2 - y1 + 1.0f;
      bool keep_box = fmin(width, height) >= min_size_scaled;

      // We are not deleting the box right now even if !keep_box
      // we want to keep the relative order of the elements stable
      // we'll do it in such a way later
      // d_boxes_keep_flags size: (num_images,prenms_nboxes)
      // d_out_boxes size: (num_images,prenms_nboxes)
      const int out_index = image_index * prenms_nboxes + ibox;

      d_boxes_keep_flags[out_index] = keep_box;
      d_out_boxes[out_index] = {x1, y1, x2, y2};
      // d_inout_scores size: (num_images,KA)
      if (!keep_box)
        d_inout_scores[image_index * KA + ibox] = FLT_MIN;  // for NMS
    }
  }
}

// Copy the selected boxes and scores to output tensors.
//
__global__ void WriteUprightBoxesOutput(
    const CudaLaunchConfig nboxes, const float4* d_image_boxes,
    const float* d_image_scores, const int* d_image_boxes_keep_list,
    const int n_rois, float* d_image_out_rois, float* d_image_out_rois_probs) {
  CUDA_1D_KERNEL_LOOP(i, nboxes.virtual_thread_count) {
    if (i < n_rois) {  // copy rois to output
      const int ibox = d_image_boxes_keep_list[i];
      const float4 box = d_image_boxes[ibox];
      const float score = d_image_scores[ibox];
      // Scattered memory accesses
      // postnms_nboxes is small anyway
      d_image_out_rois_probs[i] = score;
      const int base_idx = 4 * i;
      d_image_out_rois[base_idx + 0] = box.y;
      d_image_out_rois[base_idx + 1] = box.x;
      d_image_out_rois[base_idx + 2] = box.w;
      d_image_out_rois[base_idx + 3] = box.z;
    } else {  // set trailing entries to 0
      d_image_out_rois_probs[i] = 0.;
      const int base_idx = 4 * i;
      d_image_out_rois[base_idx + 0] = 0.;
      d_image_out_rois[base_idx + 1] = 0.;
      d_image_out_rois[base_idx + 2] = 0.;
      d_image_out_rois[base_idx + 3] = 0.;
    }
  }
}

template <typename T>
void ResetTensor(Tensor* t, const Eigen::GpuDevice& d) {
  CudaLaunchConfig zconfig = GetCudaLaunchConfig(t->NumElements(), d);
  TF_CHECK_OK(GpuLaunchKernel(
      SetZero<T>, zconfig.block_count, zconfig.thread_per_block, 0, d.stream(),
      zconfig.virtual_thread_count, (*t).flat<T>().data()));
}
// Allocate scratch spaces that are needed for operation
//

Status AllocateGenerationTempTensors(
    OpKernelContext* context, Tensor* d_conv_layer_indexes,
    Tensor* d_image_offset, Tensor* d_cub_sort_buffer,
    Tensor* d_cub_select_buffer, Tensor* d_sorted_conv_layer_indexes,
    Tensor* d_sorted_scores, Tensor* dev_boxes, Tensor* dev_boxes_keep_flags,
    int num_images, int conv_layer_nboxes, size_t cub_sort_temp_storage_bytes,
    size_t cub_select_temp_storage_bytes, int nboxes_to_generate, int box_dim) {
  auto d = context->eigen_gpu_device();
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_images, conv_layer_nboxes}),
      d_conv_layer_indexes));
  ResetTensor<int>(d_conv_layer_indexes, d);
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_images + 1}), d_image_offset));
  ResetTensor<int>(d_image_offset, d);
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({(int64)cub_sort_temp_storage_bytes}),
      d_cub_sort_buffer));
  ResetTensor<int8>(d_cub_sort_buffer, d);
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({(int64)cub_select_temp_storage_bytes}),
      d_cub_select_buffer));
  ResetTensor<int8>(d_cub_select_buffer, d);
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_images, conv_layer_nboxes}),
      d_sorted_conv_layer_indexes));
  ResetTensor<int32>(d_sorted_conv_layer_indexes, d);
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({num_images, conv_layer_nboxes}),
      d_sorted_scores));
  ResetTensor<float>(d_sorted_scores, d);
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT,
      TensorShape({num_images, box_dim * nboxes_to_generate}), dev_boxes));
  ResetTensor<float>(dev_boxes, d);
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({num_images, nboxes_to_generate}),
      dev_boxes_keep_flags));
  ResetTensor<int8>(dev_boxes_keep_flags, d);
  return Status::OK();
}

// Allocate workspace for NMS operation
Status AllocatePreNMSTempTensors(
    OpKernelContext* context, Tensor* dev_image_prenms_boxes,
    Tensor* dev_image_prenms_scores, Tensor* dev_image_boxes_keep_list,
    Tensor* dev_postnms_rois, Tensor* dev_postnms_rois_probs,
    Tensor* dev_prenms_nboxes, int num_images, int num_boxes_to_generate,
    int box_dim, int post_nms_topn, int pre_nms_topn) {
  auto d = context->eigen_gpu_device();
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({box_dim * num_boxes_to_generate}),
      dev_image_prenms_boxes));
  ResetTensor<float>(dev_image_prenms_boxes, d);

  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({num_boxes_to_generate}),
      dev_image_prenms_scores));
  ResetTensor<float>(dev_image_prenms_scores, d);

  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_boxes_to_generate}),
      dev_image_boxes_keep_list));
  ResetTensor<int32>(dev_image_boxes_keep_list, d);

  const int max_postnms_nboxes = std::min(num_boxes_to_generate, post_nms_topn);
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT,
      TensorShape({box_dim * num_images * max_postnms_nboxes}),
      dev_postnms_rois));
  ResetTensor<float>(dev_postnms_rois, d);

  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({num_images * max_postnms_nboxes}),
      dev_postnms_rois_probs));
  ResetTensor<float>(dev_postnms_rois_probs, d);

  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_images}), dev_prenms_nboxes));
  ResetTensor<int32>(dev_prenms_nboxes, d);

  return Status::OK();
}

// Initialize index and offset arrays.
// num_images is the batch size, KA is the number of anchors
__global__ void InitializeDataKernel(const Cuda2DLaunchConfig config,
                                     int* d_image_offsets,
                                     int* d_boxes_keys_iota) {
  const int KA = config.virtual_thread_count.x;
  const int num_images = config.virtual_thread_count.y;
  // printf("num_images %d KA %d\n",num_images,KA);
  CUDA_AXIS_KERNEL_LOOP(img_idx, config.virtual_thread_count.y, Y) {
    CUDA_AXIS_KERNEL_LOOP(box_idx, config.virtual_thread_count.x, X) {
      // CUDA_2D_KERNEL_LOOP(box_idx, KA, img_idx, num_images) {
      d_boxes_keys_iota[img_idx * KA + box_idx] = box_idx;

      // One 1D line sets the 1D data
      if (box_idx == 0) {
        d_image_offsets[img_idx] = KA * img_idx;
        // One thread sets the last+1 offset
        if (img_idx == 0) d_image_offsets[num_images] = KA * num_images;
      }
    }
  }
}

}  // namespace

class GenerateBoundingBoxProposals : public tensorflow::AsyncOpKernel {
 public:
  explicit GenerateBoundingBoxProposals(
      tensorflow::OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("post_nms_topn", &post_nms_topn_));
    // compatibility for detectron like networks. False for generic case
    OP_REQUIRES_OK(context, context->GetAttr("correct_transform_coords",
                                             &correct_transform_coords_));
    CHECK_GT(post_nms_topn_, 0);
    bbox_xform_clip_default_ = log(1000.0 / 16.);
  }

  template <typename T>
  Status GetScalarValue(OpKernelContext* context, int input, T* value) {
    const Tensor& scalar_tensor = context->input(input);
    if (!TensorShapeUtils::IsScalar(scalar_tensor.shape())) {
      return errors::InvalidArgument("Expected a scalar in input ", input,
                                     "but got shape ",
                                     scalar_tensor.shape().DebugString());
    }
    *value = scalar_tensor.scalar<T>()();
    return Status::OK();
  }

  void ComputeAsync(tensorflow::OpKernelContext* context,
                    DoneCallback done) override {
    // .Input("scores: float")
    // .Input("bbox_deltas: float")
    // .Input("image_info: float")
    // .Input("anchors: float")

    const auto scores = context->input(0);
    const auto bbox_deltas = context->input(1);
    const auto image_info = context->input(2);
    const auto anchors = context->input(3);
    const auto num_images = scores.dim_size(0);
    const auto A = scores.dim_size(3);
    const auto H = scores.dim_size(1);
    const auto W = scores.dim_size(2);
    const auto box_dim = anchors.dim_size(0) / A;
    CHECK_EQ(box_dim, 4);
    // TODO(skama): make sure that inputs are ok.
    const int K = H * W;
    // VLOG(0)<<"num_images="<<num_images<<" A="<<A<<" H="<<H<<" W="<<W;
    const int conv_layer_nboxes =
        K * A;  // total number of boxes when decoded on anchors.
    // The following calls to CUB primitives do nothing
    // (because the first arg is nullptr)
    // except setting cub_*_temp_storage_bytes
    float nms_threshold;
    int pre_nms_topn;
    float min_size;
    OP_REQUIRES_OK_ASYNC(context, GetScalarValue(context, 4, &nms_threshold),
                         done);
    if (nms_threshold < 0 || nms_threshold > 1.0) {
      context->SetStatus(errors::InvalidArgument(
          "nms_threshold should be between 0 and 1. Got ", nms_threshold));
      done();
      return;
    }
    OP_REQUIRES_OK_ASYNC(context, GetScalarValue(context, 5, &pre_nms_topn),
                         done);
    if (pre_nms_topn <= 0) {
      context->SetStatus(errors::InvalidArgument(
          "pre_nms_topn should be greater than 0", pre_nms_topn));
      done();
      return;
    }

    OP_REQUIRES_OK_ASYNC(context, GetScalarValue(context, 6, &min_size), done);
    auto cuda_stream = GetCudaStream(context);
    size_t cub_sort_temp_storage_bytes = 0;
    float* flt_ptr = nullptr;
    int* int_ptr = nullptr;
    cudaError_t cuda_ret = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, cub_sort_temp_storage_bytes, flt_ptr, flt_ptr, int_ptr,
        int_ptr, num_images * conv_layer_nboxes, num_images, int_ptr, int_ptr,
        0, 8 * sizeof(float),  // sort all bits
        cuda_stream);
    CHECK_EQ(cuda_ret, cudaSuccess);
    // get the size of select temp buffer
    size_t cub_select_temp_storage_bytes = 0;
    char* char_ptr = nullptr;
    float4* f4_ptr = nullptr;
    TF_OP_REQUIRES_CUDA_SUCCESS_ASYNC(
        context,
        cub::DeviceSelect::Flagged(nullptr, cub_select_temp_storage_bytes,
                                   f4_ptr, char_ptr, f4_ptr, int_ptr, K * A,
                                   cuda_stream),
        done);
    Tensor d_conv_layer_indexes;  // box indices on device
    Tensor d_image_offset;        // starting offsets boxes for each image
    Tensor d_cub_sort_buffer;     // buffer for cub sorting
    Tensor d_cub_select_buffer;   // buffer for cub selection
    Tensor d_sorted_conv_layer_indexes;  // output of cub sorting, indices of
                                         // the sorted boxes
    Tensor dev_sorted_scores;            // sorted scores, cub output
    Tensor dev_boxes;                    // boxes on device
    Tensor dev_boxes_keep_flags;  // bitmask for keeping the boxes or rejecting
                                  // from output
    const int nboxes_to_generate = std::min(conv_layer_nboxes, pre_nms_topn);
    OP_REQUIRES_OK_ASYNC(
        context,
        AllocateGenerationTempTensors(
            context, &d_conv_layer_indexes, &d_image_offset, &d_cub_sort_buffer,
            &d_cub_select_buffer, &d_sorted_conv_layer_indexes,
            &dev_sorted_scores, &dev_boxes, &dev_boxes_keep_flags, num_images,
            conv_layer_nboxes, cub_sort_temp_storage_bytes,
            cub_select_temp_storage_bytes, nboxes_to_generate, box_dim),
        done);
    const GPUDevice& d = context->eigen_device<GPUDevice>();
    Cuda2DLaunchConfig conf2d =
        GetCuda2DLaunchConfig(conv_layer_nboxes, num_images, d);
    // create box indices and offsets for each image on device
    TF_CHECK_OK(GpuLaunchKernel(InitializeDataKernel, conf2d.block_count,
                                conf2d.thread_per_block, 0, d.stream(), conf2d,
                                d_image_offset.flat<int>().data(),
                                d_conv_layer_indexes.flat<int>().data()));

    // sort boxes with their scores.
    // d_sorted_conv_layer_indexes will hold the pointers to old indices.

    TF_OP_REQUIRES_CUDA_SUCCESS_ASYNC(
        context,
        cub::DeviceSegmentedRadixSort::SortPairsDescending(
            d_cub_sort_buffer.flat<int8>().data(), cub_sort_temp_storage_bytes,
            scores.flat<float>().data(), dev_sorted_scores.flat<float>().data(),
            d_conv_layer_indexes.flat<int>().data(),
            d_sorted_conv_layer_indexes.flat<int>().data(),
            num_images * conv_layer_nboxes, num_images,
            d_image_offset.flat<int>().data(),
            d_image_offset.flat<int>().data() + 1, 0,
            8 * sizeof(float),  // sort all bits
            cuda_stream),
        done);
    // Keeping only the topN pre_nms
    conf2d = GetCuda2DLaunchConfig(nboxes_to_generate, num_images, d);

    // create box y1,x1,y2,x2 from box_deltas and anchors (decode the boxes) and
    // mark the boxes which are smaller that min_size ignored.
    TF_CHECK_OK(GpuLaunchKernel(
        GeneratePreNMSUprightBoxesKernel, conf2d.block_count,
        conf2d.thread_per_block, 0, d.stream(), conf2d,
        d_sorted_conv_layer_indexes.flat<int>().data(),
        reinterpret_cast<const float4*>(bbox_deltas.flat<float>().data()),
        reinterpret_cast<const float4*>(anchors.flat<float>().data()), H, W, A,
        min_size, image_info.flat<float>().data(), bbox_xform_clip_default_,
        correct_transform_coords_,
        reinterpret_cast<float4*>(dev_boxes.flat<float>().data()),
        nboxes_to_generate, dev_sorted_scores.flat<float>().data(),
        (char*)dev_boxes_keep_flags.flat<int8>().data()));
    const int nboxes_generated = nboxes_to_generate;
    const int roi_cols = box_dim;
    const int max_postnms_nboxes = std::min(nboxes_generated, post_nms_topn_);
    Tensor dev_image_prenms_boxes;
    Tensor dev_image_prenms_scores;
    Tensor dev_image_boxes_keep_list;
    Tensor dev_postnms_rois;
    Tensor dev_postnms_rois_probs;
    Tensor dev_prenms_nboxes;
    // Allocate workspaces needed for NMS
    OP_REQUIRES_OK_ASYNC(
        context,
        AllocatePreNMSTempTensors(
            context, &dev_image_prenms_boxes, &dev_image_prenms_scores,
            &dev_image_boxes_keep_list, &dev_postnms_rois,
            &dev_postnms_rois_probs, &dev_prenms_nboxes, num_images,
            nboxes_generated, box_dim, post_nms_topn_, pre_nms_topn),
        done);
    // get the pointers for temp storages
    int* d_prenms_nboxes = dev_prenms_nboxes.flat<int>().data();
    int h_prenms_nboxes = 0;
    char* d_cub_select_temp_storage =
        (char*)d_cub_select_buffer.flat<int8>().data();
    float* d_image_prenms_boxes = dev_image_prenms_boxes.flat<float>().data();
    float* d_image_prenms_scores = dev_image_prenms_scores.flat<float>().data();
    int* d_image_boxes_keep_list = dev_image_boxes_keep_list.flat<int>().data();

    int nrois_in_output = 0;
    // get the pointers to boxes and scores
    char* d_boxes_keep_flags = (char*)dev_boxes_keep_flags.flat<int8>().data();
    float* d_boxes = dev_boxes.flat<float>().data();
    float* d_sorted_scores = dev_sorted_scores.flat<float>().data();

    // Create output tensors
    Tensor* output_rois = nullptr;
    Tensor* output_roi_probs = nullptr;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_output(
            0, TensorShape({num_images, post_nms_topn_, roi_cols}),
            &output_rois),
        done);
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_output(1, TensorShape({num_images, post_nms_topn_}),
                                 &output_roi_probs),
        done);
    float* d_postnms_rois = (*output_rois).flat<float>().data();
    float* d_postnms_rois_probs = (*output_roi_probs).flat<float>().data();

    // Do  per-image nms
    for (int image_index = 0; image_index < num_images; ++image_index) {
      // reset output workspaces
      ResetTensor<int32>(&dev_image_boxes_keep_list, d);
      // Sub matrices for current image
      // boxes
      const float* d_image_boxes =
          &d_boxes[image_index * nboxes_generated * box_dim];
      // scores
      const float* d_image_sorted_scores =
          &d_sorted_scores[image_index * K * A];
      // keep flags
      char* d_image_boxes_keep_flags =
          &d_boxes_keep_flags[image_index * nboxes_generated];

      // Output buffer for image
      float* d_image_postnms_rois =
          &d_postnms_rois[image_index * roi_cols * post_nms_topn_];
      float* d_image_postnms_rois_probs =
          &d_postnms_rois_probs[image_index * post_nms_topn_];

      // Moving valid boxes (ie the ones with d_boxes_keep_flags[ibox] == true)
      // to the output tensors
      TF_OP_REQUIRES_CUDA_SUCCESS_ASYNC(
          context,
          cub::DeviceSelect::Flagged(
              d_cub_select_temp_storage, cub_select_temp_storage_bytes,
              reinterpret_cast<const float4*>(d_image_boxes),
              d_image_boxes_keep_flags,
              reinterpret_cast<float4*>(d_image_prenms_boxes), d_prenms_nboxes,
              nboxes_generated, d.stream()),
          done);

      TF_OP_REQUIRES_CUDA_SUCCESS_ASYNC(
          context,
          cub::DeviceSelect::Flagged(
              d_cub_select_temp_storage, cub_select_temp_storage_bytes,
              d_image_sorted_scores, d_image_boxes_keep_flags,
              d_image_prenms_scores, d_prenms_nboxes, nboxes_generated,
              d.stream()),
          done);
      d.memcpyDeviceToHost(&h_prenms_nboxes, d_prenms_nboxes, sizeof(int));
      d.synchronize();
      // We know prenms_boxes <= topN_prenms, because nboxes_generated <=
      // topN_prenms. Calling NMS on the generated boxes
      const int prenms_nboxes = h_prenms_nboxes;
      int nkeep;
      OP_REQUIRES_OK_ASYNC(
          context,
          NmsGpu(d_image_prenms_boxes, prenms_nboxes, nms_threshold,
                 d_image_boxes_keep_list, &nkeep, context),
          done);
      // All operations done after previous sort were keeping the relative order
      // of the elements the elements are still sorted keep topN <=> truncate
      // the array
      const int postnms_nboxes = std::min(nkeep, post_nms_topn_);
      // Moving the out boxes to the output tensors,
      // adding the image_index dimension on the fly
      CudaLaunchConfig config = GetCudaLaunchConfig(post_nms_topn_, d);
      // make this single kernel
      TF_CHECK_OK(GpuLaunchKernel(
          WriteUprightBoxesOutput, config.block_count, config.thread_per_block,
          0, d.stream(), config,
          reinterpret_cast<const float4*>(d_image_prenms_boxes),
          d_image_prenms_scores, d_image_boxes_keep_list, postnms_nboxes,
          d_image_postnms_rois, d_image_postnms_rois_probs));
      nrois_in_output += postnms_nboxes;
      TF_OP_REQUIRES_CUDA_SUCCESS_ASYNC(context, cudaGetLastError(), done);
    }
    done();
  }

 private:
  int post_nms_topn_;
  float bbox_xform_clip_default_;
  bool correct_transform_coords_;
};

REGISTER_KERNEL_BUILDER(
    Name("GenerateBoundingBoxProposals").Device(tensorflow::DEVICE_GPU),
    tensorflow::GenerateBoundingBoxProposals);
}  // namespace tensorflow
#endif