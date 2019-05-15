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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/non_max_suppression_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "third_party/cub/device/device_radix_sort.cuh"
#include "third_party/cub/device/device_segmented_radix_sort.cuh"
#include "third_party/cub/device/device_select.cuh"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#define NMS_BLOCK_DIM 16
#define NMS_BLOCK_DIM_MAX 16
#define NMS_CHUNK_SIZE 2000

#define CUDA_CHECK(result)                                    \
  do {                                                        \
    cudaError_t error(result);                                \
    CHECK(error == cudaSuccess) << cudaGetErrorString(error); \
  } while (0)

struct __align__(16) Box {
  float x1, y1, x2, y2;
};

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::ThreadPoolDevice CPUDevice;

const int NMS_BOXES_PER_THREAD = 8 * sizeof(int);

__launch_bounds__(NMS_BLOCK_DIM* NMS_BLOCK_DIM, 4) __global__
    void NMSKernel(const Box* d_desc_sorted_boxes, const int nboxes,
                   const float thresh, const int mask_ld, int* d_delete_mask,
                   bool flip_boxes = false) {
  // Storing boxes used by this CUDA block in the shared memory
  __shared__ Box shared_i_boxes[NMS_BLOCK_DIM];
  // Same thing with areas
  __shared__ float shared_i_areas[NMS_BLOCK_DIM];
  // The condition of the for loop is common to all threads in the block
  // This is necessary to be able to call __syncthreads() inside of the loop
  for (int i_block_offset = blockIdx.x * blockDim.x; i_block_offset < nboxes;
       i_block_offset += blockDim.x * gridDim.x) {
    const int i_to_load = i_block_offset + threadIdx.x;
    if (i_to_load < nboxes) {
      // One 1D line load the boxes for x-dimension
      if (threadIdx.y == 0) {
        const Box box = d_desc_sorted_boxes[i_to_load];
        Box flipped = box;
        if (flip_boxes) {
          if (box.x1 > box.x2) {
            flipped.x1 = box.x2;
            flipped.x2 = box.x1;
          }
          if (box.y1 > box.y2) {
            flipped.y1 = box.y2;
            flipped.y2 = box.y1;
          }
        }
        shared_i_boxes[threadIdx.x] = flipped;
        shared_i_areas[threadIdx.x] =
            (flipped.x2 - flipped.x1 + 1.0f) * (flipped.y2 - flipped.y1 + 1.0f);
      }
    }
    __syncthreads();
    const int i = i_block_offset + threadIdx.x;
    for (int j_thread_offset =
             NMS_BOXES_PER_THREAD * (blockIdx.y * blockDim.y + threadIdx.y);
         j_thread_offset < nboxes;
         j_thread_offset += NMS_BOXES_PER_THREAD * blockDim.y * gridDim.y) {
      // Note : We can do everything using multiplication,
      // and use fp16 - we are comparing against a low precision
      // threshold
      int above_thresh = 0;
      bool valid = false;
      for (int ib = 0; ib < NMS_BOXES_PER_THREAD; ++ib) {
        // This thread will compare Box i and Box j
        const int j = j_thread_offset + ib;
        if (i < j && i < nboxes && j < nboxes) {
          valid = true;
          const Box o_box = d_desc_sorted_boxes[j];
          const Box i_box = shared_i_boxes[threadIdx.x];
          Box j_box = o_box;
          if (flip_boxes) {
            if (j_box.x1 > j_box.x2) {
              float tmp = j_box.x2;
              j_box.x2 = j_box.x1;
              j_box.x1 = tmp;
            }
            if (j_box.y1 > j_box.y2) {
              float tmp = j_box.y2;
              j_box.y2 = j_box.y1;
              j_box.y1 = tmp;
            }
          }
          const float j_area =
              (j_box.x2 - j_box.x1 + 1.0f) * (j_box.y2 - j_box.y1 + 1.0f);
          const float i_area = shared_i_areas[threadIdx.x];
          // The following code will not be valid with empty boxes
          if (i_area == 0.0f || j_area == 0.0f) continue;
          const float xx1 = fmaxf(i_box.x1, j_box.x1);
          const float yy1 = fmaxf(i_box.y1, j_box.y1);
          const float xx2 = fminf(i_box.x2, j_box.x2);
          const float yy2 = fminf(i_box.y2, j_box.y2);

          // fdimf computes the positive difference between xx2+1 and xx1
          const float w = fdimf(xx2 + 1.0f, xx1);
          const float h = fdimf(yy2 + 1.0f, yy1);
          const float intersection = w * h;

          // Testing for a/b > t
          // eq with a > b*t (b is !=0)
          // avoiding divisions
          const float a = intersection;
          const float b = i_area + j_area - intersection;
          const float bt = b * thresh;
          // eq. to if ovr > thresh
          if (a > bt) {
            // we have score[j] <= score[i]
            above_thresh |= (1U << ib);
          }
        }
      }
      if (valid) {
        d_delete_mask[i * mask_ld + j_thread_offset / NMS_BOXES_PER_THREAD] =
            above_thresh;
      }
    }
    __syncthreads();  // making sure everyone is done reading smem
  }
}

template <typename T, typename Index>
__global__ void IndexSelect(const CudaLaunchConfig config, const T* original,
                            const Index* indices, T* selected) {
  CUDA_AXIS_KERNEL_LOOP(idx, config.virtual_thread_count, X) {
    selected[idx] = original[indices[idx]];
  }
}

template <typename P1, typename P2, typename Index>
__global__ void PairedSelect(const CudaLaunchConfig config, const P1* p1,
                             const P2* p2, const Index* indices,
                             P1* selected_p1, P2* selected_p2) {
  for (auto idx : CudaGridRangeX(config.virtual_thread_count)) {
    selected_p1[idx] = p1[indices[idx]];
    selected_p2[idx] = p2[indices[idx]];
  }
}

template <typename T>
__global__ void Iota(const CudaLaunchConfig config, const T offset,
                     T* to_fill) {
  for (int idx : CudaGridRangeX(config.virtual_thread_count)) {
    to_fill[idx] = static_cast<T>(idx) + offset;
  }
}

tensorflow::Status nms_gpu(const float* d_desc_sorted_boxes_float_ptr,
                           const int N, const float thresh,
                           int* d_keep_sorted_list, int* h_nkeep,
                           int* dev_delete_mask, int* host_delete_mask,
                           OpKernelContext* context, bool flip_boxes) {
  // d_desc_sorted_boxes_float_ptr is a pointer
  //    to device memory float array containing the box corners for N boxes.
  // N is number of boxes.
  // threshold is the iou threshold for elimination
  // d_keep_sorted_list is a device pointer to int array containing sorted
  //    indices of the boxes to keep h_nkeep is the pointer to number of
  //    elements to keep for the host.
  // h_nkeep is a host pointer for returning number of items to keep
  // flip_boxes flag reorders the boxes use lower left and upper right corners
  // if they are given in mixed format. Making sure we respect the __align(16)__
  // we promised to the compiler
  auto iptr = reinterpret_cast<std::uintptr_t>(d_desc_sorted_boxes_float_ptr);
  CHECK_EQ((iptr & 15), 0);

  // The next kernel expects squares

  const int mask_ld = (N + NMS_BOXES_PER_THREAD - 1) / NMS_BOXES_PER_THREAD;
  const Box* d_desc_sorted_boxes =
      reinterpret_cast<const Box*>(d_desc_sorted_boxes_float_ptr);
  auto stream_exec = context->op_device_context()->stream();
  auto device = context->eigen_gpu_device();
  dim3 grid, block;
  int block_size = (N + NMS_BLOCK_DIM - 1) / NMS_BLOCK_DIM;
  block_size = std::max(std::min(block_size, NMS_BLOCK_DIM_MAX), 1);
  grid.x = block_size;
  grid.y = block_size;
  block.x = NMS_BLOCK_DIM;
  block.y = NMS_BLOCK_DIM;
  NMSKernel<<<grid, block, 0, device.stream()>>>(
      d_desc_sorted_boxes, N, thresh, mask_ld, dev_delete_mask, flip_boxes);
  CHECK_EQ(cudaGetLastError(), cudaSuccess);
  // Overlapping CPU computes and D2H memcpy
  // both take about the same time
  int nto_copy = std::min(NMS_CHUNK_SIZE, N);
  cudaEvent_t copy_done;
  cudaEventCreate(&copy_done);
  device.memcpyDeviceToHost(&host_delete_mask[0], &dev_delete_mask[0],
                            nto_copy * mask_ld * sizeof(int));
  CUDA_CHECK(cudaEventRecord(copy_done, device.stream()));
  int offset = 0;
  std::vector<int> h_keep_sorted_list;
  std::vector<int> rmv(mask_ld, 0);
  memset(host_delete_mask, N, sizeof(int));
  while (offset < N) {
    const int ncopied = nto_copy;
    int next_offset = offset + ncopied;
    nto_copy = std::min(NMS_CHUNK_SIZE, N - next_offset);
    if (nto_copy > 0) {
      device.memcpyDeviceToHost(&host_delete_mask[next_offset * mask_ld],
                                &dev_delete_mask[next_offset * mask_ld],
                                nto_copy * mask_ld * sizeof(int));
    }
    // Waiting for previous copy
    CUDA_CHECK(cudaEventSynchronize(copy_done));
    if (nto_copy > 0) CUDA_CHECK(cudaEventRecord(copy_done, device.stream()));
    for (int i = offset; i < next_offset; ++i) {
      int iblock = i / NMS_BOXES_PER_THREAD;
      int inblock = i % NMS_BOXES_PER_THREAD;
      if (!(rmv[iblock] & (1 << inblock))) {
        h_keep_sorted_list.push_back(i);
        int* p = &host_delete_mask[i * mask_ld];
        for (int ib = 0; ib < mask_ld; ++ib) {
          rmv[ib] |= p[ib];
        }
      }
    }
    offset = next_offset;
  }
  cudaEventDestroy(copy_done);

  const int nkeep = h_keep_sorted_list.size();
  device.memcpyHostToDevice(d_keep_sorted_list, &h_keep_sorted_list[0],
                            nkeep * sizeof(int));

  *h_nkeep = nkeep;
  return Status::OK();
}

class NonMaxSuppressionV2GPUOp : public OpKernel {
 public:
  explicit NonMaxSuppressionV2GPUOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // boxes: [num_boxes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()),
                errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                        iou_threshold.shape().DebugString()));
    const float iou_threshold_val = iou_threshold.scalar<float>()();

    OP_REQUIRES(context, iou_threshold_val >= 0 && iou_threshold_val <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    OP_REQUIRES(context, boxes.dims() == 2,
                errors::InvalidArgument("boxes must be a rank 2 tensor!"));
    int num_boxes = boxes.dim_size(0);
    OP_REQUIRES(context, boxes.dim_size(1) == 4,
                errors::InvalidArgument("boxes must be Nx4"));
    OP_REQUIRES(context, scores.dims() == 1,
                errors::InvalidArgument("scores must be a vector!"));
    OP_REQUIRES(
        context, scores.dim_size(0) == num_boxes,
        errors::InvalidArgument(
            "scores has incompatible shape"));  // message must be exactly this
                                                // otherwise tests fail!
    if (!context->status().ok()) {
      return;
    }
    if (num_boxes == 0) {
      Tensor* output_indices = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({0}),
                                                       &output_indices));
      return;
    }
    const int output_size = max_output_size.scalar<int>()();
    // const float score_threshold_val =
    // std::numeric_limits<float>::lowest();
    size_t cub_sort_temp_storage_bytes = 0;
    float* flt_ptr = nullptr;
    int* int_ptr = nullptr;
    auto cuda_stream = GetCudaStream(context);
    auto device = context->eigen_gpu_device();
    cudaError_t cuda_ret = cub::DeviceRadixSort::SortPairsDescending(
        nullptr, cub_sort_temp_storage_bytes,
        flt_ptr,               // scores
        flt_ptr,               // sorted scores
        int_ptr,               // input indices
        int_ptr,               // sorted indices
        num_boxes,             // num items
        0, 8 * sizeof(float),  // sort all bits
        cuda_stream);
    CHECK_EQ(cuda_ret, cudaSuccess);
    Tensor d_cub_sort_buffer;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(
                       DataType::DT_INT8,
                       TensorShape({(int64)cub_sort_temp_storage_bytes}),
                       &d_cub_sort_buffer));
    Tensor d_indices;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataType::DT_INT32,
                                        TensorShape({num_boxes}), &d_indices));
    Tensor d_sorted_indices;
    OP_REQUIRES_OK(context, context->allocate_temp(DataType::DT_INT32,
                                                   TensorShape({num_boxes}),
                                                   &d_sorted_indices));
    Tensor d_selected_indices;
    OP_REQUIRES_OK(context, context->allocate_temp(DataType::DT_INT32,
                                                   TensorShape({num_boxes}),
                                                   &d_selected_indices));
    Tensor d_sorted_scores;
    OP_REQUIRES_OK(context, context->allocate_temp(DataType::DT_FLOAT,
                                                   TensorShape({num_boxes}),
                                                   &d_sorted_scores));
    Tensor d_sorted_boxes;
    OP_REQUIRES_OK(context, context->allocate_temp(DataType::DT_FLOAT,
                                                   TensorShape({num_boxes, 4}),
                                                   &d_sorted_boxes));
    Tensor host_nms_mask, d_nms_mask;
    int64 max_nms_mask_size =
        num_boxes *
        ((num_boxes + NMS_BOXES_PER_THREAD - 1) / NMS_BOXES_PER_THREAD);
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataType::DT_INT32,
                                TensorShape({max_nms_mask_size}), &d_nms_mask));
    // reset data sensitive tensors
    auto config = GetCudaLaunchConfig(d_nms_mask.NumElements(), device);
    SetZero<<<config.block_count, config.thread_per_block, 0,
              device.stream()>>>(config.virtual_thread_count,
                                 d_nms_mask.flat<int32>().data());

    AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    alloc_attr.set_gpu_compatible(true);
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataType::DT_INT32,
                                          TensorShape({max_nms_mask_size}),
                                          &host_nms_mask, alloc_attr));

    // this will return sorted scores and their indices
    config = GetCudaLaunchConfig(num_boxes, device);
    // initialize box and score indices
    Iota<<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
        config, 0, d_indices.flat<int>().data());
    cuda_ret = cudaGetLastError();
    CUDA_CHECK(cuda_ret);
    CHECK_EQ(cuda_ret, cudaSuccess);
    cuda_ret = cub::DeviceRadixSort::SortPairsDescending(
        d_cub_sort_buffer.flat<int8>().data(), cub_sort_temp_storage_bytes,
        scores.flat<float>().data(), d_sorted_scores.flat<float>().data(),
        d_indices.flat<int>().data(), d_sorted_indices.flat<int>().data(),
        num_boxes, 0,
        8 * sizeof(float),  // sort all bits
        cuda_stream);
    CUDA_CHECK(cuda_ret);
    CHECK_EQ(cuda_ret, cudaSuccess);

    // get pointers for easy access
    const float4* original_boxes =
        reinterpret_cast<const float4*>(boxes.flat<float>().data());
    float4* sorted_boxes =
        reinterpret_cast<float4*>(d_sorted_boxes.flat<float>().data());
    const int* sorted_indices = d_sorted_indices.flat<int>().data();
    // sort boxes using indices
    IndexSelect<<<config.block_count, config.thread_per_block, 0,
                  device.stream()>>>(config, original_boxes, sorted_indices,
                                     sorted_boxes);
    CHECK_EQ(cuda_ret, cudaSuccess);
    int num_to_keep = 0;
    bool flip_boxes = true;  // there is no guarantee that boxes are given in
                             // lower left-upper right corners!
    auto status =
        nms_gpu(d_sorted_boxes.flat<float>().data(), num_boxes,
                iou_threshold_val, d_selected_indices.flat<int>().data(),
                &num_to_keep, d_nms_mask.flat<int>().data(),
                host_nms_mask.flat<int>().data(), context, flip_boxes);
    cuda_ret = cudaGetLastError();
    CUDA_CHECK(cuda_ret);
    CHECK_EQ(cuda_ret, cudaSuccess);
    if (!status.ok()) {
      context->SetStatus(status);
      return;
    }
    Tensor* output_indices = nullptr;
    int num_outputs = std::min(num_to_keep, output_size);  // no padding!
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({num_outputs}),
                                            &output_indices));
    if (num_outputs == 0) return;
    config = GetCudaLaunchConfig(num_outputs, device);
    IndexSelect<<<config.block_count, config.thread_per_block, 0,
                  device.stream()>>>(config, sorted_indices,
                                     d_selected_indices.flat<int>().data(),
                                     (*output_indices).flat<int>().data());
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
  }
};
REGISTER_KERNEL_BUILDER(
    Name("NonMaxSuppressionV2").TypeConstraint<float>("T").Device(DEVICE_GPU),
    NonMaxSuppressionV2GPUOp);

}  // namespace tensorflow
#endif
