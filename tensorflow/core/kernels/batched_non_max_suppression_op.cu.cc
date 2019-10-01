#include <limits>
#include "tensorflow/core/kernels/batched_non_max_suppression_op.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "third_party/cub/device/device_radix_sort.cuh"
#include "third_party/cub/device/device_segmented_radix_sort.cuh"
#include "third_party/cub/device/device_select.cuh"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
// DoTranspose is broken we need to use conv2d swapdimensions1and2intensor3
#include "tensorflow/core/kernels/conv_2d_gpu.h"
#define TF_RETURN_IF_CUDA_ERROR(result)                                        \
  do {                                                                         \
    cudaError_t error(result);                                                 \
    if (!SE_PREDICT_TRUE(error == cudaSuccess)) {                              \
      return errors::Internal("Cuda call failed with ",                        \
                              cudaGetErrorString(error), " at ", __FUNCTION__, \
                              ":", __LINE__);                                  \
    }                                                                          \
  } while (0)

#define TF_OP_REQUIRES_CUDA_SUCCESS(context, result)                   \
  do {                                                                 \
    cudaError_t error(result);                                         \
    if (!SE_PREDICT_TRUE(error == cudaSuccess)) {                      \
      context->SetStatus(errors::Internal("Cuda call failed with",     \
                                          cudaGetErrorString(error))); \
      return;                                                          \
    }                                                                  \
  } while (0)

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

struct __align__(16) Box {
  float x1, y1, x2, y2;
};

using absl::StrAppend;
using absl::StrCat;

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::ThreadPoolDevice CPUDevice;

constexpr int NumBits(int n) { return (n == 0) ? 0 : NumBits(n >> 1) + 1; }

constexpr int kNmsReductionChunkSize = 256;

template <typename T>
__device__ EIGEN_STRONG_INLINE void Swap(T& a, T& b) {
  T c(a);
  a = b;
  b = c;
}

// Check whether two boxes have an IoU greater than threshold.
template <typename T>
__device__ EIGEN_STRONG_INLINE bool OverThreshold(const Box* a, const Box* b,
                                                  const float a_area,
                                                  const T iou_threshold) {
  const float b_area = (b->x2 - b->x1) * (b->y2 - b->y1);
  if (a_area == 0.0f || b_area == 0.0f) return false;
  const float xx1 = fmaxf(a->x1, b->x1);
  const float yy1 = fmaxf(a->y1, b->y1);
  const float xx2 = fminf(a->x2, b->x2);
  const float yy2 = fminf(a->y2, b->y2);

  // fdimf computes the positive difference between xx2+1 and xx1.
  const float w = fdimf(xx2, xx1);
  const float h = fdimf(yy2, yy1);
  const float intersection = w * h;

  // Testing for aa/bb > t
  // eq with aa > bb*t (b is !=0)
  // avoiding divisions.
  const float aa = intersection;
  const float bb = a_area + b_area - intersection;
  const float bt = bb * iou_threshold;
  return aa >= bt;
}

template <bool flip_box>
__device__ EIGEN_STRONG_INLINE void Flipped(Box& box);

template <>
__device__ EIGEN_STRONG_INLINE void Flipped<false>(Box& box) {}

template <>
__device__ EIGEN_STRONG_INLINE void Flipped<true>(Box& box) {
  if (box.x1 > box.x2) Swap(box.x1, box.x2);
  if (box.y1 > box.y2) Swap(box.y1, box.y2);
}
template <bool clip_box>
__device__ EIGEN_STRONG_INLINE Box Clipped(Box box);

template <>
__device__ EIGEN_STRONG_INLINE Box Clipped<false>(Box box) {
  return std::move(box);
}

// This could lead to box size=0 if used incorrectly!
template <>
__device__ EIGEN_STRONG_INLINE Box Clipped<true>(Box box) {
  box.x1 = fminf(fmaxf(0.0, box.x1), 1.0);
  box.y1 = fminf(fmaxf(0.0, box.y1), 1.0);
  box.x2 = fminf(fmaxf(0.0, box.x2), 1.0);
  box.y2 = fminf(fmaxf(0.0, box.y2), 1.0);
  return std::move(box);
}

template <typename T>
__device__ EIGEN_STRONG_INLINE bool CheckBit(const T* bit_mask, int bit) {
  constexpr int kShiftLen = NumBits(8 * sizeof(T)) - 1;
  constexpr int kRemainderMask = 8 * sizeof(T) - 1;
  const int bin = bit >> kShiftLen;
  return (bit_mask[bin] >> (bit & kRemainderMask)) & 1;
}
template <typename T>
__device__ EIGEN_STRONG_INLINE void SetBit(T* bit_mask, int bit) {
  constexpr int kShiftLen = NumBits(8 * sizeof(T)) - 1;
  constexpr int kRemainderMask = 8 * sizeof(T) - 1;
  int bin = bit >> kShiftLen;
  atomicOr(bit_mask + bin, T(1) << (bit & kRemainderMask));
}

template <typename T>
__device__ EIGEN_STRONG_INLINE void ClearBit(T* bit_mask, int bit) {
  constexpr int kShiftLen = NumBits(8 * sizeof(T)) - 1;
  constexpr int kRemainderMask = 8 * sizeof(T) - 1;
  int bin = bit >> kShiftLen;
  atomicAnd(bit_mask + bin, ~(T(1) << (bit & kRemainderMask)));
}

__global__ void FlipBoxes(Box* boxes, const int* num_batch_boxes,
                          const int* box_strides, const int batch_size) {
  for (const int y : CudaGridRangeY(batch_size)) {
    int box_offset = box_strides[y];
    Box* curr_boxes = boxes + box_offset;
    for (int i : GpuGridRangeX(num_batch_boxes[y])) {
      Flipped<true>(curr_boxes[i]);
    }
  }
}

// This op applies Look-Forward style nms. More efficient when max_accept is
// closer to the maximum number of boxes
__launch_bounds__(1024) __global__
    void NMSApplyAndReduceForward(const Box* boxes, const int* num_batch_boxes,
                                  const int* box_strides, const int max_accept,
                                  int max_number_of_boxes, float iou_threshold,
                                  int* selected_counts,
                                  char* selection_results) {
  // boxes, pointer to boxes.
  // num_batch_boxes, number of boxes in each batch entry.
  // box_strides, start offset for each batch wrt boxes.
  // max_accept, maximum box from each batch entry to accept.
  // iou_threshold, threshold for suppression.
  // selected_counts, number of selected boxes in each batch entry.
  // selection_results, selected mask for gathering selected boxes.
  // Requires batch_size==number of blocks launched since algorithm can only
  // work in a single block and will fail if expanded to multiple blocks

  extern __shared__ int selected[];
  int num_boxes =
      num_batch_boxes[blockIdx.x];  // boxes in current block/image/class
  int bit_mask_len = (num_boxes + sizeof(int) * 8 - 1) / (sizeof(int) * 8);
  int box_stride = box_strides[blockIdx.x];
  const Box* current_boxes = boxes + box_stride;
  for (int box = threadIdx.x; box < bit_mask_len; box += blockDim.x) {
    selected[box] = 0xFFFFFFFF;
  }
  __syncthreads();
  int accepted_boxes = 0;  // box 0 is always accepted
  int last_accepted = -1;
  for (int box = 0; box < num_boxes; ++box) {
    if (!CheckBit(selected, box)) {
      continue;
    }
    accepted_boxes += 1;
    last_accepted = box;
    if (accepted_boxes >= max_accept) {
      break;
    }
    const Box b = current_boxes[box];
    float box_area = (b.x2 - b.x1) * (b.y2 - b.y1);
    // int start = ((box + 1) / blockDim.x) * blockDim.x;
    for (int target = threadIdx.x; target < num_boxes; target += blockDim.x) {
      if (target <= box) continue;
      if (OverThreshold(&b, current_boxes + target, box_area, iou_threshold)) {
        ClearBit<int>(selected, target);
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    selected_counts[blockIdx.x] = accepted_boxes;
  }
  __syncthreads();
  char* curr_results = selection_results + box_stride;

  for (int i = threadIdx.x; i < max_number_of_boxes; i += blockDim.x) {
    if (i <= last_accepted) {
      curr_results[i] = CheckBit(selected, i);
    } else {
      curr_results[i] = 0;
    }
  }
}

// This kernel uses Look-Behind algorithm for nms. More efficient when
// max_accept/num_of_boxes is close to 0.
__launch_bounds__(1024) __global__ void NMSApplyAndReduceChunkedInWarp(
    const Box* boxes, const int* num_batch_boxes, const int* box_strides,
    const int max_accept, int max_number_of_boxes, float iou_threshold,
    int* selected_counts, char* selection_results) {
  // boxes, pointer to boxes.
  // num_batch_boxes, number of boxes in each batch entry.
  // box_strides, start offset for each batch wrt boxes.
  // max_accept, maximum box from each batch entry to accept.
  // iou_threshold, threshold for suppression.
  // selected_counts, number of selected boxes in each batch entry.
  // selection_results, selected mask for gathering selected boxes.
  // Requires batch_size==number of blocks launched since algorithm can only
  // work in a single block and will fail if expanded to multiple blocks

  extern __shared__ int shared_buffer[];
  // Shared data layout
  // ChunkSize x box -> chunk boxes
  // ChunkSize x float ->chunk areas
  // bit_mask_len x int -> selected bitmask
  // chunk_mask_len x int ->chunk_mask

  const int num_boxes = num_batch_boxes[blockIdx.x];
  const int box_stride = box_strides[blockIdx.x];
  const int bit_mask_len =
      (num_boxes + sizeof(int) * 8 - 1) / (sizeof(int) * 8);
  const int chunk_mask_size =
      (kNmsReductionChunkSize + 31) / 32;  // in integers
  const int num_chunks =
      (num_boxes + kNmsReductionChunkSize - 1) / kNmsReductionChunkSize;

  // shared memory pointers
  Box* const chunk = (Box*)shared_buffer;
  float* const areas = (float*)(chunk + kNmsReductionChunkSize);
  int* const selected = (int*)(areas + kNmsReductionChunkSize);
  int* const chunk_mask = selected + bit_mask_len;
  // global memory pointer
  const Box* current_boxes = boxes + box_stride;

  for (int box = threadIdx.x; box < bit_mask_len; box += blockDim.x) {
    selected[box] = 0xFFFFFFFF;
  }
  __syncthreads();

  int accepted_boxes = 0;  // box 0 is always accepted
  int last_accepted = -1;

  for (int c = 0; c < num_chunks; ++c) {
    const int chunk_begin = c * kNmsReductionChunkSize;
    const int chunk_size = min(kNmsReductionChunkSize, num_boxes - chunk_begin);
    for (int i = threadIdx.x; i < chunk_size; i += blockDim.x) {
      SetBit(chunk_mask, i);
      chunk[i] = current_boxes[chunk_begin + i];
      Box* b = chunk + i;
      areas[i] = (b->y2 - b->y1) * (b->x2 - b->x1);
    }
    __syncthreads();
    // previous chunks
    for (int target = threadIdx.x; target < chunk_begin; target += blockDim.x) {
      bool active = CheckBit(selected, target);
      bool any_accept = __any_sync(0xFFFFFFFF, active);
      if (!any_accept) {  // if all warp is disabled, continue
        continue;
      }
      int eliminated_chunks = 0;
      for (int in_chunk = 0; in_chunk < chunk_mask_size; in_chunk++) {
        int chunk_state = chunk_mask[in_chunk];
        bool warp_active = __any_sync(0xFFFFFFFF, chunk_state);
        if (!warp_active) {
          eliminated_chunks += 1;
          continue;
        }
        int in_warp_mask = 0xFFFFFFFF;
        int const warp_offset = in_chunk * 32;
        if (active) {
          for (int in_warp = 0; in_warp < 32; ++in_warp) {
            if (OverThreshold(chunk + warp_offset + in_warp,
                              current_boxes + target,
                              areas[warp_offset + in_warp], iou_threshold)) {
              in_warp_mask ^= (1 << in_warp);
            }
          }
        }
        for (int o = 16; o > 0; o >>= 1) {
          in_warp_mask &= __shfl_down_sync(0xFFFFFFFF, in_warp_mask, o);
        }
        if ((threadIdx.x & 31) == 0) {
          atomicAnd(chunk_mask + in_chunk, in_warp_mask);
        }
      }
      bool all_eliminated = (eliminated_chunks == chunk_mask_size);
      bool dont_have_boxes = __all_sync(0xFFFFFFFF, all_eliminated);
      if (dont_have_boxes) break;
    }
    __syncthreads();
    // Within a chunk
    for (int i = 0; i < chunk_size; ++i) {
      // look backward
      bool active = CheckBit<int>(chunk_mask, i);
      int accepted = 1;
      if (active) {
        for (int j = threadIdx.x; j < i; j += blockDim.x) {
          if (CheckBit(chunk_mask, j) &&
              OverThreshold(chunk + i, chunk + j, areas[i], iou_threshold)) {
            accepted = 0;
          }
        }
      }
      bool all_accept = __all_sync(0xFFFFFFFF, accepted);
      // 0th thread in warp has reduced value
      if (!all_accept && ((threadIdx.x & 31) == 0)) {
        ClearBit<int>(chunk_mask, i);
      }
      __syncthreads();
      if (CheckBit(chunk_mask, i)) {
        accepted_boxes += 1;
        last_accepted = chunk_begin + i;
        if (accepted_boxes >= max_accept) {
          break;
        }
      } else {
        if (threadIdx.x == 0) {
          ClearBit<int>(selected, chunk_begin + i);
        }
      }
    }
    __syncthreads();
    if (accepted_boxes >= max_accept) break;
  }

  if (threadIdx.x == 0) {
    selected_counts[blockIdx.x] = accepted_boxes;
  }
  __syncthreads();
  char* curr_results = selection_results + box_stride;
  for (int i = threadIdx.x; i < max_number_of_boxes; i += blockDim.x) {
    if (i <= last_accepted) {
      curr_results[i] = CheckBit(selected, i);
    } else {
      curr_results[i] = 0;
    }
  }
}

template <typename Index>
__device__ __forceinline__ void SelectHelper(const Index i_selected,
                                             const Index i_original) {}

template <typename Index, typename T, typename... Args>
__device__ __forceinline__ void SelectHelper(const Index i_selected,
                                             const Index i_original,
                                             const T* original, T* selected,
                                             Args... args) {
  selected[i_selected] = original[i_original];
  SelectHelper(i_selected, i_original, args...);
}

__global__ void BroadcastBoxes(const int num_boxes, const int num_classes,
                               const int batch_size, const Box* input_boxes,
                               Box* output_boxes) {
  constexpr int cache_size = 2048;
  __shared__ Box BoxCache[cache_size];
  int num_chunks = (num_boxes + cache_size - 1) / cache_size;
  int batch_offset =
      blockIdx.y * num_boxes;  // input boxes are given only once per batch
  for (int c = 0; c < num_chunks; ++c) {
    int chunk_start = c * cache_size;
    int chunk_end = min(num_boxes - chunk_start, cache_size);
    if (threadIdx.y == 0) {
      for (int x = threadIdx.x; x < chunk_end; x += blockDim.x) {
        BoxCache[x] = input_boxes[batch_offset + chunk_start + x];
      }
    }
    __syncthreads();

    for (int cl = blockIdx.x * blockDim.y + threadIdx.y; cl < num_classes;
         cl += blockDim.y * gridDim.x) {
      int cl_offset = cl * num_boxes + batch_offset * num_classes + chunk_start;

      for (int x = threadIdx.x; x < chunk_end; x += blockDim.x) {
        output_boxes[cl_offset + x] = BoxCache[x];
      }
    }
    __syncthreads();
  }
}

// Batch version of IndexMultiSelect, num_elemets contains number of elements in
// each entry offsets is the offsets of batch entries,
template <typename Index, typename T, typename... Args>
__global__ void BatchedIndexMultiSelect(const int* num_elements,
                                        const int* input_strides,
                                        const int* output_strides,
                                        int batch_size, const Index* indices,
                                        const T* original, T* selected,
                                        Args... args) {
  for (const int y : CudaGridRangeY(batch_size)) {
    int istride = input_strides[y];
    int ostride = output_strides[y];
    for (const int idx : CudaGridRangeX(num_elements[y])) {
      SelectHelper(idx + ostride, istride + indices[idx + istride], original,
                   selected, args...);
    }
  }
}
// Batch version of IndexMultiSelect, num_elemets contains number of elements in
// each entry offsets is the offsets of batch entries,
__global__ void TransposedBoxSelect(
    const int* num_elements,     // Number of selected elements for each class
    const int batch_size,        // B
    const int num_classes,       // NC
    const int num_box_classes,   // NbC could be 1
    const int num_boxes,         // NB
    const int max_output_boxes,  // Max num selected (NS)
    const int* input_indices,    // [B,NC,NB]
    const Box* original,         // [B,NB,NbC]
    Box* selected                // [B,NC,NS]
) {
  for (const int batch : GpuGridRangeZ(batch_size)) {
    // CombinedNMS can compress input boxes by specifying each box once
    // regardless of class count
    const Box* batch_boxes = original + batch * num_boxes * num_box_classes;
    for (const int y : GpuGridRangeY(num_classes)) {
      Box* output_boxes = selected + batch * num_classes * max_output_boxes +
                          y * max_output_boxes;
      const int* class_indices =
          input_indices + batch * num_classes * num_boxes + y * num_boxes;
      for (const int idx :
           GpuGridRangeX(num_elements[y + batch * num_classes])) {
        int input_pos = class_indices[idx] * num_box_classes +
                        ((num_box_classes == 1) ? 0 : y);  // Box Id+class
        output_boxes[idx] = batch_boxes[input_pos];
      }
    }
  }
}

template <typename T>
__global__ void Iota(const int num_elements, const T offset, T* to_fill,
                     int batch_size = 1) {
  for (int i : CudaGridRangeY(batch_size)) {
    T* img = to_fill + (i * num_elements);
    for (int idx : CudaGridRangeX(num_elements)) {
      img[idx] = static_cast<T>(idx) + offset;
    }
  }
}
__global__ void BatchIndexScatter(const int* batch_counts, const int* offsets,
                                  int batch_size, const int* input_indices,
                                  int* output_indices) {
  for (int i : GpuGridRangeY(batch_size)) {
    int input_offset = 0;
    for (int k = 0; k < i; ++k) input_offset += batch_counts[k];
    int* output = output_indices + offsets[i];
    const int* input = input_indices + input_offset;
    for (int idx : GpuGridRangeX(batch_counts[i])) {
      output[idx] = input[idx];
    }
  }
}

__launch_bounds__(1024) __global__
    void FindPartitionIndex(const float* sorted_scores, const int* num_boxes,
                            const int max_boxes, const float threshold,
                            int* partitions) {
  // do this block by block it shouldn't matter much
  size_t offset = 0;
  int n_boxes = num_boxes[blockIdx.x];
  if (n_boxes == 0) {
    partitions[blockIdx.x] = 0;
    return;
  }
  for (int i = 0; i < blockIdx.x; ++i) offset += max_boxes;
  const float* scores = sorted_scores + offset;
  __syncthreads();
  for (int i = threadIdx.x; i < n_boxes - 1; i += blockDim.x) {
    if (scores[i] > threshold) {
      if (scores[i + 1] <= threshold) {
        partitions[blockIdx.x] = i + 1;
      }
    } else {
      if (i == 0) {
        partitions[blockIdx.x] = 0;
      };
      break;
    }
  }
  if (threadIdx.x == 0 && scores[n_boxes - 1] > threshold) {
    partitions[blockIdx.x] = n_boxes;
  }
}

template <bool clipped>
__global__ void CombineClasses(
    const int* indices,  // Selected box indices per batch per class from
                         // DoNMSBatched [batch,num_classes,per_class_stride]
    const int per_class_stride,  // output per_class stride from DoNmsBatched
    const Box* boxes,  // transposed boxes, [batch_size*numclasses,num_boxes,4]
    const float* scores,  //  transposed scores [batch,num_classes,num_boxes]
    const int* num_selections,  // number of selected boxes in each class from
                                // [batch*num_classes] DoNMSBatched
    const int num_boxes,        // number of input boxes
    const int num_batches,      // number of batches
    const int num_classes,      // number of classes per batch
    bool pad_per_class,         // whether final ouput is padded per class
    int max_output,             // maximum output for final tensors
    Box* out_boxes,             // output boxes concatenated
    float* out_scores,          // output_scores
    float* out_classes,         // output classes
    int* merged_counts,         // output_counts and start offsets for sorting
    int* final_counts,          // counts for final ouput tensors
    int* out_sort_indices) {    // indices for sorting

  for (int b : CudaGridRangeZ(num_batches)) {
    const Box* input_boxes = boxes + (b * num_classes) * num_boxes;
    const int* batch_indices = indices + (b * (per_class_stride * num_classes));
    const float* batch_in_scores = scores + (b * num_boxes * num_classes);
    Box* batch_out_boxes = out_boxes + (b * (per_class_stride * num_classes));
    float* batch_out_scores =
        out_scores + (b * (per_class_stride * num_classes));
    float* batch_out_classes =
        out_classes + (b * (per_class_stride * num_classes));
    int* batch_out_sort_indices =
        out_sort_indices + (b * (per_class_stride * num_classes));
    for (int c : CudaGridRangeY(num_classes)) {
      const Box* class_boxes = input_boxes + (c * num_boxes);
      const int* class_indices = batch_indices + (c * per_class_stride);
      const float* class_scores = batch_in_scores + c * num_boxes;
      int out_offset = 0;
      for (int i = b * num_classes; i < (b * num_classes + c); ++i) {
        out_offset += num_selections[i];
      }
      for (int idx : CudaGridRangeX(num_selections[b * num_classes + c])) {
        int input_index = class_indices[idx];
        batch_out_boxes[out_offset + idx] =
            Clipped<clipped>(class_boxes[input_index]);
        batch_out_scores[out_offset + idx] = class_scores[input_index];
        batch_out_classes[out_offset + idx] = c;
        batch_out_sort_indices[out_offset + idx] = out_offset + idx;
      }
    }
    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (blockIdx.x == 0) &&
        (blockIdx.y == 0)) {
      int sum = 0;
      for (int i = 0; i < num_classes; ++i) {
        sum += num_selections[b * num_classes + i];
      }
      merged_counts[b] = b * (per_class_stride * num_classes);
      merged_counts[num_batches + b] =
          b * per_class_stride * num_classes + sum;  // end of the batch
      int max_output_size = num_classes * per_class_stride;
      if (pad_per_class) {
        max_output_size = min(min(max_output, max_output_size), sum);
      } else {
        max_output_size = min(max_output, sum);
      }
      final_counts[b] = max_output_size;
    }
  }
}

Status DoNMSBatched(OpKernelContext* context, const Tensor& boxes,
                    const Tensor& scores, const Tensor& box_counts_tensor,
                    const int max_output_size, const float iou_threshold_val,
                    const float score_threshold, bool pad_to_max_output,
                    int* num_saved_outputs, Tensor** output_indices, int kernel,
                    bool pre_sorted_inputs = false) {
  int batch_size = boxes.dim_size(0);
  auto cuda_stream = GetGpuStream(context);
  auto device = context->eigen_gpu_device();
  const int* box_counts = box_counts_tensor.flat<int>().data();
  int max_boxes = 0;
  AllocatorAttributes alloc_attr;
  alloc_attr.set_on_host(true);
  alloc_attr.set_gpu_compatible(true);
  Tensor begin_end_offsets_host_tensor;
  TF_RETURN_IF_ERROR(
      context->allocate_temp(DataType::DT_INT32, TensorShape({2 * batch_size}),
                             &begin_end_offsets_host_tensor, alloc_attr));
  int* begin_end_offsets = begin_end_offsets_host_tensor.flat<int>().data();

  for (int i = 0; i < batch_size; ++i) {
    if (max_boxes < box_counts[i]) {
      max_boxes = box_counts[i];
    }
  }
  for (int i = 0; i < batch_size; ++i) {
    begin_end_offsets[i] = i * max_boxes;
    begin_end_offsets[batch_size + i] = begin_end_offsets[i] + box_counts[i];
  }
  if (max_boxes == 0) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataType::DT_INT32, TensorShape({0}), *output_indices));
    device.memset(num_saved_outputs, 0, sizeof(int) * batch_size);
    return Status::OK();
  }
  // This is actually true for post score selection number of boxes rather than
  // maximum input but that would require copying selection results to host and
  // synchronize.
  if (max_boxes > device.sharedMemPerBlock() * 8) {
    return errors::InvalidArgument(StrCat(
        "This op can not handle more than ", device.sharedMemPerBlock() * 8,
        " boxes in any of the batch elements while maximum number of boxes in "
        "at least one of the elements is ",
        max_boxes));
  }

  size_t cub_temp_storage_bytes = 0;
  cudaError_t cuda_ret = cudaSuccess;
  if (!pre_sorted_inputs) {
    // Calling cub with nullptrs as inputs will make it return
    // workspace size needed for the operation instead of doing the operation.
    // In this specific instance, cub_sort_storage_bytes will contain the
    // necessary workspace size for sorting after the call.
    cuda_ret = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, cub_temp_storage_bytes,
        static_cast<float*>(nullptr),  // scores
        static_cast<float*>(nullptr),  // sorted scores
        static_cast<int*>(nullptr),    // input indices
        static_cast<int*>(nullptr),    // sorted indices
        batch_size * max_boxes,        // Total number of boxes in batch
        batch_size,                    // num segments
        static_cast<int*>(nullptr), static_cast<int*>(nullptr), 0,
        8 * sizeof(float),  // sort all bits
        cuda_stream);
    TF_RETURN_IF_CUDA_ERROR(cuda_ret);
    TF_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  }
  size_t flagged_buffer_size = 0;
  cuda_ret =
      cub::DeviceSelect::Flagged(static_cast<void*>(nullptr),  // temp_storage
                                 flagged_buffer_size,
                                 static_cast<int*>(nullptr),   // input
                                 static_cast<char*>(nullptr),  // selection flag
                                 static_cast<int*>(nullptr),   // selected items
                                 static_cast<int*>(nullptr),   // num_selected
                                 max_boxes * batch_size, device.stream());
  TF_RETURN_IF_CUDA_ERROR(cuda_ret);
  TF_RETURN_IF_CUDA_ERROR(cudaGetLastError());

  cub_temp_storage_bytes =
      std::max(cub_temp_storage_bytes, flagged_buffer_size);
  Tensor d_cub_temp_buffer;
  Tensor d_indices;
  Tensor d_selected_indices;
  Tensor d_sorted_boxes;
  Tensor d_sorted_indices;
  Tensor d_sorted_scores;
  Tensor d_selection_mask;

  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({(int64)cub_temp_storage_bytes}),
      &d_cub_temp_buffer));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({batch_size, max_boxes}), &d_indices));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({batch_size * max_boxes * 2}),
      &d_selected_indices));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({max_boxes * batch_size}),
      &d_selection_mask));
  // If the inputs are already sorted, we don't need to sort them again.
  if (!pre_sorted_inputs) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataType::DT_INT32, TensorShape({max_boxes * batch_size}),
        &d_sorted_indices));
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataType::DT_FLOAT, TensorShape({max_boxes * batch_size}),
        &d_sorted_scores));
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataType::DT_FLOAT, TensorShape({max_boxes * batch_size, 4}),
        &d_sorted_boxes));
  } else {
    d_sorted_indices = d_indices;
    d_sorted_scores = scores;
    d_sorted_boxes = boxes;
  }
  // this will return sorted scores and their indices
  // initialize box and score indices
  auto config2d = GetGpu2DLaunchConfig(max_boxes, batch_size, device);

  TF_RETURN_IF_ERROR(GpuLaunchKernel(
      Iota<int>, config2d.block_count, config2d.thread_per_block, 0,
      device.stream(), max_boxes, 0, d_indices.flat<int>().data(), batch_size));
  Tensor device_box_counts_tensor;
  TF_RETURN_IF_ERROR(context->allocate_temp(DataType::DT_INT32,
                                            TensorShape({batch_size}),
                                            &device_box_counts_tensor));
  Tensor device_begin_end_offsets_tensor;
  TF_RETURN_IF_ERROR(context->allocate_temp(DataType::DT_INT32,
                                            TensorShape({2 * batch_size}),
                                            &device_begin_end_offsets_tensor));
  Tensor device_selected_counts_tensor;
  TF_RETURN_IF_ERROR(context->allocate_temp(DataType::DT_INT32,
                                            TensorShape({batch_size}),
                                            &device_selected_counts_tensor));

  int* device_box_counts = device_box_counts_tensor.flat<int>().data();

  int* device_begin_offsets =
      device_begin_end_offsets_tensor.flat<int>().data();

  int* device_selected_counts =
      device_selected_counts_tensor.flat<int>().data();
  device.memcpyHostToDevice(device_begin_offsets, begin_end_offsets,
                            sizeof(int) * batch_size * 2);
  if (!pre_sorted_inputs) {
    cuda_ret = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_cub_temp_buffer.flat<int8>().data(), cub_temp_storage_bytes,
        scores.flat<float>().data(),           // scores
        d_sorted_scores.flat<float>().data(),  // sorted scores
        d_indices.flat<int>().data(),          // input indices
        d_sorted_indices.flat<int>().data(),   // sorted indices
        batch_size * max_boxes,                // Total number of boxes in batch
        batch_size,                            // num segments
        device_begin_offsets, device_begin_offsets + batch_size, 0,
        8 * sizeof(float),  // sort all bits
        cuda_stream);
    TF_RETURN_IF_CUDA_ERROR(cuda_ret);
  }
  if (score_threshold > std::numeric_limits<float>::lowest()) {
    device.memcpyHostToDevice(device_box_counts, box_counts,
                              sizeof(int) * batch_size);
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        FindPartitionIndex, batch_size, 1024, 0, device.stream(),
        d_sorted_scores.flat<float>().data(), device_box_counts, max_boxes,
        score_threshold, device_box_counts));
  } else {
    device.memcpyHostToDevice(device_box_counts, box_counts,
                              sizeof(int) * batch_size);
  }
  if (!pre_sorted_inputs) {
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        BatchedIndexMultiSelect<int, float4>, config2d.block_count,
        config2d.thread_per_block, 0, device.stream(), device_box_counts,
        device_begin_offsets, device_begin_offsets, batch_size,
        d_sorted_indices.flat<int>().data(),
        reinterpret_cast<const float4*>(boxes.flat<float>().data()),
        reinterpret_cast<float4*>(d_sorted_boxes.flat<float>().data())));
  }
  config2d = GetGpu2DLaunchConfig(max_boxes, batch_size, device);
  Box* sorted_boxes =
      reinterpret_cast<Box*>(d_sorted_boxes.flat<float>().data());
  char* selection_mask =
      reinterpret_cast<char*>(d_selection_mask.flat<int8>().data());
  // Make sure that the boxes are flipped to ensure x1<x2 and y1<y2

  TF_RETURN_IF_ERROR(GpuLaunchKernel(
      FlipBoxes, config2d.block_count, config2d.thread_per_block, 0,
      device.stream(), sorted_boxes, device_box_counts, device_begin_offsets,
      batch_size));
  int bitmask_length_bytes =
      ((max_boxes + sizeof(int) * 8 - 1) / (sizeof(int) * 8)) * sizeof(int);
  // simple heuristics for auto selecting which kernel to use
  if (kernel == -1) {
    float suppression_ratio = (double)max_output_size / (double)max_boxes;
    kernel = 0;
    if (max_output_size <= 2000) {
      kernel = 0;
    } else if ((max_boxes >= 10000) && (suppression_ratio >= .20)) {
      kernel = 1;
    } else if (max_output_size >= 50000) {
      kernel = 1;
    }
  }
  if (kernel == 1) {  // use Look-Forward kernel
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        NMSApplyAndReduceForward, batch_size, 1024, bitmask_length_bytes,
        device.stream(), sorted_boxes, device_box_counts, device_begin_offsets,
        max_output_size, max_boxes, iou_threshold_val, num_saved_outputs,
        selection_mask));
  } else if (kernel == 0) {  // Use Look-Backward kernel
    int shm_size = bitmask_length_bytes +
                   (sizeof(float) * 5) * kNmsReductionChunkSize +
                   (kNmsReductionChunkSize + sizeof(int) * 8 - 1) / 8;
    shm_size += 8;
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        NMSApplyAndReduceChunkedInWarp, batch_size, 1024, shm_size,
        device.stream(), sorted_boxes, device_box_counts, device_begin_offsets,
        max_output_size, max_boxes, iou_threshold_val, num_saved_outputs,
        selection_mask));
  }
  // There is no guarantee that boxes are given in the for x1<x2 and/or y1<y2,
  Tensor selected_counts_host_tensor;
  TF_RETURN_IF_ERROR(
      context->allocate_temp(DataType::DT_INT32, TensorShape({batch_size}),
                             &selected_counts_host_tensor, alloc_attr));

  int* selected_counts = selected_counts_host_tensor.flat<int>().data();
  device.memcpyDeviceToHost(selected_counts, num_saved_outputs,
                            sizeof(int) * batch_size);
  device.synchronize();
  int num_outputs = 0;
  int max_selected = 0;
  for (int i = 0; i < batch_size; ++i) {
    max_selected = std::max(max_selected, selected_counts[i]);
  }

  num_outputs = std::min(max_selected, (int)max_output_size);
  if (pad_to_max_output && num_outputs != max_output_size) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataType::DT_INT32, TensorShape({batch_size, max_output_size}),
        *output_indices));
    auto config = GetGpuLaunchConfig((*output_indices)->NumElements(), device);
    TF_CHECK_OK(GpuLaunchKernel(SetZero<int>, config.block_count,
                                config.thread_per_block, 0, device.stream(),
                                config.virtual_thread_count,
                                (*output_indices)->flat<int>().data()));
    num_outputs = max_output_size;
  } else {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataType::DT_INT32, TensorShape({batch_size, num_outputs}),
        *output_indices));
  }
  if (num_outputs == 0) {
    device.memset(num_saved_outputs, 0, sizeof(int) * batch_size);
    return Status::OK();
  }

  int* device_selected_indices = d_selected_indices.flat<int>().data();
  cuda_ret = cub::DeviceSelect::Flagged(
      d_cub_temp_buffer.flat<int8>().data(),  // temp_storage
      cub_temp_storage_bytes,
      d_indices.flat<int>().data(),  // input
      selection_mask,                // selection flag
      device_selected_indices,       // output selected items
      device_selected_counts,        // output counts
      max_boxes * batch_size,        // number of elements
      device.stream());
  TF_RETURN_IF_CUDA_ERROR(cuda_ret);


  config2d = GetGpu2DLaunchConfig(max_boxes, batch_size, device);
  TF_RETURN_IF_ERROR(GpuLaunchKernel(
      BatchIndexScatter, config2d.block_count, config2d.thread_per_block, 0,
      device.stream(), num_saved_outputs, device_begin_offsets, batch_size,
      device_selected_indices,
      device_selected_indices + (max_boxes * batch_size)));
  for (int i = 0; i < batch_size; ++i) {
    begin_end_offsets[i] = i * num_outputs;
  }
  device.memcpyHostToDevice(device_begin_offsets + batch_size,
                            begin_end_offsets, sizeof(int) * batch_size);

  TF_RETURN_IF_ERROR(
      // input and output offsets are different!
      GpuLaunchKernel(BatchedIndexMultiSelect<int, int>, config2d.block_count,
                      config2d.thread_per_block, 0, device.stream(),
                      num_saved_outputs, device_begin_offsets,
                      device_begin_offsets + batch_size, batch_size,
                      device_selected_indices + (max_boxes * batch_size),
                      d_sorted_indices.flat<int>().data(),
                      (*output_indices)->flat<int>().data()));

  return Status::OK();
}

Status CheckValidInputs(const Tensor& boxes, const Tensor& scores,
                        const Tensor& max_output_size,
                        const Tensor& iou_threshold,
                        const Tensor* box_counts = nullptr,
                        bool is_batch = false) {
  if (!TensorShapeUtils::IsScalar(max_output_size.shape())) {
    return errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                   max_output_size.shape().DebugString());
  }
  if (!TensorShapeUtils::IsScalar(iou_threshold.shape())) {
    return errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                   iou_threshold.shape().DebugString());
  }
  const float iou_threshold_val = iou_threshold.scalar<float>()();
  if (iou_threshold_val < 0 || iou_threshold_val > 1) {
    return errors::InvalidArgument("iou_threshold must be in [0, 1]");
  }
  int batch_offset = (is_batch ? 1 : 0);
  if (boxes.dims() != 2 + batch_offset) {
    return errors::InvalidArgument(
        StrCat("boxes must be a rank ", 2 + batch_offset, " tensor!"));
  }
  int num_boxes = boxes.dim_size(batch_offset);
  if (boxes.dim_size(1 + batch_offset) != 4) {
    return errors::InvalidArgument(
        StrCat("boxes must be", (is_batch ? " Bx" : " "), "Nx4"));
  }
  if (scores.dims() != 1 + batch_offset) {
    return errors::InvalidArgument(
        StrCat("scores must be ", (is_batch ? "2D" : "1D"), "!"));
  }
  if (scores.dim_size(batch_offset) != num_boxes) {
    return errors::InvalidArgument(
        "scores has incompatible shape");  // message must be exactly this
                                           // otherwise tests fail!
  }
  if (is_batch && box_counts) {
    if (box_counts->dim_size(0) != boxes.dim_size(0)) {
      return errors::InvalidArgument(
          "Box counts size should be equal to batch size");
    }
  }
  return Status::OK();
}

Status SortScores(OpKernelContext* context, int batch_size, int num_classes,
                  int num_boxes, Tensor* input_scores, Tensor* indices,
                  Tensor* sorted_scores, Tensor* sorted_indices,
                  Tensor* temp_buffer, Tensor* device_offsets,
                  bool use_given_offsets = false) {
  size_t cub_temp_storage_bytes = 0;
  cudaError_t cuda_ret = cudaSuccess;
  auto cuda_stream = GetGpuStream(context);
  auto device = context->eigen_gpu_device();
  cuda_ret = cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr, cub_temp_storage_bytes,
      static_cast<float*>(nullptr),          // scores
      static_cast<float*>(nullptr),          // sorted scores
      static_cast<int*>(nullptr),            // input indices
      static_cast<int*>(nullptr),            // sorted indices
      batch_size * num_classes * num_boxes,  // Total number of boxes in batch
      batch_size * num_classes,              // num segments
      static_cast<int*>(nullptr), static_cast<int*>(nullptr), 0,
      8 * sizeof(float),  // sort all bits
      cuda_stream);
  TF_RETURN_IF_CUDA_ERROR(cuda_ret);
  TF_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  if (temp_buffer->NumElements() == 0 ||
      temp_buffer->NumElements() < cub_temp_storage_bytes) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataType::DT_INT8, TensorShape({(int64)cub_temp_storage_bytes}),
        temp_buffer));
  }
  if (device_offsets->NumElements() == 0 || !use_given_offsets) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataType::DT_INT32, TensorShape({3 * batch_size * num_classes}),
        device_offsets));
    AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    alloc_attr.set_gpu_compatible(true);

    Tensor begin_end_offsets_tensor;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataType::DT_INT32, TensorShape({batch_size * num_classes * 3}),
        &begin_end_offsets_tensor, alloc_attr));
    int* begin_end_offsets = begin_end_offsets_tensor.flat<int>().data();
    for (int i = 0; i < batch_size * num_classes; ++i) {
      begin_end_offsets[i] = i * num_boxes;
      begin_end_offsets[batch_size * num_classes + i] =
          begin_end_offsets[i] + num_boxes;
      begin_end_offsets[i + 2 * batch_size * num_classes] = num_boxes;
    }
    device.memcpyHostToDevice(
        device_offsets->flat<int>().data(), begin_end_offsets,
        sizeof(int) * begin_end_offsets_tensor.NumElements());
  }
  int* device_begin_offsets = device_offsets->flat<int>().data();

  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({batch_size * num_classes, num_boxes}),
      indices));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({batch_size * num_classes, num_boxes}),
      sorted_indices));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({batch_size * num_classes, num_boxes}),
      sorted_scores));

  auto config2d =
      GetGpu2DLaunchConfig(num_boxes, batch_size * num_classes, device);

  TF_RETURN_IF_ERROR(GpuLaunchKernel(
      Iota<int>, config2d.block_count, config2d.thread_per_block, 0,
      device.stream(), num_boxes, 0, indices->template flat<int>().data(),
      batch_size * num_classes));

  TF_RETURN_IF_CUDA_ERROR(cub::DeviceSegmentedRadixSort::SortPairsDescending(
      temp_buffer->flat<int8>().data(), cub_temp_storage_bytes,
      input_scores->flat<float>().data(),    // scores
      sorted_scores->flat<float>().data(),   // sorted scores
      indices->flat<int>().data(),           // input indices
      sorted_indices->flat<int>().data(),    // sorted indices
      batch_size * num_classes * num_boxes,  // Total number of boxes in batch
      batch_size * num_classes,              // num segments
      device_begin_offsets, device_begin_offsets + batch_size * num_classes, 0,
      8 * sizeof(float),  // sort all bits
      cuda_stream));
  return Status::OK();
}

class BatchedNonMaxSuppressionGPUOp : public OpKernel {
 public:
  explicit BatchedNonMaxSuppressionGPUOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("pad_to_max_output_size",
                                             &pad_to_max_output_size_));
    OP_REQUIRES_OK(context, context->GetAttr("algorithm", &kernel_));
  }

  void Compute(OpKernelContext* context) override {
    // boxes: [N ,num_boxes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [N, num_boxes]
    const Tensor& scores = context->input(1);
    // Box_counts in each batch element [N]
    const Tensor& box_counts = context->input(2);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(3);
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(4);
    auto valid = CheckValidInputs(boxes, scores, max_output_size, iou_threshold,
                                  &box_counts, true);
    if (!valid.ok()) {
      context->SetStatus(valid);
      return;
    }

    const Tensor& score_threshold = context->input(5);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const float score_threshold_val = score_threshold.scalar<float>()();

    Tensor* num_outputs_t = nullptr;
    int batch_size = boxes.dim_size(0);
    OP_REQUIRES_OK(
        context, context->allocate_output(
                     1, tensorflow::TensorShape({batch_size}), &num_outputs_t));
    auto device = context->eigen_gpu_device();
    int num_boxes = boxes.dim_size(1);
    if (num_boxes == 0) {
      Tensor* output_indices = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
                                                       &output_indices));
      return;
    }

    const float iou_threshold_val = iou_threshold.scalar<float>()();
    const int64_t output_size = max_output_size.scalar<int>()();
    int* num_outputs = num_outputs_t->flat<int>().data();
    Tensor output_tensor;
    Tensor* output_ptr = &output_tensor;
    OP_REQUIRES_OK(
        context, DoNMSBatched(context, boxes, scores, box_counts, output_size,
                              iou_threshold_val, score_threshold_val,
                              pad_to_max_output_size_, num_outputs, &output_ptr,
                              kernel_));
    context->set_output(0, output_tensor);
    return;
  }

 private:
  bool pad_to_max_output_size_;
  int kernel_;
};

class CombinedNonMaxSuppressionGPUOp : public OpKernel {
 public:
  explicit CombinedNonMaxSuppressionGPUOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("pad_per_class", &pad_per_class_));
    OP_REQUIRES_OK(context, context->GetAttr("clip_boxes", &clip_boxes_));
  }

  void Compute(OpKernelContext* context) override {
    // Unfortunately, data layout of inputs the this op is choosen in unoptimal
    // way So we will have to do a bunch of transposes and temporary copies.
    // boxes: [batch_size, num_boxes, num_classes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [batch_size, num_boxes, num_classes]
    const Tensor& scores = context->input(1);
    OP_REQUIRES(
        context, (boxes.dim_size(0) == scores.dim_size(0)),
        errors::InvalidArgument("boxes and scores must have same batch size"));

    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_size_per_class must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));
    const int max_size_per_class = max_output_size.scalar<int>()();
    // max_total_size: scalar
    const Tensor& max_total_size = context->input(3);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_total_size.shape()),
        errors::InvalidArgument("max_total_size must be 0-D, got shape ",
                                max_total_size.shape().DebugString()));
    const int max_total_size_per_batch = max_total_size.scalar<int>()();
    OP_REQUIRES(context, max_total_size_per_batch > 0,
                errors::InvalidArgument("max_total_size must be > 0"));
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(4);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()),
                errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                        iou_threshold.shape().DebugString()));
    const float iou_threshold_val = iou_threshold.scalar<float>()();

    // score_threshold: scalar
    const Tensor& score_threshold = context->input(5);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const float score_threshold_val = score_threshold.scalar<float>()();

    OP_REQUIRES(context, iou_threshold_val >= 0 && iou_threshold_val <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));

    const int batch_size = boxes.dim_size(0);
    const int num_boxes = boxes.dim_size(1);
    const int num_classes = scores.dim_size(2);
    int total_output = max_total_size_per_batch;
    if (pad_per_class_) {
      total_output = std::min(total_output, num_classes * max_size_per_class);
    }
    OP_REQUIRES(
        context,
        (boxes.dim_size(2) == scores.dim_size(2)) || (boxes.dim_size(2) == 1),
        errors::InvalidArgument("Number of classes in scores and boxes either "
                                "identical or boxes.dim_size(2) should be 1"));
    if (boxes.NumElements() == 0) {  // Empty input
      OP_REQUIRES_OK(context,
                     ReturnEmptyTensors(context, batch_size, total_output));
      return;
    }
    AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    alloc_attr.set_gpu_compatible(true);
    Tensor box_counts_tensor;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataType::DT_INT32,
                                        TensorShape({batch_size * num_classes}),
                                        &box_counts_tensor, alloc_attr));
    Tensor num_selected_per_class_tensor;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataType::DT_INT32,
                                        TensorShape({batch_size * num_classes}),
                                        &num_selected_per_class_tensor));
    auto device = context->eigen_device<GPUDevice>();
    Tensor reshaped_boxes_tensor;
    Tensor reshaped_scores_tensor, transposed_scores_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(
                       DataType::DT_FLOAT,
                       TensorShape({batch_size, num_classes, num_boxes}),
                       &transposed_scores_tensor));
    int* box_counts = box_counts_tensor.flat<int>().data();
    for (int i = 0; i < batch_size * num_classes; ++i) {
      box_counts[i] = num_boxes;
    }
    int max_boxes = 0;
    OP_REQUIRES_OK(context, tensorflow::DoTranspose(device, scores, {{0, 2, 1}},
                                                    &transposed_scores_tensor));
    // do sort and select before broadcast to reduce the box count.
    Tensor indices, sorted_scores, sorted_indices, cub_temp_buffer,
        device_offsets_tensor, device_selected_counts_tensor;
    OP_REQUIRES_OK(
        context,
        SortScores(context, batch_size, num_classes, num_boxes,
                   &transposed_scores_tensor, &indices, &sorted_scores,
                   &sorted_indices, &cub_temp_buffer, &device_offsets_tensor));
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataType::DT_INT32,
                                        TensorShape({batch_size * num_classes}),
                                        &device_selected_counts_tensor));
    if (score_threshold_val > std::numeric_limits<float>::lowest()) {
      int* device_offsets = device_offsets_tensor.flat<int>().data();
      OP_REQUIRES_OK(
          context,
          GpuLaunchKernel(FindPartitionIndex, batch_size * num_classes, 1024, 0,
                          device.stream(), sorted_scores.flat<float>().data(),
                          device_offsets + 2 * batch_size * num_classes,
                          num_boxes, score_threshold_val,
                          device_selected_counts_tensor.flat<int>().data()));
      device.memcpyDeviceToHost(
          box_counts, device_selected_counts_tensor.flat<int>().data(),
          sizeof(int) * device_selected_counts_tensor.NumElements());
      device.synchronize();

      for (int i = 0; i < num_classes * batch_size; ++i) {
        max_boxes = std::max(max_boxes, box_counts[i]);
      }
      if (max_boxes == 0) {
        OP_REQUIRES_OK(context,
                       ReturnEmptyTensors(context, batch_size, total_output));
        return;
      }
      OP_REQUIRES_OK(context,
                     context->allocate_temp(
                         DataType::DT_FLOAT,
                         TensorShape({batch_size * num_classes, max_boxes, 4}),
                         &reshaped_boxes_tensor));
      Tensor sliced_scores;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(
                         DataType::DT_FLOAT,
                         TensorShape({batch_size * num_classes, max_boxes}),
                         &sliced_scores));
      auto config2d =
          GetGpu2DLaunchConfig(max_boxes, batch_size * num_classes, device);
      Tensor score_counts_tensor;
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataType::DT_INT32,
                                  TensorShape({batch_size * num_classes}),
                                  &score_counts_tensor, alloc_attr));
      int* score_counts = score_counts_tensor.flat<int>().data();
      for (int i = 0; i < batch_size * num_classes; ++i) {
        score_counts[i] = i * max_boxes;
      }
      int* device_selected_counts =
          device_selected_counts_tensor.flat<int>().data();

      device.memcpyHostToDevice(
          device_offsets + 2 * batch_size * num_classes, score_counts,
          sizeof(int) * score_counts_tensor.NumElements());
      OP_REQUIRES_OK(
          context, GpuLaunchKernel(
                       BatchedIndexMultiSelect<int, float>,
                       config2d.block_count, config2d.thread_per_block, 0,
                       device.stream(), device_selected_counts, device_offsets,
                       device_offsets + 2 * batch_size * num_classes,
                       batch_size * num_classes, indices.flat<int>().data(),
                       sorted_scores.flat<float>().data(),
                       sliced_scores.flat<float>().data()));
      sorted_scores = sliced_scores;
      auto config3d =
          GetGpu3DLaunchConfig(max_boxes, num_classes, batch_size, device,
                               TransposedBoxSelect, 0, 1024);
      const Box* in_boxes =
          reinterpret_cast<const Box*>(boxes.flat<float>().data());
      Box* out_boxes =
          reinterpret_cast<Box*>(reshaped_boxes_tensor.flat<float>().data());

      OP_REQUIRES_OK(
          context,
          GpuLaunchKernel(
              TransposedBoxSelect, config3d.block_count,
              config3d.thread_per_block, 0, device.stream(),
              device_selected_counts_tensor.flat<int>().data(), batch_size,
              num_classes, (int)boxes.dim_size(2), num_boxes, max_boxes,
              sorted_indices.flat<int>().data(), in_boxes, out_boxes));
    } else {
      auto config3d =
          GetGpu3DLaunchConfig(num_boxes, num_classes, batch_size, device,
                               TransposedBoxSelect, 0, 1024);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(
                         DataType::DT_FLOAT,
                         TensorShape({batch_size * num_classes, num_boxes, 4}),
                         &reshaped_boxes_tensor));

      const Box* in_boxes =
          reinterpret_cast<const Box*>(boxes.flat<float>().data());

      Box* out_boxes =
          reinterpret_cast<Box*>(reshaped_boxes_tensor.flat<float>().data());

      max_boxes = num_boxes;
      int* device_selected_counts = device_offsets_tensor.flat<int>().data() +
                                    batch_size * num_classes * 2;

      OP_REQUIRES_OK(
          context, GpuLaunchKernel(
                       TransposedBoxSelect, config3d.block_count,
                       config3d.thread_per_block, 0, device.stream(),
                       device_selected_counts, batch_size, num_classes,
                       (int)boxes.dim_size(2), num_boxes, num_boxes,
                       sorted_indices.flat<int>().data(), in_boxes, out_boxes));
    }
    Tensor output_indices_tensor;
    Tensor* output_indices_ptr = &output_indices_tensor;
    Tensor per_class_outputs_tensor;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataType::DT_INT32,
                                        TensorShape({batch_size * num_classes}),
                                        &per_class_outputs_tensor));
    int* per_class_outputs = per_class_outputs_tensor.flat<int>().data();
    OP_REQUIRES_OK(
        context, DoNMSBatched(context, reshaped_boxes_tensor, sorted_scores,
                              box_counts_tensor, max_size_per_class,
                              iou_threshold_val, score_threshold_val, true,
                              per_class_outputs, &output_indices_ptr, 0, true));
    Tensor gathered_boxes_tensor;
    OP_REQUIRES_OK(
        context,
        context->allocate_temp(
            DataType::DT_FLOAT,
            TensorShape({batch_size, num_classes * max_size_per_class, 4}),
            &gathered_boxes_tensor));
    Tensor gathered_scores_tensor;
    OP_REQUIRES_OK(
        context,
        context->allocate_temp(
            DataType::DT_FLOAT,
            TensorShape({batch_size, num_classes * max_size_per_class}),
            &gathered_scores_tensor));
    Tensor gathered_classes_tensor;
    OP_REQUIRES_OK(
        context,
        context->allocate_temp(
            DataType::DT_FLOAT,
            TensorShape({batch_size, num_classes * max_size_per_class}),
            &gathered_classes_tensor));
    Tensor out_sort_indices_tensor;
    OP_REQUIRES_OK(
        context,
        context->allocate_temp(
            DataType::DT_INT32,
            TensorShape({batch_size, num_classes * max_size_per_class}),
            &out_sort_indices_tensor));
    Tensor merged_box_counts_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataType::DT_INT32,
                                          TensorShape({batch_size * 2}),
                                          &merged_box_counts_tensor));
    Tensor final_counts_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DataType::DT_INT32,
                                                   TensorShape({batch_size}),
                                                   &final_counts_tensor));

    int* post_nms_indices = output_indices_tensor.flat<int>().data();
    Box* reshaped_boxes =
        reinterpret_cast<Box*>(reshaped_boxes_tensor.flat<float>().data());
    float* reshaped_scores = sorted_scores.flat<float>().data();
    Box* out_boxes =
        reinterpret_cast<Box*>(gathered_boxes_tensor.flat<float>().data());
    float* out_classes = gathered_classes_tensor.flat<float>().data();
    float* out_scores = gathered_scores_tensor.flat<float>().data();
    int* merged_counts = merged_box_counts_tensor.flat<int>().data();
    int* out_sort_indices = out_sort_indices_tensor.flat<int>().data();
    int* final_counts = final_counts_tensor.flat<int>().data();

    if (clip_boxes_) {
      Gpu3DLaunchConfig config3d =
          GetGpu3DLaunchConfig(max_size_per_class, num_classes, batch_size,
                               device, CombineClasses<true>, 0, 1024);

      OP_REQUIRES_OK(
          context,
          GpuLaunchKernel(
              CombineClasses<true>, config3d.block_count,
              config3d.thread_per_block, 0, device.stream(), post_nms_indices,
              max_size_per_class, reshaped_boxes, reshaped_scores,
              per_class_outputs, max_boxes, batch_size, num_classes,
              pad_per_class_, max_total_size_per_batch, out_boxes, out_scores,
              out_classes, merged_counts, final_counts, out_sort_indices));
    } else {
      Gpu3DLaunchConfig config3d =
          GetGpu3DLaunchConfig(max_size_per_class, num_classes, batch_size,
                               device, CombineClasses<false>, 0, 1024);

      OP_REQUIRES_OK(
          context,
          GpuLaunchKernel(
              CombineClasses<false>, config3d.block_count,
              config3d.thread_per_block, 0, device.stream(), post_nms_indices,
              max_size_per_class, reshaped_boxes, reshaped_scores,
              per_class_outputs, max_boxes, batch_size, num_classes,
              pad_per_class_, max_total_size_per_batch, out_boxes, out_scores,
              out_classes, merged_counts, final_counts, out_sort_indices));
    }

    Tensor post_nms_sorted_scores, post_nms_sorted_indices,
        post_nms_device_offsets;
    OP_REQUIRES_OK(
        context,
        SortScores(context, batch_size, 1, max_size_per_class * num_classes,
                   &gathered_scores_tensor, &out_sort_indices_tensor,
                   &post_nms_sorted_scores, &post_nms_sorted_indices,
                   &cub_temp_buffer, &merged_box_counts_tensor,true));
    Tensor* output_boxes_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({batch_size, total_output, 4}),
                                &output_boxes_tensor));
    Tensor* output_scores_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({batch_size, total_output}),
                                &output_scores_tensor));
    Tensor* output_classes_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                2, TensorShape({batch_size, total_output}),
                                &output_classes_tensor));
    context->set_output(3, final_counts_tensor);
    Box* final_boxes =
        reinterpret_cast<Box*>(output_boxes_tensor->flat<float>().data());
    float* final_classes = output_classes_tensor->flat<float>().data();
    float* final_scores = output_scores_tensor->flat<float>().data();
    auto zero_config = GetGpuLaunchConfig(batch_size * total_output, device);
    OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(SetZero<float>, zero_config.block_count,
                        zero_config.thread_per_block, 0, device.stream(),
                        zero_config.virtual_thread_count, final_classes));
    OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(SetZero<float>, zero_config.block_count,
                        zero_config.thread_per_block, 0, device.stream(),
                        zero_config.virtual_thread_count, final_scores));
    zero_config = GetGpuLaunchConfig(batch_size * total_output * 4, device);
    OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(SetZero<float>, zero_config.block_count,
                        zero_config.thread_per_block, 0, device.stream(),
                        zero_config.virtual_thread_count,
                        reinterpret_cast<float*>(final_boxes)));
    for (int i = 0; i < batch_size; ++i) {
      box_counts[i] = i * total_output;
    }

    // reuse GPU tensor for selection strides
    device.memcpyHostToDevice(per_class_outputs, box_counts,
                              sizeof(int) * batch_size);
    auto config2d = GetGpu2DLaunchConfig(total_output, batch_size, device);
    int* psorted_indices = post_nms_sorted_indices.flat<int>().data();

    OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(
            BatchedIndexMultiSelect<int, Box, float*, float*, float*, float*>,
            config2d.block_count, config2d.thread_per_block, 0, device.stream(),
            final_counts, merged_counts, per_class_outputs, batch_size,
            psorted_indices, out_boxes, final_boxes, out_scores, final_scores,
            out_classes, final_classes));
  }

 private:
  Status ReturnEmptyTensors(OpKernelContext* context, int batch_size,
                            int total_output) {
    Tensor* output_boxes_tensor = nullptr;
    TF_RETURN_IF_ERROR(context->allocate_output(
        0, TensorShape({batch_size, total_output, 4}), &output_boxes_tensor));
    Tensor* output_scores_tensor = nullptr;

    TF_RETURN_IF_ERROR(context->allocate_output(
        1, TensorShape({batch_size, total_output}), &output_scores_tensor));
    Tensor* output_classes_tensor = nullptr;
    TF_RETURN_IF_ERROR(context->allocate_output(
        2, TensorShape({batch_size, total_output}), &output_classes_tensor));
    Tensor* final_counts = nullptr;
    TF_RETURN_IF_ERROR(
        context->allocate_output(3, TensorShape({batch_size}), &final_counts));
    auto device = context->eigen_gpu_device();
    device.memset(final_counts->flat<int>().data(), 0,
                  final_counts->NumElements() * sizeof(int));
    auto zero_config = GetGpuLaunchConfig(batch_size * total_output, device);
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        SetZero<float>, zero_config.block_count, zero_config.thread_per_block,
        0, device.stream(), zero_config.virtual_thread_count,
        output_scores_tensor->flat<float>().data()));
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        SetZero<float>, zero_config.block_count, zero_config.thread_per_block,
        0, device.stream(), zero_config.virtual_thread_count,
        output_classes_tensor->flat<float>().data()));
    zero_config = GetGpuLaunchConfig(batch_size * total_output * 4, device);
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        SetZero<float>, zero_config.block_count, zero_config.thread_per_block,
        0, device.stream(), zero_config.virtual_thread_count,
        output_boxes_tensor->flat<float>().data()));
    return Status::OK();
  }

  bool pad_per_class_;
  bool clip_boxes_;
};

REGISTER_KERNEL_BUILDER(Name("CombinedNonMaxSuppression")
                            .Device(DEVICE_GPU)
                            .HostMemory("max_output_size_per_class")
                            .HostMemory("max_total_size")
                            .HostMemory("iou_threshold")
                            .HostMemory("score_threshold"),
                        tensorflow::CombinedNonMaxSuppressionGPUOp);

REGISTER_KERNEL_BUILDER(Name("BatchedNonMaxSuppression")
                            .TypeConstraint<float>("T")
                            .Device(DEVICE_GPU)
                            .HostMemory("box_counts")
                            .HostMemory("iou_threshold")
                            .HostMemory("max_output_size")
                            .HostMemory("score_threshold"),
                        tensorflow::BatchedNonMaxSuppressionGPUOp);

}  // namespace tensorflow
#endif
