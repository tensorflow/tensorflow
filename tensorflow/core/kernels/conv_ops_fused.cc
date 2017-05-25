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

// Implements convolution operations with other kernels baked into the
// processing, to optimize latency and memory usage.

#define EIGEN_USE_THREADS

#include <string.h>
#include <map>
#include <vector>
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/kernels/gemm_functors.h"
#include "tensorflow/core/kernels/image_resizer_state.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/util/mirror_pad_mode.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

namespace {

// We don't want to allocate a buffer to hold all the patches if the size is
// going to be extremely large, so break it into chunks if it's bigger than
// a limit. Each chunk will be processed serially, so we can refill the
// buffer for the next chunk and reuse it, keeping maximum memory size down.
// In this case, we've picked 16 megabytes as a reasonable limit for Android and
// other platforms using Eigen, and 1MB for iOS devices, from experimentation.
#if defined(__APPLE__) && defined(IS_MOBILE_PLATFORM)
const size_t kMaxChunkSize = (1 * 1024 * 1024);
#else
const size_t kMaxChunkSize = (16 * 1024 * 1024);
#endif
const size_t kResizeCacheSize = (8 * 1024 * 1024);

// Lookup method used when resizing.
enum SamplingMode {
  BILINEAR = 0,
  NEAREST = 1,
};

// Simple utility function used by FusedConv to multithread basic workloads. To
// use it, pass begin and end values for the full workload and a std::function
// that receives a subset of that through the begin and end values for each
// worker's task. The division of the full workload into worker tasks is handled
// by the multithreading logic. Here's an example of how to use it:
// std::vector<float> my_vector(100);
// ...
// FusedConvParallelFor(context, 0, 100,
//   [&my_vector](int64 task_begin, int64 task_end) {
//     for (int64 current = task_begin; current != task_end; ++current) {
//       my_vector[current] *= 10.0f;
//     }
// });
void FusedConvParallelFor(
    OpKernelContext* context, int64 begin, int64 end,
    const std::function<void(int64, int64)>& task_function) {
// On iOS, the thread management imposes a very big performance penalty, so
// just call the function directly with no multithreading.
#if defined(__APPLE__) && defined(IS_MOBILE_PLATFORM)
  task_function(begin, end);
#else
  auto& worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
  thread::ThreadPool* thread_pool = worker_threads.workers;
  const int64 total_elements = end - begin;
  // This is a bit of an arbitrary number, but was found to work well for
  // typical models we've been profiling on various devices.
  const int64 element_cost = 10000000;
  thread_pool->ParallelFor(
      total_elements, element_cost,
      [begin, task_function](int64 begin_offset, int64 end_offset) {
        const int64 task_begin = begin + begin_offset;
        const int64 task_end = begin + end_offset;
        task_function(task_begin, task_end);
      });
#endif
}

// Holds the state needed for the resizing subtasks.
template <class T1>
struct ResizeTaskParameters {
  ResizeTaskParameters() : st(false) {}

  int cache_height;
  T1* resize_cache;
  int cache_line_width;
  int input_width;
  int input_depth;
  int top_padding;
  int pad_offset;
  int64 resized_height;
  ImageResizerState st;
  const T1* input_batch_start;
  int64 cache_start_x;
  int64 cache_end_x;
  int left_padding;
  int64 resized_width;
  int64 padded_width;
  int64 padded_height;
};

template <class T1>
struct PerCacheLineParameters {
  PerCacheLineParameters() {}
  PerCacheLineParameters(const PerCacheLineParameters<T1>& other)
      : cache_line_start(other.cache_line_start),
        input_top_row_start(other.input_top_row_start),
        input_bottom_row_start(other.input_bottom_row_start),
        y_lerp(other.y_lerp) {}

  T1* cache_line_start;
  const T1* input_top_row_start;
  const T1* input_bottom_row_start;
  T1 y_lerp;
};

// Helper class to simplify bilinear filtering
template <class T1>
struct SampleRect {
  EIGEN_ALWAYS_INLINE SampleRect(const T1* in_top_left, const T1* in_top_right,
                                 const T1* in_bottom_left,
                                 const T1* in_bottom_right)
      : top_left(in_top_left),
        top_right(in_top_right),
        bottom_left(in_bottom_left),
        bottom_right(in_bottom_right) {}

  EIGEN_ALWAYS_INLINE T1 BilinearSample(int channel, T1 x_lerp,
                                        T1 y_lerp) const {
    const T1 top =
        top_left[channel] + (top_right[channel] - top_left[channel]) * x_lerp;
    const T1 bottom = bottom_left[channel] +
                      (bottom_right[channel] - bottom_left[channel]) * x_lerp;
    return top + (bottom - top) * y_lerp;
  }

  const T1* top_left;
  const T1* top_right;
  const T1* bottom_left;
  const T1* bottom_right;
};

// Calculates parameters which remain constant through a resize cache row.
template <class T1>
EIGEN_ALWAYS_INLINE PerCacheLineParameters<T1> CalculatePerCacheLineParameters(
    int64 cache_height, int64 cache_y, T1* resize_cache, int64 cache_line_width,
    int64 input_width, int64 input_depth, int64 top_padding, int64 pad_offset,
    int64 resized_height, const ImageResizerState& st,
    const T1* input_batch_start) {
  PerCacheLineParameters<T1> result;
  // The cache is organized so that the real y values of the resized image map
  // onto the actual cache values through a modulo scheme. This means that as we
  // progress downwards through the image, we keep reusing a small cache and so
  // keep memory usage down.
  int64 cache_index_y;
  if (cache_y < 0) {
    cache_index_y = cache_height + (cache_y % cache_height);
  } else {
    cache_index_y = cache_y % cache_height;
  }
  result.cache_line_start =
      resize_cache + (cache_index_y * cache_line_width * input_depth);
  // This part is implementing the mirror padding that happens before resizing.
  float in_y = (cache_y - top_padding);
  if (in_y < 0) {
    in_y = -(in_y + 1.0f - pad_offset);
  } else if (in_y >= resized_height) {
    in_y = (resized_height * 2.0f) - (in_y + 1.0f + pad_offset);
  }
  // Here's where do do the actual resize.
  in_y *= st.height_scale;
  const int64 top_y_index = static_cast<int64>(std::floor(in_y));
  const int64 bottom_y_index =
      std::min(static_cast<int64>(std::ceil(in_y)), (st.in_height - 1));
  // Lerp is used for bilinear filtering when that's needed.
  result.y_lerp = in_y - top_y_index;
  // Which rows of the original input image to pull the values from.
  result.input_top_row_start =
      input_batch_start + (top_y_index * input_width * input_depth);
  result.input_bottom_row_start =
      input_batch_start + (bottom_y_index * input_width * input_depth);
  return result;
}

template <class T1>
struct PerCachePixelParameters {
  PerCachePixelParameters() {}
  PerCachePixelParameters(const PerCachePixelParameters<T1>& other)
      : cache_line_pixel(other.cache_line_pixel),
        left_x_index(other.left_x_index),
        right_x_index(other.right_x_index),
        x_lerp(other.x_lerp) {}

  T1* cache_line_pixel;
  int64 left_x_index;
  int64 right_x_index;
  T1 x_lerp;
};

// Pulls out common parameters used for every resized pixel.
template <class T1>
EIGEN_ALWAYS_INLINE PerCachePixelParameters<T1>
CalculatePerCachePixelParameters(int64 cache_x, int64 cache_start_x,
                                 T1* cache_line_start, int64 input_depth,
                                 int64 left_padding, int64 pad_offset,
                                 int64 resized_width,
                                 const ImageResizerState& st) {
  PerCachePixelParameters<T1> result;
  // Figure out where we're going to store the results of our transform.
  const int cache_index_x = cache_x - cache_start_x;
  result.cache_line_pixel = cache_line_start + (cache_index_x * input_depth);
  // Implement mirror padding by flipping in_x if it's off the edge.
  float in_x = (cache_x - left_padding);
  if (in_x < 0) {
    in_x = -(in_x + 1.0f - pad_offset);
  } else if (in_x >= resized_width) {
    in_x = (resized_width * 2.0f) - (in_x + 1.0f + pad_offset);
  }
  // Resize the x parameters.
  in_x *= st.width_scale;
  // Get the x coordinates for the left and right pixels to pull from.
  result.left_x_index = static_cast<int64>(std::floor(in_x));
  result.right_x_index =
      std::min(static_cast<int64>(std::ceil(in_x)), (st.in_width - 1));
  // This x_lerp is used to blend pixels in bilinear filtering.
  result.x_lerp = in_x - result.left_x_index;
  return result;
}

// Combines bilinear resizing and mirror padding into the im2col transformation
// stage of convolution.
template <class T1, class T2, class T3, class TGemmFunctor,
          SamplingMode SampleMode>
class FusedResizeAndPadConvFunctor {
 public:
  void operator()(OpKernelContext* context, const Tensor& input,
                  int input_batches, int resized_height, int resized_width,
                  int padded_height, int padded_width, int input_depth,
                  const T2* filter_data, int filter_height, int filter_width,
                  int filter_count, int stride_rows, int stride_cols,
                  Padding padding, T3* output_data, int output_height,
                  int output_width, const ImageResizerState& st,
                  int top_padding, int bottom_padding, int left_padding,
                  int right_padding, int pad_offset) {
    if ((input_batches <= 0) || (padded_width <= 0) || (padded_height <= 0) ||
        (input_depth <= 0)) {
      LOG(WARNING) << "Conv2D was called with bad input dimensions: "
                   << input_batches << ", " << padded_height << ", "
                   << padded_width << ", " << input_depth;
      return;
    }
    if ((filter_width <= 0) || (filter_height <= 0) || (filter_count <= 0)) {
      LOG(WARNING) << "Conv2D was called with bad filter dimensions: "
                   << filter_width << ", " << filter_height << ", "
                   << filter_count;
      return;
    }
    if ((output_width <= 0) || (output_height <= 0)) {
      LOG(WARNING) << "Conv2D was called with bad output width or height: "
                   << output_width << ", " << output_height;
      return;
    }
    OP_REQUIRES(
        context, ((SampleMode == NEAREST) || (SampleMode == BILINEAR)),
        errors::InvalidArgument("Bad sample mode passed in", SampleMode));

    // These calculations define how the patches will be positioned within the
    // input image. The actual definitions are quite complex, and rely on the
    // previously-calculated output size.
    int filter_left_offset;
    int filter_top_offset;
    if (padding == VALID) {
      filter_left_offset =
          ((output_width - 1) * stride_cols + filter_width - padded_width + 1) /
          2;
      filter_top_offset = ((output_height - 1) * stride_rows + filter_height -
                           padded_height + 1) /
                          2;
    } else {
      filter_left_offset =
          ((output_width - 1) * stride_cols + filter_width - padded_width) / 2;
      filter_top_offset =
          ((output_height - 1) * stride_rows + filter_height - padded_height) /
          2;
    }

    ResizeTaskParameters<T1> task_params;
    task_params.input_depth = input_depth;
    task_params.top_padding = top_padding;
    task_params.pad_offset = pad_offset;
    task_params.resized_height = resized_height;
    task_params.st = st;
    task_params.left_padding = left_padding;
    task_params.resized_width = resized_width;
    task_params.padded_width = padded_width;
    task_params.padded_height = padded_height;

    // The im2col buffer has # of patches rows, and # of filters cols.
    // It's laid out like this, in row major order in memory:
    //        < filter value count >
    //   ^   +---------------------+
    // patch |                     |
    // count |                     |
    //   v   +---------------------+
    // Each patch row contains a filter_width x filter_height patch of the
    // input, with the depth channel as the most contiguous in memory, followed
    // by the width, then the height. This is the standard memory order in the
    // image world if it helps to visualize it.
    const int filter_value_count = filter_width * filter_height * input_depth;

    OP_REQUIRES(context, (filter_value_count * sizeof(T1)) <= kMaxChunkSize,
                errors::InvalidArgument("Im2Col patch too large for buffer"));
    const size_t patches_per_chunk =
        kMaxChunkSize / (filter_value_count * sizeof(T1));
    // Because memory allocation is very expensive on mobile platforms, try to
    // allocate a persistent buffer that will be kept around between calls. We
    // use TensorFlow's resource management to ensure that the memory will be
    // released when the session is over.
    Im2ColBufferResource<T1, kMaxChunkSize>* im2col_buffer_resource;
    std::function<Status(Im2ColBufferResource<T1, kMaxChunkSize>**)> creator =
        [](Im2ColBufferResource<T1, kMaxChunkSize>** resource) {
          *resource = new Im2ColBufferResource<T1, kMaxChunkSize>();
          return Status::OK();
        };
    OP_REQUIRES_OK(context, context->resource_manager()->LookupOrCreate(
                                "Conv2d", "im2col_buffer",
                                &im2col_buffer_resource, creator));

    // Create a resize cache memory buffer that will hold the rows of
    // transformed and mirror padded input pixels, ready to be copied
    // into filter patches by im2col.
    // It's laid out like this, in row major order in memory:
    //         < cache line width >
    //   ^    +--------------------+
    // cache  |                    |
    // height |                    |
    //   v    +--------------------+
    // Each cache row contains a cache_line_width number of resized pixels,
    // each with input_depth channels. The cache height is typically less than
    // the full height the resized image would be, so it's filled up
    // incrementally as we progress downwards through the input creating im2col
    // patches.
    task_params.cache_start_x = -filter_left_offset;
    task_params.cache_end_x =
        (((output_width - 1) * stride_cols) - filter_left_offset) +
        filter_width;
    task_params.cache_line_width =
        task_params.cache_end_x - task_params.cache_start_x;
    task_params.cache_height =
        kResizeCacheSize / (task_params.cache_line_width * input_depth);
    const int needed_resize_cache_count =
        filter_height * task_params.cache_line_width * input_depth;
    OP_REQUIRES(context,
                (needed_resize_cache_count * sizeof(T1)) <= kResizeCacheSize,
                errors::InvalidArgument("Input too large for resize cache"));
    Im2ColBufferResource<T1, kResizeCacheSize>* resize_cache_resource;
    std::function<Status(Im2ColBufferResource<T1, kResizeCacheSize>**)>
        resize_creator =
            [](Im2ColBufferResource<T1, kResizeCacheSize>** resource) {
              *resource = new Im2ColBufferResource<T1, kResizeCacheSize>();
              return Status::OK();
            };
    OP_REQUIRES_OK(context, context->resource_manager()->LookupOrCreate(
                                "Conv2d", "resize_cache",
                                &resize_cache_resource, resize_creator));

    // This means that multiple ops can't be run simultaneously on different
    // threads, because we have a single shared resource. The platforms this is
    // aimed at have intra-op parallelism as their focus though, so it shouldn't
    // be an issue.
    mutex_lock lock_buffer(im2col_buffer_resource->mu);
    core::ScopedUnref unref_buffer(im2col_buffer_resource);
    T1* im2col_buffer = im2col_buffer_resource->data;

    // This buffer is used as a fairly heavy-weight cache for the resized and
    // mirrored inputs to the im2col operation. The problem is that we want to
    // keep the memory usage down by not rendering the fully resized and padded
    // input tensor to the convolution into an entire buffer. The first approach
    // to avoid this was to fold the bilinear filtering and padding spatial
    // transformations into the im2col lookup itself. This successfully reduced
    // memory usage, but because im2col can access an individual pixel for many
    // different patches, the extra overhead of doing the same bilinear lookups
    // repeatedly became too expensive.
    // The resize cache is designed to avoid this problem by keeping a
    // horizontal slice of the resized and padded input to the im2col
    // precalculated, so that repeated accesses to the same pixel from different
    // filter patches can just be copied from this cache. It's organized as a
    // horizontal slice stretching across the whole virtual image, and as high
    // as the filter window, so that as the patch processing moves across all
    // the pixels are present, and before a new row of patches is started any
    // previously calculated rows that are needed are maintained, with new rows
    // calculated as required.
    mutex_lock resize_lock_buffer(resize_cache_resource->mu);
    core::ScopedUnref unref_resized_cache(resize_cache_resource);
    task_params.resize_cache = resize_cache_resource->data;

    const T1* input_data = input.flat<T1>().data();
    const int64 input_height = input.shape().dim_sizes()[1];
    task_params.input_width = input.shape().dim_sizes()[2];

    int end_cached_lines = std::numeric_limits<int>::min();

    for (int batch = 0; batch < input_batches; ++batch) {
      task_params.input_batch_start =
          input_data +
          (batch * input_height * task_params.input_width * input_depth);
      const int in_y_end =
          ((output_height * stride_rows) - filter_top_offset) + filter_height;
      for (int out_y = 0; out_y < output_height; ++out_y) {
        const int in_y_origin = (out_y * stride_rows) - filter_top_offset;
        const int cache_start_y = std::max(in_y_origin, end_cached_lines);
        const int cache_end_y = std::min(
            in_y_end, std::max((in_y_origin + task_params.cache_height),
                               end_cached_lines));
        if (end_cached_lines < (in_y_origin + filter_height)) {
          // This call breaks up the work required for calculating the mirror
          // padding and resizing across multiple threads.
          FusedConvParallelFor(
              context, cache_start_y, cache_end_y,
              [task_params](int64 task_cache_start_y, int64 task_cache_end_y) {
                // This is a long and confusing function, but it's been laid out
                // this way to help with performance on some intensive models.
                // What it's doing is populating a cache of the original input
                // image, after it's been bilinear resized and had its edges
                // mirrored. This allows the following im2col code to access the
                // transformed pixels from this cache, without having to
                // repeatedly apply the expensive bilinear calculations as the
                // same pixels are accessed by different patches.
                // This is most effective when the stride is small and the
                // filter size is large, since that's when pixels are reused
                // most frequently as patches overlap.
                for (int cache_y = task_cache_start_y;
                     cache_y < task_cache_end_y; ++cache_y) {
                  // We organize the cache as a series of rows, each containing
                  // all the transformed pixels for a given line in the image.
                  // This cache is big enough to hold at least a filter's height
                  // worth of rows, but typically more, limited by the size of
                  // the cache buffer.
                  // We don't allocate an entire image's worth of rows though,
                  // because we're trying to keep memory usage down, so as we
                  // progress downwards through the im2col we periodically
                  // refresh the cache so that the next lines that are needed
                  // for that operation are always present.
                  // Work out the parameters that remain constant across the
                  // row we're calculating.
                  PerCacheLineParameters<float> line_params(
                      CalculatePerCacheLineParameters<float>(
                          task_params.cache_height, cache_y,
                          task_params.resize_cache,
                          task_params.cache_line_width, task_params.input_width,
                          task_params.input_depth, task_params.top_padding,
                          task_params.pad_offset, task_params.resized_height,
                          task_params.st, task_params.input_batch_start));
                  // Iterate through the resize cache row we're filling in.
                  for (int cache_x = task_params.cache_start_x;
                       cache_x < task_params.cache_end_x; ++cache_x) {
                    // Figure out what we need for the cache pixel we're
                    // populating.
                    PerCachePixelParameters<T1> pixel_params(
                        CalculatePerCachePixelParameters<T1>(
                            cache_x, task_params.cache_start_x,
                            line_params.cache_line_start,
                            task_params.input_depth, task_params.left_padding,
                            task_params.pad_offset, task_params.resized_width,
                            task_params.st));
                    // If the access is off the left, right, top, or bottom of
                    // the resized image, the conv padding means we should set
                    // it to zero.
                    if ((cache_x < 0) ||
                        (cache_x >= task_params.padded_width) ||
                        (cache_y < 0) ||
                        (cache_y >= task_params.padded_height)) {
                      std::fill_n(pixel_params.cache_line_pixel,
                                  task_params.input_depth, T1(0));
                    } else {
                      // There are two different sampling strategies for
                      // resizing. When using nearest, we can just do a
                      // straight copy of the pixel closest to our sample point,
                      // but bilinear requires a more complex calculation.
                      if (SampleMode == NEAREST) {
                        const T1* input_top_left_pixel =
                            line_params.input_top_row_start +
                            (pixel_params.left_x_index *
                             task_params.input_depth);

                        std::copy_n(input_top_left_pixel,
                                    task_params.input_depth,
                                    pixel_params.cache_line_pixel);
                      } else {
                        const SampleRect<T1> rect(
                            line_params.input_top_row_start +
                                (pixel_params.left_x_index *
                                 task_params.input_depth),
                            line_params.input_top_row_start +
                                (pixel_params.right_x_index *
                                 task_params.input_depth),
                            line_params.input_bottom_row_start +
                                (pixel_params.left_x_index *
                                 task_params.input_depth),
                            line_params.input_bottom_row_start +
                                (pixel_params.right_x_index *
                                 task_params.input_depth));
                        for (int in_channel = 0;
                             in_channel < task_params.input_depth;
                             ++in_channel) {
                          pixel_params.cache_line_pixel[in_channel] =
                              rect.BilinearSample(in_channel,
                                                  pixel_params.x_lerp,
                                                  line_params.y_lerp);
                        }
                      }
                    }
                  }
                }
              });
          end_cached_lines = cache_end_y;
        }
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = (out_x * stride_cols) - filter_left_offset;
          const int patch_index = (batch * output_width * output_height) +
                                  (out_y * output_width) + out_x;
          const int patch_index_within_chunk = patch_index % patches_per_chunk;
          T1* im2col_patch_start =
              im2col_buffer + (patch_index_within_chunk * filter_value_count);
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            T1* im2col_row_start =
                im2col_patch_start +
                (filter_y * filter_width * task_params.input_depth);
            const int conv_in_y = in_y_origin + filter_y;
            int cache_index_y;
            if (conv_in_y < 0) {
              cache_index_y = task_params.cache_height +
                              (conv_in_y % task_params.cache_height);
            } else {
              cache_index_y = conv_in_y % task_params.cache_height;
            }
            T1* cache_line_start =
                task_params.resize_cache +
                (cache_index_y * task_params.cache_line_width *
                 task_params.input_depth);
            T1* cache_filter_row_start =
                cache_line_start + ((in_x_origin - task_params.cache_start_x) *
                                    task_params.input_depth);
            std::copy_n(cache_filter_row_start,
                        (filter_width * task_params.input_depth),
                        im2col_row_start);
          }
          const bool is_last_in_chunk =
              (patch_index_within_chunk == (patches_per_chunk - 1));
          const bool is_last_overall =
              ((batch == (input_batches - 1)) &&
               (out_y == (output_height - 1)) && (out_x == (output_width - 1)));
          if (is_last_in_chunk || is_last_overall) {
            // Now we've assembled a set of image patches into a matrix, apply
            // a GEMM matrix multiply of the patches as rows, times the filter
            // weights in columns, to get partial results in the output
            // matrix.
            const int how_many_patches = patch_index_within_chunk + 1;
            const int m = how_many_patches;
            const int n = filter_count;
            const int k = filter_value_count;
            const int lda = filter_value_count;
            const int ldb = filter_count;
            const int ldc = filter_count;
            const size_t start_patch_index =
                patch_index - (how_many_patches - 1);
            T3* chunk_output_data =
                output_data + (start_patch_index * filter_count);
            TGemmFunctor gemm_functor;
            gemm_functor(context, m, n, k, im2col_buffer, lda, filter_data, ldb,
                         chunk_output_data, ldc);
          }
        }
      }
    }
  }
};

}  // namespace

// Implements a version of convolution with bilinear resizing and mirror padding
// included.
template <class T, class TConvFunctor, bool DoResize>
class FusedResizeConv2DUsingGemmOp : public OpKernel {
 public:
  explicit FusedResizeConv2DUsingGemmOp(OpKernelConstruction* context)
      : OpKernel(context) {
    if (DoResize) {
      OP_REQUIRES_OK(context,
                     context->GetAttr("resize_align_corners", &align_corners_));
    }
    MirrorPadMode mode;
    OP_REQUIRES_OK(context, context->GetAttr("mode", &mode));

    switch (mode) {
      case MirrorPadMode::SYMMETRIC: {
        offset_ = 0;
        break;
      }
      case MirrorPadMode::REFLECT: {
        offset_ = 1;
        break;
      }
      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument(
                        "mode must be either REFLECT or SYMMETRIC."));
    }
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    const int64 stride_n = GetTensorDim(strides_, FORMAT_NHWC, 'N');
    const int64 stride_c = GetTensorDim(strides_, FORMAT_NHWC, 'C');
    OP_REQUIRES(
        context, stride_n == 1 && stride_c == 1,
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, (input.shape().num_elements() > 0),
                errors::InvalidArgument("Input tensor can't be empty"));

    ImageResizerState st(false);
    if (DoResize) {
      st = ImageResizerState(align_corners_);
      st.ValidateAndCalculateOutputSize(context, input);
      if (!context->status().ok()) return;
    } else {
      // Set up the resize parameters to do no scaling at all.
      st.batch_size = input.dim_size(0);
      st.out_height = input.dim_size(1);
      st.out_width = input.dim_size(2);
      st.in_height = input.dim_size(1);
      st.in_width = input.dim_size(2);
      st.channels = input.dim_size(3);
      st.height_scale = 1.0f;
      st.width_scale = 1.0f;
    }
    TensorShape resized_shape(
        {input.dim_size(0), st.out_height, st.out_width, input.dim_size(3)});
    int paddings_index;
    int filter_index;
    if (DoResize) {
      paddings_index = 2;
      filter_index = 3;
    } else {
      paddings_index = 1;
      filter_index = 2;
    }
    const Tensor& paddings = context->input(paddings_index);

    const int dims = resized_shape.dims();
    OP_REQUIRES(
        context, TensorShapeUtils::IsMatrix(paddings.shape()) &&
                     paddings.dim_size(1) == 2,
        errors::InvalidArgument("paddings must be a matrix with 2 columns: ",
                                paddings.shape().DebugString()));
    const int fixed_dims =
        (allow_legacy_scalars() && dims == 0 && paddings.dim_size(0) == 1)
            ? 1
            : dims;
    OP_REQUIRES(
        context, fixed_dims == paddings.dim_size(0),
        errors::InvalidArgument(
            "The first dimension of paddings must be the rank of inputs: ",
            fixed_dims, " ", paddings.shape().DebugString(), " ",
            resized_shape.DebugString()));
    OP_REQUIRES(
        context, dims == paddings.dim_size(0),
        errors::InvalidArgument(
            "The first dimension of paddings must be the rank of inputs: ",
            dims, " ", paddings.shape().DebugString(), " ",
            resized_shape.DebugString()));

    OP_REQUIRES(
        context, dims == 4,
        errors::InvalidArgument(
            "Fused mirror padding only supports four-dimensional inputs, but ",
            dims, " requested"));

    // Compute the shape of the output tensor, and allocate it.
    TensorShape padded_shape;
    TTypes<int32>::ConstMatrix paddings_matrix = paddings.matrix<int32>();
    for (int d = 0; d < dims; ++d) {
      const int32 before =
          paddings_matrix(d, 0);  // Pad before existing elements.
      const int32 after =
          paddings_matrix(d, 1);  // Pad after existing elements.
      OP_REQUIRES(context, before >= 0 && after >= 0,
                  errors::InvalidArgument("paddings must be non-negative: ",
                                          before, " ", after));
      if (offset_ == 0) {  // SYMMETRIC mode.
        OP_REQUIRES(
            context, before <= resized_shape.dim_size(d) &&
                         after <= resized_shape.dim_size(d),
            errors::InvalidArgument("paddings must be no greater "
                                    "than the dimension size: ",
                                    before, ", ", after, " greater than ",
                                    resized_shape.dim_size(d)));
      } else if (offset_ == 1) {  // REFLECT mode.
        OP_REQUIRES(
            context, before < resized_shape.dim_size(d) &&
                         after < resized_shape.dim_size(d),
            errors::InvalidArgument("paddings must be less than"
                                    " the dimension size: ",
                                    before, ", ", after, " not less than ",
                                    resized_shape.dim_size(d)));
      }
      padded_shape.AddDim(before + resized_shape.dim_size(d) + after);
    }

    OP_REQUIRES(
        context, ((paddings_matrix(0, 0) == 0) && (paddings_matrix(0, 1) == 0)),
        errors::InvalidArgument(
            "Fused mirror padding only support spatial padding, not batches: ",
            paddings.DebugString()));
    OP_REQUIRES(
        context, ((paddings_matrix(3, 0) == 0) && (paddings_matrix(3, 1) == 0)),
        errors::InvalidArgument(
            "Fused mirror padding only support spatial padding, not channels: ",
            paddings.DebugString()));
    const int32 top_padding = paddings_matrix(1, 0);
    const int32 bottom_padding = paddings_matrix(1, 1);
    const int32 left_padding = paddings_matrix(2, 0);
    const int32 right_padding = paddings_matrix(2, 1);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& filter = context->input(filter_index);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, padded_shape.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        padded_shape.DebugString()));
    OP_REQUIRES(context, filter.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter.shape().DebugString()));

    // We only check the first three dims, since the depth is accessed as an
    // int64 below.
    for (int i = 0; i < 3; i++) {
      OP_REQUIRES(context, FastBoundsCheck(filter.dim_size(i),
                                           std::numeric_limits<int>::max()),
                  errors::InvalidArgument("filter too large"));
    }

    // The last dimension for input is in_depth. It must be the same as the
    // filter's in_depth.
    const int64 in_depth = padded_shape.dim_size(3);
    OP_REQUIRES(
        context, in_depth == filter.dim_size(2),
        errors::InvalidArgument("input and filter must have the same depth: ",
                                in_depth, " vs ", filter.dim_size(2)));

    // The last dimension for filter is out_depth.
    const int out_depth = static_cast<int>(filter.dim_size(3));

    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64 padded_rows_raw = padded_shape.dim_size(1);
    OP_REQUIRES(context, FastBoundsCheck(padded_rows_raw,
                                         std::numeric_limits<int>::max()),
                errors::InvalidArgument("Input rows too large"));
    const int padded_rows = static_cast<int>(padded_rows_raw);
    const int filter_rows = static_cast<int>(filter.dim_size(0));
    const int resized_rows = static_cast<int>(resized_shape.dim_size(1));

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64 padded_cols_raw = padded_shape.dim_size(2);
    OP_REQUIRES(context, FastBoundsCheck(padded_cols_raw,
                                         std::numeric_limits<int>::max()),
                errors::InvalidArgument("Input cols too large"));
    const int padded_cols = static_cast<int>(padded_cols_raw);
    const int filter_cols = static_cast<int>(filter.dim_size(1));
    const int resized_cols = static_cast<int>(resized_shape.dim_size(2));

    // The first dimension for input is batch.
    const int64 batch_raw = padded_shape.dim_size(0);
    OP_REQUIRES(context,
                FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                errors::InvalidArgument("batch is too large"));
    const int batch = static_cast<int>(batch_raw);

    // For now we take the stride from the second and third dimensions only (we
    // do not support striding on the batch or depth dimension).
    const int stride_rows = GetTensorDim(strides_, FORMAT_NHWC, 'H');
    const int stride_cols = GetTensorDim(strides_, FORMAT_NHWC, 'W');

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(padded_rows, filter_rows, stride_rows,
                                         padding_, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(padded_cols, filter_cols, stride_cols,
                                         padding_, &out_cols, &pad_cols));
    TensorShape out_shape =
        ShapeFromFormat(FORMAT_NHWC, batch, out_rows, out_cols, out_depth);
    OP_REQUIRES(context, (out_shape.num_elements() > 0),
                errors::InvalidArgument("Output tensor can't be empty"));

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "FusedConv2D: " << name() << ", in_depth = " << in_depth
            << ", padded_cols = " << padded_cols
            << ", resized_cols = " << resized_cols
            << ", filter_cols = " << filter_cols
            << ", padded_rows = " << padded_rows
            << ", resized_rows = " << resized_rows
            << ", filter_rows = " << filter_rows
            << ", stride_rows = " << stride_rows
            << ", stride_cols = " << stride_cols
            << ", out_depth = " << out_depth << ", DoResize=" << DoResize;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }
    TConvFunctor conv_functor;
    conv_functor(context, input, batch, resized_rows, resized_cols, padded_rows,
                 padded_cols, in_depth, filter.flat<T>().data(), filter_rows,
                 filter_cols, out_depth, stride_rows, stride_cols, padding_,
                 output->flat<T>().data(), out_rows, out_cols, st, top_padding,
                 bottom_padding, left_padding, right_padding, offset_);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  bool align_corners_;
  int offset_;

  TF_DISALLOW_COPY_AND_ASSIGN(FusedResizeConv2DUsingGemmOp);
};

#define REGISTER_FUSED(T)                                                    \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("FusedResizeAndPadConv2D")                                        \
          .Device(DEVICE_CPU)                                                \
          .TypeConstraint<T>("T"),                                           \
      FusedResizeConv2DUsingGemmOp<                                          \
          T, FusedResizeAndPadConvFunctor<T, T, T, FastGemmFunctor<T, T, T>, \
                                          BILINEAR>,                         \
          true>);

TF_CALL_float(REGISTER_FUSED);

#define REGISTER_PAD_ONLY_FUSED(T)                                           \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("FusedPadConv2D").Device(DEVICE_CPU).TypeConstraint<T>("T"),      \
      FusedResizeConv2DUsingGemmOp<                                          \
          T, FusedResizeAndPadConvFunctor<T, T, T, FastGemmFunctor<T, T, T>, \
                                          NEAREST>,                          \
          false>);

TF_CALL_float(REGISTER_PAD_ONLY_FUSED);

}  // namespace tensorflow
