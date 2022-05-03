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

// Implements quantized eight-bit versions of the convolution operations.

#include <algorithm>
#include <vector>

#define EIGEN_USE_THREADS

#define GEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK
#include "public/gemmlowp.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/kernels/meta_support.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/kernels/reference_gemm.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/padding.h"

namespace tensorflow {

// This functor implements the convolution operation in as simple a form as
// possible. It won't give great performance, but it is very useful for
// stepping through and instrumenting for debugging, creating minimal benchmarks
// to prototype with, and sharing with teams that want to run this outside of
// our environment.
// With that in mind, I've avoided using anything except pretty standard C++
// types. This is especially noticeable in the data access through raw array
// indexing. It's deliberate in this case though, since it makes the underlying
// memory order very explicit, which is important for both inspecting memory
// contents during debugging and for specifying what we expect to others.
// The memory layout of the data is, from biggest stride to smallest:
// input_data = [input_batches, input_height, input_width, input_depth]
// filter_data = [filter_height, filter_width, input_depth, filter_count]
// output_data = [input_batches, output_height, output_width, filter_count]
template <class T1, class T2, class T3>
class ReferenceConvFunctor {
 public:
  void operator()(OpKernelContext* context, const T1* input_data,
                  int input_batches, int input_height, int input_width,
                  int input_depth, int input_offset, const T2* filter_data,
                  int filter_height, int filter_width, int filter_count,
                  int filter_offset, int stride, Padding padding,
                  T3* output_data, int output_height, int output_width,
                  int output_shift, int output_offset, int output_mult) {
    // Set up some constants we need for the output down-shifting and
    // saturation.
    const int32_t highest = static_cast<int32>(Eigen::NumTraits<T3>::highest());
    const int32_t lowest = static_cast<int32>(Eigen::NumTraits<T3>::lowest());

    // When we're converting the 32 bit accumulator to a lower bit depth, we
    // need to add on 0.5 in fixed-point terms to make the operation round half
    // up towards positive infinity, rather than a floor.
    // We also need to watch out for the case when there's no down shift,
    // because a left shift by a negative number gives undefined results.
    const int32_t rounding = (output_shift < 1) ? 0 : (1 << (output_shift - 1));

    // The two different padding modes we support can be a bit confusing. SAME
    // means we're trying to produce an output image that's the same size as the
    // input. It's complicated by stride, which shrinks the output image by a
    // a factor, but it means we end up sampling from outside the borders of the
    // input. These out-of-bounds values are read as zeroes. VALID means only
    // produce output values where the filters can read all their values from
    // within the input image. It effectively removes the margins of the output
    // image compared to the one produced by SAME. Stride complicates this
    // definition though, because it can result in the right and bottom filter
    // patches sampling from outside the borders if it's greater than 1.
    // Most of the logic for sorting this all out is done before this function,
    // when we calculate the output size, but the positioning of the origin of
    // the filters is different between the two modes, since SAME positions the
    // first filter off the edge of the input.
    int filter_left_offset;
    int filter_top_offset;
    if (padding == VALID) {
      filter_left_offset =
          ((output_width - 1) * stride + filter_width - input_width + 1) / 2;
      filter_top_offset =
          ((output_height - 1) * stride + filter_height - input_height + 1) / 2;
    } else {
      filter_left_offset =
          ((output_width - 1) * stride + filter_width - input_width) / 2;
      filter_top_offset =
          ((output_height - 1) * stride + filter_height - input_height) / 2;
    }

    // If we've got multiple images in our input, work through each of them.
    for (int batch = 0; batch < input_batches; ++batch) {
      // Walk through all the output image values, sliding the filter to
      // different
      // positions in the input.
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          // Each filter kernel produces one output channel.
          for (int out_channel = 0; out_channel < filter_count; ++out_channel) {
            // We're going to calculate a single output value, which means we
            // need to multiply a three dimensional kernel of weights against
            // the current location within the input image.
            /*
              *-------------------------------...
              |\ ^
              | \in_depth
              |  \ v
              |   *-------------------------------...
              |   |            ^
              |   |       in_y_origin
              |   |            v   \
              |   |<in_x_origin>*---*^
              |   |            \|   |filter_height
              .   |             *---*v
              .   |             <--->
                  .         filter_width
                  .
            */
            const int in_x_origin = (out_x * stride) - filter_left_offset;
            const int in_y_origin = (out_y * stride) - filter_top_offset;
            int32_t total = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                for (int in_channel = 0; in_channel < input_depth;
                     ++in_channel) {
                  const int in_x = in_x_origin + filter_x;
                  const int in_y = in_y_origin + filter_y;
                  int32_t input_value;
                  // If the location is outside the bounds of the input image,
                  // use zero as a default value.
                  if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                      (in_y < input_height)) {
                    const T1 input_source_value =
                        input_data[(batch * input_height * input_width *
                                    input_depth) +
                                   (in_y * input_width * input_depth) +
                                   (in_x * input_depth) + in_channel];
                    // We're promoting the T1 type to a higher bit depth here as
                    // we do the subtraction.
                    input_value =
                        static_cast<int32>(input_source_value) - input_offset;
                  } else {
                    input_value = 0;
                  }
                  const T2 filter_source_value =
                      filter_data[(filter_y * filter_width * input_depth *
                                   filter_count) +
                                  (filter_x * input_depth * filter_count) +
                                  (in_channel * filter_count) + out_channel];
                  // Another promotion to 32 bit, as above.
                  const int32_t filter_value =
                      static_cast<int32>(filter_source_value) - filter_offset;
                  total += (input_value * filter_value);
                }
              }
            }
            // Here we're applying scale factors to compress the 32 bit
            // accumulated total to a potentially lower bit depth.
            const int32_t output =
                ((((total + output_offset) * output_mult) + rounding) >>
                 output_shift);
            // We need to saturate the results against the largest and smallest
            // values that can be represented in this type.
            const int32_t top_clamped_output = std::min(output, highest);
            const int32_t clamped_output = std::max(top_clamped_output, lowest);
            output_data[(batch * output_height * output_width * filter_count) +
                        (out_y * output_width * filter_count) +
                        (out_x * filter_count) + out_channel] = clamped_output;
          }
        }
      }
    }
  }
};

// We don't want to allocate a buffer to hold all the patches if the size is
// going to be extremely large, so break it into chunks if it's bigger than
// a limit. Each chunk will be processed serially, so we can refill the
// buffer for the next chunk and reuse it, keeping maximum memory size down.
// In this case, we've picked 1 megabyte as a reasonable limit, from
// experimentation.
const size_t kMaxChunkSize = (1 * 1024 * 1024);

// Implements convolution as a two stage process, first packing the patches of
// the input image into columns (im2col) and then running GEMM to produce the
// final result.
template <class T1, class T2, class T3>
class Im2ColConvFunctor {
 public:
  void operator()(OpKernelContext* context, const T1* input_data,
                  int input_batches, int input_height, int input_width,
                  int input_depth, int input_offset, const T2* filter_data,
                  int filter_height, int filter_width, int filter_count,
                  int filter_offset, int stride, Padding padding,
                  T3* output_data, int output_height, int output_width,
                  int output_shift, int output_offset, int output_mult) {
    if (input_offset < 0) {
      // Only log the first few occurrences of this warning.
      static int warning_count = 0;
      if (warning_count < 10) {
        ++warning_count;
        LOG(WARNING)
            << "For kernel '" << context->op_kernel().name() << "' from input '"
            << context->op_kernel().requested_input(0)
            << "': Zero is not representable in the quantized range used by the"
            << " input. This means QuantizedConv2d has to fall back to a slow"
            << " implementation, since the border of zero values can't be"
            << " represented easily. You should try to construct graphs that"
            << " avoid this situation.";
      }
      ReferenceConvFunctor<T1, T2, T3> conv_functor;
      conv_functor(context, input_data, input_batches, input_height,
                   input_width, input_depth, input_offset, filter_data,
                   filter_height, filter_width, filter_count, filter_offset,
                   stride, padding, output_data, output_height, output_width,
                   output_shift, output_offset, output_mult);
      return;
    }

    OP_REQUIRES(
        context, output_width > 0,
        errors::InvalidArgument("output_width must be strictly positive"));
    OP_REQUIRES(
        context, output_height > 0,
        errors::InvalidArgument("output_height must be strictly positive"));
    int filter_left_offset;
    int filter_top_offset;
    if (padding == VALID) {
      filter_left_offset =
          ((output_width - 1) * stride + filter_width - input_width + 1) / 2;
      filter_top_offset =
          ((output_height - 1) * stride + filter_height - input_height + 1) / 2;
    } else {
      filter_left_offset =
          ((output_width - 1) * stride + filter_width - input_width) / 2;
      filter_top_offset =
          ((output_height - 1) * stride + filter_height - input_height) / 2;
    }

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
    OP_REQUIRES(context, filter_value_count > 0,
                errors::InvalidArgument(
                    "filter patch must contain at least one element"));
    const int64_t patches_per_chunk =
        kMaxChunkSize / (filter_value_count * sizeof(T1));
    const int64_t chunk_value_count =
        (kMaxChunkSize + (sizeof(T1) - 1)) / sizeof(T1);
    // TODO(petewarden) - Memory allocation can be very slow on Android. Can we
    // optimize this by keeping the scratch buffer around?
    // Because memory allocation is very expensive on mobile platforms, try to
    // allocate a persistent buffer that will be kept around between calls. We
    // use TensorFlow's resource management to ensure that the memory will be
    // released when the session is over.
    Im2ColBufferResource<T1, chunk_value_count>* im2col_buffer_resource;
    std::function<Status(Im2ColBufferResource<T1, chunk_value_count>**)>
        creator = [](Im2ColBufferResource<T1, chunk_value_count>** resource) {
#ifdef _MSC_VER
          // MSVC complains about the capture of chunk_value_count which oddly
          // works fine in conv_ops_using_gemm.cc for example.
          // Define chunk_value_count inside the lambda for now.
          const int64 chunk_value_count =
              (kMaxChunkSize + (sizeof(T1) - 1)) / sizeof(T1);
#endif
          *resource = new Im2ColBufferResource<T1, chunk_value_count>();
          return Status::OK();
        };
    OP_REQUIRES_OK(context, context->resource_manager()->LookupOrCreate(
                                "Conv2d", "im2col_buffer",
                                &im2col_buffer_resource, creator));
    // This means that multiple ops can't be run simultaneously on different
    // threads, because we have a single shared resource. The platforms this is
    // aimed at have intra-op parallelism as their focus though, so it shouldn't
    // be an issue.
    mutex_lock lock_buffer(im2col_buffer_resource->mu);
    core::ScopedUnref unref_buffer(im2col_buffer_resource);
    T1* im2col_buffer = im2col_buffer_resource->data;

    const int64_t patch_count = (input_batches * output_height * output_width);
    const int64_t chunk_count =
        (patch_count + (patches_per_chunk - 1)) / patches_per_chunk;

    for (int64_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
      const int64_t patch_index_start = chunk_index * patches_per_chunk;
      const int64_t patch_index_end =
          std::min(patch_index_start + patches_per_chunk, patch_count);
      for (int64_t patch_index = patch_index_start;
           patch_index < patch_index_end; ++patch_index) {
        const int64_t batch = patch_index / (output_height * output_width);
        const int64_t out_y = (patch_index / output_width) % output_height;
        const int64_t out_x = patch_index % output_width;
        const T1* input_batch_start =
            input_data + (batch * input_height * input_width * input_depth);
        const int in_y_origin = (out_y * stride) - filter_top_offset;
        const int in_x_origin = (out_x * stride) - filter_left_offset;
        const int patch_index_within_chunk = patch_index % patches_per_chunk;
        T1* im2col_patch_start =
            im2col_buffer + (patch_index_within_chunk * filter_value_count);
        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
          const int in_y = in_y_origin + filter_y;
          T1* im2col_row_start =
              im2col_patch_start + (filter_y * filter_width * input_depth);
          // If we're off the top or the bottom of the input, fill the
          // whole row with zeroes.
          if ((in_y < 0) || (in_y >= input_height)) {
            // On Android, memset and memcpy are significantly faster than the
            // more modern std::set and std::copy equivalents.
            memset(im2col_row_start, input_offset,
                   (filter_width * input_depth));
          } else {
            // What we're doing here is trying to copy and fill the im2col
            // buffer as efficiently as possible, using functions to set or
            // duplicate values en masse. We know we don't have to worry about
            // vertical edges because we dealt with that case above, so we
            // just need to handle filters that overlap the left or right
            // edges. Here's what that looks like:
            //
            // < left_zero_count > < center_copy_count > < right_zero_count >
            // +------------------+---------------------+--------------------+
            // |     (filter)     |       (image)       |      (filter)      |
            // +------------------+---------------------+--------------------+
            // in_x_origin        0                 input_width       in_x_end
            //
            // In reality it's unlikely that a filter patch will be wider
            // than an input, but this shows all the edge cases.
            // We use memset() to set the left and right sections to zeroes
            // and memcpy() to copy over the input data for the center. These
            // are preferred to std::fill and std::copy because they're much
            // faster on Android.
            const int in_x_end = in_x_origin + filter_width;
            const int left_zero_count = std::max(0, 0 - in_x_origin);
            const int right_zero_count = std::max(0, in_x_end - input_width);
            const int center_copy_count =
                filter_width - (left_zero_count + right_zero_count);
            if (left_zero_count > 0) {
              T1* im2col_left_start = im2col_row_start;
              memset(im2col_left_start, input_offset,
                     (left_zero_count * input_depth));
            }
            if (center_copy_count > 0) {
              const T1* input_row_start =
                  input_batch_start + (in_y * input_width * input_depth) +
                  (std::max(0, in_x_origin) * input_depth);
              T1* im2col_center_start =
                  im2col_row_start + (left_zero_count * input_depth);
              memcpy(im2col_center_start, input_row_start,
                     (center_copy_count * input_depth));
            }
            if (right_zero_count > 0) {
              T1* im2col_right_start =
                  im2col_row_start +
                  ((left_zero_count + center_copy_count) * input_depth);
              memset(im2col_right_start, input_offset,
                     (right_zero_count * input_depth));
            }
          }
        }
      }
      // Now we've assembled a set of image patches into a matrix, apply a
      // GEMM matrix multiply of the patches as rows, times the filter
      // weights in columns, to get partial results in the output matrix.
      const int how_many_patches = patch_index_end - patch_index_start;
      const bool transpose_a = false;
      const bool transpose_b = false;
      const bool transpose_c = false;
      const int m = how_many_patches;
      const int n = filter_count;
      const int k = filter_value_count;
      const int lda = filter_value_count;
      const int ldb = filter_count;
      const int ldc = filter_count;
      T3* chunk_output_data = output_data + (patch_index_start * filter_count);

      if (meta::IsSupportedAndEnabled() && std::is_same<T1, quint8>() &&
          std::is_same<T2, quint8>() && std::is_same<T3, qint32>() &&
          (output_offset == 0) && (output_mult == 1) && (output_shift == 0) &&
          (transpose_c == false) && (k <= 2048)) {
        meta::QuantizedGemm(context, transpose_a, transpose_b, im2col_buffer,
                            filter_data, chunk_output_data, m, n, k,
                            -input_offset, -filter_offset, lda, ldb, ldc);
      } else if (std::is_same<T1, quint8>() && std::is_same<T2, quint8>() &&
                 std::is_same<T3, qint32>() && (output_offset == 0) &&
                 (output_mult == 1) && (output_shift == 0)) {
        // The gemmlowp optimized library only works for a particular set of
        // data types, so check if we meet those requirements and fall back to a
        // slower reference implementation if not.
        const uint8* im2col_data_as_uint8 = &(im2col_buffer->value);
        const uint8* filter_data_as_uint8 = &(filter_data->value);
        int32* output_data_as_int32 = &(chunk_output_data->value);
        // All of the transpose_* variables are currently compile-time consts,
        // so we could just hard-code these values too, but that would break if
        // anybody changed those values in the future (e.g. to match the ability
        // of MatMul to specify them as attributes). We're using a verbose
        // approach of deriving the order values from the transpose variables to
        // be able to catch any changes like that.
        static const gemmlowp::MapOrder ResultOrder =
            !transpose_c ? gemmlowp::MapOrder::RowMajor
                         : gemmlowp::MapOrder::ColMajor;
        static const gemmlowp::MapOrder LhsOrder =
            !transpose_a ? gemmlowp::MapOrder::RowMajor
                         : gemmlowp::MapOrder::ColMajor;
        static const gemmlowp::MapOrder RhsOrder =
            !transpose_b ? gemmlowp::MapOrder::RowMajor
                         : gemmlowp::MapOrder::ColMajor;
        gemmlowp::MatrixMap<const std::uint8_t, LhsOrder> lhs(
            im2col_data_as_uint8, m, k, lda);
        gemmlowp::MatrixMap<const std::uint8_t, RhsOrder> rhs(
            filter_data_as_uint8, k, n, ldb);
        gemmlowp::MatrixMap<std::int32_t, ResultOrder> result(
            output_data_as_int32, m, n, ldc);
        const std::tuple<> empty_pipeline = {};

        auto& worker_threads =
            *(context->device()->tensorflow_cpu_worker_threads());
        TensorflowGemmContext context(worker_threads.num_threads,
                                      worker_threads.workers);
        gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t,
                                         gemmlowp::DefaultL8R8BitDepthParams>(
            &context, lhs, rhs, &result, -input_offset, -filter_offset,
            empty_pipeline);
        // Since gemmlowp uses assembly to write to the output, msan won't
        // detect the output buffer as written to, so we mark it manually.
        TF_ANNOTATE_MEMORY_IS_INITIALIZED(output_data_as_int32,
                                          m * n * sizeof(int32));
      } else {
        ReferenceGemm<T1, T2, T3>(
            transpose_a, transpose_b, transpose_c, m, n, k, im2col_buffer,
            input_offset, lda, filter_data, filter_offset, ldb,
            chunk_output_data, output_shift, output_offset, output_mult, ldc);
      }
    }
  }
};

template <class T1, class T2, class T3,
          template <class TF1, class TF2, class TF3> class ConvFunctor>
class QuantizedConv2DOp : public OpKernel {
 public:
  explicit QuantizedConv2DOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, strides_[1] == strides_[2],
                errors::InvalidArgument(
                    "Current implementation only supports equal length "
                    "strides in the row and column dimensions."));
    OP_REQUIRES(
        context, (strides_[0] == 1 && strides_[3] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    std::vector<int32> dilations;
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations));
    OP_REQUIRES(context, dilations.size() == 4,
                errors::InvalidArgument("Dilations field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, dilations[1] == 1 && dilations[2] == 1,
                errors::InvalidArgument(
                    "Current implementation only supports dilated rate as 1 "
                    "in the row and column dimensions."));
    OP_REQUIRES(context, (dilations[0] == 1 && dilations[3] == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilations in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& filter = context->input(1);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be rank 4 but is rank ",
                                        input.shape().dims()));
    OP_REQUIRES(context, filter.dims() == 4,
                errors::InvalidArgument("filter must be rank 4 but is rank ",
                                        filter.shape().dims()));

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(context->input(2).shape()),
                errors::InvalidArgument("min_input must be rank 0 but is rank ",
                                        context->input(2).shape().dims()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(context->input(3).shape()),
                errors::InvalidArgument("max_input must be rank 0 but is rank ",
                                        context->input(3).shape().dims()));
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(context->input(4).shape()),
        errors::InvalidArgument("min_filter must be rank 0 but is rank ",
                                context->input(4).shape().dims()));
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(context->input(5).shape()),
        errors::InvalidArgument("max_filter must be rank 0 but is rank ",
                                context->input(5).shape().dims()));

    const float min_input = context->input(2).flat<float>()(0);
    const float max_input = context->input(3).flat<float>()(0);
    const float min_filter = context->input(4).flat<float>()(0);
    const float max_filter = context->input(5).flat<float>()(0);
    const int32_t offset_input =
        FloatToQuantizedUnclamped<T1>(0.0f, min_input, max_input);
    const int32_t offset_filter =
        FloatToQuantizedUnclamped<T2>(0.0f, min_filter, max_filter);
    const int32_t offset_output = 0;
    const int32_t mult_output = 1;
    const int32_t shift_output = 0;

    // The last dimension for input is in_depth. It must be the same as the
    // filter's in_depth.
    const int64_t in_depth = input.dim_size(3);
    OP_REQUIRES(context, in_depth == filter.dim_size(2),
                errors::InvalidArgument(
                    "input and filter must have the same depth: ", in_depth,
                    " vs ", filter.dim_size(2)));

    // The last dimension for filter is out_depth.
    const int64_t out_depth = filter.dim_size(3);

    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64_t input_rows = input.dim_size(1);
    const int64_t filter_rows = filter.dim_size(0);

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64_t input_cols = input.dim_size(2);
    const int64_t filter_cols = filter.dim_size(1);

    // The first dimension for input is batch.
    const int64_t batch = input.dim_size(0);

    // For now we take the stride from the second dimension only (we
    // assume row = col stride, and do not support striding on the
    // batch or depth dimension).
    const int stride = strides_[1];

    int64_t out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_rows, filter_rows, stride,
                                         padding_, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_cols, filter_cols, stride,
                                         padding_, &out_cols, &pad_cols));
    CHECK_GT(batch, 0);
    CHECK_GT(out_rows, 0);
    CHECK_GT(out_cols, 0);
    CHECK_GT(out_depth, 0);
    TensorShape out_shape({batch, out_rows, out_cols, out_depth});

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // This will call different implementations (e.g. reference or optimized)
    // depending on the template parameter.
    ConvFunctor<T1, T2, T3> conv_functor;
    conv_functor(context, input.flat<T1>().data(), batch, input_rows,
                 input_cols, in_depth, offset_input, filter.flat<T2>().data(),
                 filter_rows, filter_cols, out_depth, offset_filter, stride,
                 padding_, output->flat<T3>().data(), out_rows, out_cols,
                 shift_output, offset_output, mult_output);

    float min_output_value;
    float max_output_value;
    QuantizationRangeForMultiplication<T1, T2, T3>(
        min_input, max_input, min_filter, max_filter, &min_output_value,
        &max_output_value);

    Tensor* output_min = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {}, &output_min));
    output_min->flat<float>()(0) = min_output_value;

    Tensor* output_max = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, {}, &output_max));
    output_max->flat<float>()(0) = max_output_value;
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
};

// Right now we only support taking two eight bit inputs, and returning the
// results as signed 32-bit integers.
REGISTER_KERNEL_BUILDER(
    Name("QuantizedConv2D")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<quint8>("Tfilter")
        .TypeConstraint<qint32>("out_type"),
    QuantizedConv2DOp<quint8, quint8, qint32, Im2ColConvFunctor>);

}  // namespace tensorflow
