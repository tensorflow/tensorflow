/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "external/gemmlowp/public/gemmlowp.h"
#include "tensorflow/contrib/quantization/kernels/quantization_utils.h"
#include "tensorflow/contrib/quantization/kernels/reference_gemm.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
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
  void operator()(const T1* input_data, int input_batches, int input_height,
                  int input_width, int input_depth, int input_offset,
                  const T2* filter_data, int filter_height, int filter_width,
                  int filter_count, int filter_offset, int stride,
                  Padding padding, T3* output_data, int output_height,
                  int output_width, int output_shift, int output_offset,
                  int output_mult) {
    // Set up some constants we need for the output down-shifting and
    // saturation.
    const int32 highest = static_cast<int32>(Eigen::NumTraits<T3>::highest());
    const int32 lowest = static_cast<int32>(Eigen::NumTraits<T3>::lowest());

    // When we're converting the 32 bit accumulator to a lower bit depth, we
    // need to add on 0.5 in fixed-point terms to make the operation round half
    // up towards positive infinity, rather than a floor.
    // We also need to watch out for the case when there's no down shift,
    // because a left shift by a negative number gives undefined results.
    const int32 rounding = (output_shift < 1) ? 0 : (1 << (output_shift - 1));

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
          ((output_width - 1) * stride + filter_width - input_width) / 2;
      filter_top_offset =
          ((output_height - 1) * stride + filter_height - input_height) / 2;
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
            int32 total = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                for (int in_channel = 0; in_channel < input_depth;
                     ++in_channel) {
                  const int in_x = in_x_origin + filter_x;
                  const int in_y = in_y_origin + filter_y;
                  int32 input_value;
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
                  const int32 filter_value =
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
            const int32 top_clamped_output = std::min(output, highest);
            const int32 clamped_output = std::max(top_clamped_output, lowest);
            output_data[(batch * output_height * output_width * filter_count) +
                        (out_y * output_width * filter_count) +
                        (out_x * filter_count) + out_channel] = clamped_output;
          }
        }
      }
    }
  }
};

// Implements convolution as a two stage process, first packing the patches of
// the input image into columns (im2col) and then running GEMM to produce the
// final result.
// TODO(petewarden) - We need to update gemmlowp to support 32-bit outputs
// before we can re-enable this path.
template <class T1, class T2, class T3>
class Im2ColConvFunctor {
 public:
  void operator()(const T1* input_data, int input_batches, int input_height,
                  int input_width, int input_depth, int input_offset,
                  const T2* filter_data, int filter_height, int filter_width,
                  int filter_count, int filter_offset, int stride,
                  Padding padding, T3* output_data, int output_height,
                  int output_width, int output_shift, int output_offset,
                  int output_mult) {
    if (input_offset < 0) {
      // Only log the first few occurrences of this warning.
      static int warning_count = 0;
      if (warning_count < 10) {
        ++warning_count;
        LOG(WARNING)
            << "Zero is not representable in the quantized range used by the"
            << " input. This means QuantizedConv2d has to fall back to a slow"
            << " implementation, since the border of zero values can't be"
            << " represented easily. You should try to construct graphs that"
            << " avoid this situation.";
      }
      ReferenceConvFunctor<T1, T2, T3> conv_functor;
      conv_functor(input_data, input_batches, input_height, input_width,
                   input_depth, input_offset, filter_data, filter_height,
                   filter_width, filter_count, filter_offset, stride, padding,
                   output_data, output_height, output_width, output_shift,
                   output_offset, output_mult);
      return;
    }

    CHECK_GT(output_width, 0);
    CHECK_GT(output_height, 0);
    int filter_left_offset;
    int filter_top_offset;
    if (padding == VALID) {
      filter_left_offset =
          ((output_width - 1) * stride + filter_width - input_width) / 2;
      filter_top_offset =
          ((output_height - 1) * stride + filter_height - input_height) / 2;
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
    const int patch_count = input_batches * output_width * output_height;
    const int im2col_size = patch_count * filter_value_count;
    // TODO(petewarden) - Memory allocation can be very slow on Android. Can we
    // optimize this by keeping the scratch buffer around?
    std::unique_ptr<T1[]> im2col_buffer(new T1[im2col_size]);

    for (int batch = 0; batch < input_batches; ++batch) {
      const T1* input_batch_start =
          input_data + (batch * input_height * input_width * input_depth);
      for (int out_y = 0; out_y < output_height; ++out_y) {
        const int in_y_origin = (out_y * stride) - filter_top_offset;
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = (out_x * stride) - filter_left_offset;
          const int patch_index = (batch * output_width * output_height) +
                                  (out_y * output_width) + out_x;
          T1* im2col_patch_start =
              im2col_buffer.get() + (patch_index * filter_value_count);
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + filter_y;
            T1* im2col_row_start =
                im2col_patch_start + (filter_y * filter_width * input_depth);
            // If we're off the top or the bottom of the input, fill the whole
            // row with zeroes.
            if ((in_y < 0) || (in_y >= input_height)) {
              T1* im2col_row_end =
                  im2col_row_start + (filter_width * input_depth);
              // We'll be subtracting this offset during the calculations
              // so to get an actual zero after that bias we need to set
              // it to input_offset here.
              std::fill(im2col_row_start, im2col_row_end, input_offset);
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
              // We use std::fill() to set the left and right sections to zeroes
              // and std::copy() to copy over the input data for the center.
              const int in_x_end = in_x_origin + filter_width;
              const int left_zero_count = std::max(0, 0 - in_x_origin);
              const int right_zero_count = std::max(0, in_x_end - input_width);
              const int center_copy_count =
                  filter_width - (left_zero_count + right_zero_count);
              if (left_zero_count > 0) {
                T1* im2col_left_start = im2col_row_start;
                T1* im2col_left_end =
                    im2col_left_start + (left_zero_count * input_depth);
                std::fill(im2col_left_start, im2col_left_end, input_offset);
              }
              if (center_copy_count > 0) {
                const T1* input_row_start =
                    input_batch_start + (in_y * input_width * input_depth) +
                    (std::max(0, in_x_origin) * input_depth);
                const T1* input_row_end =
                    input_row_start + (center_copy_count * input_depth);
                T1* im2col_center_start =
                    im2col_row_start + (left_zero_count * input_depth);
                std::copy(input_row_start, input_row_end, im2col_center_start);
              }
              if (right_zero_count > 0) {
                T1* im2col_right_start =
                    im2col_row_start +
                    ((left_zero_count + center_copy_count) * input_depth);
                T1* im2col_right_end =
                    im2col_right_start + (right_zero_count * input_depth);
                std::fill(im2col_right_start, im2col_right_end, input_offset);
              }
            }
          }
        }
      }
    }

    CHECK_GT(patch_count, 0);
    CHECK_GT(filter_count, 0);
    CHECK_GT(filter_value_count, 0);

    const bool transpose_a = false;
    const bool transpose_b = false;
    const bool transpose_c = false;
    const int m = patch_count;
    const int n = filter_count;
    const int k = filter_value_count;
    const int lda = filter_value_count;
    const int ldb = filter_count;
    const int ldc = filter_count;
    // The gemmlowp optimized library only works for a particular set of data
    // types, so check if we meet those requirements and
    // fall back to a slower reference implementation if not.
    if (std::is_same<T1, quint8>() && std::is_same<T2, quint8>() &&
        std::is_same<T3, qint32>() && (output_offset == 0) &&
        (output_mult == 1) && (output_shift == 0)) {
      const uint8* im2col_data_as_uint8 = &(im2col_buffer.get()->value);
      const uint8* filter_data_as_uint8 = &(filter_data->value);
      int32* output_data_as_int32 = &(output_data->value);
      // All of the transpose_* variables are currently compile-time consts, so
      // we could just hard-code these values too, but that would break if
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
      gemmlowp::GemmContext context;
      gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t,
                                       gemmlowp::DefaultL8R8BitDepthParams>(
          &context, lhs, rhs, &result, -input_offset, -filter_offset,
          empty_pipeline);
    } else {
      ReferenceGemm<T1, T2, T3>(transpose_a, transpose_b, transpose_c, m, n, k,
                                im2col_buffer.get(), input_offset, lda,
                                filter_data, filter_offset, ldb, output_data,
                                output_shift, output_offset, output_mult, ldc);
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
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, filter.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter.shape().DebugString()));

    const float min_input = context->input(2).flat<float>()(0);
    const float max_input = context->input(3).flat<float>()(0);
    const float min_filter = context->input(4).flat<float>()(0);
    const float max_filter = context->input(5).flat<float>()(0);
    const int32 offset_input =
        FloatToQuantizedUnclamped<T1>(0.0f, min_input, max_input);
    const int32 offset_filter =
        FloatToQuantizedUnclamped<T2>(0.0f, min_filter, max_filter);
    const int32 offset_output = 0;
    const int32 mult_output = 1;
    const int32 shift_output = 0;

    // The last dimension for input is in_depth. It must be the same as the
    // filter's in_depth.
    const int64 in_depth = input.dim_size(3);
    OP_REQUIRES(
        context, in_depth == filter.dim_size(2),
        errors::InvalidArgument("input and filter must have the same depth: ",
                                in_depth, " vs ", filter.dim_size(2)));

    // The last dimension for filter is out_depth.
    const int64 out_depth = filter.dim_size(3);

    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64 input_rows = input.dim_size(1);
    const int64 filter_rows = filter.dim_size(0);

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64 input_cols = input.dim_size(2);
    const int64 filter_cols = filter.dim_size(1);

    // The first dimension for input is batch.
    const int64 batch = input.dim_size(0);

    // For now we take the stride from the second dimension only (we
    // assume row = col stride, and do not support striding on the
    // batch or depth dimension).
    const int stride = strides_[1];

    int out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    if (filter_cols == filter_rows && filter_rows == 1 && stride == 1) {
      // For 1x1 kernel, the 2D convolution is reduced to matrix
      // multiplication.
      out_rows = input_rows;
      out_cols = input_cols;
    } else {
      OP_REQUIRES_OK(
          context, Get2dOutputSize(input_rows, input_cols, filter_rows,
                                   filter_cols, stride, stride, padding_,
                                   &out_rows, &out_cols, &pad_rows, &pad_cols));
    }
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
    conv_functor(input.flat<T1>().data(), batch, input_rows, input_cols,
                 in_depth, offset_input, filter.flat<T2>().data(), filter_rows,
                 filter_cols, out_depth, offset_filter, stride, padding_,
                 output->flat<T3>().data(), out_rows, out_cols, shift_output,
                 offset_output, mult_output);

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
