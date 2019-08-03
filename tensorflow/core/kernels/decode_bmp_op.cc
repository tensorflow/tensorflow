/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// Decode the contents of a BMP file
class DecodeBmpOp : public OpKernel {
 public:
  explicit DecodeBmpOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("channels", &channels_));
    OP_REQUIRES(
        context,
        channels_ == 0 || channels_ == 1 || channels_ == 3 || channels_ == 4,
        errors::InvalidArgument("channels must be 0, 1, 3 or 4, got ",
                                channels_));
  }
  inline int32 ByteSwapInt32ForBigEndian(int32 x) {
#if (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    return le32toh(x);
#else
    return x;
#endif
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& contents = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents.shape()),
                errors::InvalidArgument("contents must be scalar, got shape ",
                                        contents.shape().DebugString()));

    // Start decoding image to get shape details
    const StringPiece input = contents.scalar<string>()();

    OP_REQUIRES(context, (32 <= input.size()),
                errors::InvalidArgument("Incomplete bmp content, requires at "
                                        "least 32 bytes to find the header "
                                        "size, width, height, and bpp, got ",
                                        input.size(), " bytes"));

    const uint8* img_bytes = reinterpret_cast<const uint8*>(input.data());
    int32 header_size_ = internal::SubtleMustCopy(
        *(reinterpret_cast<const int32*>(img_bytes + 10)));
    const int32 header_size = ByteSwapInt32ForBigEndian(header_size_);
    int32 width_ = internal::SubtleMustCopy(
        *(reinterpret_cast<const int32*>(img_bytes + 18)));
    const int32 width = ByteSwapInt32ForBigEndian(width_);
    int32 height_ = internal::SubtleMustCopy(
        *(reinterpret_cast<const int32*>(img_bytes + 22)));
    const int32 height = ByteSwapInt32ForBigEndian(height_);
    int32 bpp_ = internal::SubtleMustCopy(
        *(reinterpret_cast<const int32*>(img_bytes + 28)));
    const int32 bpp = ByteSwapInt32ForBigEndian(bpp_);

    if (channels_) {
      OP_REQUIRES(context, (channels_ == bpp / 8),
                  errors::InvalidArgument(
                      "channels attribute ", channels_,
                      " does not match bits per pixel from file ", bpp / 8));
    } else {
      channels_ = bpp / 8;
    }

    // Current implementation only supports 1, 3 or 4 channel
    // bitmaps.
    OP_REQUIRES(context, (channels_ == 1 || channels_ == 3 || channels_ == 4),
                errors::InvalidArgument(
                    "Number of channels must be 1, 3 or 4, was ", channels_));

    OP_REQUIRES(context, width > 0,
                errors::InvalidArgument("Width must be positive"));
    OP_REQUIRES(context, height != 0,
                errors::InvalidArgument("Height must be nonzero"));
    OP_REQUIRES(context, header_size >= 0,
                errors::InvalidArgument("header size must be nonnegative"));

    // The real requirement is < 2^31 minus some headers and channel data,
    // so rounding down to something that's still ridiculously big.
    OP_REQUIRES(
        context,
        (static_cast<int64>(width) * std::abs(static_cast<int64>(height))) <
            static_cast<int64>(std::numeric_limits<int32_t>::max() / 8),
        errors::InvalidArgument(
            "Total possible pixel bytes must be less than 2^30"));

    const int32 abs_height = abs(height);

    // there may be padding bytes when the width is not a multiple of 4 bytes
    const int row_size = (channels_ * width + 3) / 4 * 4;

    const int64 last_pixel_offset = static_cast<int64>(header_size) +
                                    (abs_height - 1) * row_size +
                                    (width - 1) * channels_;

    // [expected file size] = [last pixel offset] + [last pixel size=channels]
    const int64 expected_file_size = last_pixel_offset + channels_;

    OP_REQUIRES(
        context, (expected_file_size <= input.size()),
        errors::InvalidArgument("Incomplete bmp content, requires at least ",
                                expected_file_size, " bytes, got ",
                                input.size(), " bytes"));

    // if height is negative, data layout is top down
    // otherwise, it's bottom up
    bool top_down = (height < 0);

    // Decode image, allocating tensor once the image size is known
    Tensor* output = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(
                     0, TensorShape({abs_height, width, channels_}), &output));

    const uint8* bmp_pixels = &img_bytes[header_size];

    Decode(bmp_pixels, row_size, output->flat<uint8>().data(), width,
           abs_height, channels_, top_down);
  }

  uint8* Decode(const uint8* input, const int row_size, uint8* const output,
                const int width, const int height, const int channels,
                bool top_down);

 private:
  int channels_;
};
REGISTER_KERNEL_BUILDER(Name("DecodeBmp").Device(DEVICE_CPU), DecodeBmpOp);

uint8* DecodeBmpOp::Decode(const uint8* input, const int row_size,
                           uint8* const output, const int width,
                           const int height, const int channels,
                           bool top_down) {
  for (int i = 0; i < height; i++) {
    int src_pos;
    int dst_pos;

    for (int j = 0; j < width; j++) {
      if (!top_down) {
        src_pos = ((height - 1 - i) * row_size) + j * channels;
      } else {
        src_pos = i * row_size + j * channels;
      }

      dst_pos = (i * width + j) * channels;

      switch (channels) {
        case 1:
          output[dst_pos] = input[src_pos];
          break;
        case 3:
          // BGR -> RGB
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];
          break;
        case 4:
          // BGRA -> RGBA
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];
          output[dst_pos + 3] = input[src_pos + 3];
          break;
        default:
          LOG(FATAL) << "Unexpected number of channels: " << channels;
          break;
      }
    }
  }

  return output;
}

}  // namespace tensorflow
