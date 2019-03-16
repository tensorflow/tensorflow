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

#include <memory>
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/jpeg/jpeg_mem.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// Encode an image to a JPEG stream
class EncodeJpegOp : public OpKernel {
 public:
  explicit EncodeJpegOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("format", &format_));
    if (format_.empty()) {
      flags_.format = static_cast<jpeg::Format>(0);
    } else if (format_ == "grayscale") {
      flags_.format = jpeg::FORMAT_GRAYSCALE;
    } else if (format_ == "rgb") {
      flags_.format = jpeg::FORMAT_RGB;
    } else {
      OP_REQUIRES(context, false,
                  errors::InvalidArgument(
                      "format must be '', grayscale or rgb, got ", format_));
    }

    OP_REQUIRES_OK(context, context->GetAttr("quality", &flags_.quality));
    OP_REQUIRES(context, 0 <= flags_.quality && flags_.quality <= 100,
                errors::InvalidArgument("quality must be in [0,100], got ",
                                        flags_.quality));
    OP_REQUIRES_OK(context,
                   context->GetAttr("progressive", &flags_.progressive));
    OP_REQUIRES_OK(
        context, context->GetAttr("optimize_size", &flags_.optimize_jpeg_size));
    OP_REQUIRES_OK(context, context->GetAttr("chroma_downsampling",
                                             &flags_.chroma_downsampling));

    string density_unit;
    OP_REQUIRES_OK(context, context->GetAttr("density_unit", &density_unit));
    if (density_unit == "in") {
      flags_.density_unit = 1;
    } else if (density_unit == "cm") {
      flags_.density_unit = 2;
    } else {
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("density_unit must be 'in' or 'cm'",
                                          density_unit));
    }

    OP_REQUIRES_OK(context, context->GetAttr("x_density", &flags_.x_density));
    OP_REQUIRES_OK(context, context->GetAttr("y_density", &flags_.y_density));
    OP_REQUIRES_OK(context, context->GetAttr("xmp_metadata", &xmp_metadata_));
    flags_.xmp_metadata = xmp_metadata_;  // StringPiece doesn't own data
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& image = context->input(0);
    OP_REQUIRES(context, image.dims() == 3,
                errors::InvalidArgument("image must be 3-dimensional",
                                        image.shape().DebugString()));

    OP_REQUIRES(
        context,
        FastBoundsCheck(image.NumElements(), std::numeric_limits<int32>::max()),
        errors::InvalidArgument(
            "Cannot encode images with >= max int32 elements"));

    const int32 dim_size0 = static_cast<int32>(image.dim_size(0));
    const int32 dim_size1 = static_cast<int32>(image.dim_size(1));
    const int32 dim_size2 = static_cast<int32>(image.dim_size(2));

    // Autodetect format if desired, otherwise make sure format and
    // image channels are consistent.
    int channels;
    jpeg::CompressFlags adjusted_flags = flags_;
    if (flags_.format == 0) {
      channels = dim_size2;
      if (channels == 1) {
        adjusted_flags.format = jpeg::FORMAT_GRAYSCALE;
      } else if (channels == 3) {
        adjusted_flags.format = jpeg::FORMAT_RGB;
      } else {
        OP_REQUIRES(
            context, false,
            errors::InvalidArgument("image must have 1 or 3 channels, got ",
                                    image.shape().DebugString()));
      }
    } else {
      if (flags_.format == jpeg::FORMAT_GRAYSCALE) {
        channels = 1;
      } else {  // RGB
        channels = 3;
      }
      OP_REQUIRES(context, channels == dim_size2,
                  errors::InvalidArgument("format ", format_, " expects ",
                                          channels, " channels, got ",
                                          image.shape().DebugString()));
    }

    // Encode image to jpeg string
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    OP_REQUIRES(context,
                jpeg::Compress(image.flat<uint8>().data(), dim_size1, dim_size0,
                               adjusted_flags, &output->scalar<string>()()),
                errors::Internal("JPEG encoding failed"));
  }

 private:
  string format_;
  string xmp_metadata_;  // Owns data referenced by flags_
  jpeg::CompressFlags flags_;
};
REGISTER_KERNEL_BUILDER(Name("EncodeJpeg").Device(DEVICE_CPU), EncodeJpegOp);

}  // namespace tensorflow
