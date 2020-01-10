// See docs in ../ops/image_ops.cc

#include <memory>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/jpeg/jpeg_mem.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

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
                                        image.shape().ShortDebugString()));

    // Autodetect format if desired, otherwise make sure format and
    // image channels are consistent.
    int channels;
    jpeg::CompressFlags adjusted_flags = flags_;
    if (flags_.format == 0) {
      channels = image.dim_size(2);
      if (channels == 1) {
        adjusted_flags.format = jpeg::FORMAT_GRAYSCALE;
      } else if (channels == 3) {
        adjusted_flags.format = jpeg::FORMAT_RGB;
      } else {
        OP_REQUIRES(context, false, errors::InvalidArgument(
                                        "image must have 1 or 3 channels, got ",
                                        image.shape().ShortDebugString()));
      }
    } else {
      if (flags_.format == jpeg::FORMAT_GRAYSCALE) {
        channels = 1;
      } else {  // RGB
        channels = 3;
      }
      OP_REQUIRES(context, channels == image.dim_size(2),
                  errors::InvalidArgument("format ", format_, " expects ",
                                          channels, " channels, got ",
                                          image.shape().ShortDebugString()));
    }

    // Encode image to jpeg string
    Tensor* output = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    OP_REQUIRES(context,
                jpeg::Compress(image.flat<uint8>().data(), image.dim_size(1),
                               image.dim_size(0), adjusted_flags,
                               &output->scalar<string>()()),
                errors::Internal("JPEG encoding failed"));
  }

 private:
  string format_;
  string xmp_metadata_;  // Owns data referenced by flags_
  jpeg::CompressFlags flags_;
};
REGISTER_KERNEL_BUILDER(Name("EncodeJpeg").Device(DEVICE_CPU), EncodeJpegOp);

}  // namespace tensorflow
