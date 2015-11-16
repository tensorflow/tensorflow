// See docs in ../ops/image_ops.cc

#include <memory>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/png/png_io.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

// Encode an image to a PNG stream
class EncodePngOp : public OpKernel {
 public:
  explicit EncodePngOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("compression", &compression_));
    OP_REQUIRES(context, -1 <= compression_ && compression_ <= 9,
                errors::InvalidArgument("compression should be in [-1,9], got ",
                                        compression_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& image = context->input(0);
    OP_REQUIRES(context, image.dims() == 3,
                errors::InvalidArgument("image must be 3-dimensional",
                                        image.shape().ShortDebugString()));
    const int64 channels = image.dim_size(2);
    OP_REQUIRES(context, channels == 1 || channels == 3 || channels == 4,
                errors::InvalidArgument(
                    "image must have 1, 3, or 4 channels, got ", channels));

    // Encode image to png string
    Tensor* output = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    OP_REQUIRES(context,
                png::WriteImageToBuffer(
                    image.flat<uint8>().data(), image.dim_size(1),
                    image.dim_size(0), image.dim_size(1) * channels, channels,
                    8, compression_, &output->scalar<string>()(), nullptr),
                errors::Internal("PNG encoding failed"));
  }

 private:
  int compression_;
};
REGISTER_KERNEL_BUILDER(Name("EncodePng").Device(DEVICE_CPU), EncodePngOp);

}  // namespace tensorflow
