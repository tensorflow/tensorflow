// See docs in ../ops/image_ops.cc

#include <memory>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"
#include "tensorflow/core/lib/png/png_io.h"

namespace tensorflow {

// Decode the contents of a PNG file
class DecodePngOp : public OpKernel {
 public:
  explicit DecodePngOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("channels", &channels_));
    OP_REQUIRES(context, channels_ == 0 || channels_ == 1 || channels_ == 3 ||
                             channels_ == 4,
                errors::InvalidArgument("channels must be 0, 1, 3, or 4, got ",
                                        channels_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& contents = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents.shape()),
                errors::InvalidArgument("contents must be scalar, got shape ",
                                        contents.shape().ShortDebugString()));

    // Start decoding image to get shape details
    const StringPiece data = contents.scalar<string>()();
    png::DecodeContext decode;
    OP_REQUIRES(
        context, png::CommonInitDecode(data, channels_, 8, &decode),
        errors::InvalidArgument("Invalid PNG header, data size ", data.size()));

    // Verify that width and height don't overflow int
    const int width = decode.width;
    const int height = decode.height;
    if (width != static_cast<int64>(decode.width) ||
        height != static_cast<int64>(decode.height)) {
      png::CommonFreeDecode(&decode);
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("PNG size too large for int: ",
                                          decode.width, " by ", decode.height));
    }

    // Allocate tensor
    Tensor* output = nullptr;
    const auto status = context->allocate_output(
        0, TensorShape({height, width, decode.channels}), &output);
    if (!status.ok()) png::CommonFreeDecode(&decode);
    OP_REQUIRES_OK(context, status);

    // Finish decoding image
    OP_REQUIRES(
        context, png::CommonFinishDecode(output->flat<uint8>().data(),
                                         decode.channels * width, &decode),
        errors::InvalidArgument("Invalid PNG data, size ", data.size()));
  }

 private:
  int channels_;
};
REGISTER_KERNEL_BUILDER(Name("DecodePng").Device(DEVICE_CPU), DecodePngOp);

}  // namespace tensorflow
