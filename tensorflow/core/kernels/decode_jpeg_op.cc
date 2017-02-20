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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/jpeg/jpeg_mem.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// Decode the contents of a JPEG file
class DecodeJpegOp : public OpKernel {
 public:
  explicit DecodeJpegOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("channels", &flags_.components));
    OP_REQUIRES(context, flags_.components == 0 || flags_.components == 1 ||
                             flags_.components == 3,
                errors::InvalidArgument("channels must be 0, 1, or 3, got ",
                                        flags_.components));
    OP_REQUIRES_OK(context, context->GetAttr("ratio", &flags_.ratio));
    OP_REQUIRES(context, flags_.ratio == 1 || flags_.ratio == 2 ||
                             flags_.ratio == 4 || flags_.ratio == 8,
                errors::InvalidArgument("ratio must be 1, 2, 4, or 8, got ",
                                        flags_.ratio));
    OP_REQUIRES_OK(
        context, context->GetAttr("fancy_upscaling", &flags_.fancy_upscaling));
    OP_REQUIRES_OK(context,
                   context->GetAttr("try_recover_truncated",
                                    &flags_.try_recover_truncated_jpeg));
    OP_REQUIRES_OK(context, context->GetAttr("acceptable_fraction",
                                             &flags_.min_acceptable_fraction));

    string dct_method;
    OP_REQUIRES_OK(context, context->GetAttr("dct_method", &dct_method));
    OP_REQUIRES(
        context, (dct_method.empty() || dct_method == "INTEGER_FAST" ||
                  dct_method == "INTEGER_ACCURATE"),
        errors::InvalidArgument("dct_method must be one of "
                                "{'', 'INTEGER_FAST', 'INTEGER_ACCURATE'}"));
    if (dct_method == "INTEGER_FAST") {
      flags_.dct_method = JDCT_IFAST;
    } else if (dct_method == "INTEGER_ACCURATE") {
      flags_.dct_method = JDCT_ISLOW;
    } else {
      // The TensorFlow-chosen default is IFAST, sacrificing decoding
      // image quality for speed.
      flags_.dct_method = JDCT_IFAST;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& contents = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents.shape()),
                errors::InvalidArgument("contents must be scalar, got shape ",
                                        contents.shape().DebugString()));
    const StringPiece input = contents.scalar<string>()();
    OP_REQUIRES(context, input.size() <= std::numeric_limits<int>::max(),
                errors::InvalidArgument("JPEG contents are too large for int: ",
                                        input.size()));

    // Decode image, allocating tensor once the image size is known
    Tensor* output = NULL;
    OP_REQUIRES(
        context,
        jpeg::Uncompress(
            input.data(), input.size(), flags_, nullptr /* nwarn */,
            [=, &output](int width, int height, int channels) -> uint8* {
              Status status(context->allocate_output(
                  0, TensorShape({height, width, channels}), &output));
              if (!status.ok()) {
                VLOG(1) << status;
                context->SetStatus(status);
                return nullptr;
              }
              return output->flat<uint8>().data();
            }),
        errors::InvalidArgument("Invalid JPEG data, size ", input.size()));
  }

 private:
  jpeg::UncompressFlags flags_;
};
REGISTER_KERNEL_BUILDER(Name("DecodeJpeg").Device(DEVICE_CPU), DecodeJpegOp);

}  // namespace tensorflow
