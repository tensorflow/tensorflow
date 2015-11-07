// See docs in ../ops/io_ops.cc.

#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/match.h"
#include "tensorflow/core/public/env.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

class MatchingFilesOp : public OpKernel {
 public:
  using OpKernel::OpKernel;
  void Compute(OpKernelContext* context) override {
    const Tensor* pattern;
    OP_REQUIRES_OK(context, context->input("pattern", &pattern));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(pattern->shape()),
                errors::InvalidArgument(
                    "Input pattern tensor must be scalar, but had shape: ",
                    pattern->shape().DebugString()));
    std::vector<string> fnames;
    OP_REQUIRES_OK(context,
                   io::GetMatchingFiles(context->env(),
                                        pattern->scalar<string>()(), &fnames));
    const int num_out = fnames.size();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                "filenames", TensorShape({num_out}), &output));
    auto output_vec = output->vec<string>();
    for (int i = 0; i < num_out; ++i) {
      output_vec(i) = fnames[i];
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MatchingFiles").Device(DEVICE_CPU),
                        MatchingFilesOp);

}  // namespace tensorflow
