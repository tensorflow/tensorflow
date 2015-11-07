// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/public/tensor_shape.h"
#include "tensorflow/core/public/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

template <typename T>
class InTopK : public OpKernel {
 public:
  explicit InTopK(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
  }

  void Compute(OpKernelContext* context) override {
    const auto& predictions_in = context->input(0);
    const auto& targets_in = context->input(1);
    OP_REQUIRES(context, predictions_in.dims() == 2,
                errors::InvalidArgument("predictions must be 2-dimensional"));
    OP_REQUIRES(context, targets_in.dims() == 1,
                errors::InvalidArgument("targets must be 1-dimensional"));
    OP_REQUIRES(context, predictions_in.dim_size(0) == targets_in.dim_size(0),
                errors::InvalidArgument("First dimension of predictions ",
                                        predictions_in.dim_size(0),
                                        " must match length of targets ",
                                        targets_in.dim_size(0)));
    const auto& predictions = predictions_in.matrix<T>();
    const auto& targets = targets_in.vec<int>();

    Tensor* t_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({targets_in.dim_size(0)}), &t_out));
    auto out = t_out->vec<bool>();

    const auto size = targets.size();
    const auto num_classes = predictions.dimension(1);
    for (int b = 0; b < size; b++) {
      T target_prediction = predictions(b, targets(b));
      int more_probable_classes = 0;
      for (int i = 0; i < num_classes; ++i) {
        if (predictions(b, i) > target_prediction) ++more_probable_classes;
      }
      out(b) = more_probable_classes < k_;
    }
  }

 private:
  int k_;
};

REGISTER_KERNEL_BUILDER(Name("InTopK").Device(DEVICE_CPU), InTopK<float>);

}  // namespace tensorflow
