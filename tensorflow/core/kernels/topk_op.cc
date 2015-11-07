// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/gtl/top_n.h"
#include "tensorflow/core/public/tensor_shape.h"
#include "tensorflow/core/public/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

template <typename T>
class TopK : public OpKernel {
 public:
  explicit TopK(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
  }

  void Compute(OpKernelContext* context) override {
    const auto& input_in = context->input(0);
    OP_REQUIRES(context, input_in.dims() == 2,
                errors::InvalidArgument("input must be 2-dimensional"));
    OP_REQUIRES(context, input_in.dim_size(1) >= k_,
                errors::InvalidArgument("input must have at least k columns"));

    const auto& input = input_in.matrix<T>();

    const auto num_rows = input_in.dim_size(0);  // generally batch_size
    const auto num_cols = input_in.dim_size(1);

    Tensor* values_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({num_rows, k_}), &values_out));
    Tensor* indices_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({num_rows, k_}), &indices_out));
    auto values = values_out->matrix<T>();
    auto indices = indices_out->matrix<int32>();

    gtl::TopN<std::pair<T, int32>> filter(k_);

    for (int r = 0; r < num_rows; r++) {
      for (int32 c = 0; c < num_cols; ++c) {
        // The second element is the negated index, so that lower-index elements
        // are considered larger than higher-index elements in case of ties.
        filter.push(std::make_pair(input(r, c), -c));
      }

      std::unique_ptr<std::vector<std::pair<T, int32>>> top_k(filter.Extract());
      for (int32 i = 0; i < k_; ++i) {
        values(r, i) = (*top_k)[i].first;
        indices(r, i) = -(*top_k)[i].second;
      }
      filter.Reset();
    }
  }

 private:
  int k_;
};

#define REGISTER_KERNELS(type) \
  REGISTER_KERNEL_BUILDER(     \
      Name("TopK").Device(DEVICE_CPU).TypeConstraint<type>("T"), TopK<type>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  // namespace tensorflow
