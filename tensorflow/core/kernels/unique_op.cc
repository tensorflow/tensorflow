#include <unordered_map>
#include <utility>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename T>
class UniqueOp : public OpKernel {
 public:
  explicit UniqueOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt}, {dt, DT_INT32}));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input.shape()),
                errors::InvalidArgument("unique expects a 1D vector."));
    auto Tin = input.vec<T>();
    const int N = Tin.size();

    Tensor* idx = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, input.shape(), &idx));
    auto idx_vec = idx->template vec<int32>();

    std::unordered_map<T, int32> uniq;
    uniq.reserve(2 * N);
    for (int i = 0, j = 0; i < N; ++i) {
      auto it = uniq.insert(std::make_pair(Tin(i), j));
      idx_vec(i) = it.first->second;
      if (it.second) {
        ++j;
      }
    }
    int32 uniq_size = uniq.size();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({uniq_size}), &output));
    auto output_vec = output->template vec<T>();

    for (auto it : uniq) {
      output_vec(it.second) = it.first;
    }
  }
};

#define REGISTER_UNIQUE(type)                                      \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Unique").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      UniqueOp<type>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_UNIQUE);
#undef REGISTER_UNIQUE
}  // namespace tensorflow
