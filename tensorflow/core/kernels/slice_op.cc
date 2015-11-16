// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/slice_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {

namespace {

gtl::InlinedVector<int64, 4> IntTensorToInt64Vec(const Tensor& tensor) {
  gtl::InlinedVector<int64, 4> out;
  if (tensor.dtype() == DT_INT32) {
    for (int64 i = 0; i < tensor.NumElements(); ++i) {
      out.push_back(tensor.flat<int32>()(i));
    }
  } else if (tensor.dtype() == DT_INT64) {
    for (int64 i = 0; i < tensor.NumElements(); ++i) {
      out.push_back(tensor.flat<int64>()(i));
    }
  } else {
    LOG(FATAL) << "begin must be either int32 or int64";
  }
  return out;
}

}  // namespace

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Shared code that is not dependent on the type of T.  We do this to reduce
// code size by not duplicating all this for all T (float, double, int32, etc.)
static void SharedValidation(OpKernelContext* context,
                             TensorShape* output_shape, bool* is_identity,
                             bool* slice_dim0,
                             gtl::InlinedVector<int64, 4>* begin,
                             gtl::InlinedVector<int64, 4>* size) {
  const Tensor& input = context->input(0);
  const Tensor& begin_tensor = context->input(1);
  const Tensor& size_tensor = context->input(2);

  OP_REQUIRES(
      context, TensorShapeUtils::IsLegacyVector(begin_tensor.shape()) &&
                   TensorShapeUtils::IsLegacyVector(size_tensor.shape()) &&
                   begin_tensor.NumElements() == input.dims() &&
                   size_tensor.NumElements() == input.dims(),
      errors::InvalidArgument(
          "Expected begin and size arguments to be 1-D tensors of size ",
          input.dims(), ", but got ", begin_tensor.NumElements(), " and ",
          size_tensor.NumElements(), " instead."));

  const int input_dims = input.dims();
  *begin = IntTensorToInt64Vec(begin_tensor);
  *size = IntTensorToInt64Vec(size_tensor);
  for (int i = 0; i < input_dims; ++i) {
    if ((*size)[i] == -1) {
      // A size[i] of -1 means "all elements from begin[i] to dim_size(i)".
      (*size)[i] = input.dim_size(i) - (*begin)[i];
    }
  }

  *is_identity = true;
  *slice_dim0 = true;
  for (int i = 0; i < input_dims; ++i) {
    int64 b = (*begin)[i];
    int64 s = (*size)[i];
    if (input.dim_size(i) == 0) {
      OP_REQUIRES(
          context, b == 0 && s == 0,
          errors::InvalidArgument("Expected begin[", i, "] == 0 (got ", b,
                                  ") and size[", i, "] == 0 ", "(got ", s,
                                  ") when ", "input.dim_size(", i, ") == 0"));
    } else {
      OP_REQUIRES(context, 0 <= b && b <= input.dim_size(i),
                  errors::InvalidArgument("Expected begin[", i, "] in [0, ",
                                          input.dim_size(i), "], but got ", b));
      OP_REQUIRES(
          context, 0 <= s && b + s <= input.dim_size(i),
          errors::InvalidArgument("Expected size[", i, "] in [0, ",
                                  input.dim_size(i) - b, "], but ", "got ", s));
    }
    output_shape->AddDim(s);
    const bool take_all = (b == 0) && (s == input.dim_size(i));
    (*is_identity) &= take_all;
    (*slice_dim0) &= (i == 0) || take_all;
  }
}

template <typename Device, typename T>
class SliceOp : public OpKernel {
 public:
  explicit SliceOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TensorShape output_shape;
    bool is_identity = true;
    bool slice_dim0 = true;
    gtl::InlinedVector<int64, 4> begin;
    gtl::InlinedVector<int64, 4> size;
    SharedValidation(context, &output_shape, &is_identity, &slice_dim0, &begin,
                     &size);
    if (!context->status().ok()) return;
    const Tensor& input = context->input(0);
    if (is_identity) {
      VLOG(1) << "Slice identity";
      context->set_output(0, input);
      return;
    }

    if (slice_dim0 && IsInnerDimsSizeAligned<T>(input.shape())) {
      VLOG(1) << "Slice dim 0: " << input.shape().DebugString();
      CHECK_GE(input.dims(), 1);  // Otherwise, is_identity should be true.
      context->set_output(0, input.Slice(begin[0], begin[0] + size[0]));
      return;
    }

    Tensor* result = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &result));
    const int input_dims = input.dims();

    if (output_shape.num_elements() > 0) {
      if (std::is_same<Device, CPUDevice>::value && input_dims == 2 &&
          DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
        auto input = context->input(0).tensor<T, 2>();
        auto output = result->tensor<T, 2>();
        // TODO(agarwal): Consider multi-threading this loop for cases where
        // size[0] is very large.
        for (int i = 0; i < size[0]; ++i) {
          const int row = begin[0] + i;
          if (i + 1 < size[0]) {
            port::prefetch<port::PREFETCH_HINT_T0>(&output(i + 1, 0));
            port::prefetch<port::PREFETCH_HINT_T0>(&input(row + 1, begin[1]));
          }
          memcpy(&output(i, 0), &input(row, begin[1]), size[1] * sizeof(T));
        }
        return;
      }
#define HANDLE_DIM(NDIM)                            \
  if (input_dims == NDIM) {                         \
    HandleCase<NDIM>(context, begin, size, result); \
    return;                                         \
  }

      HANDLE_DIM(1);
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);

#undef HANDLE_DIM

      OP_REQUIRES(context, false, errors::Unimplemented(
                                      "SliceOp : Unhandled input dimensions"));
    }
  }

 private:
  template <int NDIM>
  void HandleCase(OpKernelContext* context, const gtl::ArraySlice<int64>& begin,
                  const gtl::ArraySlice<int64>& size, Tensor* result) {
    Eigen::DSizes<Eigen::DenseIndex, NDIM> indices;
    Eigen::DSizes<Eigen::DenseIndex, NDIM> sizes;
    for (int i = 0; i < NDIM; ++i) {
      indices[i] = begin[i];
      sizes[i] = size[i];
    }

    functor::Slice<Device, T, NDIM>()(
        context->eigen_device<Device>(), result->tensor<T, NDIM>(),
        context->input(0).tensor<T, NDIM>(), indices, sizes);
  }
};

#define REGISTER_SLICE(type)                             \
  REGISTER_KERNEL_BUILDER(Name("Slice")                  \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("begin")       \
                              .HostMemory("size"),       \
                          SliceOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_SLICE);
REGISTER_SLICE(bfloat16);

#undef REGISTER_SLICE

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, NDIM)                                  \
  template <>                                                      \
  void Slice<GPUDevice, T, NDIM>::operator()(                      \
      const GPUDevice& d, typename TTypes<T, NDIM>::Tensor output, \
      typename TTypes<T, NDIM>::ConstTensor input,                 \
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& indices,       \
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& sizes);        \
  extern template struct Slice<GPUDevice, T, NDIM>;

#define DECLARE_FOR_N(T)  \
  DECLARE_GPU_SPEC(T, 1); \
  DECLARE_GPU_SPEC(T, 2); \
  DECLARE_GPU_SPEC(T, 3); \
  DECLARE_GPU_SPEC(T, 4); \
  DECLARE_GPU_SPEC(T, 5);

TF_CALL_GPU_NUMBER_TYPES(DECLARE_FOR_N);
DECLARE_FOR_N(int32);

#undef DECLARE_FOR_N
#undef DECLARE_GPU_SPEC
}  // namespace functor

#define REGISTER_GPU(type)                                     \
  REGISTER_KERNEL_BUILDER(Name("Slice")                        \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<type>("T")       \
                              .HostMemory("begin")             \
                              .HostMemory("size")              \
                              .TypeConstraint<int32>("Index"), \
                          SliceOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
REGISTER_GPU(int32);

#undef REGISTER_GPU

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
