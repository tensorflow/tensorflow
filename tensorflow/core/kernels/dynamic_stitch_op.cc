// See docs in ../ops/data_flow_ops.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {

template <class T>
class DynamicStitchOp : public OpKernel {
 public:
  explicit DynamicStitchOp(OpKernelConstruction* c) : OpKernel(c) {
    // Compute expected input signature
    const DataType dt = DataTypeToEnum<T>::v();
    const int n = c->num_inputs() / 2;
    DataTypeVector expected;
    for (int i = 0; i < n; i++) {
      expected.push_back(DT_INT32);
    }
    for (int i = 0; i < n; i++) {
      expected.push_back(dt);
    }
    OP_REQUIRES_OK(c, c->MatchSignature(expected, {dt}));
    OP_REQUIRES(
        c, c->num_inputs() > 0,
        errors::InvalidArgument("DynamicStitchOp: Must have some inputs"));
    OP_REQUIRES(c, c->num_inputs() % 2 == 0,
                errors::InvalidArgument(
                    "DynamicStitchOp: Must have even number of arguments"));
  }

  void Compute(OpKernelContext* c) override {
    // Find maximum index in the indices vectors
    OpInputList indices_inputs;
    OP_REQUIRES_OK(c, c->input_list("indices", &indices_inputs));

    int32 max_index = -1;
    for (const Tensor& indices : indices_inputs) {
      Eigen::Tensor<int32, 0, Eigen::RowMajor> m =
          indices.flat<int32>().maximum();
      max_index = std::max(m(), max_index);
    }
    const int first_dim_size = max_index + 1;

    // Validate that data[i].shape = indices[i].shape + constant
    OpInputList data_inputs;
    OP_REQUIRES_OK(c, c->input_list("data", &data_inputs));
    const Tensor& data0 = data_inputs[0];
    const Tensor& indices0 = indices_inputs[0];
    for (int input_num = 0; input_num < indices_inputs.size(); input_num++) {
      const Tensor& indices = indices_inputs[input_num];
      const Tensor& data = data_inputs[input_num];
      OP_REQUIRES(
          c, TensorShapeUtils::StartsWith(data.shape(), indices.shape()),
          errors::InvalidArgument(
              "data[", input_num, "].shape = ", data.shape().ShortDebugString(),
              " does not start with indices[", input_num, "].shape = ",
              indices.shape().ShortDebugString()));
      OP_REQUIRES(
          c, input_num == 0 || SameExtraShape(data0, indices0, data, indices),
          errors::InvalidArgument(
              "Need data[0].shape[", indices0.dims(), ":] = data[", input_num,
              "].shape[", indices.dims(), ":], got data[0].shape = ",
              data0.shape().ShortDebugString(), ", data[", input_num,
              "].shape = ", data.shape().ShortDebugString(),
              ", indices[0].shape = ", indices0.shape().ShortDebugString(),
              ", indices[", input_num, "].shape = ",
              indices.shape().ShortDebugString()));
    }

    // Allocate result tensor of shape
    //   [first_dim_size] + data.shape[indices.dims:]
    TensorShape result_shape;
    result_shape.AddDim(first_dim_size);
    for (int d = indices0.dims(); d < data0.dims(); d++) {
      result_shape.AddDim(data0.dim_size(d));
    }
    Tensor* merged = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &merged));

    // TODO(jeff): Currently we leave uninitialized any portions of
    // merged that aren't covered by an index in indices.  What should we do?
    if (first_dim_size > 0) {
      auto merged_flat = merged->flat_outer_dims<T>();
      const int slice_size = merged_flat.dimension(1);
      for (int input_num = 0; input_num < indices_inputs.size(); input_num++) {
        const Tensor& indices = indices_inputs[input_num];
        auto indices_vec = indices.flat<int32>();
        const Tensor& data = data_inputs[input_num];
        auto data_flat =
            data.shaped<T, 2>({indices_vec.dimension(0), slice_size});

        if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
          T* merged_base = &merged_flat(0, 0);
          const T* data_base = &data_flat(0, 0);
          const size_t slice_bytes = slice_size * sizeof(T);
          for (int i = 0; i < indices_vec.size(); i++) {
            memcpy(merged_base + indices_vec(i) * slice_size,
                   data_base + i * slice_size, slice_bytes);
          }
        } else {
          Eigen::DSizes<Eigen::DenseIndex, 2> sizes(1, slice_size);
          for (int i = 0; i < indices_vec.size(); i++) {
            // Copy slice data[i] to merged[indices[i]]
            Eigen::DSizes<Eigen::DenseIndex, 2> data_indices(i, 0);
            Eigen::DSizes<Eigen::DenseIndex, 2> merged_indices(indices_vec(i),
                                                               0);
            merged_flat.slice(merged_indices, sizes) =
                data_flat.slice(data_indices, sizes);
          }
        }
      }
    }
  }

 private:
  // Check if data0.shape[indices0.dims():] == data1.shape[indices1.dims():]
  static bool SameExtraShape(const Tensor& data0, const Tensor& indices0,
                             const Tensor& data1, const Tensor& indices1) {
    const int extra0 = data0.dims() - indices0.dims();
    const int extra1 = data1.dims() - indices1.dims();
    if (extra0 != extra1) return false;
    for (int i = 0; i < extra0; i++) {
      if (data0.dim_size(indices0.dims() + i) !=
          data1.dim_size(indices1.dims() + i)) {
        return false;
      }
    }
    return true;
  }
};

#define REGISTER_DYNAMIC_STITCH(type)                    \
  REGISTER_KERNEL_BUILDER(Name("DynamicStitch")          \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices"),    \
                          DynamicStitchOp<type>)

TF_CALL_ALL_TYPES(REGISTER_DYNAMIC_STITCH);
#undef REGISTER_DYNAMIC_STITCH

#if GOOGLE_CUDA
#define REGISTER_DYNAMIC_STITCH_GPU(type)                \
  REGISTER_KERNEL_BUILDER(Name("DynamicStitch")          \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices")     \
                              .HostMemory("data")        \
                              .HostMemory("merged"),     \
                          DynamicStitchOp<type>)

TF_CALL_ALL_TYPES(REGISTER_DYNAMIC_STITCH_GPU);
#undef REGISTER_DYNAMIC_STITCH_GPU

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
