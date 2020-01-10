// See core/ops/sparse_ops.cc for documentation.
//
// NOTE: the operations in this file only are suitable for execution
// on CPUs.

#define EIGEN_USE_THREADS

#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

// Operator to convert sparse representations to dense.
template <typename T, typename Index>
class SparseToDense : public OpKernel {
 public:
  explicit SparseToDense(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* c) override {
    // sparse_indices
    const Tensor& indices = c->input(0);
    OP_REQUIRES(c, indices.dims() <= 2,
                errors::InvalidArgument(
                    "sparse_indices should be a scalar, vector, or matrix, "
                    "got shape ",
                    indices.shape().ShortDebugString()));
    const int64 num_elems = indices.dims() > 0 ? indices.dim_size(0) : 1;
    const int64 num_dims = indices.dims() > 1 ? indices.dim_size(1) : 1;

    // output_shape
    const Tensor& output_shape = c->input(1);
    OP_REQUIRES(
        c, TensorShapeUtils::IsLegacyVector(output_shape.shape()),
        errors::InvalidArgument("output_shape should be a vector, got shape ",
                                output_shape.shape().ShortDebugString()));
    OP_REQUIRES(c, output_shape.NumElements() == num_dims,
                errors::InvalidArgument(
                    "output_shape has incorrect number of elements: ",
                    output_shape.NumElements(), " should be: ", num_dims));

    // sparse_values
    const Tensor& sparse_values = c->input(2);
    const int64 num_values = sparse_values.NumElements();
    OP_REQUIRES(
        c, sparse_values.dims() == 0 ||
               (sparse_values.dims() == 1 && num_values == num_elems),
        errors::InvalidArgument("sparse_values has incorrect shape ",
                                sparse_values.shape().ShortDebugString(),
                                ", should be [] or [", num_elems, "]"));

    // default_value
    const Tensor& default_value = c->input(3);
    OP_REQUIRES(c, TensorShapeUtils::IsScalar(default_value.shape()),
                errors::InvalidArgument("default_value should be a scalar."));

    auto output_shape_vec = output_shape.flat<Index>();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShapeUtils::MakeShape(
                                                output_shape_vec.data(),
                                                output_shape_vec.size()),
                                         &output));

    TensorShape ix_shape({num_elems, num_dims});
    Tensor indices_shaped(DT_INT64, ix_shape);
    if (indices.dtype() == DT_INT64) {
      CHECK(indices_shaped.CopyFrom(indices, ix_shape));
    } else {
      indices_shaped.matrix<int64>() =
          indices.shaped<Index, 2>(ix_shape.dim_sizes()).template cast<int64>();
    }

    // If we received a scalar, we'll need to create a new
    // tensor with copies of the values as a vec.
    // TODO(ebrevdo): find a way to avoid this temp allocation.
    Tensor sparse_values_b;

    if (TensorShapeUtils::IsScalar(sparse_values.shape())) {
      OP_REQUIRES_OK(
          c, c->allocate_temp(DataTypeToEnum<T>::value,
                              TensorShape({num_elems}), &sparse_values_b));
      sparse_values_b.vec<T>().setConstant(sparse_values.scalar<T>()());
    } else {
      sparse_values_b = sparse_values;
    }

    gtl::InlinedVector<int64, 8> order(output->shape().dims());
    std::iota(order.begin(), order.end(), 0);  // Assume order is correct
    sparse::SparseTensor st(indices_shaped, sparse_values_b, output->shape(),
                            order);

    output->flat<T>().setConstant(default_value.scalar<T>()());
    OP_REQUIRES(c, st.template ToDense<T>(output, false /* initialize */),
                errors::InvalidArgument(
                    "Indices are not valid (out of bounds).  Shape: ",
                    output->shape().DebugString()));
  }
};

#define REGISTER_KERNELS(type, index_type)                             \
  REGISTER_KERNEL_BUILDER(Name("SparseToDense")                        \
                              .Device(DEVICE_CPU)                      \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          SparseToDense<type, index_type>);

#define REGISTER_KERNELS_ALL(type) \
  REGISTER_KERNELS(type, int32);   \
  REGISTER_KERNELS(type, int64);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS_ALL);
REGISTER_KERNELS_ALL(bool);
REGISTER_KERNELS_ALL(string);

#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

}  // namespace tensorflow
