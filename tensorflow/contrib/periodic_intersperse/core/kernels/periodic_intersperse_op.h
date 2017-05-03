#ifndef TENSORFLOW_KERNELS_PERIODICINTERSPERSE_OP_H_
#define TENSORFLOW_KERNELS_PERIODICINTERSPERSE_OP_H_

#include <cmath>
#include <type_traits>
#include <vector>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;
using std::vector;
using std::decay;


template<class IndexVec_t, class Index_t>
Index_t compute_input_index(IndexVec_t &Y, const Index_t &i,
                            const IndexVec_t &X, const int &q,
                            const vector<int64> &g, const vector<int64> &G,
                            Index_t &result, vector<Index_t> &output_indices,
                            const typename decay<decltype(Y.size())>::type &rank)
{
  result = 0;
  output_indices.clear();

  // un-rasterize the output index
  {
    auto last_reduced_i = i;
    for (auto r = rank - 1;; --r) {
      output_indices[r] = last_reduced_i % Y[r];
      last_reduced_i = (last_reduced_i - output_indices[r]) / Y[r];
      if (r == 0) break;
    }
  }

  // rasterize the input index
  {
    Index_t last_index_factor = 1;
    for (auto r = rank - 1;; --r) {
      Index_t index = 0;
      if (r != q)
        index = output_indices[r]/g[r];
      else {
        for (int qi = 0; qi < rank; ++qi) {
          // if (qi == r) continue;
          if (qi == q) continue;
          index += G[qi]*(output_indices[qi] % g[qi]);
        }
        index *= Y[q];
        // index *= Y[r];
        index += output_indices[r];
      }
      result += last_index_factor*index;
      last_index_factor *= X[r];
      if (r == 0) break;
    }
  }

  return result;
}


template<class T, class Vec_t>
void main_logic (OpKernelContext *context,
                 const Vec_t &desired_shape,
                 const Tensor &input_tensor)
{
  // NOTE input is a strided array (last index is fastest, C-ordered)
  auto input = input_tensor.flat<T>();
  const int rank = input_tensor.dims();
  const auto original_size = input.size();
  // NOTE original and target dimensions
  vector<int64> X(rank), Y(rank);
  int64 total_size = 1, new_sliced_size = 1;
  // NOTE factor by which X increases/decreases w.r.t. Y
  vector<float> f(rank);
  // NOTE helper arrays related to f
  vector<int64> g(rank), G(rank);
  // NOTE index of adjustable dimension
  int q;
  TensorShape output_shape;

  // NOTE requires that the rank of the input tensor and length of the desired shape are equal
  OP_REQUIRES(context, rank == desired_shape.size(),
              errors::InvalidArgument("periodic_intersperse expects the rank of the input tensor, ",
              rank, ", to be the same as the length of the desired shape, ", desired_shape.size(), "."));

  // NOTE from here on, the logic parallels notebook
  {
    bool found = false;
    for (int i = 0; i < rank; ++i) {
      if (desired_shape(i) < 1) {
        // NOTE only one index can be adjustable
        OP_REQUIRES(context, !found,
                    errors::InvalidArgument("periodic_intersperse expects only "
                    "one index to be marked as adjustable."));
        q = i;
        found = true;
      }
      else {
        Y[i] = desired_shape(i);
        new_sliced_size *= Y[i];
      }
    }
    // NOTE at least one index needs to be adjustable
    OP_REQUIRES(context, found,
                errors::InvalidArgument("periodic_intersperse expects at least "
                "one index to be marked as adjustable."));

    int count = 0;
    for (const auto dim_info : input_tensor.shape()) {
      X[count] = dim_info.size;
      total_size *= X[count];
      ++count;
    }

    Y[q] = int64(std::floor(float(total_size)/float(new_sliced_size)));

    count = 0;
    for (const auto dim_info : input_tensor.shape()) {
        f[count] = float(Y[count])/float(X[count]);
        g[count] = int64(std::ceil(f[count]));
        if (count == 0) G[count] = 1;
        else G[count] = G[count - 1]*g[count - 1];
      ++count;
    }
  }

  // NOTE ensure that the new dimension is greater than zero
  OP_REQUIRES(context, Y[q] > 0,
              errors::InvalidArgument("periodic_intersperse found that the "
              "adjustable dimension, ", q, ", isn't greater than zero, ", Y[q],
              "."));
  for (int i = 0; i < rank; ++i) {
    output_shape.AddDim(Y[i]);
  }
  const auto new_size = new_sliced_size*Y[q];

  // Create an output tensor and attach it to the current context
  Tensor* output_tensor = NULL;
  OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
    &output_tensor));
  auto output = output_tensor->flat<T>();

  {
    // NOTE memory is allocated for these variables outside the inner loop for
    //      efficiency (yes, I know I could create a separate class scope for
    //      this purpose instead)
    typename decay<decltype(new_size)>::type result = 0;
    vector<decltype(result)> output_indices(Y.size());
    const auto rank = Y.size();

    // Fill output tensor with shuffled input tensor values
    for (typename decay<decltype(new_size)>::type i = 0; i < new_size; ++i) {
      output(i) = input(compute_input_index(Y, i, X, q, g, G,
                                            result, output_indices, rank));
    }
  }
}


template<class T>
void create_output_Tensor (OpKernelContext *context,
                           const Tensor &input_tensor,
                           const DataType &input_tensor_type,
                           const Tensor &desired_shape_tensor)
{
  auto desired_shape = desired_shape_tensor.flat<T>();

  // obligatory type switch
  if (input_tensor_type == DataTypeToEnum<float>::value)
    main_logic<float>(context, desired_shape, input_tensor);
  if (input_tensor_type == DataTypeToEnum<double>::value)
    main_logic<double>(context, desired_shape, input_tensor);
  if (input_tensor_type == DataTypeToEnum<int32>::value)
    main_logic<int32>(context, desired_shape, input_tensor);
  if (input_tensor_type == DataTypeToEnum<int64>::value)
    main_logic<int64>(context, desired_shape, input_tensor);
}


class PeriodicIntersperseOp : public OpKernel {
public:
  explicit PeriodicIntersperseOp (OpKernelConstruction* context) : OpKernel(context) {}

  void Compute (OpKernelContext* context) override {

    // Grab the input tensor and desired shape
    const Tensor &input_tensor = context->input(0);
    const DataType input_tensor_type = context->input_dtype(0);
    const Tensor &desired_shape_tensor = context->input(1);
    const DataType desired_shape_tensor_type = context->input_dtype(1);

    // NOTE requires that the desired shape is a vector
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(desired_shape_tensor.shape()),
        errors::InvalidArgument(
          "periodic_intersperse expects a 1D vector for the desired shape."));

    // obligatory type switch
    if (desired_shape_tensor_type == DataTypeToEnum<int32>::value)
      create_output_Tensor<int32>(context,
                                  input_tensor,
                                  input_tensor_type,
                                  desired_shape_tensor);
    if (desired_shape_tensor_type == DataTypeToEnum<int64>::value)
      create_output_Tensor<int64>(context,
                                  input_tensor,
                                  input_tensor_type,
                                  desired_shape_tensor);
  }
};

#endif  // TENSORFLOW_KERNELS_PERIODICINTERSPERSE_OP_H_
