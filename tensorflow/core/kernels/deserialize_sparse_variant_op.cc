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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow {

namespace {

class DeserializeSparseOp : public OpKernel {
 public:
  explicit DeserializeSparseOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    OP_REQUIRES(
        context, input.dims() > 0,
        errors::InvalidArgument("Serialized sparse should have non-zero rank ",
                                input.shape().DebugString()));
    OP_REQUIRES(context, input.shape().dim_size(input.dims() - 1) == 3,
                errors::InvalidArgument(
                    "Serialized sparse should have 3 as the last dimension ",
                    input.shape().DebugString()));

    // `input_dims_to_stack` is the number of dimensions that will be added to
    // each of the elements before they are concatenated into the output.
    const int64_t input_dims_to_stack = input.dims() - 1;
    int num_sparse_tensors = 1;
    for (int i = 0; i < input_dims_to_stack; ++i) {
      num_sparse_tensors *= input.shape().dim_size(i);
    }

    if (num_sparse_tensors == 1 && input_dims_to_stack == 0) {
      // Special case with a single sparse tensor, and no dimensions to add
      // to the output indices. We can return the boxed tensors directly (after
      // validating them).
      const Tensor* output_indices;
      const Tensor* output_values;
      const Tensor* output_shape;
      const auto& input_as_vec = input.vec<Variant>();
      int64_t total_non_zeros;
      OP_REQUIRES_OK(context, GetAndValidateSparseTensorShape(
                                  input_as_vec(1), input_as_vec(2), 0,
                                  &output_shape, &total_non_zeros));
      OP_REQUIRES_OK(context, GetAndValidateSparseTensorIndicesAndValues(
                                  input_as_vec(0), input_as_vec(1), 0,
                                  output_shape->NumElements(), &output_indices,
                                  &output_values));
      context->set_output(0, *output_indices);
      context->set_output(1, *output_values);
      context->set_output(2, *output_shape);
      return;
    }

    OP_REQUIRES(
        context, num_sparse_tensors > 0,
        errors::InvalidArgument(
            "Serialized sparse should have at least 1 serialized tensor, "
            "but has a zero dimension ",
            input.shape().DebugString()));

    const auto& input_as_matrix = input.flat_inner_dims<Variant, 2>();

    // Compute the output "dense shape" of and number of non-zero elements in
    // the stacked sparse tensors. Given an input of shape (S_0, ...,
    // S_{input_dims_to_stack-1}, 3), and an element of dense shape (E_0, ...
    // E_n), the output dense shape will be (S_0, ...,
    // S_{input_dims_to_stack-1}, E_0, ..., E_n).
    Tensor* output_shape;
    int64_t total_non_zeros = 0;

    // Allocate and build the initial output shape based on the element shape of
    // the 0th sparse tensor in the input.
    //
    // NOTE(mrry): We define `element_shape` as a `const Tensor*` rather than a
    // `Tensor` to avoid the overhead of allocating and deallocating a `Tensor`
    // on the stack. While the per-`Tensor` cost is small, this op can unbox a
    // large number of tensors (3 per batch element) and these fixed overheads
    // dominate when the number of non-zeros per element is small.
    const Tensor* element_shape;
    OP_REQUIRES_OK(context, GetAndValidateSparseTensorShape(
                                input_as_matrix(0, 1), input_as_matrix(0, 2), 0,
                                &element_shape, &total_non_zeros));
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       2, {input_dims_to_stack + element_shape->NumElements()},
                       &output_shape));
    const auto element_shape_vec = element_shape->vec<int64_t>();
    auto output_shape_vec = output_shape->vec<int64_t>();
    output_shape_vec(0) = num_sparse_tensors;
    for (int64_t j = 0; j < input_dims_to_stack; ++j) {
      output_shape_vec(j) = input.dim_size(j);
    }
    for (int64_t j = 0; j < element_shape->NumElements(); ++j) {
      output_shape_vec(j + input_dims_to_stack) = element_shape_vec(j);
    }

    // Accumulate the number of non-zero elements from the remaining sparse
    // tensors, and validate that they have compatible dense shapes.
    //
    // NOTE(mrry): For compatibility with the implementations of
    // DeserializeManySparse, and many ops that generate SparseTensors to batch
    // that do not have a fixed dense_shape (e.g. `tf.parse_single_example()`),
    // we compute the maximum in each dimension to find the smallest dense_shape
    // that bounds all of the input SparseTensors.
    for (int i = 1; i < num_sparse_tensors; ++i) {
      int64_t num_non_zeros;
      OP_REQUIRES_OK(context, GetAndValidateSparseTensorShape(
                                  input_as_matrix(i, 1), input_as_matrix(i, 2),
                                  i, &element_shape, &num_non_zeros));
      total_non_zeros += num_non_zeros;
      OP_REQUIRES(
          context,
          output_shape->NumElements() - input_dims_to_stack ==
              element_shape->NumElements(),
          errors::InvalidArgument(
              "Inconsistent shape across SparseTensors: rank prior to "
              "SparseTensor[",
              i, "] was: ", output_shape->NumElements() - input_dims_to_stack,
              " but rank of SparseTensor[", i,
              "] is: ", element_shape->NumElements()));
      const auto element_shape_vec = element_shape->vec<int64_t>();
      for (int j = 0; j < element_shape->NumElements(); ++j) {
        output_shape_vec(j + input_dims_to_stack) = std::max(
            output_shape_vec(j + input_dims_to_stack), element_shape_vec(j));
      }
    }

    // Compute the output "indices" matrix and "values" vector.
    Tensor* output_indices;
    Tensor* output_values;

    const int output_rank = output_shape->NumElements();
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, {static_cast<int64_t>(total_non_zeros), output_rank},
                       &output_indices));
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, {static_cast<int64_t>(total_non_zeros)},
                                &output_values));

    // The bulk of the work in this method involves building the output indices
    // in a tight loop. For cache friendliness, we generate the indices in the
    // order that they will be laid out in memory. We use raw pointers instead
    // of Eigen element/slice indexing methods, to access the underlying index
    // buffer to minimize the amount of work in that tight loop.
    int64_t* output_indices_data = output_indices->matrix<int64_t>().data();
    size_t current_row = 0;

    for (int i = 0; i < num_sparse_tensors; ++i) {
      const Tensor* element_indices;
      const Tensor* element_values;
      OP_REQUIRES_OK(context, this->GetAndValidateSparseTensorIndicesAndValues(
                                  input_as_matrix(i, 0), input_as_matrix(i, 1),
                                  i, output_rank - input_dims_to_stack,
                                  &element_indices, &element_values));

      const size_t num_index_rows = element_values->NumElements();

      // An empty sparse tensor in the input will generate no data
      // in the output. We short-circuit the rest of the iteration to avoid
      // triggering assertions in the Eigen when manipulating empty tensors (or
      // slices of tensors).
      if (num_index_rows == 0) continue;

      const size_t start_row = current_row;
      const size_t next_start_row = current_row + num_index_rows;

      // NOTE(mrry): If the element is a scalar SparseTensor,
      // `element_indices` will be an empty tensor, and this pointer will not
      // be valid. However, we will not dereference the pointer in that case,
      // because `input_dims_to_stack == output_rank`.
      const int64_t* element_indices_data =
          element_indices->matrix<int64_t>().data();

      // Build the submatrix of `output_indices` for the i^th sparse tensor
      // in the input.
      //
      // Each row of `output_indices` comprises `input_dims_to_stack` indices
      // based on the position of the i^th sparse tensor in the input tensor,
      // followed by the indices from the corresponding row in
      // `element_indices`.
      if (input_dims_to_stack == 1 && output_rank == 2) {
        // We specialize this case because the compiler can generate
        // more efficient code when the number of indices for each element is
        // known statically. Since the most common use of this op is to
        // serialize batches of SparseTensors, and the most common source of
        // SparseTensors is the `tf.parse_single_example()` op, which generates
        // 1-D SparseTensors, we statically unroll the loop for the rank 2
        // output case.
        for (; current_row < next_start_row; ++current_row) {
          *output_indices_data++ = i;
          *output_indices_data++ = *element_indices_data++;
        }
      } else {
        // `sparse_tensor_index` is the tuple of indices that correspond to
        // mapping the flat element index (`i`) back onto the stacked
        // coordinates implied by the position of the i^th sparse tensor in the
        // input tensor.
        //
        // We build `sparse_tensor_index` in reverse (innermost/minor dimension
        // to outermost/major dimension). The `cumulative_product` represents
        // the size of the inner subtensor for which `sparse_tensor_index` has
        // already been built.
        absl::InlinedVector<int64_t, 4UL> sparse_tensor_index(
            input_dims_to_stack);
        int cumulative_product = 1;
        for (size_t j = 0; j < sparse_tensor_index.size(); ++j) {
          size_t reverse_index = sparse_tensor_index.size() - j - 1;
          sparse_tensor_index[reverse_index] =
              (i / cumulative_product) % input.dim_size(reverse_index);
          cumulative_product *= input.dim_size(reverse_index);
        }
        for (; current_row < next_start_row; ++current_row) {
          for (int64_t sparse_tensor_index_component : sparse_tensor_index) {
            *output_indices_data++ = sparse_tensor_index_component;
          }
          for (size_t k = input_dims_to_stack; k < output_rank; ++k) {
            *output_indices_data++ = *element_indices_data++;
          }
        }
      }

      // Build the subvector of `output_values` for the i^th sparse tensor
      // in the input.
      //
      // NOTE(mrry): There is a potential optimization here where we use a T*
      // to represent the current position in `output_values`, but it would
      // require some rejigging of the template parameters.
      // NOTE(mrry): Another potential optimization: if we know that this
      // operation consumes its input, we could std::move non-primitive elements
      // into the output and avoid a copy.
      Eigen::DSizes<Eigen::DenseIndex, 1> values_start(start_row);
      Eigen::DSizes<Eigen::DenseIndex, 1> values_sizes(num_index_rows);

#define HANDLE_TYPE(T)                                          \
  case DataTypeToEnum<T>::value: {                              \
    output_values->vec<T>().slice(values_start, values_sizes) = \
        element_values->vec<T>();                               \
    break;                                                      \
  }
      switch (dtype_) {
        TF_CALL_ALL_TYPES(HANDLE_TYPE);
        TF_CALL_QUANTIZED_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
        default:
          OP_REQUIRES_OK(
              context, errors::Unimplemented(
                           "DeserializeSparse Unhandled data type: ", dtype_));
      }
    }
  }

 private:
  absl::Status GetAndValidateSparseTensorShape(const Variant& serialized_values,
                                               const Variant& serialized_shape,
                                               int index,
                                               const Tensor** output_shape,
                                               int64_t* output_num_non_zeros) {
    // Deserialize and validate the shape.
    *output_shape = serialized_shape.get<Tensor>();
    if (*output_shape == nullptr) {
      return errors::InvalidArgument(
          "Could not get a tensor from serialized_sparse[", index, ", 2]");
    }
    if ((*output_shape)->dtype() != DT_INT64) {
      return errors::InvalidArgument(
          "Expected serialized_sparse[", index,
          ", 2] to be a vector of DT_INT64 but received dtype ",
          DataTypeString((*output_shape)->dtype()));
    }
    if (!TensorShapeUtils::IsVector((*output_shape)->shape())) {
      return errors::InvalidArgument(
          "Expected serialized_sparse[", index,
          ", 2] to be a shape vector but its shape is ",
          (*output_shape)->shape().DebugString());
    }
    *output_num_non_zeros = serialized_values.get<Tensor>()->NumElements();
    return absl::OkStatus();
  }

  absl::Status GetAndValidateSparseTensorIndicesAndValues(
      const Variant& serialized_indices, const Variant& serialized_values,
      int index, int expected_rank, const Tensor** output_indices,
      const Tensor** output_values) {
    // Deserialize and validate the indices.
    *output_indices = serialized_indices.get<Tensor>();
    if (*output_indices == nullptr) {
      return errors::InvalidArgument(
          "Could not get a tensor from serialized_sparse[", index, ", 0]");
    }
    if ((*output_indices)->dtype() != DT_INT64) {
      return errors::InvalidArgument(
          "Expected serialized_sparse[", index,
          ", 0] to be a matrix of DT_INT64 but received dtype ",
          DataTypeString((*output_indices)->dtype()));
    }
    if (!TensorShapeUtils::IsMatrix((*output_indices)->shape())) {
      return errors::InvalidArgument(
          "Expected serialized_sparse[", index,
          ", 0] to represent an index matrix but received shape ",
          (*output_indices)->shape().DebugString());
    }
    int64_t num_entries = (*output_indices)->dim_size(0);
    int rank = (*output_indices)->dim_size(1);
    if (rank != expected_rank) {
      return errors::InvalidArgument(
          "Expected column counts of SparseTensor[", index,
          "].indices to match size of SparseTensor[", index,
          "].shape but they do not: ", rank, " vs. ", expected_rank);
    }

    // Deserialize and validate the values.
    *output_values = serialized_values.get<Tensor>();
    if (*output_values == nullptr) {
      return errors::InvalidArgument(
          "Could not get a tensor from serialized_sparse[", index, ", 1]");
    }
    if (!TensorShapeUtils::IsVector((*output_values)->shape())) {
      return errors::InvalidArgument(
          "Expected serialized_sparse[", index,
          ", 1] to represent a values vector but received shape ",
          (*output_values)->shape().DebugString());
    }
    if (dtype_ != (*output_values)->dtype()) {
      return errors::InvalidArgument(
          "Requested SparseTensor of type ", DataTypeString(dtype_),
          " but SparseTensor[", index,
          "].values.dtype() == ", DataTypeString((*output_values)->dtype()));
    }
    if (num_entries != (*output_values)->dim_size(0)) {
      return errors::InvalidArgument(
          "Expected row counts of SparseTensor[", index,
          "].indices and SparseTensor[", index,
          "].values to match but they do not: ", num_entries, " vs. ",
          (*output_values)->dim_size(0));
    }

    return absl::OkStatus();
  }

  DataType dtype_;
};

REGISTER_KERNEL_BUILDER(Name("DeserializeSparse")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<Variant>("Tserialized"),
                        DeserializeSparseOp)
REGISTER_KERNEL_BUILDER(Name("DeserializeSparse")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Variant>("Tserialized")
                            .HostMemory("serialized_sparse")
                            .HostMemory("sparse_indices")
                            .HostMemory("sparse_values")
                            .HostMemory("sparse_shape"),
                        DeserializeSparseOp)
REGISTER_KERNEL_BUILDER(Name("DeserializeSparse")
                            .Device(DEVICE_TPU)
                            .TypeConstraint<Variant>("Tserialized")
                            .HostMemory("serialized_sparse")
                            .HostMemory("sparse_indices")
                            .HostMemory("sparse_values")
                            .HostMemory("sparse_shape"),
                        DeserializeSparseOp)

}  // namespace

}  // namespace tensorflow
