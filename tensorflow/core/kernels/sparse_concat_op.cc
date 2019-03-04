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

#define EIGEN_USE_THREADS

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/sorter.h"
#include "tensorflow/core/util/sparse/dim_comparator.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

template <typename T>
class SparseConcatOp : public OpKernel {
 public:
  explicit SparseConcatOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("concat_dim", &concat_dim_attr_));
  }

  void Compute(OpKernelContext* context) override {
    OpInputList inds;
    OP_REQUIRES_OK(context, context->input_list("indices", &inds));
    const int N = inds.size();
    for (int i = 0; i < N; i++) {
      OP_REQUIRES(context, TensorShapeUtils::IsMatrix(inds[i].shape()),
                  errors::InvalidArgument(
                      "Input indices should be a matrix but received shape ",
                      inds[i].shape().DebugString(), " at position ", i));
    }

    OpInputList vals;
    OP_REQUIRES_OK(context, context->input_list("values", &vals));
    OP_REQUIRES(context, vals.size() == N,
                errors::InvalidArgument("Expected ", N, " input values, got ",
                                        vals.size()));
    for (int i = 0; i < N; i++) {
      OP_REQUIRES(context, TensorShapeUtils::IsVector(vals[i].shape()),
                  errors::InvalidArgument(
                      "Input values should be a vector but received shape ",
                      vals[i].shape().DebugString(), " at position ", i));
    }

    OpInputList shapes;
    OP_REQUIRES_OK(context, context->input_list("shapes", &shapes));
    OP_REQUIRES(context, shapes.size() == N,
                errors::InvalidArgument("Expected ", N, " input shapes, got ",
                                        shapes.size()));
    for (int i = 0; i < N; i++) {
      OP_REQUIRES(context, TensorShapeUtils::IsVector(shapes[i].shape()),
                  errors::InvalidArgument(
                      "Input shapes should be a vector but received shape ",
                      shapes[i].shape().DebugString(), " at position ", i));
    }

    const TensorShape input_shape(shapes[0].vec<int64>());
    const int input_rank = input_shape.dims();
    const int concat_dim = (concat_dim_attr_ < 0)
                               ? input_rank + concat_dim_attr_
                               : concat_dim_attr_;
    OP_REQUIRES(context, concat_dim >= 0 && concat_dim < input_rank,
                errors::InvalidArgument("Concat dimension must be in range [",
                                        -input_rank, ", ", input_rank,
                                        "), got ", concat_dim_attr_));
    std::vector<int64> out_shape_vec(input_rank, 0);
    std::vector<int64> val_offsets(N, 0);
    std::vector<int64> concat_dim_offsets(N, 0);
    int64 num_entries = inds[0].dim_size(0);
    for (int i = 0; i < input_rank; ++i) {
      out_shape_vec[i] = shapes[0].vec<int64>()(i);
    }
    for (int i = 1; i < N; ++i) {
      const TensorShape current_shape(shapes[i].vec<int64>());
      OP_REQUIRES(
          context, current_shape.dims() == input_rank,
          errors::InvalidArgument(
              "Ranks of all input tensors must match: expected ", input_rank,
              " but got ", current_shape.dims(), " at position ", i));
      for (int j = 0; j < input_rank; ++j) {
        if (j != concat_dim) {
          OP_REQUIRES(
              context, input_shape.dim_size(j) == current_shape.dim_size(j),
              errors::InvalidArgument(
                  "Input shapes must match: expected ", input_shape.dim_size(j),
                  " for dimension ", j, " but got ", current_shape.dim_size(j),
                  " at position ", i));
        }
      }
      out_shape_vec[concat_dim] += shapes[i].vec<int64>()(concat_dim);
      concat_dim_offsets[i] =
          concat_dim_offsets[i - 1] + shapes[i - 1].vec<int64>()(concat_dim);
      val_offsets[i] = val_offsets[i - 1] + inds[i - 1].dim_size(0);
      num_entries += inds[i].dim_size(0);  // Update number of entries
    }

    // In this implementation, the concat process has 3 steps:
    // 1. Copy all tensor indices and values to an output with the input order,
    // and re-calculate the indices after concatation.
    // 2. Perform a sort on indices to get the final position of each element.
    // 3. Copy the indices and values to the final output tensor according to
    // the sorting result.
    Tensor output_ix(DT_INT64, TensorShape({num_entries, input_rank}));
    Tensor output_vals(DataTypeToEnum<T>::v(), TensorShape({num_entries}));
    Tensor output_shape(DT_INT64, TensorShape({input_rank}));
    const auto output_shape_t = output_shape.vec<int64>();
    std::copy_n(out_shape_vec.begin(), input_rank, output_shape_t.data());

    const DeviceBase::CpuWorkerThreads& worker_threads =
        *(context->device()->tensorflow_cpu_worker_threads());
    thread::ThreadPool* workers = worker_threads.workers;
    int num_threads = worker_threads.num_threads;

    // Concat on dim0, just copy and re-calculate the new indices, no further
    // work is needed.
    if (concat_dim == 0) {
      auto concat0 = [this, &output_ix, &output_vals, &inds, &vals,
                      &val_offsets,
                      &concat_dim_offsets](int64 start, int64 end) {
        ConcatTask(inds, vals, val_offsets, concat_dim_offsets, 0,
                   start, end, &output_ix, &output_vals);
      };
      // NOTE(zycao): Here we have to use 'num_threads - 1' to make sure no more
      // task fractions should be created. The cost is also a coarse estimation.
      Shard(num_threads - 1, workers, N, num_entries / N, concat0);

      context->set_output(0, output_ix);
      context->set_output(1, output_vals);
      context->set_output(2, output_shape);
      return;
    }

    // Concat along a certern lower dim. We do concat-sort-copy steps.
    std::vector<int64> reorder(num_entries);
    Tensor concat_ix(DT_INT64, TensorShape({num_entries, input_rank}));
    Tensor concat_vals(DataTypeToEnum<T>::v(), TensorShape({num_entries}));

    auto concat = [this, &reorder, &concat_ix, &concat_vals, &inds, &vals,
                   concat_dim, &concat_dim_offsets,
                   &val_offsets](int64 start, int64 end) {
      // This loop only parallel fills the reorder vector.
      for (int64 i = start; i < end; ++i) {
        const int64 val_offset = val_offsets[i];
        const int64 input_num = inds[i].dim_size(0);
        for (int64 j = 0; j < input_num; ++j) {
          reorder[val_offset + j] = val_offset + j;
        }
      }
      ConcatTask(inds, vals, val_offsets, concat_dim_offsets, concat_dim,
                 start, end, &concat_ix, &concat_vals);
    };
    // NOTE(zycao): Here we have to use 'num_threads - 1' to make sure no more
    // task fractions should be created. The cost is also a coarse estimation.
    Shard(num_threads - 1, workers, N, num_entries / N, concat);

    // Sort to get order of indices
    switch (input_rank) {
#define CASE_SORT(RANK)                                               \
  case RANK: {                                                        \
    sparse::NaiveDimComparator<RANK> comp(concat_ix.matrix<int64>()); \
    ParallelSorter sorter(num_threads, workers);                      \
    sorter.QSort(reorder, comp);                                      \
    break;                                                            \
  }
      CASE_SORT(1);
      CASE_SORT(2);
      CASE_SORT(3);
      CASE_SORT(4);
      CASE_SORT(5);
#undef CASE_SORT
      default: {
        gtl::InlinedVector<int64, 8> std_order(input_rank);
        std::iota(std_order.begin(), std_order.end(), 0);
        sparse::DimComparator comp(concat_ix.matrix<int64>(), std_order,
                                   out_shape_vec);
        ParallelSorter sorter(num_threads, workers);
        sorter.QSort(reorder, comp);
      }
    }

    // Copy to the output tensor.
    auto rewrite = [this, &reorder, &concat_ix, &concat_vals, &output_ix,
                    &output_vals](int64 start, int64 end) {
      RewriteTask(concat_ix, concat_vals, reorder, start, end, &output_ix,
                  &output_vals);
    };
    // NOTE(zycao): Here we have to use 'num_threads - 1' to make sure no more
    // task fractions should be created. The cost is also a coarse estimation.
    Shard(num_threads - 1, workers, num_entries, input_rank, rewrite);

    context->set_output(0, output_ix);
    context->set_output(1, output_vals);
    context->set_output(2, output_shape);
  }

 private:
  // This inline function is used to copy the indices and values from a range
  // of input sparse tensor to the output concatenated sparse tensor. It would
  // be called in multi-thread context.
  inline void ConcatTask(const OpInputList& inds, const OpInputList& vals,
                         const std::vector<int64> val_offsets,
                         const std::vector<int64> concat_dim_offsets,
                         int concat_dim, int64 start, int64 end,
                         Tensor* output_ix, Tensor* output_vals) {
    for (int64 i = start; i < end; ++i) {
      const int64 val_offset = val_offsets[i];
      const int64 concat_dim_offset = concat_dim_offsets[i];
      const int64 input_num = inds[i].dim_size(0);
      const int64 input_rank = inds[i].dim_size(1);
      auto in_ix_t = inds[i].flat<int64>();
      auto in_vals_t = vals[i].vec<T>();
      auto out_ix_t = output_ix->flat<int64>();
      auto out_vals_t = output_vals->vec<T>();

      for (int64 j = 0; j < input_num; ++j) {
        const int64 out_vals_offset = val_offset + j;
        out_vals_t(out_vals_offset) = in_vals_t(j);

        const int64 in_ix_offset = j * input_rank;
        const int64 out_ix_offset = out_vals_offset * input_rank;
        for (int64 k = 0; k < input_rank; ++k) {
          out_ix_t(out_ix_offset + k) = in_ix_t(in_ix_offset + k);
        }
        out_ix_t(out_ix_offset + concat_dim) += concat_dim_offset;
      }
    }
  }

  // This function does the copy indices and values from sparse tensor before
  // sorting to final output tensors, it would handle a range in output tensor
  // and would be multi-threaded called.
  inline void RewriteTask(const Tensor& in_ix, const Tensor& in_vals,
                          const std::vector<int64>& reorder,
                          int64 start, int64 end,
                          Tensor* output_ix, Tensor* output_vals) {
    const int64 input_rank = in_ix.dim_size(1);
    auto in_ix_t = in_ix.flat<int64>();
    auto in_vals_t = in_vals.vec<T>();
    auto out_ix_t = output_ix->flat<int64>();
    auto out_vals_t = output_vals->vec<T>();

    for (int i = start; i < end; ++i) {
      const int64 pos = reorder[i];
      out_vals_t(i) = in_vals_t(pos);

      const int64 out_ix_offset = i * input_rank;
      const int64 in_ix_offset = pos * input_rank;
      for (int k = 0; k < input_rank; ++k) {
        out_ix_t(out_ix_offset + k) = in_ix_t(in_ix_offset + k);
      }
    }
  }

 private:
  int concat_dim_attr_;
};

#define REGISTER_KERNELS(type)                                           \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SparseConcat").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseConcatOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS
}  // namespace tensorflow
