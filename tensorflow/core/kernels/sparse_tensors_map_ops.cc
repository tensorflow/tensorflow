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
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/overflow.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

using sparse::SparseTensor;

class SparseTensorsMap : public ResourceBase {
 public:
  explicit SparseTensorsMap(const std::string& name)
      : name_(name), counter_(0) {}

  std::string DebugString() const override { return "A SparseTensorsMap"; }

  typedef struct {
    Tensor indices;
    Tensor values;
    absl::InlinedVector<int64_t, 8UL> shape;
  } PersistentSparseTensor;

  absl::Status AddSparseTensor(OpKernelContext* ctx, const SparseTensor& sp,
                               int64_t* handle) {
    Tensor ix;
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(sp.indices().dtype(), sp.indices().shape(), &ix));
    ix = sp.indices();

    Tensor values;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(sp.indices().dtype(),
                                          sp.indices().shape(), &values));
    values = sp.values();
    {
      mutex_lock l(mu_);
      int64_t unique_st_handle = counter_++;  // increment is guarded on purpose
      sp_tensors_[unique_st_handle] =
          PersistentSparseTensor{ix, values,
                                 absl::InlinedVector<int64_t, 8UL>(
                                     sp.shape().begin(), sp.shape().end())};
      *handle = unique_st_handle;
    }
    return absl::OkStatus();
  }

  absl::Status RetrieveAndClearSparseTensors(
      OpKernelContext* ctx, const TTypes<int64_t>::ConstVec& handles,
      std::vector<SparseTensor>* sparse_tensors) {
    sparse_tensors->clear();
    sparse_tensors->reserve(handles.size());
    {
      mutex_lock l(mu_);
      for (size_t i = 0; i < handles.size(); ++i) {
        const int64_t handle = handles(i);
        auto sp_iter = sp_tensors_.find(handle);
        if (sp_iter == sp_tensors_.end()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Unable to find SparseTensor: ", handle, " in map: ", name_));
        }
        const Tensor* ix = &sp_iter->second.indices;
        const Tensor* values = &sp_iter->second.values;
        const auto& shape = sp_iter->second.shape;
        SparseTensor tensor;
        TF_RETURN_IF_ERROR(SparseTensor::Create(*ix, *values, shape, &tensor));
        sparse_tensors->push_back(std::move(tensor));
        sp_tensors_.erase(sp_iter);
      }
    }

    return absl::OkStatus();
  }

 protected:
  ~SparseTensorsMap() override {}

 private:
  std::string name_;

  mutex mu_;
  int64_t counter_ TF_GUARDED_BY(mu_);
  std::unordered_map<int64_t, PersistentSparseTensor> sp_tensors_
      TF_GUARDED_BY(mu_);
};

class SparseTensorAccessingOp : public OpKernel {
 public:
  typedef std::function<absl::Status(SparseTensorsMap**)> CreatorCallback;

  explicit SparseTensorAccessingOp(OpKernelConstruction* context)
      : OpKernel(context), sparse_tensors_map_(nullptr) {}

 protected:
  ~SparseTensorAccessingOp() override {
    if (sparse_tensors_map_) sparse_tensors_map_->Unref();
  }

  absl::Status GetMap(OpKernelContext* ctx, bool is_writing,
                      SparseTensorsMap** sparse_tensors_map) {
    mutex_lock l(mu_);

    if (sparse_tensors_map_) {
      *sparse_tensors_map = sparse_tensors_map_;
      return absl::OkStatus();
    }

    TF_RETURN_IF_ERROR(cinfo_.Init(ctx->resource_manager(), def(),
                                   is_writing /* use_node_name_as_default */));

    CreatorCallback sparse_tensors_map_creator = [this](SparseTensorsMap** c) {
      SparseTensorsMap* map = new SparseTensorsMap(cinfo_.name());
      *c = map;
      return absl::OkStatus();
    };

    TF_RETURN_IF_ERROR(
        cinfo_.resource_manager()->LookupOrCreate<SparseTensorsMap>(
            cinfo_.container(), cinfo_.name(), &sparse_tensors_map_,
            sparse_tensors_map_creator));

    *sparse_tensors_map = sparse_tensors_map_;
    return absl::OkStatus();
  }

 private:
  ContainerInfo cinfo_;

  mutex mu_;
  SparseTensorsMap* sparse_tensors_map_ TF_PT_GUARDED_BY(mu_);
};

class AddSparseToTensorsMapOp : public SparseTensorAccessingOp {
 public:
  explicit AddSparseToTensorsMapOp(OpKernelConstruction* context)
      : SparseTensorAccessingOp(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* input_indices;
    const Tensor* input_values;
    const Tensor* input_shape;
    SparseTensorsMap* map;

    OP_REQUIRES_OK(context, context->input("sparse_indices", &input_indices));
    OP_REQUIRES_OK(context, context->input("sparse_values", &input_values));
    OP_REQUIRES_OK(context, context->input("sparse_shape", &input_shape));
    OP_REQUIRES_OK(context, GetMap(context, true /* is_writing */, &map));

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices->shape()),
                absl::InvalidArgumentError(absl::StrCat(
                    "Input indices should be a matrix but received shape ",
                    input_indices->shape().DebugString())));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_values->shape()),
                absl::InvalidArgumentError(absl::StrCat(
                    "Input values should be a vector but received shape ",
                    input_values->shape().DebugString())));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape->shape()),
                absl::InvalidArgumentError(absl::StrCat(
                    "Input shape should be a vector but received shape ",
                    input_shape->shape().DebugString())));

    TensorShape input_shape_object;
    OP_REQUIRES_OK(
        context, TensorShapeUtils::MakeShape(input_shape->vec<int64_t>().data(),
                                             input_shape->NumElements(),
                                             &input_shape_object));
    SparseTensor st;
    OP_REQUIRES_OK(context, SparseTensor::Create(*input_indices, *input_values,
                                                 input_shape_object, &st));
    Tensor* sparse_handle = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &sparse_handle));
    int64_t handle;
    OP_REQUIRES_OK(context, map->AddSparseTensor(context, st, &handle));
    sparse_handle->scalar<int64_t>()() = handle;
  }
};

REGISTER_KERNEL_BUILDER(Name("AddSparseToTensorsMap").Device(DEVICE_CPU),
                        AddSparseToTensorsMapOp);

template <typename T>
class AddManySparseToTensorsMapOp : public SparseTensorAccessingOp {
 public:
  explicit AddManySparseToTensorsMapOp(OpKernelConstruction* context)
      : SparseTensorAccessingOp(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* input_indices;
    const Tensor* input_values;
    const Tensor* input_shape;
    SparseTensorsMap* map;

    OP_REQUIRES_OK(context, context->input("sparse_indices", &input_indices));
    OP_REQUIRES_OK(context, context->input("sparse_values", &input_values));
    OP_REQUIRES_OK(context, context->input("sparse_shape", &input_shape));
    OP_REQUIRES_OK(context, GetMap(context, true /* is_writing */, &map));

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices->shape()),
                absl::InvalidArgumentError(absl::StrCat(
                    "Input indices should be a matrix but received shape ",
                    input_indices->shape().DebugString())));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_values->shape()),
                absl::InvalidArgumentError(absl::StrCat(
                    "Input values should be a vector but received shape ",
                    input_values->shape().DebugString())));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape->shape()),
                absl::InvalidArgumentError(absl::StrCat(
                    "Input shape should be a vector but received shape ",
                    input_shape->shape().DebugString())));
    OP_REQUIRES(
        context,
        input_values->shape().dim_size(0) == input_indices->shape().dim_size(0),
        absl::InvalidArgumentError(absl::StrCat(
            "Number of values must match first dimension of indices. ", "Got ",
            input_values->shape().dim_size(0),
            " values, indices shape: ", input_indices->shape().DebugString())));
    OP_REQUIRES(
        context,
        input_shape->shape().dim_size(0) == input_indices->shape().dim_size(1),
        absl::InvalidArgumentError(absl::StrCat(
            "Number of dimensions must match second dimension of indices. ",
            "Got ", input_shape->shape().dim_size(0),
            " dimensions, indices shape: ",
            input_indices->shape().DebugString())));

    int rank = input_shape->NumElements();

    OP_REQUIRES(
        context, rank > 1,
        absl::InvalidArgumentError(absl::StrCat(
            "Rank of input SparseTensor should be > 1, but saw rank: ", rank)));

    auto input_shape_vec = input_shape->vec<int64_t>();

    TensorShape tensor_input_shape;
    OP_REQUIRES_OK(context, TensorShape::BuildTensorShape(input_shape_vec,
                                                          &tensor_input_shape));
    absl::InlinedVector<int64_t, 8UL> std_order(rank);
    std::iota(std_order.begin(), std_order.end(), 0);
    SparseTensor input_st;
    OP_REQUIRES_OK(context, SparseTensor::Create(*input_indices, *input_values,
                                                 tensor_input_shape, std_order,
                                                 &input_st));

    const int64_t N = input_shape_vec(0);

    Tensor* sparse_handles = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({N}), &sparse_handles));
    auto sparse_handles_t = sparse_handles->vec<int64_t>();

    OP_REQUIRES_OK(context, input_st.IndicesValid());

    // We can generate the output shape proto string now, for all
    // minibatch entries.
    TensorShape output_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                input_shape_vec.data() + 1,
                                input_shape->NumElements() - 1, &output_shape));

    // Get groups by minibatch dimension
    std::unordered_set<int64_t> visited;
    sparse::GroupIterable minibatch = input_st.group({0});
    for (const auto& subset : minibatch) {
      const int64_t b = subset.group()[0];
      visited.insert(b);
      OP_REQUIRES(
          context, b > -1 && b < N,
          absl::InvalidArgumentError(absl::StrCat(
              "Received unexpected column 0 value in input SparseTensor: ", b,
              " < 0 or >= N (= ", N, ")")));

      const auto indices = subset.indices();
      const auto values = subset.values<T>();
      const int64_t num_entries = values.size();

      Tensor output_indices;
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DT_INT64, TensorShape({num_entries, rank - 1}),
                                  &output_indices));
      Tensor output_values;
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  TensorShape({num_entries}), &output_values));

      auto output_indices_t = output_indices.matrix<int64_t>();
      auto output_values_t = output_values.vec<T>();

      for (int i = 0; i < num_entries; ++i) {
        for (int d = 1; d < rank; ++d) {
          output_indices_t(i, d - 1) = indices(i, d);
        }
        output_values_t(i) = values(i);
      }

      SparseTensor st_i;
      OP_REQUIRES_OK(context,
                     SparseTensor::Create(output_indices, output_values,
                                          output_shape, &st_i));
      int64_t handle;
      OP_REQUIRES_OK(context, map->AddSparseTensor(context, st_i, &handle));
      sparse_handles_t(b) = handle;
    }

    // Fill in any gaps; we must provide an empty ST for batch entries
    // the grouper didn't find.
    if (visited.size() < N) {
      Tensor empty_indices;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DT_INT64, TensorShape({0, rank - 1}),
                                            &empty_indices));
      Tensor empty_values;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::value,
                                            TensorShape({0}), &empty_values));
      SparseTensor empty_st;
      OP_REQUIRES_OK(context, SparseTensor::Create(empty_indices, empty_values,
                                                   output_shape, &empty_st));

      for (int64_t b = 0; b < N; ++b) {
        // We skipped this batch entry.
        if (visited.find(b) == visited.end()) {
          int64_t handle;
          OP_REQUIRES_OK(context,
                         map->AddSparseTensor(context, empty_st, &handle));
          sparse_handles_t(b) = handle;
        }
      }
    }
  }
};

#define REGISTER_KERNELS(type)                              \
  REGISTER_KERNEL_BUILDER(Name("AddManySparseToTensorsMap") \
                              .Device(DEVICE_CPU)           \
                              .TypeConstraint<type>("T"),   \
                          AddManySparseToTensorsMapOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

template <typename T>
class TakeManySparseFromTensorsMapOp : public SparseTensorAccessingOp {
 public:
  explicit TakeManySparseFromTensorsMapOp(OpKernelConstruction* context)
      : SparseTensorAccessingOp(context) {}

  void Compute(OpKernelContext* context) override {
    SparseTensorsMap* map = nullptr;
    OP_REQUIRES_OK(context, GetMap(context, false /* is_writing */, &map));
    const Tensor& sparse_handles = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(sparse_handles.shape()),
                absl::InvalidArgumentError(absl::StrCat(
                    "sparse_handles should be a vector but received shape ",
                    sparse_handles.shape().DebugString())));
    const int64_t N = sparse_handles.dim_size(0);
    OP_REQUIRES(context, N > 0,
                absl::InvalidArgumentError(
                    "Must have at least 1 serialized SparseTensor, "
                    "but input matrix has 0 rows"));

    std::vector<SparseTensor> sparse_tensors;
    OP_REQUIRES_OK(context, map->RetrieveAndClearSparseTensors(
                                context, sparse_handles.vec<int64_t>(),
                                &sparse_tensors));

    const int rank = sparse_tensors[0].dims();
    std::vector<int64_t> output_shape(rank + 1, 0);
    output_shape[0] = N;
    int64_t total_entries = 0;
    for (int64_t i = 0; i < N; ++i) {
      const SparseTensor& st = sparse_tensors[i];
      const Tensor& input_indices = st.indices();
      const Tensor& input_values = st.values();
      OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices.shape()),
                  absl::InvalidArgumentError(absl::StrCat(
                      "Expected sparse_handles[", i,
                      "] to represent an index matrix but received shape ",
                      input_indices.shape().DebugString())));
      OP_REQUIRES(context, TensorShapeUtils::IsVector(input_values.shape()),
                  absl::InvalidArgumentError(absl::StrCat(
                      "Expected sparse_handles[", i,
                      "] to represent a values vector but received shape ",
                      input_values.shape().DebugString())));
      OP_REQUIRES(
          context, DataTypeToEnum<T>::value == input_values.dtype(),
          errors::InvalidArgument(
              "Requested SparseTensor of type ",
              DataTypeString(DataTypeToEnum<T>::value), " but SparseTensor[", i,
              "].values.dtype() == ", DataTypeString(input_values.dtype())));
      const int64_t num_entries = input_indices.dim_size(0);
      OP_REQUIRES(context, num_entries == input_values.dim_size(0),
                  absl::InvalidArgumentError(absl::StrCat(
                      "Expected row counts of SparseTensor[", i,
                      "].indices and SparseTensor[", i,
                      "].values to match but they do not: ", num_entries,
                      " vs. ", input_values.dim_size(0))));
      const int tensor_rank = input_indices.dim_size(1);
      OP_REQUIRES(context, tensor_rank == st.dims(),
                  absl::InvalidArgumentError(absl::StrCat(
                      "Expected column counts of SparseTensor[", i,
                      "].indices to match size of SparseTensor[", i,
                      "].shape but they do not: ", tensor_rank, " vs. ",
                      st.dims())));
      OP_REQUIRES(context, rank == tensor_rank,
                  absl::InvalidArgumentError(absl::StrCat(
                      "Inconsistent rank across SparseTensors: rank prior to "
                      "SparseTensor[",
                      i, "] was: ", rank + 1, " but rank of SparseTensor[", i,
                      "] is: ", tensor_rank + 1)));
      total_entries = AddWithoutOverflow(total_entries, num_entries);
      OP_REQUIRES(context, total_entries >= 0,
                  absl::ResourceExhaustedError("Too many sparse entries"));
      for (int d = 0; d < rank; ++d) {
        output_shape[d + 1] = std::max(output_shape[d + 1], st.shape()[d]);
      }
    }

    // Allocate the concatenated outputs directly; SparseTensor::Concat uses
    // unchecked Tensor constructors and bypasses the context allocator.
    Tensor* indices = nullptr;
    Tensor* values = nullptr;
    Tensor* shape = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({total_entries, rank + 1}),
                                &indices));
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({total_entries}), &values));
    OP_REQUIRES_OK(context, context->allocate_output(
                                2, TensorShape({rank + 1}), &shape));
    auto indices_t = indices->matrix<int64_t>();
    auto values_t = values->vec<T>();
    std::copy(output_shape.begin(), output_shape.end(),
              shape->vec<int64_t>().data());
    int64_t offset = 0;
    for (int64_t i = 0; i < N; ++i) {
      const SparseTensor& st = sparse_tensors[i];
      const int64_t num_entries = st.num_entries();
      if (num_entries > 0) {
        const auto input_indices = st.indices().matrix<int64_t>();
        for (int64_t row = 0; row < num_entries; ++row) {
          indices_t(offset + row, 0) = i;
          for (int d = 0; d < rank; ++d) {
            indices_t(offset + row, d + 1) = input_indices(row, d);
          }
        }
        std::copy_n(st.values().vec<T>().data(), num_entries,
                    values_t.data() + offset);
      }
      offset += num_entries;
    }
  }
};

#define REGISTER_KERNELS(type)                                 \
  REGISTER_KERNEL_BUILDER(Name("TakeManySparseFromTensorsMap") \
                              .Device(DEVICE_CPU)              \
                              .TypeConstraint<type>("dtype"),  \
                          TakeManySparseFromTensorsMapOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  // namespace tensorflow
