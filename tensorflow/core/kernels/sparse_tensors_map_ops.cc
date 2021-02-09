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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

using sparse::SparseTensor;

class SparseTensorsMap : public ResourceBase {
 public:
  explicit SparseTensorsMap(const string& name) : name_(name), counter_(0) {}

  string DebugString() const override { return "A SparseTensorsMap"; }

  typedef struct {
    PersistentTensor indices;
    PersistentTensor values;
    gtl::InlinedVector<int64, 8> shape;
  } PersistentSparseTensor;

  Status AddSparseTensor(OpKernelContext* ctx, const SparseTensor& sp,
                         int64* handle) {
    PersistentTensor persistent_ix;
    Tensor* ix;
    TF_RETURN_IF_ERROR(ctx->allocate_persistent(
        sp.indices().dtype(), sp.indices().shape(), &persistent_ix, &ix));
    *ix = sp.indices();

    PersistentTensor persistent_values;
    Tensor* values;
    TF_RETURN_IF_ERROR(ctx->allocate_persistent(sp.indices().dtype(),
                                                sp.indices().shape(),
                                                &persistent_values, &values));
    *values = sp.values();
    {
      mutex_lock l(mu_);
      int64 unique_st_handle = counter_++;  // increment is guarded on purpose
      sp_tensors_[unique_st_handle] = PersistentSparseTensor{
          persistent_ix, persistent_values,
          gtl::InlinedVector<int64, 8>(sp.shape().begin(), sp.shape().end())};
      *handle = unique_st_handle;
    }
    return Status::OK();
  }

  Status RetrieveAndClearSparseTensors(
      OpKernelContext* ctx, const TTypes<int64>::ConstVec& handles,
      std::vector<SparseTensor>* sparse_tensors) {
    sparse_tensors->clear();
    sparse_tensors->reserve(handles.size());
    {
      mutex_lock l(mu_);
      for (size_t i = 0; i < handles.size(); ++i) {
        const int64 handle = handles(i);
        auto sp_iter = sp_tensors_.find(handle);
        if (sp_iter == sp_tensors_.end()) {
          return errors::InvalidArgument(
              "Unable to find SparseTensor: ", handle, " in map: ", name_);
        }
        const Tensor* ix = sp_iter->second.indices.AccessTensor(ctx);
        const Tensor* values = sp_iter->second.values.AccessTensor(ctx);
        const auto& shape = sp_iter->second.shape;
        SparseTensor tensor;
        TF_RETURN_IF_ERROR(SparseTensor::Create(*ix, *values, shape, &tensor));
        sparse_tensors->push_back(std::move(tensor));
        sp_tensors_.erase(sp_iter);
      }
    }

    return Status::OK();
  }

 protected:
  ~SparseTensorsMap() override {}

 private:
  string name_;

  mutex mu_;
  int64 counter_ TF_GUARDED_BY(mu_);
  std::unordered_map<int64, PersistentSparseTensor> sp_tensors_
      TF_GUARDED_BY(mu_);
};

class SparseTensorAccessingOp : public OpKernel {
 public:
  typedef std::function<Status(SparseTensorsMap**)> CreatorCallback;

  explicit SparseTensorAccessingOp(OpKernelConstruction* context)
      : OpKernel(context), sparse_tensors_map_(nullptr) {}

 protected:
  ~SparseTensorAccessingOp() override {
    if (sparse_tensors_map_) sparse_tensors_map_->Unref();
  }

  Status GetMap(OpKernelContext* ctx, bool is_writing,
                SparseTensorsMap** sparse_tensors_map) {
    mutex_lock l(mu_);

    if (sparse_tensors_map_) {
      *sparse_tensors_map = sparse_tensors_map_;
      return Status::OK();
    }

    TF_RETURN_IF_ERROR(cinfo_.Init(ctx->resource_manager(), def(),
                                   is_writing /* use_node_name_as_default */));

    CreatorCallback sparse_tensors_map_creator = [this](SparseTensorsMap** c) {
      SparseTensorsMap* map = new SparseTensorsMap(cinfo_.name());
      *c = map;
      return Status::OK();
    };

    TF_RETURN_IF_ERROR(
        cinfo_.resource_manager()->LookupOrCreate<SparseTensorsMap>(
            cinfo_.container(), cinfo_.name(), &sparse_tensors_map_,
            sparse_tensors_map_creator));

    *sparse_tensors_map = sparse_tensors_map_;
    return Status::OK();
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
                errors::InvalidArgument(
                    "Input indices should be a matrix but received shape ",
                    input_indices->shape().DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_values->shape()),
                errors::InvalidArgument(
                    "Input values should be a vector but received shape ",
                    input_values->shape().DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape->shape()),
                errors::InvalidArgument(
                    "Input shape should be a vector but received shape ",
                    input_shape->shape().DebugString()));

    TensorShape input_shape_object;
    OP_REQUIRES_OK(context,
                   TensorShapeUtils::MakeShape(input_shape->vec<int64>().data(),
                                               input_shape->NumElements(),
                                               &input_shape_object));
    SparseTensor st;
    OP_REQUIRES_OK(context, SparseTensor::Create(*input_indices, *input_values,
                                                 input_shape_object, &st));
    int64 handle;
    OP_REQUIRES_OK(context, map->AddSparseTensor(context, st, &handle));

    Tensor sparse_handle(DT_INT64, TensorShape({}));
    auto sparse_handle_t = sparse_handle.scalar<int64>();

    sparse_handle_t() = handle;

    context->set_output(0, sparse_handle);
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
                errors::InvalidArgument(
                    "Input indices should be a matrix but received shape ",
                    input_indices->shape().DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_values->shape()),
                errors::InvalidArgument(
                    "Input values should be a vector but received shape ",
                    input_values->shape().DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape->shape()),
                errors::InvalidArgument(
                    "Input shape should be a vector but received shape ",
                    input_shape->shape().DebugString()));

    int rank = input_shape->NumElements();

    OP_REQUIRES(
        context, rank > 1,
        errors::InvalidArgument(
            "Rank of input SparseTensor should be > 1, but saw rank: ", rank));

    TensorShape tensor_input_shape(input_shape->vec<int64>());
    gtl::InlinedVector<int64, 8> std_order(rank);
    std::iota(std_order.begin(), std_order.end(), 0);
    SparseTensor input_st;
    OP_REQUIRES_OK(context, SparseTensor::Create(*input_indices, *input_values,
                                                 tensor_input_shape, std_order,
                                                 &input_st));

    auto input_shape_t = input_shape->vec<int64>();
    const int64 N = input_shape_t(0);

    Tensor sparse_handles(DT_INT64, TensorShape({N}));
    auto sparse_handles_t = sparse_handles.vec<int64>();

    OP_REQUIRES_OK(context, input_st.IndicesValid());

    // We can generate the output shape proto string now, for all
    // minibatch entries.
    TensorShape output_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                input_shape_t.data() + 1,
                                input_shape->NumElements() - 1, &output_shape));

    // Get groups by minibatch dimension
    std::unordered_set<int64> visited;
    sparse::GroupIterable minibatch = input_st.group({0});
    for (const auto& subset : minibatch) {
      const int64 b = subset.group()[0];
      visited.insert(b);
      OP_REQUIRES(
          context, b > -1 && b < N,
          errors::InvalidArgument(
              "Received unexpected column 0 value in input SparseTensor: ", b,
              " < 0 or >= N (= ", N, ")"));

      const auto indices = subset.indices();
      const auto values = subset.values<T>();
      const int64 num_entries = values.size();

      Tensor output_indices = Tensor(DT_INT64, {num_entries, rank - 1});
      Tensor output_values = Tensor(DataTypeToEnum<T>::value, {num_entries});

      auto output_indices_t = output_indices.matrix<int64>();
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
      int64 handle;
      OP_REQUIRES_OK(context, map->AddSparseTensor(context, st_i, &handle));
      sparse_handles_t(b) = handle;
    }

    // Fill in any gaps; we must provide an empty ST for batch entries
    // the grouper didn't find.
    if (visited.size() < N) {
      Tensor empty_indices(DT_INT64, {0, rank - 1});
      Tensor empty_values(DataTypeToEnum<T>::value, {0});
      SparseTensor empty_st;
      OP_REQUIRES_OK(context, SparseTensor::Create(empty_indices, empty_values,
                                                   output_shape, &empty_st));

      for (int64 b = 0; b < N; ++b) {
        // We skipped this batch entry.
        if (visited.find(b) == visited.end()) {
          int64 handle;
          OP_REQUIRES_OK(context,
                         map->AddSparseTensor(context, empty_st, &handle));
          sparse_handles_t(b) = handle;
        }
      }
    }

    context->set_output(0, sparse_handles);
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
                errors::InvalidArgument(
                    "sparse_handles should be a vector but received shape ",
                    sparse_handles.shape().DebugString()));

    int64 N = sparse_handles.shape().dim_size(0);

    OP_REQUIRES(
        context, N > 0,
        errors::InvalidArgument("Must have at least 1 serialized SparseTensor, "
                                "but input matrix has 0 rows"));

    std::vector<Tensor> indices_to_concat;
    std::vector<Tensor> values_to_concat;
    std::vector<TensorShape> shapes_to_concat;

    const auto& sparse_handles_t = sparse_handles.vec<int64>();

    std::vector<SparseTensor> sparse_tensors;

    OP_REQUIRES_OK(context, map->RetrieveAndClearSparseTensors(
                                context, sparse_handles_t, &sparse_tensors));

    for (int64 i = 0; i < N; ++i) {
      const SparseTensor& st = sparse_tensors[i];
      const Tensor& output_indices = st.indices();
      const Tensor& output_values = st.values();
      const auto output_shape = st.shape();

      OP_REQUIRES(context, TensorShapeUtils::IsMatrix(output_indices.shape()),
                  errors::InvalidArgument(
                      "Expected sparse_handles[", i,
                      "] to represent an index matrix but received shape ",
                      output_indices.shape().DebugString()));
      OP_REQUIRES(context, TensorShapeUtils::IsVector(output_values.shape()),
                  errors::InvalidArgument(
                      "Expected sparse_handles[", i,
                      "] to represent a values vector but received shape ",
                      output_values.shape().DebugString()));
      OP_REQUIRES(
          context, DataTypeToEnum<T>::value == output_values.dtype(),
          errors::InvalidArgument(
              "Requested SparseTensor of type ",
              DataTypeString(DataTypeToEnum<T>::value), " but SparseTensor[", i,
              "].values.dtype() == ", DataTypeString(output_values.dtype())));

      int64 num_entries = output_indices.dim_size(0);
      OP_REQUIRES(context, num_entries == output_values.dim_size(0),
                  errors::InvalidArgument(
                      "Expected row counts of SparseTensor[", i,
                      "].indices and SparseTensor[", i,
                      "].values to match but they do not: ", num_entries,
                      " vs. ", output_values.dim_size(0)));
      int rank = output_indices.dim_size(1);
      OP_REQUIRES(
          context, rank == output_shape.size(),
          errors::InvalidArgument("Expected column counts of SparseTensor[", i,
                                  "].indices to match size of SparseTensor[", i,
                                  "].shape "
                                  "but they do not: ",
                                  rank, " vs. ", output_shape.size()));

      // Now we expand each SparseTensors' indices and shape by
      // prefixing a dimension
      Tensor expanded_indices(
          DT_INT64, TensorShape({num_entries, 1 + output_indices.dim_size(1)}));
      Tensor expanded_shape(DT_INT64, TensorShape({1 + rank}));
      const auto& output_indices_t = output_indices.matrix<int64>();
      auto expanded_indices_t = expanded_indices.matrix<int64>();
      auto expanded_shape_t = expanded_shape.vec<int64>();
      expanded_indices_t.chip<1>(0).setZero();
      Eigen::DSizes<Eigen::DenseIndex, 2> indices_start(0, 1);
      Eigen::DSizes<Eigen::DenseIndex, 2> indices_sizes(num_entries, rank);
      expanded_indices_t.slice(indices_start, indices_sizes) = output_indices_t;
      expanded_shape_t(0) = 1;
      // TODO: copy shape from TensorShape to &expanded_shape_t(1)
      // std::copy_n(&output_shape_t(0), rank, &expanded_shape_t(1));
      for (int i = 0; i < rank; ++i) {
        expanded_shape_t(i + 1) = output_shape[i];
      }
      TensorShape expanded_tensor_shape(expanded_shape_t);

      indices_to_concat.push_back(std::move(expanded_indices));
      values_to_concat.push_back(output_values);
      shapes_to_concat.push_back(std::move(expanded_tensor_shape));
    }

    int rank = -1;
    for (int i = 0; i < N; ++i) {
      if (rank < 0) rank = shapes_to_concat[i].dims();
      OP_REQUIRES(context, rank == shapes_to_concat[i].dims(),
                  errors::InvalidArgument(
                      "Inconsistent rank across SparseTensors: rank prior to "
                      "SparseTensor[",
                      i, "] was: ", rank, " but rank of SparseTensor[", i,
                      "] is: ", shapes_to_concat[i].dims()));
    }

    // SparseTensor::Concat requires consistent shape for all but the
    // primary order dimension (dimension 0 in this case).  So we get
    // the maximum value across all the input SparseTensors for each
    // dimension and use that.
    TensorShape preconcat_shape(shapes_to_concat[0]);
    for (int i = 0; i < N; ++i) {
      for (int d = 0; d < rank; ++d) {
        preconcat_shape.set_dim(d, std::max(preconcat_shape.dim_size(d),
                                            shapes_to_concat[i].dim_size(d)));
      }
    }

    // Dimension 0 is the primary dimension.
    gtl::InlinedVector<int64, 8> std_order(rank);
    std::iota(std_order.begin(), std_order.end(), 0);

    std::vector<SparseTensor> tensors_to_concat;
    tensors_to_concat.reserve(N);
    for (int i = 0; i < N; ++i) {
      SparseTensor tensor;
      OP_REQUIRES_OK(context,
                     SparseTensor::Create(std::move(indices_to_concat[i]),
                                          std::move(values_to_concat[i]),
                                          preconcat_shape, std_order, &tensor));
      tensors_to_concat.push_back(std::move(tensor));
    }

    auto output = SparseTensor::Concat<T>(tensors_to_concat);
    Tensor final_output_shape(DT_INT64, TensorShape({output.dims()}));

    std::copy_n(output.shape().data(), output.dims(),
                final_output_shape.vec<int64>().data());

    context->set_output(0, output.indices());
    context->set_output(1, output.values());
    context->set_output(2, final_output_shape);
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
