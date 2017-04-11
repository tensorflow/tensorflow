/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_UTIL_EXAMPLE_PROTO_HELPER_H_
#define THIRD_PARTY_TENSORFLOW_CORE_UTIL_EXAMPLE_PROTO_HELPER_H_

#include <string>
#include <vector>

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

// This is a set of helper methods that will make it possible to share
// tensorflow::Example proto Tensor conversion code inside the ExampleParserOp
// OpKernel as well as in external code.
namespace tensorflow {

// "Dense" feature configuration.
struct FixedLenFeature {
  string key;
  DataType dtype;
  TensorShape shape;
  Tensor default_value;
  string values_output_tensor_name;
};

// "Sparse" feature configuration.
struct VarLenFeature {
  string key;
  DataType dtype;
  string values_output_tensor_name;
  string indices_output_tensor_name;
  string shapes_output_tensor_name;
};

// Given a single tensorflow::Example, with an optional example name
// at a particular index within a batch, and dense and sparse feature
// configurations from fixed_len_features, var_len_features, this method
// updates the dense value tensor and the sparse values temporary vector
// of tensors. The indexing of the output vectors correspond 1:1 to the
// indexing of the feature configuration vectors.
//
// The fixed_len_features and var_len_features maps are assume to be
// have disjoint key fields from the Feature map in the tensorflow.Example
// proto.
//
// For each sparse feature, the sparse values temporary vector holds a
// tensor for each Example. Each tensor is either empty or filled, depending
// on if the sparse feature value is set for the Example. This
// temporary structure is needed because we need to know the total number
// of filled elements in the batch to get the proper final sparse tensor
// shapes allocated.  After the entire batch is processed,
// GetSparseTensorShape can be used to calculate the final shapes and
// CopyIntoSparseTensor can be used to copy from the temporary vector
// into the final allocated tensors.
Status SingleExampleProtoToTensors(
    const Example& example, const string& name, const int batch_index,
    const std::vector<FixedLenFeature>& fixed_len_features,
    const std::vector<VarLenFeature>& var_len_features,
    std::vector<Tensor*>* dense_values,
    std::vector<std::vector<Tensor>>* sparse_values_temporary_vector);

// The shape of the indices and values tensors associated with a SparseTensor
// are dependent on the contents of the batch.
struct VarLenFeatureBatchShapes {
  TensorShape indices_shape;
  TensorShape values_shape;
  int max_num_features;
};

// Get the shape of the sparse values and indices tensors for the batch,
// given how many of the tensors in the temporary sparse values vector
// are actually filled.
Status GetSparseTensorShapes(const VarLenFeature& var_len_feature,
                             const std::vector<Tensor>& sparse_values_tmp,
                             const int batch_size,
                             VarLenFeatureBatchShapes* output_shapes);

// A method to convert a batch of tensorflow::Example protos into output
// tensors. This method is useful if there already is a batch of deserialized
// Example protos in memory (such as a serving use-case) and we do not wish
// to incur an extraneous serialize/deserialize.  It is intended
// as an outside of OpKernel compatible replacement for the functionality of
// ExampleParserOp. In a serving setting, this method could be used to produce
// a feed_dict of Tensors that could bypass the ExampleParserOp.
//
// Note that unlike SingleExampleProtoToTensors, output tensors are
// allocated using a provided Allocator within this method.
Status BatchExampleProtoToTensors(
    const std::vector<const Example*>& examples,
    const std::vector<string>& names,
    const std::vector<FixedLenFeature>& fixed_len_features,
    const std::vector<VarLenFeature>& var_len_features, Allocator* allocator,
    std::vector<Tensor>* output_dense_values_tensor,
    std::vector<Tensor>* output_sparse_indices_tensor,
    std::vector<Tensor>* output_sparse_values_tensor,
    std::vector<Tensor>* output_sparse_shapes_tensor);

// Check that the given dtype is one that is compatible with
// tensorflow::Example protocol buffer feature values.
Status CheckValidType(const DataType& dtype);

// Check that the provided Feature proto message's oneof value
// matches that of the provided dtype.
Status CheckTypesMatch(const Feature& feature, const DataType& dtype,
                       bool* match);

// For a single Example, copy a dense feature value into an output
// dense value tensor Out at the provided out_index offset.
Status FeatureDenseCopy(const std::size_t out_index, const string& name,
                        const string& key, const DataType& dtype,
                        const TensorShape& shape, const Feature& feature,
                        Tensor* out);

// Copy the value a provided Tensor into an output dense_value tensor Out
// at the provided out_index offset.
void RowDenseCopy(const std::size_t& out_index, const DataType& dtype,
                  const Tensor& in, Tensor* out);

// For a single Example, and given sparse feature return a temporary output
// Tensor suitable for being collected in the temporary sparse value vector.
Tensor FeatureSparseCopy(const std::size_t batch, const string& key,
                         const DataType& dtype, const Feature& feature);

// Copy a temporary Tensor into the final sparse indices and values
// tensor at a given batch index and element offset. This method
// assumes that the indices/values Tensors have been properly allocated
// for the batch.
int64 CopyIntoSparseTensor(const Tensor& in, const int batch,
                           const int64 offset, Tensor* indices, Tensor* values);

// Parses the attributes passed to ParseSingleExample.
// REQUIRES: Init must be called after construction.
class ParseSingleExampleAttrs {
 public:
  template <typename ContextType>
  Status Init(ContextType* ctx) {
    TF_RETURN_IF_ERROR(ctx->GetAttr("sparse_types", &sparse_types));
    TF_RETURN_IF_ERROR(ctx->GetAttr("Ndense", &num_dense));
    TF_RETURN_IF_ERROR(ctx->GetAttr("Nsparse", &num_sparse));
    TF_RETURN_IF_ERROR(ctx->GetAttr("Tdense", &dense_types));
    TF_RETURN_IF_ERROR(ctx->GetAttr("dense_shapes", &dense_shapes));
    // Temporary check until we start allowing a variable length outer
    // dimension.
    for (int i = 0; i < dense_shapes.size(); ++i) {
      bool shape_ok = true;
      if (dense_shapes[i].dims() == -1) {
        shape_ok = false;
      } else {
        for (int d = 1; d < dense_shapes[i].dims(); ++d) {
          if (dense_shapes[i].dim_size(d) == -1) {
            shape_ok = false;
          }
        }
      }
      if (!shape_ok) {
        return errors::InvalidArgument(
            "dense_shapes[", i,
            "] has unknown rank or unknown inner dimensions: ",
            dense_shapes[i].DebugString());
      }
      TensorShape dense_shape;
      if (dense_shapes[i].dims() > 0 && dense_shapes[i].dim_size(0) == -1) {
        variable_length.push_back(true);
        for (int d = 1; d < dense_shapes[i].dims(); ++d) {
          dense_shape.AddDim(dense_shapes[i].dim_size(d));
        }
      } else {
        variable_length.push_back(false);
        dense_shapes[i].AsTensorShape(&dense_shape);
      }
      elements_per_stride.push_back(dense_shape.num_elements());
    }
    return FinishInit();
  }

  int64 num_sparse;
  int64 num_dense;
  std::vector<DataType> sparse_types;
  std::vector<DataType> dense_types;
  std::vector<PartialTensorShape> dense_shapes;
  std::vector<bool> variable_length;
  std::vector<std::size_t> elements_per_stride;

 private:
  Status FinishInit();  // for context-independent parts of Init.
};

// Parses the attributes passed to ParseSingleSequenceExample.
// REQUIRES: Init must be called after construction.
class ParseSingleSequenceExampleAttrs {
 public:
  template <typename ContextType>
  Status Init(ContextType* ctx) {
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("context_sparse_types", &context_sparse_types));
    TF_RETURN_IF_ERROR(ctx->GetAttr("Ncontext_dense", &num_context_dense));
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("Nfeature_list_dense", &num_feature_list_dense));
    TF_RETURN_IF_ERROR(ctx->GetAttr("Ncontext_sparse", &num_context_sparse));
    TF_RETURN_IF_ERROR(ctx->GetAttr("Tcontext_dense", &context_dense_types));
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("feature_list_sparse_types", &feature_list_sparse_types));
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("feature_list_dense_types", &feature_list_dense_types));
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("Nfeature_list_sparse", &num_feature_list_sparse));
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("context_dense_shapes", &context_dense_shapes));
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("feature_list_dense_shapes", &feature_list_dense_shapes));
    return FinishInit();
  }

  int64 num_context_sparse;
  int64 num_context_dense;
  int64 num_feature_list_sparse;
  int64 num_feature_list_dense;
  std::vector<DataType> context_sparse_types;
  std::vector<DataType> context_dense_types;
  std::vector<TensorShape> context_dense_shapes;
  std::vector<DataType> feature_list_sparse_types;
  std::vector<DataType> feature_list_dense_types;
  std::vector<TensorShape> feature_list_dense_shapes;

 private:
  Status FinishInit();  // for context-independent parts of Init.
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_UTIL_EXAMPLE_PROTO_HELPER_H_
