/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <stddef.h>

#include <algorithm>
#include <string>
#include <vector>

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/broadcast_to_op.h"
#include "tensorflow/core/kernels/list_kernels.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/ops/ragged_to_dense_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

namespace {
typedef Eigen::ThreadPoolDevice CPUDevice;
using ::std::vector;
using ::tensorflow::errors::Internal;

const int kShapeInputIndex = 0;
const int kValueInputIndex = 1;
const int kDefaultValueInputIndex = 2;
const int kFirstPartitionInputIndex = 3;

template <typename INDEX_TYPE>
class RaggedTensorToTensorBaseOp : public OpKernel {
 public:
  typedef
      typename ::tensorflow::TTypes<const INDEX_TYPE>::Flat RowPartitionTensor;

  explicit RaggedTensorToTensorBaseOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, GetRowPartitionTypes<OpKernelConstruction>(
                                context, &row_partition_types_));
    ragged_rank_ = GetRaggedRank(row_partition_types_);
  }

  // Returns the relationship between dimension and dimension + 1.
  RowPartitionType GetRowPartitionTypeByDimension(int dimension) {
    if (row_partition_types_[0] == RowPartitionType::FIRST_DIM_SIZE) {
      return row_partition_types_[dimension + 1];
    } else {
      return row_partition_types_[dimension];
    }
  }

  // Returns the relationship between dimension and dimension + 1.
  RowPartitionTensor GetRowPartitionTensor(OpKernelContext* c, int dimension) {
    if (row_partition_types_[0] == RowPartitionType::FIRST_DIM_SIZE) {
      return c->input(dimension + 1 + kFirstPartitionInputIndex)
          .flat<INDEX_TYPE>();
    } else {
      return c->input(dimension + kFirstPartitionInputIndex).flat<INDEX_TYPE>();
    }
  }

  Status GetMaxWidth(OpKernelContext* c, int dimension, INDEX_TYPE* result) {
    const RowPartitionTensor row_partition_tensor =
        GetRowPartitionTensor(c, dimension - 1);
    switch (GetRowPartitionTypeByDimension(dimension - 1)) {
      case RowPartitionType::VALUE_ROWIDS:
        *result = GetMaxWidthValueRowID(row_partition_tensor);
        return Status::OK();
      case RowPartitionType::ROW_SPLITS:
        *result = GetMaxWidthRowSplit(row_partition_tensor);
        return Status::OK();
      default:
        return errors::InvalidArgument(
            "Cannot handle partition type ",
            RowPartitionTypeToString(
                GetRowPartitionTypeByDimension(dimension - 1)));
    }
  }

  static INDEX_TYPE GetMaxWidthRowSplit(const RowPartitionTensor& row_split) {
    const INDEX_TYPE tensor_length = row_split.size();
    if (tensor_length == 0 || tensor_length == 1) {
      return 0;
    }
    INDEX_TYPE max_width = 0;
    for (INDEX_TYPE i = 0; i < tensor_length - 1; ++i) {
      const INDEX_TYPE current_width = row_split(i + 1) - row_split(i);
      if (current_width > max_width) {
        max_width = current_width;
      }
    }
    return max_width;
  }

  static INDEX_TYPE GetMaxWidthValueRowID(
      const RowPartitionTensor& value_rowids) {
    const INDEX_TYPE index_length = value_rowids.size();
    if (index_length == 0) {
      return 0;
    }
    INDEX_TYPE first_equal_index = 0;
    INDEX_TYPE first_equal_index_value = value_rowids(0);
    INDEX_TYPE max_width = 0;
    for (INDEX_TYPE i = 1; i < index_length; ++i) {
      const INDEX_TYPE value = value_rowids(i);
      if (value != first_equal_index_value) {
        first_equal_index_value = value;
        max_width = std::max(i - first_equal_index, max_width);
        first_equal_index = i;
      }
    }
    return std::max(index_length - first_equal_index, max_width);
  }

  Status CalculateOutputSize(INDEX_TYPE first_dim, OpKernelContext* c,
                             vector<INDEX_TYPE>* result) {
    TensorShapeProto value_shape_proto;
    c->input(kValueInputIndex).shape().AsProto(&value_shape_proto);

    TensorShapeProto default_value_shape_proto;
    c->input(kDefaultValueInputIndex)
        .shape()
        .AsProto(&default_value_shape_proto);

    TensorShapeProto output_shape_proto;
    TF_RETURN_IF_ERROR(ValidateDefaultValueShape(default_value_shape_proto,
                                                 value_shape_proto));

    TensorShapeProto shape_proto;
    {
      PartialTensorShape partial_tensor_shape;
      TF_RETURN_IF_ERROR(TensorShapeFromTensor(c->input(kShapeInputIndex),
                                               &partial_tensor_shape));
      partial_tensor_shape.AsProto(&shape_proto);
    }

    TF_RETURN_IF_ERROR(CombineRaggedTensorToTensorShapes(
        ragged_rank_, shape_proto, value_shape_proto, &output_shape_proto));

    result->reserve(output_shape_proto.dim_size());
    for (const TensorShapeProto::Dim& dim : output_shape_proto.dim()) {
      // Note that this may be -1 (if dimension size is unknown).
      result->push_back(dim.size());
    }

    if ((*result)[0] < 0) {
      (*result)[0] = first_dim;
    }
    for (int i = 1; i <= ragged_rank_; ++i) {
      if ((*result)[i] < 0) {
        TF_RETURN_IF_ERROR(GetMaxWidth(c, i, &(*result)[i]));
      }
    }
    return Status::OK();
  }

  /**
   * The output_index represents the index in the output tensor
   * where the first element of a particular dimension would be written.
   * If it is -1, it indicates that the index is out of scope.
   * Example, given first_dimension = 10, first_dimension_output = 6,
   * and output_index_multiplier = 100:
   * result = [0 100 200 300 400 500 -1 -1 -1 -1]
   * If first_dimension_output = 11 instead, then:
   * result = [0 100 200 300 400 500 600 700 800 900]
   */
  vector<INDEX_TYPE> CalculateFirstParentOutputIndex(
      INDEX_TYPE first_dimension, INDEX_TYPE output_index_multiplier,
      INDEX_TYPE first_dimension_output) {
    const INDEX_TYPE min_dimension =
        std::min(first_dimension, first_dimension_output);
    vector<INDEX_TYPE> result;
    result.reserve(first_dimension);
    int current_output_index = 0;
    for (INDEX_TYPE i = 0; i < min_dimension;
         ++i, current_output_index += output_index_multiplier) {
      result.push_back(current_output_index);
    }
    for (INDEX_TYPE i = min_dimension; i < first_dimension; ++i) {
      result.push_back(-1);
    }
    DCHECK_EQ(result.size(), first_dimension);
    return result;
  }

  void CalculateOutputIndexRowSplit(
      const RowPartitionTensor& row_split,
      const vector<INDEX_TYPE>& parent_output_index,
      INDEX_TYPE output_index_multiplier, INDEX_TYPE output_size,
      vector<INDEX_TYPE>* result) {
    INDEX_TYPE row_split_size = row_split.size();
    if (row_split_size > 0) {
      result->reserve(row_split(row_split_size - 1));
    }
    for (INDEX_TYPE i = 0; i < row_split_size - 1; ++i) {
      INDEX_TYPE row_length = row_split(i + 1) - row_split(i);
      INDEX_TYPE real_length = std::min(output_size, row_length);
      INDEX_TYPE parent_output_index_current = parent_output_index[i];

      if (parent_output_index_current == -1) {
        real_length = 0;
      }
      for (INDEX_TYPE j = 0; j < real_length; ++j) {
        result->push_back(parent_output_index_current);
        parent_output_index_current += output_index_multiplier;
      }
      for (INDEX_TYPE j = 0; j < row_length - real_length; ++j) {
        result->push_back(-1);
      }
    }
    if (row_split_size > 0) {
      DCHECK_EQ(result->size(), row_split(row_split_size - 1));
    }
  }

  // Calculate the output index of the first element of a list.
  // The parent_output_index is the same computation for the previous list.
  // -1 indicates an element or list that is out of range.
  // The output_index_multiplier is the number of output indices one moves
  // forward for each column.
  // E.g., given:
  // value_rowids:[0 1 2 2 2 3 5 5 6]
  // parent_output_index:[1000 1100 2000 2100 -1 3000 4000]
  // output_index_multiplier: 10
  // output_size: 2
  // You get:
  // result = [1000 1100 2000 2010 -1 2100 -1 -1 3000]
  // result[0] = parent_output_index[value_rowids[0]]
  // result[1] = parent_output_index[value_rowids[1]]
  // result[2] = parent_output_index[value_rowids[2]]
  // result[3] = parent_output_index[value_rowids[2] + 10]
  // result[4] = -1 because it is the third element the size is 2.
  // result[5] = parent_output_index[value_rowids[3]]
  // result[6] = -1 because parent_output_index[value_rowids[6]] == -1
  // result[7] = -1 because parent_output_index[value_rowids[6]] == -1
  // result[8] = parent_output_index[value_rowids[7]]
  void CalculateOutputIndexValueRowID(
      const RowPartitionTensor& value_rowids,
      const vector<INDEX_TYPE>& parent_output_index,
      INDEX_TYPE output_index_multiplier, INDEX_TYPE output_size,
      vector<INDEX_TYPE>* result) {
    const INDEX_TYPE index_size = value_rowids.size();
    result->reserve(index_size);
    if (index_size == 0) {
      return;
    }

    INDEX_TYPE current_output_column = 0;
    INDEX_TYPE current_value_rowid = value_rowids(0);
    DCHECK_LT(current_value_rowid, parent_output_index.size());
    INDEX_TYPE current_output_index = parent_output_index[current_value_rowid];
    result->push_back(current_output_index);
    for (INDEX_TYPE i = 1; i < index_size; ++i) {
      INDEX_TYPE next_value_rowid = value_rowids(i);
      if (next_value_rowid == current_value_rowid) {
        if (current_output_index >= 0) {
          ++current_output_column;
          if (current_output_column < output_size) {
            current_output_index += output_index_multiplier;
          } else {
            current_output_index = -1;
          }
        }
      } else {
        current_output_column = 0;
        current_value_rowid = next_value_rowid;
        DCHECK_LT(next_value_rowid, parent_output_index.size());
        current_output_index = parent_output_index[next_value_rowid];
      }
      result->push_back(current_output_index);
    }
    DCHECK_EQ(result->size(), value_rowids.size());
  }

  Status CalculateOutputIndex(OpKernelContext* context, int dimension,
                              const vector<INDEX_TYPE>& parent_output_index,
                              INDEX_TYPE output_index_multiplier,
                              INDEX_TYPE output_size,
                              vector<INDEX_TYPE>* result) {
    const RowPartitionTensor row_partition_tensor =
        GetRowPartitionTensor(context, dimension);
    auto partition_type = GetRowPartitionTypeByDimension(dimension);
    switch (partition_type) {
      case RowPartitionType::VALUE_ROWIDS:
        CalculateOutputIndexValueRowID(
            row_partition_tensor, parent_output_index, output_index_multiplier,
            output_size, result);
        return tensorflow::Status::OK();
      case RowPartitionType::ROW_SPLITS:
        CalculateOutputIndexRowSplit(row_partition_tensor, parent_output_index,
                                     output_index_multiplier, output_size,
                                     result);
        return tensorflow::Status::OK();
      default:
        return errors::InvalidArgument(
            "Unsupported partition type:",
            RowPartitionTypeToString(partition_type));
    }
  }

  Status GetFirstDimensionSize(OpKernelContext* context, INDEX_TYPE* result) {
    const Tensor first_partition_tensor =
        context->input(kFirstPartitionInputIndex);
    const RowPartitionType first_partition_type = row_partition_types_[0];
    switch (first_partition_type) {
      case RowPartitionType::FIRST_DIM_SIZE:
        *result = first_partition_tensor.scalar<INDEX_TYPE>()();
        return Status::OK();
      case RowPartitionType::VALUE_ROWIDS:
        return errors::InvalidArgument(
            "Cannot handle VALUE_ROWIDS in first dimension.");
      case RowPartitionType::ROW_SPLITS:
        *result = first_partition_tensor.shape().dim_size(0) - 1;
        return Status::OK();
      default:
        return errors::InvalidArgument(
            "Cannot handle type ",
            RowPartitionTypeToString(first_partition_type));
    }
  }

  void Compute(OpKernelContext* context) override {
    INDEX_TYPE first_dimension;
    OP_REQUIRES_OK(context, GetFirstDimensionSize(context, &first_dimension));
    vector<INDEX_TYPE> output_size;
    OP_REQUIRES_OK(context,
                   CalculateOutputSize(first_dimension, context, &output_size));
    vector<INDEX_TYPE> multiplier;
    multiplier.resize(output_size.size());

    multiplier[multiplier.size() - 1] = 1;
    for (int i = output_size.size() - 2; i >= 0; --i) {
      multiplier[i] = multiplier[i + 1] * output_size[i + 1];
    }
    // Full size of the tensor.
    TensorShape output_shape;
    OP_REQUIRES_OK(context,
                   TensorShapeUtils::MakeShape(output_size, &output_shape));
    Tensor* output_tensor = nullptr;

    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));
    const INDEX_TYPE full_size = multiplier[0] * output_size[0];
    if (full_size > 0) {
      vector<INDEX_TYPE> output_index = CalculateFirstParentOutputIndex(
          first_dimension, multiplier[0], output_size[0]);

      for (int i = 1; i <= ragged_rank_; ++i) {
        vector<INDEX_TYPE> new_output_index;
        OP_REQUIRES_OK(context, CalculateOutputIndex(
                                    context, i - 1, output_index, multiplier[i],
                                    output_size[i], &new_output_index));
        output_index = new_output_index;
      }

      SetOutput(context, output_index, output_tensor);
    }
  }
  virtual void SetOutput(OpKernelContext* context,
                         const vector<INDEX_TYPE>& output_index,
                         Tensor* output_tensor) = 0;

 private:
  vector<RowPartitionType> row_partition_types_;
  int ragged_rank_;
};

template <typename VALUE_TYPE, typename INDEX_TYPE>
void slow_copy_array(VALUE_TYPE* dst, const VALUE_TYPE* src, INDEX_TYPE size) {
  for (INDEX_TYPE index = 0; index < size; ++index) {
    dst[index] = src[index];
  }
}

template <typename VALUE_TYPE, typename INDEX_TYPE>
void copy_array(VALUE_TYPE* dst, const VALUE_TYPE* src, INDEX_TYPE size,
                size_t bytes) {
  memcpy(dst, src, bytes);
}

template <>
void copy_array<string, int64>(string* dst, const string* src, int64 size,
                               size_t bytes) {
  slow_copy_array(dst, src, size);
}

template <>
void copy_array<string, int32>(string* dst, const string* src, int32 size,
                               size_t bytes) {
  slow_copy_array(dst, src, size);
}

// If we don't specialize for Eigen::half, we get:
// undefined behavior, destination object type 'Eigen::half'
// is not TriviallyCopyable
template <>
void copy_array<Eigen::half, int64>(Eigen::half* dst, const Eigen::half* src,
                                    int64 size, size_t bytes) {
  slow_copy_array(dst, src, size);
}

template <>
void copy_array<Eigen::half, int32>(Eigen::half* dst, const Eigen::half* src,
                                    int32 size, size_t bytes) {
  slow_copy_array(dst, src, size);
}

template <typename VALUE_TYPE, typename INDEX_TYPE>
class RaggedTensorToTensorOp : public RaggedTensorToTensorBaseOp<INDEX_TYPE> {
 public:
  explicit RaggedTensorToTensorOp(OpKernelConstruction* context)
      : RaggedTensorToTensorBaseOp<INDEX_TYPE>(context) {}

  void SetOutput(OpKernelContext* context,
                 const vector<INDEX_TYPE>& output_index,
                 Tensor* output_tensor) override {
    typename tensorflow::TTypes<VALUE_TYPE>::Flat output_flat =
        output_tensor->flat<VALUE_TYPE>();
    const auto& value_tensor = context->input(kValueInputIndex);
    const auto& default_value_tensor = context->input(kDefaultValueInputIndex);
    if (value_tensor.shape().dims() == 1) {
      // Initialize tensor to default_value.
      VALUE_TYPE* base_output = output_flat.data();
      VALUE_TYPE default_value = default_value_tensor.scalar<VALUE_TYPE>()();

      std::fill(base_output, base_output + output_flat.size(), default_value);
      auto values = context->input(kValueInputIndex).flat<VALUE_TYPE>();
      int values_size = values.size();
      OP_REQUIRES(context, values_size == output_index.size(),
                  Internal("Values and indices must be equal"));
      for (int i = 0; i < values_size; ++i) {
        if (output_index[i] >= 0) {
          output_flat(output_index[i]) = values(i);
        }
      }
    } else {
      const auto& output_shape = output_tensor->shape();
      const auto& default_value_shape = default_value_tensor.shape();

      // Initialize tensor to default_value.

      BCast bcast(BCast::FromShape(default_value_shape),
                  BCast::FromShape(output_shape),
                  /*fewer_dims_optimization=*/true);
      OP_REQUIRES(
          context, bcast.IsValid(),
          errors::InvalidArgument(
              "Incompatible shapes: ", default_value_shape.DebugString(),
              " vs. ", default_value_shape.DebugString()));
      OP_REQUIRES(
          context, BCast::ToShape(bcast.output_shape()) == output_shape,
          errors::InvalidArgument("Unable to broadcast default_value of shape ",
                                  default_value_shape, " to tensor of shape ",
                                  output_shape));
      const CPUDevice& device = context->eigen_device<CPUDevice>();
      functor::BroadcastTo<CPUDevice, VALUE_TYPE>()(
          device, context, *output_tensor, output_shape, default_value_tensor,
          default_value_shape, bcast);

      VALUE_TYPE* base_output = output_flat.data();
      auto values = context->input(kValueInputIndex).flat<VALUE_TYPE>();
      size_t values_size = values.size();
      size_t output_index_size = output_index.size();
      //  A value "element" is a group of values that are arranged together.
      // For example, if the value shape is [3,4,5], then 20 values are in a
      // value element.
      int value_element_size = values_size / output_index_size;
      int value_element_bytesize = value_element_size * sizeof(VALUE_TYPE);
      const VALUE_TYPE* values_base = values.data();

      OP_REQUIRES(context,
                  value_tensor.shape().dim_size(0) == output_index_size,
                  Internal("Values and indices must be equal"));

      OP_REQUIRES(context,
                  values_size == output_index_size * value_element_size,
                  Internal("Values and indices must be equal"));
      INDEX_TYPE value_index = 0;
      for (int i = 0; i < output_index_size;
           ++i, value_index += value_element_size) {
        if (output_index[i] >= 0) {
          VALUE_TYPE* dst = base_output + output_index[i];
          const VALUE_TYPE* src = values_base + value_index;
          copy_array<VALUE_TYPE, INDEX_TYPE>(dst, src, value_element_size,
                                             value_element_bytesize);
        }
      }
    }
  }
};

#define REGISTER_CPU_KERNEL_INDEX_TYPE(value_type, index_type)       \
  REGISTER_KERNEL_BUILDER(Name("RaggedTensorToTensor")               \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<value_type>("T")       \
                              .TypeConstraint<index_type>("Tindex"), \
                          RaggedTensorToTensorOp<value_type, index_type>);

#define REGISTER_CPU_KERNEL(value_type)                          \
  REGISTER_CPU_KERNEL_INDEX_TYPE(value_type, tensorflow::int64); \
  REGISTER_CPU_KERNEL_INDEX_TYPE(value_type, tensorflow::int32);

TF_CALL_POD_TYPES(REGISTER_CPU_KERNEL);
TF_CALL_string(REGISTER_CPU_KERNEL);
TF_CALL_QUANTIZED_TYPES(REGISTER_CPU_KERNEL);
TF_CALL_quint16(REGISTER_CPU_KERNEL);
TF_CALL_qint16(REGISTER_CPU_KERNEL);
TF_CALL_uint32(REGISTER_CPU_KERNEL);
TF_CALL_uint64(REGISTER_CPU_KERNEL);

#undef REGISTER_CPU_KERNEL

}  // namespace
}  // namespace tensorflow
