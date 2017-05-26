/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {

// This op loads a rank-2 Tensor (matrix) from a TensorFlow checkpoint (V2) and
// swaps around the rows/columns according to row_remapping/col_remapping.
// "Missing" cells are initialized with values from initializing_values.
class LoadAndRemapMatrixOp : public OpKernel {
 public:
  explicit LoadAndRemapMatrixOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_rows", &num_rows_));
    OP_REQUIRES_OK(context, context->GetAttr("num_cols", &num_cols_));
  }

  void Compute(OpKernelContext* context) override {
    // Checks what we're remapping and inverts the relevant remapping Tensors to
    // be maps with key = old ID, value = new ID.
    std::vector<std::pair<int64, int64>> old_row_to_new_row_pairs;
    std::vector<bool> row_id_present(num_rows_);
    const Tensor* row_remapping_t;
    OP_REQUIRES_OK(context, context->input("row_remapping", &row_remapping_t));
    const auto row_remapping = row_remapping_t->vec<int64>();
    OP_REQUIRES(context, row_remapping.size() == num_rows_,
                errors::InvalidArgument(strings::StrCat(
                    "Size of row_remapping is ", row_remapping.size(),
                    " intead of being equal to num_rows=", num_rows_)));
    old_row_to_new_row_pairs.reserve(num_rows_);
    for (int i = 0; i < row_remapping.size(); ++i) {
      if (row_remapping(i) < 0) continue;
      row_id_present[i] = true;
      old_row_to_new_row_pairs.push_back(std::make_pair(row_remapping(i), i));
    }

    // Processes the remapping for columns.
    std::unordered_map<int64, int64> old_col_to_new_col_map;
    std::vector<bool> col_id_present(num_cols_);
    const Tensor* col_remapping_t;
    OP_REQUIRES_OK(context, context->input("col_remapping", &col_remapping_t));
    const auto col_remapping = col_remapping_t->vec<int64>();
    // Note that we always "remap rows", even when the row vocabulary does
    // not change, because partitioning requires a mapping from partitioned
    // Variables to the full checkpoints we load.
    const bool remap_cols = col_remapping.size() > 0;
    if (remap_cols) {
      OP_REQUIRES(
          context, col_remapping.size() == num_cols_,
          errors::InvalidArgument(strings::StrCat(
              "Provided col_remapping, but its size is ", col_remapping.size(),
              " instead of being equal to num_cols=", num_cols_)));
      for (int i = 0; i < col_remapping.size(); ++i) {
        const int64 old_col = col_remapping(i);
        if (old_col < 0) continue;
        col_id_present[i] = true;
        OP_REQUIRES(
            context,
            gtl::InsertIfNotPresent(&old_col_to_new_col_map, old_col, i),
            errors::Unimplemented(strings::StrCat(
                "Old column ID ", old_col, " is mapped to both new column ID ",
                old_col_to_new_col_map[old_col], " and ", i,
                ", which is not currently supported - but could be "
                "implemented.")));
      }
    } else {
      col_id_present.clear();
      col_id_present.resize(num_cols_, true);
    }

    // Processes the checkpoint source and the provided Tensor name.
    const Tensor* ckpt_path_t;
    OP_REQUIRES_OK(context, context->input("ckpt_path", &ckpt_path_t));
    const string ckpt_path = *(ckpt_path_t->scalar<string>().data());
    const Tensor* old_tensor_name_t;
    OP_REQUIRES_OK(context,
                   context->input("old_tensor_name", &old_tensor_name_t));
    const string old_tensor_name =
        *(old_tensor_name_t->scalar<string>().data());

    LOG(INFO) << "Processing checkpoint : " << ckpt_path;
    BundleReader reader(context->env(), ckpt_path);
    OP_REQUIRES_OK(context, reader.status());

    DataType tensor_type;
    TensorShape tensor_shape;
    OP_REQUIRES_OK(context, reader.LookupDtypeAndShape(
                                old_tensor_name, &tensor_type, &tensor_shape));
    OP_REQUIRES(context, tensor_type == DT_FLOAT,
                errors::InvalidArgument(strings::StrCat(
                    "Tensor ", old_tensor_name, " has invalid type ",
                    DataTypeString(tensor_type), " instead of expected type ",
                    DataTypeString(DT_FLOAT))));
    // This op is limited to loading Tensors of rank 2 (matrices).
    OP_REQUIRES(
        context, tensor_shape.dims() == 2,
        errors::InvalidArgument(strings::StrCat(
            "Tensor ", old_tensor_name, " has shape ",
            tensor_shape.DebugString(), " of invalid rank ",
            tensor_shape.dims(), " instead of expected shape of rank 2.")));

    if (!remap_cols) {
      // TODO(weiho): Consider relaxing this restriction to allow partial column
      // loading (even when no column remapping is specified) if there turns out
      // to be a use case for it.
      OP_REQUIRES(context, num_cols_ == tensor_shape.dim_size(1),
                  errors::InvalidArgument(strings::StrCat(
                      "Tensor ", old_tensor_name, " has shape ",
                      tensor_shape.DebugString(),
                      ", where the size of its 2nd dimension is ",
                      tensor_shape.dim_size(1),
                      " instead of being equal to num_cols=", num_cols_)));
    }

    // Uses TensorSlice to selectively read rows of interest from the old
    // tensor. Given BundleReader's use of RandomAccessFile and InputBuffer,
    // there shouldn't too many more additional disk seeks when compared to
    // loading the old tensor in chunks, once we sort the row IDs. Even if there
    // are locality concerns with some reading patterns, that just means if we
    // had read it in chunks, then we would have had to read, copy, and process
    // then discard many redundant rows - so we should come out ahead this way.
    // In addition, this frees us from having to hold the entire old tensor in
    // memory.
    std::sort(old_row_to_new_row_pairs.begin(), old_row_to_new_row_pairs.end());
    std::vector<TensorSlice> tensor_slices;
    tensor_slices.reserve(old_row_to_new_row_pairs.size());
    TensorSlice slice(tensor_shape.dims());
    for (const auto& pair : old_row_to_new_row_pairs) {
      OP_REQUIRES(
          context, pair.first < tensor_shape.dim_size(0),
          errors::InvalidArgument(strings::StrCat(
              "Trying to read row ", pair.first, " from tensor ",
              old_tensor_name, ", which only has ", tensor_shape.dim_size(0),
              " rows (with shape ", tensor_shape.DebugString(), ").")));
      slice.set_start(0, pair.first);
      slice.set_length(0, 1);
      tensor_slices.push_back(slice);
    }

    // Allocates the output matrix.
    Tensor* output_matrix_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output_matrix",
                                            TensorShape({num_rows_, num_cols_}),
                                            &output_matrix_t));
    auto output_matrix = output_matrix_t->matrix<float>();

    // Iterates through tensor slices and copies over values from the old tensor
    // to the output matrix.
    Tensor loaded_tensor_t(DT_FLOAT,
                           TensorShape({1, tensor_shape.dim_size(1)}));
    for (int i = 0; i < tensor_slices.size(); ++i) {
      const int64 new_row = old_row_to_new_row_pairs[i].second;
      if (i % 500000 == 0) {
        LOG(INFO) << "Processing slice " << i << " of " << tensor_slices.size()
                  << " - corresponding to old row "
                  << old_row_to_new_row_pairs[i].first << " of "
                  << tensor_shape.dim_size(0);
      }
      OP_REQUIRES_OK(context,
                     reader.LookupSlice(old_tensor_name, tensor_slices[i],
                                        &loaded_tensor_t));

      // Copies over the row element-by-element, in case remapping is needed
      // along the column axis.
      const auto& loaded_tensor = loaded_tensor_t.flat<float>();
      for (int old_col = 0; old_col < loaded_tensor.size(); ++old_col) {
        int64 new_col = old_col;
        if (remap_cols) {
          const int64* new_col_ptr =
              gtl::FindOrNull(old_col_to_new_col_map, old_col);
          if (new_col_ptr == nullptr) {
            // Column remapping is specified, but this column is not found in
            // old_col_to_new_col_map, so we leave it uninitialized, to be
            // filled in with initializing_values later.
            continue;
          }
          new_col = *new_col_ptr;
        }

        OP_REQUIRES(context,
                    new_row < num_rows_ && new_col < num_cols_ &&
                        new_row >= 0 && new_col >= 0,
                    errors::Internal(strings::StrCat(
                        "new_row=", new_row, " and new_col=", new_col,
                        " should have been less than num_rows_=", num_rows_,
                        " and num_cols_=", num_cols_,
                        " and non-negative. This should never have happened "
                        "if the code were correct. Please file a bug.")));
        output_matrix(new_row, new_col) = loaded_tensor(old_col);
      }
    }
    LOG(INFO) << "Copied " << tensor_slices.size()
              << " rows from old matrix (with " << tensor_shape.dim_size(0)
              << " rows) to new matrix (with " << num_rows_ << " rows).";

    // At this point, there are potentially whole rows/columns uninitialized
    // (corresponding to the indices where row_id_present/col_id_present are
    // false). We fill this in cell-by-cell using row_id_present and
    // col_id_present while dequeuing from the initializing_values vector.
    const Tensor* initializing_values_t;
    OP_REQUIRES_OK(
        context, context->input("initializing_values", &initializing_values_t));
    const auto initializing_values = initializing_values_t->flat<float>();
    int64 initializing_values_index = 0;
    for (int i = 0; i < num_rows_; ++i) {
      for (int j = 0; j < num_cols_; ++j) {
        if (!row_id_present[i] || !col_id_present[j]) {
          output_matrix(i, j) = initializing_values(initializing_values_index);
          ++initializing_values_index;
        }
      }
    }

    // Checks that we used all the given initializing values.
    OP_REQUIRES(
        context, initializing_values_index == initializing_values.size(),
        errors::InvalidArgument(
            "initializing_values contained ", initializing_values.size(),
            " elements, but only ", initializing_values_index,
            " elements were used to fill in missing values."));
  }

 private:
  int64 num_rows_;
  int64 num_cols_;
};

REGISTER_KERNEL_BUILDER(Name("LoadAndRemapMatrix").Device(DEVICE_CPU),
                        LoadAndRemapMatrixOp);

}  // namespace tensorflow
