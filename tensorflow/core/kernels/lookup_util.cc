/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/kernels/lookup_util.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace lookup {
namespace {

Status GetTableHandle(const string& input_name, OpKernelContext* ctx,
                      string* container, string* table_handle) {
  {
    mutex* mu;
    TF_RETURN_IF_ERROR(ctx->input_ref_mutex(input_name, &mu));
    mutex_lock l(*mu);
    Tensor tensor;
    TF_RETURN_IF_ERROR(ctx->mutable_input(input_name, &tensor, true));
    if (tensor.NumElements() != 2) {
      return errors::InvalidArgument(
          "Lookup table handle must be scalar, but had shape: ",
          tensor.shape().DebugString());
    }
    auto h = tensor.flat<string>();
    *container = h(0);
    *table_handle = h(1);
  }
  return Status::OK();
}

}  // namespace

Status GetLookupTable(const string& input_name, OpKernelContext* ctx,
                      LookupInterface** table) {
  string container;
  string table_handle;
  TF_RETURN_IF_ERROR(
      GetTableHandle(input_name, ctx, &container, &table_handle));
  return ctx->resource_manager()->Lookup(container, table_handle, table);
}

Status GetInitializableLookupTable(const string& input_name,
                                   OpKernelContext* ctx,
                                   InitializableLookupTable** table) {
  string container;
  string table_handle;
  TF_RETURN_IF_ERROR(
      GetTableHandle(input_name, ctx, &container, &table_handle));
  LookupInterface* lookup_table;
  TF_RETURN_IF_ERROR(
      ctx->resource_manager()->Lookup(container, table_handle, &lookup_table));
  *table = lookup_table->GetInitializableLookupTable();
  if (*table == nullptr) {
    lookup_table->Unref();
    return errors::InvalidArgument("Table ", container, " ", table_handle,
                                   " is not initializable");
  }
  return Status::OK();
}

Status CheckTableDataTypes(const LookupInterface& table, DataType key_dtype,
                           DataType value_dtype, const string& table_name) {
  if (table.key_dtype() != key_dtype || table.value_dtype() != value_dtype) {
    return errors::InvalidArgument(
        "Conflicting key/value dtypes ", key_dtype, "->", value_dtype, " with ",
        table.key_dtype(), "-", table.value_dtype(), " for table ", table_name);
  }
  return Status::OK();
}

}  // namespace lookup
}  // namespace tensorflow
