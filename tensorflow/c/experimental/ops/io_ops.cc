/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/c/experimental/ops/io_ops.h"

#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/tracing_utils.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/errors.h"

using tensorflow::tracing::MaybeSetOpName;

namespace tensorflow {
namespace ops {

// Op: RestoreV2()
// Summary: Restores tensors from a V2 checkpoint.
//
// Description:
//   For backward compatibility with the V1 format, this Op currently allows
//   restoring from a V1 checkpoint as well:
//     - This Op first attempts to find the V2 index file pointed to by
//     "prefix", and
//       if found proceed to read it as a V2 checkpoint;
//     - Otherwise the V1 read path is invoked.
//   Relying on this behavior is not recommended, as the ability to fall back to
//   read V1 might be deprecated and eventually removed.
//
//   By default, restores the named tensors in full.  If the caller wishes to
//   restore specific slices of stored tensors, "shape_and_slices" should be
//   non-empty strings and correspondingly well-formed.
//
//   Callers must ensure all the named tensors are indeed stored in the
//   checkpoint.
absl::Status RestoreV2(AbstractContext* ctx, AbstractTensorHandle* const prefix,
                       AbstractTensorHandle* const tensor_names,
                       AbstractTensorHandle* const shape_and_slices,
                       absl::Span<AbstractTensorHandle*> tensors,
                       absl::Span<DataType> dtypes, const char* name,
                       const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("RestoreV2", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(prefix));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(tensor_names));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(shape_and_slices));
  TF_RETURN_IF_ERROR(
      op_ptr->SetAttrTypeList("dtypes", dtypes.data(), dtypes.length()));
  int num_retvals = tensors.size();
  return op_ptr->Execute(tensors, &num_retvals);
}

// Op: SaveV2()
// Summary: Saves tensors in V2 checkpoint format.
//
// Description:
//   By default, saves the named tensors in full.  If the caller wishes to save
//   specific slices of full tensors, "shape_and_slices" should be non-empty
//   strings and correspondingly well-formed.
absl::Status SaveV2(AbstractContext* ctx, AbstractTensorHandle* const prefix,
                    AbstractTensorHandle* const tensor_names,
                    AbstractTensorHandle* const shape_and_slices,
                    absl::Span<AbstractTensorHandle* const> tensors,
                    const char* name, const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("SaveV2", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(prefix));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(tensor_names));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(shape_and_slices));
  TF_RETURN_IF_ERROR(op_ptr->AddInputList(tensors));
  int num_retvals = 0;
  std::vector<AbstractTensorHandle*> dummy_outputs;
  return op_ptr->Execute(absl::MakeSpan(dummy_outputs), &num_retvals);
}

}  // namespace ops
}  // namespace tensorflow
