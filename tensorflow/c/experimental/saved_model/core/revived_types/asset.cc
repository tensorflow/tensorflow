/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/experimental/saved_model/core/revived_types/asset.h"

#include <string>

#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"

namespace tensorflow {

Asset::Asset(ImmediateTensorHandlePtr handle)
    : TensorHandleConvertible(std::move(handle)) {}

Status Asset::Create(ImmediateExecutionContext* ctx,
                     const std::string& saved_model_dir,
                     const std::string& asset_filename,
                     std::unique_ptr<Asset>* output) {
  std::string abs_path =
      io::JoinPath(saved_model_dir, kSavedModelAssetsDirectory, asset_filename);
  AbstractTensorPtr tensor(ctx->CreateStringScalar(abs_path));
  if (tensor.get() == nullptr) {
    return errors::Internal(
        "Failed to create scalar string tensor for Asset at path ", abs_path);
  }

  ImmediateTensorHandlePtr handle(ctx->CreateLocalHandle(tensor.get()));
  output->reset(new Asset(std::move(handle)));
  return Status();
}

}  // namespace tensorflow
