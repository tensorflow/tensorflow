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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_ASSET_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_ASSET_H_

#include <string>

#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tensorhandle_convertible.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/core/framework/tensor.pb.h"

namespace tensorflow {

class Asset : public TensorHandleConvertible {
 public:
  static Status Create(ImmediateExecutionContext* ctx,
                       const std::string& saved_model_dir,
                       const std::string& asset_filename,
                       std::unique_ptr<Asset>* output);

  // Asset is movable, but not copyable.
  Asset(Asset&& other) = default;
  Asset& operator=(Asset&& other) = default;

  ~Asset() override = default;

 private:
  explicit Asset(ImmediateTensorHandlePtr handle);
  Asset(const Asset&) = delete;
  Asset& operator=(const Asset&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_ASSET_H_
