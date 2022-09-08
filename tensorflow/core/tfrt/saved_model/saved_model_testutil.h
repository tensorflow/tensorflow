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
#ifndef TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_TESTUTIL_H_
#define TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_TESTUTIL_H_

#include <stdlib.h>

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/saved_model/saved_model.h"
#include "third_party/tensorflow_serving/apis/predict.pb.h"
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

ABSL_DECLARE_FLAG(bool, enable_optimizer);
ABSL_DECLARE_FLAG(std::string, force_data_format);

namespace tensorflow {
namespace tfrt_stub {

std::unique_ptr<tensorflow::tfrt_stub::Runtime> DefaultTfrtRuntime(
    int num_threads);

SavedModel::Options DefaultSavedModelOptions(
    tensorflow::tfrt_stub::Runtime* runtime);

class TFRTSavedModelTest {
 public:
  explicit TFRTSavedModelTest(const std::string& saved_model_dir);
  TFRTSavedModelTest(const std::string& saved_model_dir,
                     std::unique_ptr<tensorflow::tfrt_stub::Runtime> runtime);

  SavedModel* GetSavedModel() { return saved_model_.get(); }

  tfrt::HostContext* GetHostContext() const {
    return saved_model_->GetHostContext();
  }

 private:
  std::unique_ptr<tensorflow::tfrt_stub::Runtime> runtime_;
  std::unique_ptr<SavedModel> saved_model_;
};

template <typename T, typename U = T>
tensorflow::Tensor CreateTfTensor(absl::Span<const int64_t> shape,
                                  absl::Span<const U> data) {
  tensorflow::Tensor tensor(tensorflow::DataTypeToEnum<T>::value,
                            tensorflow::TensorShape(shape));
  auto flat = tensor.flat<T>();
  for (int i = 0; i < data.size(); ++i) {
    flat(i) = data[i];
  }
  return tensor;
}

template <typename T>
std::vector<T> GetTfTensorData(const tensorflow::Tensor& tensor) {
  return std::vector<T>(tensor.flat<T>().data(),
                        tensor.flat<T>().data() + tensor.NumElements());
}

inline tensorflow::Tensor CreateTfStringTensor(
    absl::Span<const int64_t> shape, absl::Span<const std::string> data) {
  return CreateTfTensor<tensorflow::tstring>(shape, data);
}

void ComputeCurrentTFResult(const std::string& saved_model_dir,
                            const std::string& signature_name,
                            const std::vector<std::string>& input_names,
                            const std::vector<tensorflow::Tensor>& inputs,
                            const std::vector<std::string>& output_names,
                            std::vector<tensorflow::Tensor>* outputs,
                            bool enable_mlir_bridge = false,
                            bool disable_grappler = false);

// Compute the results using TF1 session loaded from the saved model. In
// addition to returning the result tensors, it also fills `bundle` with the
// loaded savedmodel. This is useful as sometimes the result tensors may only be
// valid when the bundle is alive.
void ComputeCurrentTFResult(const std::string& saved_model_dir,
                            const std::string& signature_name,
                            const std::vector<std::string>& input_names,
                            const std::vector<tensorflow::Tensor>& inputs,
                            const std::vector<std::string>& output_names,
                            std::vector<tensorflow::Tensor>* outputs,
                            tensorflow::SavedModelBundle* bundle,
                            bool enable_mlir_bridge = false,
                            bool disable_grappler = false);

void ExpectTensorEqual(const tensorflow::Tensor& x, const tensorflow::Tensor& y,
                       std::optional<double> error = std::nullopt);

SavedModel::Options DefaultTpuModelOptions(
    tensorflow::tfrt_stub::Runtime* runtime,
    tensorflow::TfrtTpuInfraTarget tpu_target);

tensorflow::StatusOr<std::vector<tensorflow::serving::PredictRequest>>
GetWarmupRequests(absl::string_view saved_model_dir);

void ProcessPredictRequestsAndMaybeProfile(
    const std::vector<tensorflow::serving::PredictRequest>& requests,
    SavedModel* saved_model, const bool profile = false,
    const int32_t num_steps = 1);

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_TESTUTIL_H_
