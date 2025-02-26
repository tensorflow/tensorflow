// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_COMPILED_MODEL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_COMPILED_MODEL_H_

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_compiled_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_compiled_model_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/cc/litert_environment.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_handle.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_requirements.h"

namespace litert {

// The CompiledModel is a higher level inference API. It is created by
// provided model with compilation options. Internally, it instantiates runtime
// and applies Delegates mapped to the compilation options.
// It also supports getting BufferRequirements to create input/output
// TensorBuffers, and it allows to invoke the model with the input/output
// TensorBuffers.
//
// Example user flow:
//
// 1. Create CompiledModel
// 2. Query the model input/output requirements
// 3. Create input/output TensorBuffers
// 4. Fill the input TensorBuffers with input data
// 5. Invoke the model with the input/output TensorBuffers
// 6. Evaluate the output TensorBuffers

class CompiledModel
    : public internal::Handle<LiteRtCompiledModel, LiteRtDestroyCompiledModel> {
 public:
  class Options {
    struct Deleter {
      void operator()(LiteRtCompilationOptions options) {
        LiteRtDestroyCompilationOptions(options);
      }
    };

    // Were making the default constructor private to avoid  null options
    // created by mistake.
    Options() = default;
    explicit Options(LiteRtCompilationOptions options) : impl_(options) {}

   public:
    using Ptr = std::unique_ptr<LiteRtCompilationOptionsT, Deleter>;

    // Creates a new LiteRtCompilationOptions object wrapped in a `unique_ptr`.
    static Expected<Options> Create() {
      LiteRtCompilationOptions options;
      if (auto status = LiteRtCreateCompilationOptions(&options);
          status != kLiteRtStatusOk) {
        return Error(status, "Could not create default compilation options");
      }
      return Options(options);
    }

    // Create a NULL pointer.
    static Options None() { return {}; }

    Ptr GetUnderlyingPtr() { return Ptr(release()); }

    Expected<void> SetHardwareAccelerators(
        LiteRtHwAcceleratorSet accelerators) {
      if (auto status = LiteRtSetCompilationOptionsHardwareAccelerators(
              get(), accelerators);
          status != kLiteRtStatusOk) {
        return Error(
            status,
            "Could not set hardware accelerators in compilation options");
      }
      return {};
    }

    // Mimic unique_ptr API.

    Ptr::pointer release() noexcept { return impl_.release(); }
    void reset(Ptr::pointer ptr = nullptr) noexcept { return impl_.reset(ptr); }

    Ptr::pointer get() const noexcept { return impl_.get(); }
    Ptr::pointer operator->() const noexcept { return impl_.operator->(); }
    auto& operator*() const noexcept { return impl_.operator*(); }

   private:
    Ptr impl_;
  };

  CompiledModel() = default;

  // Parameter `owned` indicates if the created CompiledModel object should take
  // ownership of the provided `compiled_model` handle.
  explicit CompiledModel(LiteRtModel litert_model,
                         LiteRtCompiledModel compiled_model, bool owned = true)
      : internal::Handle<LiteRtCompiledModel, LiteRtDestroyCompiledModel>(
            compiled_model, owned),
        model_(Model::CreateFromNonOwnedHandle(litert_model)) {}

  // Creates a CompiledModel from a TFLite file.
  // The model is loaded into memory and the caller takes ownership of the
  // returned CompiledModel object. The caller should keep the model alive
  // until the CompiledModel is destroyed.
  static Expected<CompiledModel> Create(litert::Environment& env,
                                        litert::Model& model,
                                        Options&& compilation_options) {
    LiteRtModel litert_model = model.Get();
    LiteRtCompiledModel compiled_model;
    if (auto status = LiteRtCreateCompiledModel(env.Get(), litert_model,
                                                compilation_options.release(),
                                                &compiled_model);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to create compiled model");
    }
    return CompiledModel(litert_model, compiled_model);
  }

  static Expected<CompiledModel> Create(
      litert::Environment& env, litert::Model& model,
      LiteRtHwAccelerators hardware_accelerator = kLiteRtHwAcceleratorCpu) {
    LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
    options.SetHardwareAccelerators(hardware_accelerator);
    return Create(env, model, std::move(options));
  }

  // Returns the buffer requirements for the given n-th input tensor. The
  // returned TensorBufferRequirements is used to create the input tensor
  // buffer.
  Expected<TensorBufferRequirements> GetInputBufferRequirements(
      size_t signature_index, size_t input_index) const {
    LiteRtTensorBufferRequirements buffer_requirements;
    if (auto status = LiteRtGetCompiledModelInputBufferRequirements(
            Get(), signature_index, input_index, &buffer_requirements);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to get input buffer requirements");
    }
    return TensorBufferRequirements(buffer_requirements, /*owned=*/false);
  }

  // The same as above except this function takes input tensor name.
  Expected<TensorBufferRequirements> GetInputBufferRequirements(
      size_t signature_index, absl::string_view input_name) const {
    auto signature = model_.GetSignature(signature_index);
    auto input_index = FindInputIndex(signature_index, input_name);
    if (!input_index) {
      return Unexpected(kLiteRtStatusErrorNotFound, "Failed to find input");
    }
    return GetInputBufferRequirements(signature_index, *input_index);
  }

  // Returns the buffer requirements for the given output tensor. The returned
  // TensorBufferRequirements is used to create the output tensor
  // buffer.
  Expected<TensorBufferRequirements> GetOutputBufferRequirements(
      size_t signature_index, size_t output_index) const {
    LiteRtTensorBufferRequirements buffer_requirements;
    if (auto status = LiteRtGetCompiledModelOutputBufferRequirements(
            Get(), signature_index, output_index, &buffer_requirements);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to get output buffer requirements");
    }
    return TensorBufferRequirements(buffer_requirements, /*owned=*/false);
  }

  // The same as above except this function takes output tensor name.
  Expected<TensorBufferRequirements> GetOutputBufferRequirements(
      size_t signature_index, absl::string_view output_name) const {
    auto signature = model_.GetSignature(signature_index);
    auto output_index = FindOutputIndex(signature_index, output_name);
    if (!output_index) {
      return Unexpected(kLiteRtStatusErrorNotFound, "Failed to find output");
    }
    return GetOutputBufferRequirements(signature_index, *output_index);
  }

  // Creates an input tensor buffer for the given signature and input name.
  Expected<TensorBuffer> CreateInputBuffer(absl::string_view signature_name,
                                           absl::string_view input_name) const {
    return CreateInputOutputBuffer(signature_name, input_name,
                                   /*is_input=*/true);
  }

  // Creates an output tensor buffer for the given signature and output name.
  Expected<TensorBuffer> CreateOutputBuffer(
      absl::string_view signature_name, absl::string_view output_name) const {
    return CreateInputOutputBuffer(signature_name, output_name,
                                   /*is_input=*/false);
  }

  // A helper function to creates the input tensor buffers for the given
  // signature. It uses BufferRequirements and RankedTensorType to create the
  // input tensor buffers.
  Expected<std::vector<TensorBuffer>> CreateInputBuffers(
      size_t signature_index) const {
    return CreateInputOutputBuffers(signature_index, /*is_input=*/true);
  }

  // A helper function to creates the output tensor buffers for the given
  // signature. It uses BufferRequirements and RankedTensorType to create the
  // output tensor buffers.
  Expected<std::vector<TensorBuffer>> CreateOutputBuffers(
      size_t signature_index) const {
    return CreateInputOutputBuffers(signature_index, /*is_input=*/false);
  }

  // Runs the model of the given signature index synchronously with the provided
  // input/output TensorBuffers.
  Expected<void> Run(size_t signature_index,
                     const std::vector<TensorBuffer>& input_buffers,
                     const std::vector<TensorBuffer>& output_buffers) const {
    bool async = false;
    return RunHelper(signature_index, input_buffers, output_buffers, async);
  }

  // Runs the model of the given signature index asynchronously, if possible,
  // with the provided input/output TensorBuffers. If asynchronous execution is
  // possible then the function returns true in parameter `async`; otherwise the
  // function runs the model synchronously.
  Expected<void> RunAsync(size_t signature_index,
                          const std::vector<TensorBuffer>& input_buffers,
                          const std::vector<TensorBuffer>& output_buffers,
                          bool& async) const {
    async = true;
    return RunHelper(signature_index, input_buffers, output_buffers, async);
  }

  // Runs the model of the given signature key synchronously with the provided
  // input/output TensorBuffer map.
  Expected<void> Run(
      absl::string_view signature_key,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map)
      const {
    bool async = false;
    return RunHelper(signature_key, input_map, output_map, async);
  }

  // Runs the model of the given signature key asynchronously, if possible, with
  // the provided input/output TensorBuffer map. If asynchronous execution is
  // possible then the function returns true in parameter `async`; otherwise the
  // function runs the model synchronously.
  Expected<void> RunAsync(
      absl::string_view signature_key,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
      bool& async) const {
    async = true;
    return RunHelper(signature_key, input_map, output_map, async);
  }

 private:
  // Returns the signature input index for the given input tensor name.
  Expected<size_t> FindInputIndex(size_t signature_index,
                                  absl::string_view input_name) const;

  // Returns the signature output index for the given output tensor name.
  Expected<size_t> FindOutputIndex(size_t signature_index,
                                   absl::string_view output_name) const;

  // Creates a TensorBuffer with the given buffer requirements and tensor type.
  static Expected<TensorBuffer> CreateBufferImpl(
      const TensorBufferRequirements& buffer_requirements,
      const RankedTensorType& tensor_type);

  // Creates a TensorBuffer for the given signature and tensor name.
  Expected<TensorBuffer> CreateInputOutputBuffer(
      absl::string_view signature_name, absl::string_view tensor_name,
      bool is_input) const;

  // Creates a vector of TensorBuffers for the given signature subgraph.
  Expected<std::vector<TensorBuffer>> CreateInputOutputBuffers(
      size_t signature_index, bool is_input) const;

  Expected<void> RunCApiHelper(LiteRtParamIndex signature_index,
                               size_t num_input_buffers,
                               LiteRtTensorBuffer* input_buffers,
                               size_t num_output_buffers,
                               LiteRtTensorBuffer* output_buffers,
                               bool& async) const;

  Expected<void> RunHelper(size_t signature_index,
                           const std::vector<TensorBuffer>& input_buffers,
                           const std::vector<TensorBuffer>& output_buffers,
                           bool& async) const;

  Expected<void> RunHelper(
      absl::string_view signature_key,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
      bool& async) const;

  Model model_;
};

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_COMPILED_MODEL_H_
