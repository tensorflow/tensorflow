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
#include "tensorflow/lite/experimental/delegates/hexagon/hexagon_delegate.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/utils.h"
#include "tensorflow/lite/experimental/delegates/hexagon/hexagon_delegate_kernel.h"
#include "tensorflow/lite/experimental/delegates/hexagon/hexagon_implementation.h"
#include "tensorflow/lite/experimental/delegates/hexagon/utils.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace {
// Should be > 0. > 16 causes problems.
constexpr int kMaxHexagonGraphs = 4;
constexpr int kMaxMaxHexagonGraphs = 16;
constexpr int kMinNodesPerHexagonGraph = 2;

TfLiteRegistration GetHexagonKernelRegistration() {
  // This is the registration for the Delegate Node that gets added to
  // the TFLite graph instead of the subGraph it replaces it.
  // It is treated as a an OP node. But in our case
  // Init will initialize the delegate
  // Invoke will run the delegate graph.
  // Prepare for prearing the delegate.
  // Free for any cleaning needed by the delegate.
  TfLiteRegistration kernel_registration;
  kernel_registration.profiling_string = nullptr;
  kernel_registration.builtin_code = kTfLiteBuiltinDelegate;
  kernel_registration.custom_name = "TfLiteHexagonDelegate";
  kernel_registration.free = [](TfLiteContext* context, void* buffer) -> void {
    delete reinterpret_cast<HexagonDelegateKernel*>(buffer);
  };
  kernel_registration.init = [](TfLiteContext* context, const char* buffer,
                                size_t length) -> void* {
    const TfLiteDelegateParams* params =
        reinterpret_cast<const TfLiteDelegateParams*>(buffer);
    auto hexagon_kernel = std::make_unique<HexagonDelegateKernel>();
    if (hexagon_kernel->Init(context, params) != kTfLiteOk) {
      return nullptr;
    }
    return hexagon_kernel.release();
  };
  kernel_registration.invoke = [](TfLiteContext* context,
                                  TfLiteNode* node) -> TfLiteStatus {
    HexagonDelegateKernel* kernel =
        reinterpret_cast<HexagonDelegateKernel*>(node->user_data);
    if (!kernel) {
      context->ReportError(context, "Hexagon Kernel was not initialized");
      return kTfLiteError;
    }
    return kernel->Invoke(context, node);
  };
  kernel_registration.prepare = [](TfLiteContext* context,
                                   TfLiteNode* node) -> TfLiteStatus {
    if (node->user_data == nullptr) {
      context->ReportError(context, "Hexagon Kernel was not initialized");
      return kTfLiteError;
    }
    HexagonDelegateKernel* kernel =
        reinterpret_cast<HexagonDelegateKernel*>(node->user_data);
    return kernel->Prepare(context, node);
  };

  return kernel_registration;
}

class HexagonDelegate : public TfLiteDelegate {
 public:
  explicit HexagonDelegate(const TfLiteHexagonDelegateOptions* params)
      : params_(params != nullptr ? *params
                                  : TfLiteHexagonDelegateOptions({0})) {
    if (params_.max_delegated_partitions <= 0) {
      params_.max_delegated_partitions = kMaxHexagonGraphs;
    } else if (params_.max_delegated_partitions > kMaxMaxHexagonGraphs) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                      "Hexagon delegate: cannot have this many %d partitions, "
                      "and will cap to at most %d partitions.\n",
                      params_.max_delegated_partitions, kMaxMaxHexagonGraphs);
      params_.max_delegated_partitions = kMaxMaxHexagonGraphs;
    }
    if (params_.min_nodes_per_partition <= 0) {
      params_.min_nodes_per_partition = kMinNodesPerHexagonGraph;
    }
  }

  TfLiteHexagonDelegateOptions* params() { return &params_; }

  bool VerifyDelegate() {
    auto* hexagon_nn = HexagonNNImplementation();
    if (hexagon_nn == nullptr) {
      return false;
    }
    if (hexagon_nn->hexagon_nn_version != nullptr &&
        hexagon_nn->hexagon_nn_hexagon_interface_version) {
      int hexagon_nn_version = -1;
      int hexagon_interface_version =
          hexagon_nn->hexagon_nn_hexagon_interface_version();
      if (hexagon_nn->hexagon_nn_version(&hexagon_nn_version) != 0) {
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                        "Failed to fetch Hexagon NN version. This might be "
                        "because you're using incompatible versions of "
                        "libhexagon_interface and libhexagon_nn_skel. "
                        "You must use compatible versions. "
                        "Refer to Tensorflow Lite Hexagon Delegate Guide.");
        return false;
      }
      if (hexagon_nn_version != hexagon_interface_version) {
        TFLITE_LOG_PROD(
            tflite::TFLITE_LOG_WARNING,
            "Incompatible versions between interface library and "
            "libhexagon_skel %d vs %d. You must use compatible versions. "
            "Refer to Tensorflow Lite Hexagon Delegate Guide.",
            hexagon_interface_version, hexagon_nn_version);
        return false;
      }
    }
    return hexagon_nn->hexagon_nn_is_device_supported &&
           hexagon_nn->hexagon_nn_is_device_supported();
  }

  ~HexagonDelegate() {
    TfLiteIntArrayFree(params_.input_batch_dimensions);
    TfLiteIntArrayFree(params_.output_batch_dimensions);
  }

 private:
  TfLiteHexagonDelegateOptions params_;
};

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  delegates::IsNodeSupportedFn node_supported_fn =
      [=](TfLiteContext* context, TfLiteNode* node,
          TfLiteRegistration* registration,
          std::string* unsupported_details) -> bool {
    return IsNodeSupportedByHexagon(registration, node, context);
  };
  delegates::GraphPartitionHelper helper(context, node_supported_fn);
  TF_LITE_ENSURE_STATUS(helper.Partition(nullptr));

  TfLiteHexagonDelegateOptions* params =
      static_cast<TfLiteHexagonDelegateOptions*>(delegate->data_);
  const auto delegate_partitions = helper.GetFirstNLargestPartitions(
      params->max_delegated_partitions, params->min_nodes_per_partition);

  // To avoid creating a new TfLiteIntArray and free it later, we reserve one
  // element to represent TfLiteIntArray.size which is the 1st element of
  // TfLiteIntArray C struct.
  std::vector<int> supported_nodes(1);
  for (const auto partition : delegate_partitions) {
    auto* nodes = partition->nodes_to_replace;
    supported_nodes.insert(supported_nodes.end(), nodes->data,
                           nodes->data + nodes->size);
  }
  // Set first element to the number of nodes to replace.
  supported_nodes[0] = supported_nodes.size() - 1;
  auto* hexagon_delegate = static_cast<HexagonDelegate*>(delegate);
  // Make sure dynamic batch is requested on fully delegated graph only.
  if (supported_nodes[0] != helper.num_total_nodes() &&
      hexagon_delegate != nullptr &&
      hexagon_delegate->params()->enable_dynamic_batch_size) {
    TF_LITE_KERNEL_LOG(
        context, "Dynamic batch requested on non-fully delegated graph !!.");
    return kTfLiteError;
  }
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                  "Hexagon delegate: %d nodes delegated out of %d nodes with "
                  "%d partitions.\n",
                  supported_nodes[0], helper.num_total_nodes(),
                  delegate_partitions.size());

  return context->ReplaceNodeSubsetsWithDelegateKernels(
      context, GetHexagonKernelRegistration(),
      reinterpret_cast<TfLiteIntArray*>(supported_nodes.data()), delegate);
}

TfLiteDelegate* CreateDelegate(const TfLiteHexagonDelegateOptions* params) {
  TfLiteDelegate* delegate = new HexagonDelegate(params);
  if (!static_cast<HexagonDelegate*>(delegate)->VerifyDelegate()) {
    delete delegate;
    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                         "Hexagon Delegate is not supported.\n");
    return nullptr;
  }

  delegate->data_ = static_cast<HexagonDelegate*>(delegate)->params();
  delegate->flags = kTfLiteDelegateFlagsAllowDynamicTensors;
  delegate->Prepare = &DelegatePrepare;
  delegate->CopyFromBufferHandle = nullptr;
  delegate->CopyToBufferHandle = nullptr;
  delegate->FreeBufferHandle = nullptr;

  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                       "Created TensorFlow Lite delegate for Hexagon.");

  return delegate;
}

}  // namespace
}  // namespace tflite

TfLiteDelegate* TfLiteHexagonDelegateCreate(
    const TfLiteHexagonDelegateOptions* options) {
  return tflite::CreateDelegate(options);
}

void TfLiteHexagonDelegateDelete(TfLiteDelegate* delegate) { delete delegate; }

void TfLiteHexagonInit() { tflite::HexagonDelegateKernel::InitState(); }

void TfLiteHexagonInitWithPath(const char* lib_directory_path) {
  if (lib_directory_path != nullptr) {
    std::string env_var_value = lib_directory_path;
    env_var_value += ";/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp";
    setenv("ADSP_LIBRARY_PATH", env_var_value.c_str(), 1 /* overwrite */);
  }
  tflite::HexagonDelegateKernel::InitState();
}
void TfLiteHexagonTearDown() { tflite::HexagonDelegateKernel::Teardown(); }
