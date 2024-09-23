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

#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "Eigen/Core"  // from @eigen_archive
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "pthreadpool.h"  // from @pthreadpool
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/delegates/xnnpack/file_util.h"
#include "tensorflow/lite/delegates/xnnpack/flexbuffers_util.h"
#include "tensorflow/lite/delegates/xnnpack/quantization_util.h"
#include "tensorflow/lite/delegates/xnnpack/weight_cache.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/utils/sparsity_format_converter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/optimize/reduced_precision_support.h"

struct TfLiteXNNPackDelegateWeightsCache;

namespace tflite {
namespace xnnpack {
namespace {

constexpr char kOdmlSDPA[] = "odml.scaled_dot_product_attention";

template <typename T>
void SafeCopyCustomData(const TfLiteNode& node, T* target) {
  const size_t safe_size =
      std::min(static_cast<size_t>(node.custom_initial_data_size), sizeof(T));
  std::memcpy(target, node.custom_initial_data, safe_size);
}

void CopyTensorDataInt32OrInt64(int64_t* dst, const TfLiteTensor& tensor,
                                size_t n) {
  if (tensor.type == kTfLiteInt32) {
    const int32_t* data = GetTensorData<int32_t>(&tensor);
    std::copy(data, data + n, dst);
  } else if (tensor.type == kTfLiteInt64) {
    const int64_t* data = GetTensorData<int64_t>(&tensor);
    std::copy(data, data + n, dst);
  }
}

xnn_datatype GetXNNPackDatatype(TfLiteContext* context,
                                const TfLiteTensor& tensor, int t) {
  switch (tensor.type) {
    case kTfLiteFloat32:
      return xnn_datatype_fp32;
    case kTfLiteFloat16:
      return xnn_datatype_fp16;
    case kTfLiteUInt8: {
      if (tensor.quantization.type != kTfLiteAffineQuantization) {
        TF_LITE_KERNEL_LOG(context,
                           "unsupported quantization type %d for UINT8 "
                           "tensor %d in XNNPACK delegate",
                           tensor.quantization.type, t);
        return xnn_datatype_invalid;
      }
      const auto quantization_params =
          static_cast<const TfLiteAffineQuantization*>(
              tensor.quantization.params);
      if (quantization_params->scale == nullptr) {
        TF_LITE_KERNEL_LOG(context,
                           "missing scale quantization parameters for UINT8 "
                           "tensor %d in XNNPACK delegate",
                           t);
        return xnn_datatype_invalid;
      }
      if (quantization_params->zero_point == nullptr) {
        TF_LITE_KERNEL_LOG(context,
                           "missing zero point quantization parameters for "
                           "UINT8 tensor %d in XNNPACK delegate",
                           t);
        return xnn_datatype_invalid;
      }
      if (quantization_params->scale->size != 1) {
        TF_LITE_KERNEL_LOG(
            context,
            "unsupported number (%d) of scale quantization parameters for "
            "UINT8 tensor %d in XNNPACK delegate",
            quantization_params->scale->size, t);
        return xnn_datatype_invalid;
      }
      if (quantization_params->zero_point->size != 1) {
        TF_LITE_KERNEL_LOG(
            context,
            "unsupported number (%d) of zero point quantization parameters "
            "for UINT8 tensor %d in XNNPACK delegate",
            quantization_params->zero_point->size, t);
        return xnn_datatype_invalid;
      }

      const float scale = quantization_params->scale->data[0];
      if (!std::isnormal(scale) || scale <= 0.0f) {
        TF_LITE_KERNEL_LOG(context,
                           "unsupported scale value (%f) for UINT8 tensor %d "
                           "in XNNPACK delegate",
                           scale, t);
        return xnn_datatype_invalid;
      }

      const int zero_point = quantization_params->zero_point->data[0];
      if (zero_point < std::numeric_limits<uint8_t>::min() ||
          zero_point > std::numeric_limits<uint8_t>::max()) {
        TF_LITE_KERNEL_LOG(context,
                           "unsupported zero-point value (%d) for UINT8 tensor "
                           "%d in XNNPACK delegate",
                           zero_point, t);
        return xnn_datatype_invalid;
      }

      return xnn_datatype_quint8;
    }
    case kTfLiteInt8:
    case kTfLiteInt4: {
      if (tensor.quantization.type != kTfLiteAffineQuantization) {
        TF_LITE_KERNEL_LOG(context,
                           "unsupported quantization type %d for %s "
                           "tensor %d in XNNPACK delegate",
                           tensor.quantization.type,
                           TfLiteTypeGetName(tensor.type), t);
        return xnn_datatype_invalid;
      }
      const auto quantization_params =
          static_cast<const TfLiteAffineQuantization*>(
              tensor.quantization.params);
      if (quantization_params->scale == nullptr) {
        TF_LITE_KERNEL_LOG(context,
                           "missing scale quantization parameters for %s "
                           "tensor %d in XNNPACK delegate",
                           TfLiteTypeGetName(tensor.type), t);
        return xnn_datatype_invalid;
      }
      if (quantization_params->zero_point == nullptr) {
        TF_LITE_KERNEL_LOG(context,
                           "missing zero point quantization parameters for "
                           "%s tensor %d in XNNPACK delegate",
                           TfLiteTypeGetName(tensor.type), t);
        return xnn_datatype_invalid;
      }
      if (quantization_params->scale->size !=
          quantization_params->zero_point->size) {
        TF_LITE_KERNEL_LOG(context,
                           "mismatching number of scale (%d) and zero "
                           "point (%d) quantization parameters for %s "
                           "tensor %d in XNNPACK delegate",
                           quantization_params->scale->size,
                           quantization_params->zero_point->size,
                           TfLiteTypeGetName(tensor.type), t);
        return xnn_datatype_invalid;
      }

      for (int i = 0; i < quantization_params->scale->size; i++) {
        const float scale = quantization_params->scale->data[i];
        if (!std::isnormal(scale) || scale <= 0.0f) {
          TF_LITE_KERNEL_LOG(context,
                             "unsupported scale value (%f) in channel %d for "
                             "%s tensor %d in XNNPACK delegate",
                             scale, i, TfLiteTypeGetName(tensor.type), t);
          return xnn_datatype_invalid;
        }
      }

      if (quantization_params->scale->size == 1) {
        // Per-tensor quantization parameters
        if (kTfLiteInt8 != tensor.type) {
          TF_LITE_KERNEL_LOG(context,
                             "unsupported per-tensor quantization scale "
                             "parameter for %s tensor %d in XNNPACK delegate",
                             TfLiteTypeGetName(tensor.type), t);
          return xnn_datatype_invalid;
        }

        const int zero_point = quantization_params->zero_point->data[0];
        if (zero_point < std::numeric_limits<int8_t>::min() ||
            zero_point > std::numeric_limits<int8_t>::max()) {
          TF_LITE_KERNEL_LOG(context,
                             "unsupported zero-point value (%d) for INT8 "
                             "tensor %d in XNNPACK delegate",
                             zero_point, t);
          return xnn_datatype_invalid;
        }
        return xnn_datatype_qint8;
      } else if (NumDimensions(&tensor) >= 1 &&
                 quantization_params->scale->size ==
                     SizeOfDimension(
                         &tensor, quantization_params->quantized_dimension)) {
        // Per-channel quantization parameters
        for (int c = 0;
             c <
             SizeOfDimension(&tensor, quantization_params->quantized_dimension);
             c++) {
          if (quantization_params->zero_point->data[c] != 0 &&
              (tensor.type != kTfLiteInt4 &&
               quantization_params->zero_point->data[c] != 8)) {
            TF_LITE_KERNEL_LOG(context,
                               "unsupported zero-point value %d in channel "
                               "%d of %s tensor %d in XNNPACK delegate",
                               quantization_params->zero_point->data[c], c,
                               TfLiteTypeGetName(tensor.type), t);
            return xnn_datatype_invalid;
          }
        }
        switch (tensor.type) {
          case kTfLiteInt4:
            return xnn_datatype_qcint4;
          case kTfLiteInt8:
            return xnn_datatype_qcint8;
          default:
            return xnn_datatype_invalid;
        }
      } else {
        TF_LITE_KERNEL_LOG(
            context,
            "mismatching number of quantization parameters %d and outer "
            "dimension %d for INT8 tensor %d in XNNPACK delegate",
            quantization_params->scale->size,
            SizeOfDimension(&tensor, quantization_params->quantized_dimension),
            t);
        return xnn_datatype_invalid;
      }
      break;
    }
    case kTfLiteInt32: {
      if (tensor.quantization.type != kTfLiteAffineQuantization) {
        TF_LITE_KERNEL_LOG(context,
                           "unsupported quantization type %d for INT32 "
                           "tensor %d in XNNPACK delegate",
                           tensor.quantization.type, t);
        return xnn_datatype_invalid;
      }
      const auto quantization_params =
          static_cast<const TfLiteAffineQuantization*>(
              tensor.quantization.params);
      if (quantization_params->scale == nullptr) {
        TF_LITE_KERNEL_LOG(context,
                           "missing scale quantization parameters for "
                           "INT32 tensor %d in XNNPACK delegate",
                           t);
        return xnn_datatype_invalid;
      }
      if (quantization_params->zero_point == nullptr) {
        TF_LITE_KERNEL_LOG(context,
                           "missing zero point quantization parameters for "
                           "INT32 tensor %d in XNNPACK delegate",
                           t);
        return xnn_datatype_invalid;
      }
      if (quantization_params->scale->size !=
          quantization_params->zero_point->size) {
        TF_LITE_KERNEL_LOG(context,
                           "mismatching number of scale (%d) and zero "
                           "point (%d) quantization parameters for INT32 "
                           "tensor %d in XNNPACK delegate",
                           quantization_params->scale->size,
                           quantization_params->zero_point->size, t);
        return xnn_datatype_invalid;
      }
      if (quantization_params->quantized_dimension != 0) {
        TF_LITE_KERNEL_LOG(context,
                           "unsupported quantized dimension %d for INT32 "
                           "tensor %d in XNNPACK delegate",
                           quantization_params->quantized_dimension, t);
        return xnn_datatype_invalid;
      }
      if (quantization_params->scale->size == 1) {
        // Per-tensor quantization parameters
        if (quantization_params->zero_point->data[0] != 0) {
          TF_LITE_KERNEL_LOG(context,
                             "unsupported zero-point value %d for INT32 "
                             "tensor %d in XNNPACK delegate",
                             quantization_params->zero_point->data[0], t);
          return xnn_datatype_invalid;
        }
        return xnn_datatype_qint32;
      } else if (NumDimensions(&tensor) >= 1 &&
                 quantization_params->scale->size ==
                     SizeOfDimension(&tensor, 0)) {
        // Per-channel quantization parameters
        for (int c = 0; c < SizeOfDimension(&tensor, 0); c++) {
          if (quantization_params->zero_point->data[c] != 0) {
            TF_LITE_KERNEL_LOG(context,
                               "unsupported zero-point value %d in channel "
                               "%d of INT32 tensor %d in XNNPACK delegate",
                               quantization_params->zero_point->data[c], c, t);
            return xnn_datatype_invalid;
          }
        }
        return xnn_datatype_qcint32;
      } else {
        TF_LITE_KERNEL_LOG(
            context,
            "mismatching number of quantization parameters %d and outer "
            "dimension %d for INT32 tensor %d in XNNPACK delegate",
            quantization_params->scale->size, SizeOfDimension(&tensor, 0), t);
        return xnn_datatype_invalid;
      }
      break;
    }
    default:
      break;
  }
  return xnn_datatype_invalid;
}

std::vector<size_t> TfLiteDimensionsToXNNPackDimensions(
    const std::vector<int>& tflite_dims) {
  std::vector<size_t> dims(tflite_dims.size());
  for (size_t i = 0; i < tflite_dims.size(); i++) {
    dims[i] = static_cast<size_t>(tflite_dims[i]);
  }
  return dims;
}

// Forward declaration.
TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate);

// hash_combine from smhasher/boost.
template <typename T>
inline void hash_combine(size_t seed, T v) {
  seed ^= std::hash<T>{}(v) + 0x9e3779b9U + (seed << 6) + (seed >> 2);
}

struct PairHash {
  std::size_t operator()(const std::pair<std::string, std::string>& s) const {
    size_t seed = 0;
    hash_combine(seed, s.first);
    hash_combine(seed, s.second);
    return seed;
  }
};

// Variable tensors are tensors that can are persistent across graph
// invocations. A handle to a variable tensor is given by the VAR_HANDLE
// operation, the result of this operation is a tensor of type kTfLiteResource,
// which represents the name/id of a variable tensor. READ_VARIABLE (RV) and
// ASSIGN_VARIABLE (AV) access variable tensors using the result of VAR_HANDLE.
// XNNPACK does not materialize any resource tensor. In order for RV/AV to know
// which variable tensor it is accessing, we track:
// - the name in each VAR_HANDLE node,
// - the output tensor of VAR_HANDLE in each Subgraph
// - the input tensor of RV/AV in each Subgraph
// and match these up.
// Each unique name is given a "global variable id". The output tensor of
// VAR_HANDLE is mapped to this global variable id using its name.
// Then RV/AV's input resource tensor id is used to lookup the global variable
// id, and using that we get a pointer to the underlying buffer.
// This is performed in two pass because:
// - XNNPACK requires tensor declaration upfront and the dimensions are fixed
// - VAR_HANDLE node has no dimensions information, only RV/AV has it
// The two passes are:
// - PrepareOpsToDelegate will record a mapping of variable name to the global
// variable id and also record the dimensions based on RV/AV. This is called per
// subgraph in the model.
// - Subgraph::Create will actually define the tensors. This is called per
// subgraph in the model.
class VariableHolder {
 public:
  // Defines a variable with a given name. This variable is given a global
  // variable id. The global id is also associated with a subgraph-local
  // tensor id. tensor_id should be the id of a VAR_HANDLE output tensor.
  TfLiteStatus DefineVariable(const std::pair<std::string, std::string>& name,
                              int tensor_id, TfLiteContext* logging_context,
                              int node_index) {
    const auto variable_name_global_id_it = variable_name_to_global_id_.insert(
        {name, variable_name_to_global_id_.size()});
    const uint32_t global_id = variable_name_global_id_it.first->second;

    const auto it = tensor_id_to_global_id_.insert({tensor_id, global_id});
    if (!it.second && global_id != it.first->second) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "global id mismatch for tensor "
          "%d, expected %u, found %u at VAR_HANDLE node %d",
          tensor_id, global_id, it.first->second, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  // Get the global variable id associated with this local tensor.
  uint32_t GetGlobalId(int local) const {
    return tensor_id_to_global_id_.at(local);
  }

  bool HasLocalId(int local) const {
    return tensor_id_to_global_id_.count(local) != 0;
  }

  // Variable tensors don't have dimensions or type, because VAR_HANDLE don't
  // have that information. When a node (READ_VARIABLE or ASSIGN_VARIABLE) uses
  // a variable, we associate the variable tensor with an underlying TFLite
  // tensor via its global_id.
  TfLiteStatus AssociateVariableWithTensor(int local_id,
                                           const TfLiteTensor* tensor,
                                           TfLiteContext* logging_context) {
    if (tensor->type != kTfLiteFloat32 && tensor->type != kTfLiteInt8 &&
        tensor->type != kTfLiteUInt8) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "failed to associate variable tensors with tensor %d: "
          "only kTfLiteFloat32, kTfLiteInt8, or kTfLiteUint8 variable tensors "
          "are supported",
          local_id);
      return kTfLiteError;
    }

    const uint32_t global_id = GetGlobalId(local_id);
    const std::vector<int> dims(tensor->dims->data,
                                tensor->dims->data + tensor->dims->size);
    const auto it =
        global_id_to_dims_and_type_.insert(std::make_pair(global_id, tensor));
    if (!it.second) {
      // Not inserted.
      const TfLiteTensor* found_tensor = it.first->second;
      if (found_tensor->type != tensor->type) {
        // Make sure that existing type matches.
        TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                                 "mismatch between existing type of "
                                 "variable tensor id %d: expected %d, got %d",
                                 local_id, tensor->type, found_tensor->type);
        return kTfLiteError;
      }
      auto const& dims = found_tensor->dims;
      for (size_t i = 0; i < NumDimensions(found_tensor); i++) {
        if (found_tensor->dims->data[i] != tensor->dims->data[i]) {
          TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                                   "mismatch between dimension %zu of "
                                   "variable tensor id %d: expected %d, got %d",
                                   i, local_id, dims->data[i],
                                   tensor->dims->data[i]);
          return kTfLiteError;
        }
      }
    }
    return kTfLiteOk;
  }

  const std::map<uint32_t, const TfLiteTensor*>& GetAllTensors() const {
    return global_id_to_dims_and_type_;
  }

  // Global ids are per-delegate/per-model. However, the mapping from local
  // tensor id to global id is per-subgraph. This functions allows clearing
  // this mapping, which should be called at the start of visiting each
  // subgraph in the model.
  void ClearTensorIdToGlobalId() { tensor_id_to_global_id_.clear(); }

 private:
  std::unordered_map<std::pair<std::string, std::string>, uint32_t, PairHash>
      variable_name_to_global_id_;
  std::unordered_map<int, uint32_t> tensor_id_to_global_id_;
  // Variable tensors need to be defined in the same order across all XNNPACK
  // subgraphs, so we want the global ids to be ordered.
  std::map<uint32_t, const TfLiteTensor*> global_id_to_dims_and_type_;
};

class Subgraph;

class Delegate {
  friend class Subgraph;

 public:
  explicit Delegate(const TfLiteXNNPackDelegateOptions* options,
                    xnn_workspace_t workspace,
                    TfLiteContext* context = nullptr) {
#if !defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__)
    pthreadpool_t threadpool = nullptr;
#ifdef TFLITE_KERNEL_USE_XNNPACK
    if (context != nullptr) {
      threadpool =
          CpuBackendContext::GetFromContext(context)->get_xnnpack_threadpool();
    }
#endif
    if (threadpool != nullptr) {
      // Note that by passing a valid threadpool via context, your xnnpack
      // threadpool will have the same number of threads as
      // CpuBackendContext::max_num_threads_. If this is not desired behavior,
      // pass a null threadpool, and then set num_threads through
      // TfLiteXNNPackDelegateOptions.
      threadpool_.reset(threadpool);
      own_threadpool_ = false;
    } else {
      own_threadpool_ = true;
      if (options != nullptr && options->num_threads > 1) {
        threadpool_.reset(
            pthreadpool_create(static_cast<size_t>(options->num_threads)));
        threadpool = threadpool_.get();
      }
    }

#endif
    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                         "Created TensorFlow Lite XNNPACK delegate for CPU.");

    options_ =
        options != nullptr ? *options : TfLiteXNNPackDelegateOptionsDefault();
    delegate_.flags = GetXNNPackDelegateFlags();
    workspace_.reset(workspace);

    // If no weight cache is provided, add one when requested.
    if (!options_.weights_cache) {
      if (options_.weight_cache_file_path) {
        if (weight_cache_provider_.LoadOrStartBuild(
                options_.weight_cache_file_path)) {
          options_.weights_cache =
              reinterpret_cast<TfLiteXNNPackDelegateWeightsCache*>(
                  weight_cache_provider_.GetCacheProvider().context);
          options_.weight_cache_file_path =
              weight_cache_provider_.GetFilePath().data();
        } else {
          TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                          "XNNPack weight cache could neither be loaded from "
                          "or saved to '%s'. Check that this location is "
                          "readable and writable.",
                          options_.weight_cache_file_path);
          options_.weight_cache_file_path = nullptr;
        }
      } else {
        TFLITE_LOG(tflite::TFLITE_LOG_VERBOSE,
                   "XNNPack weight cache not enabled.");
      }
    }
  }

  TfLiteIntArray* PrepareOpsToDelegate(TfLiteContext* context);
  TfLiteDelegate* tflite_delegate() { return &delegate_; }

  bool support_signed_8bit_quantization() const {
    return (options_.flags & TFLITE_XNNPACK_DELEGATE_FLAG_QS8) != 0;
  }

  bool support_unsigned_8bit_quantization() const {
    return (options_.flags & TFLITE_XNNPACK_DELEGATE_FLAG_QU8) != 0;
  }

  bool support_any_8bit_quantization() const {
    return (options_.flags & (TFLITE_XNNPACK_DELEGATE_FLAG_QU8 |
                              TFLITE_XNNPACK_DELEGATE_FLAG_QS8)) != 0;
  }

  bool support_dynamic_fully_connected_operator() const {
    return (options_.flags &
            TFLITE_XNNPACK_DELEGATE_FLAG_DYNAMIC_FULLY_CONNECTED) != 0;
  }

  bool force_fp16() const {
#ifdef XNNPACK_DELEGATE_FORCE_PRECISION_FP16
    return true;
#else
    return (options_.flags & TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16) != 0;
#endif
  }

  bool enable_latest_operators() const {
#ifdef XNNPACK_DELEGATE_USE_LATEST_OPS
    return true;
#else
    return (options_.flags &
            TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS) != 0;
#endif
  }

  bool enable_subgraph_reshaping() const {
#ifdef XNNPACK_DELEGATE_ENABLE_SUBGRAPH_RESHAPING
    return true;
#else
    return (options_.flags &
            TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_SUBGRAPH_RESHAPING) != 0;
#endif
  }

  bool support_variable_ops() const {
    if (options_.flags & TFLITE_XNNPACK_DELEGATE_FLAG_VARIABLE_OPERATORS) {
      return true;
    } else if (options_.handle_variable_ops) {
      TFLITE_LOG_PROD_ONCE(
          tflite::TFLITE_LOG_ERROR,
          "TfLiteXNNPackDelegateOptions::handle_variable_ops "
          "is deprecated and will be removed in the future. "
          "Use TfLiteXNNPackDelegateOptions::flags with "
          "TFLITE_XNNPACK_DELEGATE_FLAG_VARIABLE_OPERATORS mask");
      return true;
    }
    return false;
  }

  bool transient_indirection_buffer() const {
#ifdef XNNPACK_DELEGATE_USE_TRANSIENT_INDIRECTION_BUFFERS
    return true;
#else
    return (options_.flags &
            TFLITE_XNNPACK_DELEGATE_FLAG_TRANSIENT_INDIRECTION_BUFFER) != 0;
#endif
  }

  pthreadpool_t threadpool() const {
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    return nullptr;
#else
    return threadpool_.get();
#endif
  }

  xnn_weights_cache_t weights_cache() const {
    if (options_.weights_cache == nullptr) {
      return nullptr;
    } else {
      return reinterpret_cast<xnn_weights_cache_t>(options_.weights_cache);
    }
  }

  xnn_workspace_t workspace() const { return workspace_.get(); }

  TfLiteStatus AssociateVariableWithTensor(int local_id,
                                           const TfLiteTensor* tensor,
                                           TfLiteContext* logging_context) {
    return variable_holder_.AssociateVariableWithTensor(local_id, tensor,
                                                        logging_context);
  }

  TfLiteStatus DefineVariable(const TfLiteVarHandleParams* params, int local,
                              TfLiteContext* logging_context, int node_index) {
    const std::pair<std::string, std::string> name = std::make_pair(
        std::string(params->container ? params->container : ""),
        std::string(params->shared_name ? params->shared_name : ""));
    return variable_holder_.DefineVariable(name, local, logging_context,
                                           node_index);
  }

  uint32_t GetGlobalId(int local) const {
    return variable_holder_.GetGlobalId(local);
  }

  uint32_t HasLocalId(int local) const {
    return variable_holder_.HasLocalId(local);
  }

  const std::map<uint32_t, const TfLiteTensor*>& GetAllVariableTensors() const {
    return variable_holder_.GetAllTensors();
  }

  bool CanVariableBeDelegated(int local_id) const {
    if (HasLocalId(local_id)) {
      return GetAllVariableTensors().count(GetGlobalId(local_id)) != 0;
    } else {
      return false;
    }
  }

  void maybe_release_threadpool_ownership() {
#if !defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__)
    if (!own_threadpool_) {
      threadpool_.release();
    }
#endif
  }

  const TfLiteXNNPackDelegateOptions& options() const { return options_; }

  int64_t GetXNNPackDelegateFlags() {
    if (enable_subgraph_reshaping()) {
      return kTfLiteDelegateFlagsPerOperatorProfiling |
             kTfLiteDelegateFlagsAllowDynamicTensors;
    } else {
      return kTfLiteDelegateFlagsPerOperatorProfiling;
    }
  }

 private:
  TfLiteDelegate delegate_ = {
      reinterpret_cast<void*>(this),  // .data_
      DelegatePrepare,                // .Prepare
      nullptr,                        // .CopyFromBufferHandle
      nullptr,                        // .CopyToBufferHandle
      nullptr,                        // .FreeBufferHandle
      0,                              // GetXNNPackDelegateFlags(),
  };

  // Unpacked data for quasi-static tensors, i.e. tensors produced by
  // dequantizing or unpacking static buffers.
  std::vector<char> static_unpacked_data_;
  // Mapping from a tensor index for a quasi-static tensor to the offset to
  // its unpacked data within static_unpacked_data_.
  std::unordered_map<int, size_t> static_unpacked_data_map_;
  // Set of indices of nodes which unpack static data, e.g. Dequantize
  // operators which convert FP16 static weights to FP32. These nodes are simply
  // ignored in the delegate implementation, because their outputs are
  // pre-unpacked in DelegatePrepare.
  std::unordered_set<int> static_unpack_nodes_;
  // Set of indices of tensors with unpacked static sparse weights.
  std::unordered_set<int> static_sparse_weights_;
#if !defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__)
  // Thread pool with smart-pointer for lifetime management.
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool_{
      nullptr, &pthreadpool_destroy};
  // Boolean that indicates if threadpool_ was created by xnnpack_delegate.
  bool own_threadpool_;
#endif
  std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)> workspace_{
      nullptr, &xnn_release_workspace};

  TfLiteXNNPackDelegateOptions options_{};
  VariableHolder variable_holder_;
  std::mutex workspace_mutex_;

  // If no weight cache is provided and a cache is set in the delegate options,
  // this will be used as a weight cache.
  MMapWeightCacheProvider weight_cache_provider_;
};

class Subgraph {
 public:
  // Defines all variable tensors in this subgraph. global_id_to_xnnpack_id is
  // updated to record mapping from global variable tensor id to XNNPACK value
  // id.
  static TfLiteStatus DefineVariableTensors(
      const Delegate& delegate, xnn_subgraph_t subgraph, TfLiteContext* context,
      std::unordered_map<uint32_t, uint32_t>& global_id_to_xnnpack_id) {
    for (auto const& it : delegate.GetAllVariableTensors()) {
      const xnn_datatype datatype =
          GetXNNPackDatatype(context, *it.second, it.first);
      if (datatype == xnn_datatype_invalid) {
        TF_LITE_KERNEL_LOG(
            context,
            "unsupported datatype (%s) of tensor %d in XNNPACK delegate",
            TfLiteTypeGetName(it.second->type), it.first);
        return kTfLiteError;
      }

      uint32_t out_id = XNN_INVALID_VALUE_ID;

      const std::vector<size_t> dims(
          &it.second->dims->data[0],
          &it.second->dims->data[NumDimensions(it.second)]);
      xnn_status status{};
      switch (datatype) {
        case xnn_datatype_qint8:
        case xnn_datatype_quint8: {
          auto quantization = static_cast<const TfLiteAffineQuantization*>(
              it.second->quantization.params);
          status = xnn_define_quantized_tensor_value(
              subgraph, datatype, quantization->zero_point->data[0],
              quantization->scale->data[0], dims.size(), dims.data(), nullptr,
              XNN_INVALID_VALUE_ID, XNN_VALUE_FLAG_PERSISTENT, &out_id);
          break;
        }
        case xnn_datatype_qint32:
        case xnn_datatype_qcint8:
        case xnn_datatype_qcint32: {
          // These are used for bias for Convolution/Fully Connected ops, and
          // are always static.
          TF_LITE_KERNEL_LOG(
              context,
              "unsupported datatype (%s) of tensor %d in XNNPACK delegate",
              TfLiteTypeGetName(it.second->type), it.first);
          return kTfLiteError;
        }
        default:
          status = xnn_define_tensor_value(
              subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr,
              XNN_INVALID_VALUE_ID, XNN_VALUE_FLAG_PERSISTENT, &out_id);
          break;
      }

      if (out_id == XNN_INVALID_VALUE_ID || status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(context,
                           "failed to define tensor for variable global id %d",
                           it.first);
        return kTfLiteError;
      }
      global_id_to_xnnpack_id[it.first] = out_id;
    }
    return kTfLiteOk;
  }

  static Subgraph* Create(TfLiteContext* context,
                          const TfLiteDelegateParams* params,
                          Delegate& delegate) {
    // Map tensors identifiers before packing anything.
    if (delegate.weight_cache_provider_.IsActive()) {
      delegate.weight_cache_provider_.MapTensorIdentifiers(
          context->tensors, context->tensors_size,
          reinterpret_cast<tflite::Subgraph*>(context->impl_)
              ->GetTensorBufferIdentifiers());
    }
    // Convert subgraph inputs and outputs to hash sets for faster lookup.
    const std::unordered_set<int> inputs(
        &params->input_tensors->data[0],
        &params->input_tensors->data[params->input_tensors->size]);
    std::unordered_set<int> outputs;
    for (int o = 0; o < params->output_tensors->size; o++) {
      const int output_tensor_idx = params->output_tensors->data[o];
      // Exclude quasi-static tensors and shared variable tensors which may have
      // become subgraph outputs after partitioning.
      if (delegate.static_unpacked_data_map_.count(output_tensor_idx) == 0 &&
          context->tensors[output_tensor_idx].type != kTfLiteResource) {
        outputs.insert(output_tensor_idx);
      }
    }
    std::unordered_set<int> externals(outputs);

    TfLiteIntArray* execution_plan;
    if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
      return nullptr;
    }

    bool has_sparse_weights = false;
    // Detect which tensors are used as inputs or outputs of any subgraph nodes.
    // -1 denotes tensor not used in the subgraph. These indexes will be
    // filtered out and removed later.
    std::vector<int> tensors(context->tensors_size, -1);
    for (int i = 0; i < params->nodes_to_replace->size; i++) {
      const int node_index = params->nodes_to_replace->data[i];

      TfLiteNode* node = nullptr;
      TfLiteRegistration* registration = nullptr;
      if (context->GetNodeAndRegistration(context, node_index, &node,
                                          &registration) != kTfLiteOk) {
        return nullptr;
      }

      // Detect if any of the node's inputs are sparse weights.
      if (!has_sparse_weights) {
        for (int i = 0; i < node->inputs->size; i++) {
          if (delegate.static_sparse_weights_.count(node->inputs->data[i]) !=
              0) {
            has_sparse_weights = true;
          }
        }
      }

      if (delegate.static_unpack_nodes_.count(node_index) != 0) {
        // The node unpacks static input and can be skipped because its input
        // was pre-unpacked in DelegatePrepare.
        continue;
      }

      switch (registration->builtin_code) {
        case kTfLiteBuiltinMean:
        case kTfLiteBuiltinPad:
        case kTfLiteBuiltinSum:
        case kTfLiteBuiltinReshape:
        case kTfLiteBuiltinResizeBilinear:
        case kTfLiteBuiltinStridedSlice:
        case kTfLiteBuiltinSlice:
          // Ignore all but the first input (axes, static padding, new shape,
          // begins/offsets, sizes), because other inputs are represented as
          // parameters of the XNNPACK operator rather than extra input.
          {
            const int t = node->inputs->data[0];
            tensors[t] = t;
          }
          break;
        case kTfLiteBuiltinSplit:
          // Ignore the first input (split_dim), as it is represented as
          // parameters of the XNNPACK operator rather than extra input.
          {
            const int t = node->inputs->data[1];
            tensors[t] = t;
            break;
          }
        case kTfLiteBuiltinTranspose:
          // Ignore the second input (perm), as it is represented as
          // parameters of the XNNPACK operator rather than extra input.
          {
            const int t = node->inputs->data[0];
            tensors[t] = t;
            break;
          }
        default:
          // All other operators: process all inputs
          for (int k = 0; k < node->inputs->size; k++) {
            if (registration->builtin_code == kTfLiteBuiltinTransposeConv &&
                k == 0) {
              // Ignore the output size parameter (see above).
              continue;
            }
            const int t = node->inputs->data[k];
            if (t >= 0) {
              tensors[t] = t;
            }
          }
      }
      for (int k = 0; k < node->outputs->size; k++) {
        const int t = node->outputs->data[k];
        if (t >= 0) {
          tensors[t] = t;
        }
      }
    }
    // Filter out and remove -1 (unused) indexes.
    tensors.erase(std::remove_if(tensors.begin(), tensors.end(),
                                 [](int i) { return i < 0; }),
                  tensors.end());
    std::sort(tensors.begin(), tensors.end());

    xnn_subgraph_t subgraph_ptr = nullptr;
    xnn_status status = xnn_create_subgraph(
        /*external_value_ids=*/tensors.size(), /*flags=*/0, &subgraph_ptr);
    if (status != xnn_status_success) {
      TF_LITE_KERNEL_LOG(context, "failed to create XNNPACK subgraph");
      return nullptr;
    }

    // Smart pointer to automatically release subgraph on exit.
    std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph(
        subgraph_ptr, &xnn_delete_subgraph);

    // Persistent tensors need to be defined in same order in all XNNPACK
    // runtimes. This is because they are allocated in order of their XNNPACK
    // value id. We cannot do this inside the subsequent for-loop that walks
    // through all the tensors in the subgraph, because the same 2 VAR_HANDLE
    // in 2 different subgraphs can be iterated over in different
    // order, thus breaking our requirement that persistent tensors are
    // defined in the same order.
    // For example, given subgraph 1 with VAR_HANDLE1 then VAR_HANDLE2, and
    // subgraph 2 with VAR_HANDLE2 then VAR_HANDLE1.
    // 1. Create subgraph 1
    // 2. Define persistent tensor for VAR_HANDLE1 (global id 0, xnn id 0)
    // 3. Define persistent tensor for VAR_HANDLE2 (global id 1, xnn id 1)
    // 4. Create runtime 1, tensor for VAR_HANDLE1 comes before VAR_HANDLE2
    // 5. Create subgraph 2
    // 6. Define persistent tensor for VAR_HANDLE2 (global id 1, xnn id 0)
    // 7. Define persistent tensor for VAR_HANDLE1 (global id 0, xnn id 1)
    // 8. Create runtime 2, tensor for VAR_HANDLE2 comes before VAR_HANDLE1,
    // which is wrong.
    std::unordered_map<uint32_t, uint32_t> global_id_to_xnnpack_id;
    if (DefineVariableTensors(delegate, subgraph.get(), context,
                              global_id_to_xnnpack_id) != kTfLiteOk) {
      return nullptr;
    }

    std::unordered_map<int, uint32_t> tflite_tensor_to_xnnpack;
    std::vector<int> external_inputs;
    std::vector<int> external_outputs;
    for (int t : tensors) {
      if (context->tensors[t].type == kTfLiteResource) {
        // We should never see a resource tensor if we are not handling variable
        // ops.
        if (!delegate.support_variable_ops()) {
          TF_LITE_KERNEL_LOG(
              context,
              "unexpected resource tensor when XNNPACK delegate is "
              "not configured to handle variable operations");
          return nullptr;
        }
        // Resource tensors are not materialized directly. We instead create a
        // tensor that is the same type as how the resource is used, and all
        // references to the resource tensor (that is produced by a VarHandle
        // node) refers directly to this backing tensor.
        const uint32_t global_id = delegate.GetGlobalId(t);
        const auto it = global_id_to_xnnpack_id.find(global_id);
        if (it == global_id_to_xnnpack_id.end()) {
          TF_LITE_KERNEL_LOG(context,
                             "could not find variable with global id %u in "
                             "context %p for local tensor %d",
                             global_id, context, t);
          return nullptr;
        }
        tflite_tensor_to_xnnpack[t] = it->second;
        // Proceed with processing the next tensor.
        continue;
      }
      const xnn_datatype datatype =
          GetXNNPackDatatype(context, context->tensors[t], t);
      if (datatype == xnn_datatype_invalid) {
        TF_LITE_KERNEL_LOG(
            context,
            "unsupported datatype (%s) of tensor %d in XNNPACK delegate",
            TfLiteTypeGetName(context->tensors[t].type), t);
        return nullptr;
      }

      uint32_t flags = 0;
      const void* data = nullptr;
      if (context->tensors[t].allocation_type == kTfLiteMmapRo) {
        data = context->tensors[t].data.raw_const;
      } else {
        // Check for quasi-static data.
        const auto it = delegate.static_unpacked_data_map_.find(t);
        if (it != delegate.static_unpacked_data_map_.end()) {
          data = delegate.static_unpacked_data_.data() + it->second;
        }
      }
      if (inputs.count(t) != 0) {
        flags |= XNN_VALUE_FLAG_EXTERNAL_INPUT;
        if (data == nullptr) {
          externals.insert(t);
          external_inputs.push_back(t);
        }
      }
      if (outputs.count(t) != 0) {
        flags |= XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
        external_outputs.push_back(t);
      }

      std::vector<size_t> dims(
          &context->tensors[t].dims->data[0],
          &context->tensors[t].dims->data[NumDimensions(&context->tensors[t])]);

      xnn_status status = xnn_status_success;
      uint32_t xnnpack_id = XNN_INVALID_VALUE_ID;
      switch (datatype) {
        case xnn_datatype_qint8:
        case xnn_datatype_quint8:
        case xnn_datatype_qint32:
          status = xnn_define_quantized_tensor_value(
              subgraph.get(), datatype,
              static_cast<const TfLiteAffineQuantization*>(
                  context->tensors[t].quantization.params)
                  ->zero_point->data[0],
              static_cast<const TfLiteAffineQuantization*>(
                  context->tensors[t].quantization.params)
                  ->scale->data[0],
              dims.size(), dims.data(), data, XNN_INVALID_VALUE_ID, flags,
              &xnnpack_id);
          break;
        case xnn_datatype_qcint4:
        case xnn_datatype_qcint8:
        case xnn_datatype_qcint32:
          status = xnn_define_channelwise_quantized_tensor_value(
              subgraph.get(), datatype,
              static_cast<const TfLiteAffineQuantization*>(
                  context->tensors[t].quantization.params)
                  ->scale->data,
              dims.size(),
              static_cast<const TfLiteAffineQuantization*>(
                  context->tensors[t].quantization.params)
                  ->quantized_dimension,
              dims.data(), data, XNN_INVALID_VALUE_ID, flags, &xnnpack_id);
          break;
        default:
          status = xnn_define_tensor_value(
              subgraph.get(), datatype, dims.size(), dims.data(), data,
              XNN_INVALID_VALUE_ID, flags, &xnnpack_id);
          break;
      }
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(context,
                           "failed to create XNNPACK Value for tensor %d", t);
        return nullptr;
      }
      tflite_tensor_to_xnnpack[t] = xnnpack_id;
    }

    // Create a set of quasi-static tensors for VisitNode function
    std::unordered_set<int> quasi_static_tensors;
    for (const std::pair<const int, size_t>& entry :
         delegate.static_unpacked_data_map_) {
      quasi_static_tensors.insert(entry.first);
    }

    // Create XNNPACK nodes for TFLite delegate nodes
    for (int i = 0; i < params->nodes_to_replace->size; i++) {
      const int node_index = params->nodes_to_replace->data[i];
      if (delegate.static_unpack_nodes_.count(node_index)) {
        // The node unpacks static input and can be skipped because its input
        // was pre-unpacked in DelegatePrepare.
        continue;
      }

      TfLiteNode* node = nullptr;
      TfLiteRegistration* registration = nullptr;
      if (context->GetNodeAndRegistration(context, node_index, &node,
                                          &registration) != kTfLiteOk) {
        return nullptr;
      }

      if (VisitNode(subgraph.get(), delegate, context, registration, node,
                    node_index, quasi_static_tensors,
                    tflite_tensor_to_xnnpack) != kTfLiteOk) {
        return nullptr;
      }
    }

    xnn_runtime_t runtime_ptr = nullptr;
    uint32_t flags = XNN_FLAG_YIELD_WORKERS;
    if (has_sparse_weights) {
      flags |= XNN_FLAG_HINT_SPARSE_INFERENCE;
    }
    if (delegate.transient_indirection_buffer()) {
      flags |= XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER;
    }
    if (delegate.force_fp16()) {
      flags |= XNN_FLAG_FORCE_FP16_INFERENCE;
    } else {
      const char* precision_metadata_ptr = nullptr;
      size_t precision_metadata_size = 0;
      if (context->GetModelMetadata(
              context, optimize::kTfLiteReducedPrecisionKey,
              &precision_metadata_ptr, &precision_metadata_size) == kTfLiteOk) {
        const std::string precision_metadata(precision_metadata_ptr,
                                             precision_metadata_size);
        optimize::ReducedPrecisionSupport precision_mask =
            optimize::ReducedPrecisionSupport::None;
        if (optimize::SetMaskFromReducedPrecisionMetadata(precision_metadata,
                                                          &precision_mask)) {
          if (optimize::SupportsFP16Inference(precision_mask) &&
              optimize::SupportsFP16Accumulation(precision_mask)) {
            flags |= XNN_FLAG_HINT_FP16_INFERENCE;
          }
        }
      }
    }
    if (context->profiler) {
      flags |= XNN_FLAG_BASIC_PROFILING;
    }

    if (delegate.weight_cache_provider_.IsActive() &&
        delegate.weight_cache_provider_.CanStartBuildStep()) {
      if (!delegate.weight_cache_provider_.StartBuildStep()) {
        TF_LITE_KERNEL_LOG(
            context, "XNNPack delegate failed to start cache build step.");
        return nullptr;
      }
    }
    status = xnn_create_runtime_v4(subgraph.get(), delegate.weights_cache(),
                                   delegate.workspace(), delegate.threadpool(),
                                   flags, &runtime_ptr);
    if (delegate.weight_cache_provider_.IsActive() &&
        delegate.weight_cache_provider_.CanStartBuildStep()) {
      if (!delegate.weight_cache_provider_.StopBuildStep()) {
        TF_LITE_KERNEL_LOG(context,
                           "XNNPack delegate failed to stop cache build step.");
        return nullptr;
      }
    }
    if (status != xnn_status_success) {
      TF_LITE_KERNEL_LOG(context, "failed to create XNNPACK runtime");
      return nullptr;
    }

    return new Subgraph(delegate, runtime_ptr, externals, external_inputs,
                        external_outputs, tflite_tensor_to_xnnpack);
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node,
                       bool enable_subgraph_reshaping, Delegate* delegate) {
    std::lock_guard<std::mutex> lock(delegate->workspace_mutex_);

    if (enable_subgraph_reshaping) {
      xnn_status status = xnn_status_invalid_state;
      for (int i = 0; i < inputs_.size(); ++i) {
        const TfLiteTensor* tensor = &context->tensors[inputs_[i]];
        const int dims_count = NumDimensions(tensor);
        std::array<size_t, XNN_MAX_TENSOR_DIMS> xnn_dims;
        std::copy(&tensor->dims->data[0], &tensor->dims->data[dims_count],
                  xnn_dims.begin());
        status = xnn_reshape_external_value(
            runtime_.get(), tflite_tensor_to_xnnpack_[inputs_[i]], dims_count,
            xnn_dims.data());
        if (status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(
              context, "XNNPack delegate failed to reshape external value");
          return kTfLiteError;
        }
        // signal that setup must be called.
        externals_[inputs_[i]] = nullptr;
      }
      status = xnn_reshape_runtime(runtime_.get());
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(context,
                           "XNNPack delegate failed to reshape runtime");
        return kTfLiteError;
      }

      for (int i = 0; i < outputs_.size(); ++i) {
        TfLiteTensor* tensor = &context->tensors[outputs_[i]];
        size_t num_out_dims;
        size_t out_dims[XNN_MAX_TENSOR_DIMS];
        status = xnn_get_external_value_shape(
            runtime_.get(),
            static_cast<uint32_t>(tflite_tensor_to_xnnpack_[outputs_[i]]),
            &num_out_dims, &out_dims[0]);
        if (status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(
              context, "XNNPack delegate failed to get external value shape");
          return kTfLiteError;
        }
        TfLiteIntArray* output_shape = TfLiteIntArrayCreate(num_out_dims);
        for (int k = 0; k < num_out_dims; ++k) {
          output_shape->data[k] = out_dims[k];
        }
        if (context->ResizeTensor(context, tensor, output_shape) != kTfLiteOk) {
          TF_LITE_KERNEL_LOG(
              context, "XNNPack delegate failed to get resize output tensor");
          return kTfLiteError;
        }
      }
    }
    return kTfLiteOk;
  }

  TfLiteStatus Invoke(TfLiteContext* context, bool enable_subgraph_reshaping,
                      Delegate* delegate) {
    std::lock_guard<std::mutex> lock(delegate->workspace_mutex_);

    bool any_pointers_changed = false;
    for (std::pair<int, void*> io_info : externals_) {
      const TfLiteTensor& tensor = context->tensors[io_info.first];
      void* data_pointer = &dummy_data_;
      if (tensor.data.raw != nullptr) {
        data_pointer = tensor.data.raw;
      } else {
        if (tensor.bytes != 0) {
          TF_LITE_KERNEL_LOG(
              context, "unexpected null data pointer in external tensor %d",
              io_info.first);
          return kTfLiteError;
        }
      }
      if (data_pointer != io_info.second) {
        any_pointers_changed = true;
        externals_[io_info.first] = data_pointer;
      }
    }

    // Even with no externals, we need to setup the runtime if there are
    // variables.
    if (any_pointers_changed || NeedToSetUpVariableTensors()) {
      std::vector<xnn_external_value> external_values;
      for (std::pair<int, void*> io_info : externals_) {
        xnn_external_value value = {0};
        value.id =
            static_cast<uint32_t>(tflite_tensor_to_xnnpack_[io_info.first]);
        value.data = io_info.second;
        external_values.push_back(value);
      }

      xnn_status status = xnn_status_invalid_state;
      if (enable_subgraph_reshaping) {
        status = xnn_setup_runtime_v2(runtime_.get(), external_values.size(),
                                      external_values.data());
      } else {
        status = xnn_setup_runtime(runtime_.get(), external_values.size(),
                                   external_values.data());
      }
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(context, "failed to setup XNNPACK runtime");
        return kTfLiteError;
      }
      variables_set_up_ = true;
    }

    xnn_status status = xnn_invoke_runtime(runtime_.get());
    if (status != xnn_status_success) {
      TF_LITE_KERNEL_LOG(context, "failed to invoke XNNPACK runtime");
      return kTfLiteError;
    }

    if (context->profiler) {
      if (AddEventsToProfiler(reinterpret_cast<Profiler*>(context->profiler),
                              runtime_.get()) != kTfLiteOk) {
        TF_LITE_KERNEL_LOG(context,
                           "failed to get XNNPACK profile information.");
      }
    }

    return kTfLiteOk;
  }

  // Fetch the profile information from XNNPACK and add the events to TfLite's
  // profiler.
  static TfLiteStatus AddEventsToProfiler(Profiler* profiler,
                                          const xnn_runtime_t runtime) {
    size_t required_size = 0;

    // xnn_get_runtime_profiling_info is called twice. The first time it sets
    // required_size to the required size of the buffer to store the result
    // and returns xnn_status_out_of_memory. The second time it writes the
    // result to the buffer provided that the buffer is large enough and
    // returns xnn_status_success.
    xnn_status status = xnn_get_runtime_profiling_info(
        runtime, xnn_profile_info_operator_name, /*param_value_size*/ 0,
        /*param_value*/ nullptr, &required_size);
    std::vector<char> operator_names;
    if (status == xnn_status_out_of_memory) {
      operator_names.resize(required_size);
      status = xnn_get_runtime_profiling_info(
          runtime, xnn_profile_info_operator_name, operator_names.size(),
          operator_names.data(), &required_size);
    }
    if (status != xnn_status_success) {
      return kTfLiteError;
    }
    size_t num_operators;
    status = xnn_get_runtime_profiling_info(
        runtime, xnn_profile_info_num_operators, sizeof(num_operators),
        &num_operators, &required_size);
    if (status != xnn_status_success) {
      return kTfLiteError;
    }
    status = xnn_get_runtime_profiling_info(
        runtime, xnn_profile_info_operator_timing, /*param_value_size*/ 0,
        /*param_value*/ nullptr, &required_size);
    std::vector<uint64_t> operator_timings;
    if (status == xnn_status_out_of_memory) {
      operator_timings.resize(required_size / sizeof(uint64_t));
      status = xnn_get_runtime_profiling_info(
          runtime, xnn_profile_info_operator_timing,
          operator_timings.size() * sizeof(uint64_t), operator_timings.data(),
          &required_size);
    }
    if (status != xnn_status_success) {
      return kTfLiteError;
    }
    const char* operator_name = nullptr;
    size_t name_len = 0;
    for (size_t node_index = 0; node_index < num_operators; ++node_index) {
      operator_name = &operator_names[name_len];
      name_len += strlen(operator_name) + 1;
      profiler->AddEvent(
          operator_name,
          Profiler::EventType::DELEGATE_PROFILED_OPERATOR_INVOKE_EVENT,
          operator_timings[node_index], node_index);
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CalculatePadding(TfLiteContext* context,
                                       TfLitePadding padding, uint32_t* flags,
                                       int node_index) {
    switch (padding) {
      case kTfLitePaddingSame: {
        *flags = XNN_FLAG_TENSORFLOW_SAME_PADDING;
        return kTfLiteOk;
      }
      case kTfLitePaddingValid:
        *flags = 0;
        return kTfLiteOk;
      default:
        TF_LITE_MAYBE_KERNEL_LOG(context,
                                 "invalid padding mode (%d) in node #%d",
                                 static_cast<int>(padding), node_index);
        return kTfLiteError;
    }
  }

  static TfLiteStatus CalculateTransposeConvPaddings(
      TfLiteContext* context, TfLitePadding padding, int input_height,
      int input_width, int kernel_height, int kernel_width, int dilation_height,
      int dilation_width, int stride_height, int stride_width, int node_index,
      int output_height, int output_width, int* padding_top,
      int* padding_bottom, int* padding_left, int* padding_right,
      int* adjustment_height, int* adjustment_width) {
    const int effective_kernel_height =
        (kernel_height - 1) * dilation_height + 1;
    const int effective_kernel_width = (kernel_width - 1) * dilation_width + 1;
    switch (padding) {
      case kTfLitePaddingValid: {
        if (effective_kernel_height > output_height ||
            effective_kernel_width > output_width) {
          TF_LITE_MAYBE_KERNEL_LOG(
              context,
              "output smaller than effective kernel dimensions unsupported "
              "with VALID padding in TRANSPOSE_CONV node #%d: "
              "effective kernel size %dx%d (HxW), output %dx%d",
              node_index, effective_kernel_height, effective_kernel_width,
              output_height, output_width);
          return kTfLiteError;
        }

        *padding_top = *padding_bottom = *padding_left = *padding_right = 0;
        *adjustment_height = (output_height - kernel_height) % stride_height;
        *adjustment_width = (output_width - kernel_width) % stride_width;
        break;
      }
      case kTfLitePaddingSame: {
        int expected_input_height = 0;
        int expected_input_width = 0;
        TfLitePaddingValues paddings = ComputePaddingHeightWidth(
            stride_height, stride_width, dilation_height, dilation_width,
            output_height, output_width, kernel_height, kernel_width, padding,
            &expected_input_height, &expected_input_width);
        if (expected_input_height != input_height ||
            expected_input_width != input_width) {
          TF_LITE_MAYBE_KERNEL_LOG(
              context,
              "inconsistent combination of parameters for TRANSPOSE_CONV op "
              "in node #%d: computed input size %dx%d (HxW), actual %dx%d",
              node_index, expected_input_height, expected_input_width,
              input_height, input_width);
          return kTfLiteError;
        }

        // Note: In the derivation of the adjustments below, it was assumed
        // that
        //       `effective_kernel_...` >= `stride_...` so that
        //       `ComputePadding` in TFLite doesn't encounter a negative value
        //       clamped to zero.
        if (kernel_height < stride_height || kernel_width < stride_width) {
          TF_LITE_MAYBE_KERNEL_LOG(context,
                                   "strides larger than effective kernel "
                                   "dimensions unsupported in "
                                   "TRANSPOSE_CONV node #%d: kernel size "
                                   "%dx%d (HxW), strides %dx%d",
                                   node_index, effective_kernel_height,
                                   effective_kernel_width, stride_height,
                                   stride_width);
          return kTfLiteError;
        }

        *padding_top = paddings.height;
        *padding_bottom = paddings.height + paddings.height_offset;
        *adjustment_height = 0;
        *padding_left = paddings.width;
        *padding_right = paddings.width + paddings.width_offset;
        *adjustment_width = 0;
        break;
      }
      default:
        TF_LITE_MAYBE_KERNEL_LOG(context,
                                 "invalid padding mode (%d) in node #%d",
                                 static_cast<int>(padding), node_index);
        return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus ConvertActivationToOutputRange(
      TfLiteContext* context, int node_index, TfLiteFusedActivation activation,
      float* output_min, float* output_max) {
    switch (activation) {
      case kTfLiteActNone:
        *output_min = -std::numeric_limits<float>::infinity();
        *output_max = +std::numeric_limits<float>::infinity();
        return kTfLiteOk;
      case kTfLiteActRelu:
        *output_min = 0.0f;
        *output_max = +std::numeric_limits<float>::infinity();
        return kTfLiteOk;
      case kTfLiteActReluN1To1:
        *output_min = -1.0f;
        *output_max = +1.0f;
        return kTfLiteOk;
      case kTfLiteActRelu6:
        *output_min = 0.0f;
        *output_max = 6.0f;
        return kTfLiteOk;
      case kTfLiteActTanh:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Tanh) in node #%d",
            node_index);
        return kTfLiteError;
      case kTfLiteActSignBit:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Sign) in node #%d",
            node_index);
        return kTfLiteError;
      case kTfLiteActSigmoid:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Sigmoid) in node #%d",
            node_index);
        return kTfLiteError;
      default:
        TF_LITE_MAYBE_KERNEL_LOG(context,
                                 "invalid fused activation (%d) in node #%d",
                                 static_cast<int>(activation), node_index);
        return kTfLiteError;
    }
  }

  static TfLiteStatus CheckConvolutionParams(TfLiteContext* context,
                                             const TfLiteConvParams* params,
                                             int node_index) {
    if (params->stride_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride width %d in node #%d",
                               params->stride_width, node_index);
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride height %d in node #%d",
                               params->stride_height, node_index);
      return kTfLiteError;
    }

    if (params->dilation_width_factor <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "invalid dilation width factor %d in node #%d",
                               params->dilation_width_factor, node_index);
      return kTfLiteError;
    }
    if (params->dilation_height_factor <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "invalid dilation height factor %d in node #%d",
                               params->dilation_height_factor, node_index);
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckDepthwiseConvolutionParams(
      TfLiteContext* context, const TfLiteDepthwiseConvParams* params,
      int output_channels, int node_index) {
    if (params->stride_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride width %d in node #%d",
                               params->stride_width, node_index);
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride height %d in node #%d",
                               params->stride_height, node_index);
      return kTfLiteError;
    }

    if (params->depth_multiplier <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "invalid depth multiplier %d in node #%d",
                               params->depth_multiplier, node_index);
      return kTfLiteError;
    }
    if (output_channels % params->depth_multiplier != 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "depth multiplier %d is incompatible with "
                               "number of output channels %d in node #%d",
                               params->depth_multiplier, output_channels,
                               node_index);
      return kTfLiteError;
    }

    if (params->dilation_width_factor <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "invalid dilation width factor %d in node #%d",
                               params->dilation_width_factor, node_index);
      return kTfLiteError;
    }
    if (params->dilation_height_factor <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "invalid dilation height factor %d in node #%d",
                               params->dilation_height_factor, node_index);
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckMediaPipeTransposedConvolutionParams(
      TfLiteContext* context, const TfLiteTransposeConvParams* params,
      int node_index) {
    if (params->stride_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride width %d in node #%d",
                               params->stride_width, node_index);
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride height %d in node #%d",
                               params->stride_height, node_index);
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckMediaPipePoolParams(TfLiteContext* context,
                                               const TfLitePoolParams* params,
                                               int node_index) {
    if (params->stride_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride width %d in node #%d",
                               params->stride_width, node_index);
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride height %d in node #%d",
                               params->stride_height, node_index);
      return kTfLiteError;
    }
    if (params->filter_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid filter width %d in node #%d",
                               params->filter_width, node_index);
      return kTfLiteError;
    }
    if (params->filter_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid filter height %d in node #%d",
                               params->filter_height, node_index);
      return kTfLiteError;
    }
    if (params->filter_width != params->stride_width) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "filter width %d does not match stride width %d in node #%d",
          params->filter_width, params->stride_width, node_index);
      return kTfLiteError;
    }
    if (params->filter_height != params->stride_height) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context,
          "filter height %d does not match stride height %d in node #%d",
          params->filter_height, params->stride_height, node_index);
      return kTfLiteError;
    }
    switch (params->activation) {
      case kTfLiteActNone:
        break;
      case kTfLiteActRelu:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Relu) in node #%d",
            node_index);
        return kTfLiteOk;
      case kTfLiteActReluN1To1:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (ReluMinus1To1) in node #%d",
            node_index);
        return kTfLiteOk;
      case kTfLiteActRelu6:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Relu6) in node #%d",
            node_index);
        return kTfLiteOk;
      case kTfLiteActTanh:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Tanh) in node #%d",
            node_index);
        return kTfLiteError;
      case kTfLiteActSignBit:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Sign) in node #%d",
            node_index);
        return kTfLiteError;
      case kTfLiteActSigmoid:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Sigmoid) in node #%d",
            node_index);
        return kTfLiteError;
      default:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "invalid fused activation (%d) in node #%d",
            static_cast<int>(params->activation), node_index);
        return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckFullyConnectedParams(
      TfLiteContext* context, const TfLiteFullyConnectedParams* params,
      int node_index) {
    if (params->weights_format != kTfLiteFullyConnectedWeightsFormatDefault) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unsupported non-default weights format in node #%d",
          node_index);
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckPoolingParams(TfLiteContext* context,
                                         const TfLitePoolParams* params,
                                         BuiltinOperator op_type,
                                         int node_index) {
    if (params->stride_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "invalid stride width %d in %s node #%d",
          params->stride_width, EnumNameBuiltinOperator(op_type), node_index);
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "invalid stride height %d in %s node #%d",
          params->stride_height, EnumNameBuiltinOperator(op_type), node_index);
      return kTfLiteError;
    }

    if (params->filter_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "invalid filter width %d in %s node #%d",
          params->filter_width, EnumNameBuiltinOperator(op_type), node_index);
      return kTfLiteError;
    }
    if (params->filter_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "invalid filter height %d in %s node #%d",
          params->filter_height, EnumNameBuiltinOperator(op_type), node_index);
      return kTfLiteError;
    }

    if (params->stride_width > params->filter_width) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unsupported width stride %d exceeding filter "
                               "width %d in %s node #%d",
                               params->stride_width, params->filter_width,
                               EnumNameBuiltinOperator(op_type), node_index);
      return kTfLiteError;
    }

    if (params->stride_height > params->filter_height) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unsupported height stride %d exceeding filter "
                               "height %d in %s node #%d",
                               params->stride_height, params->filter_height,
                               EnumNameBuiltinOperator(op_type), node_index);
      return kTfLiteError;
    }

    if (params->filter_width == 1 && params->filter_height == 1 &&
        std::max(params->stride_width, params->stride_height) > 1) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unsupported pooling with 1x1 filter "
                               "and %dx%d stride in %s node #%d",
                               params->stride_width, params->stride_height,
                               EnumNameBuiltinOperator(op_type), node_index);
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckNumInputs(TfLiteContext* context, TfLiteNode* node,
                                     int expected_num_inputs,
                                     BuiltinOperator op_type, int node_index) {
    if (node->inputs->size != expected_num_inputs) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unexpected number of inputs (%d != %d) in node %s #%d",
          node->inputs->size, expected_num_inputs,
          EnumNameBuiltinOperator(op_type), node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckNumInputs(TfLiteContext* context, TfLiteNode* node,
                                     int min_num_inputs, int max_num_inputs,
                                     BuiltinOperator op_type, int node_index) {
    if (node->inputs->size < min_num_inputs ||
        node->inputs->size > max_num_inputs) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unexpected number of inputs (%d) in %s node #%d",
          node->inputs->size, EnumNameBuiltinOperator(op_type), node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckNumOutputs(TfLiteContext* context, TfLiteNode* node,
                                      int expected_num_outputs,
                                      BuiltinOperator op_type, int node_index) {
    if (node->outputs->size != expected_num_outputs) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unexpected number of outputs (%d != %d) in %s node #%d",
          node->outputs->size, expected_num_outputs,
          EnumNameBuiltinOperator(op_type), node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckNumOutputs(TfLiteContext* context, TfLiteNode* node,
                                      int min_num_outputs, int max_num_outputs,
                                      BuiltinOperator op_type, int node_index) {
    if (node->outputs->size < min_num_outputs ||
        node->outputs->size > max_num_outputs) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unexpected number of outputs (%d) in %s node #%d",
          node->outputs->size, EnumNameBuiltinOperator(op_type), node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckNumInputsAndOutputs(
      TfLiteContext* context, TfLiteNode* node, int min_num_inputs,
      int max_num_inputs, int expected_num_outputs, BuiltinOperator op_type,
      int node_index) {
    TF_LITE_ENSURE_STATUS(CheckNumInputs(context, node, min_num_inputs,
                                         max_num_inputs, op_type, node_index));
    TF_LITE_ENSURE_STATUS(CheckNumOutputs(context, node, expected_num_outputs,
                                          op_type, node_index));
    return kTfLiteOk;
  }

  static TfLiteStatus CheckNumInputsAndOutputs(
      TfLiteContext* context, TfLiteNode* node, int expected_num_inputs,
      int expected_num_outputs, BuiltinOperator op_type, int node_index) {
    TF_LITE_ENSURE_STATUS(CheckNumInputs(context, node, expected_num_inputs,
                                         op_type, node_index));
    TF_LITE_ENSURE_STATUS(CheckNumOutputs(context, node, expected_num_outputs,
                                          op_type, node_index));
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorType(TfLiteContext* context,
                                      const TfLiteTensor& tensor,
                                      TfLiteType expected_type,
                                      int tensor_index, int node_index) {
    if (tensor.type != expected_type) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unsupported type %s in tensor #%d in node #%d",
          TfLiteTypeGetName(tensor.type), tensor_index, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorFloat32Type(TfLiteContext* context,
                                             const TfLiteTensor& tensor,
                                             int tensor_index, int node_index) {
    return CheckTensorType(context, tensor, kTfLiteFloat32, tensor_index,
                           node_index);
  }

  static TfLiteStatus CheckTensorFloat32OrQInt8Type(const Delegate& delegate,
                                                    TfLiteContext* context,
                                                    const TfLiteTensor& tensor,
                                                    int tensor_index,
                                                    int node_index) {
    switch (tensor.type) {
      case kTfLiteFloat32:
        return kTfLiteOk;
      case kTfLiteInt8:
        if (delegate.support_signed_8bit_quantization()) {
          const auto* quantization_params =
              static_cast<const TfLiteAffineQuantization*>(
                  tensor.quantization.params);
          if (tensor.quantization.type != kTfLiteAffineQuantization ||
              quantization_params->quantized_dimension != 0 ||
              quantization_params->scale == nullptr ||
              quantization_params->scale->size != 1) {
            TF_LITE_MAYBE_KERNEL_LOG(
                context,
                "unsupported quantization type %d in tensor #%d in node #%d",
                tensor.quantization.type, tensor_index, node_index);
            return kTfLiteError;
          }
          return kTfLiteOk;
        }
        break;
      default:
        break;
    }

    TF_LITE_MAYBE_KERNEL_LOG(
        context, "unsupported type %s in tensor #%d in node #%d",
        TfLiteTypeGetName(tensor.type), tensor_index, node_index);
    return kTfLiteError;
  }

  static TfLiteStatus CheckTensorQInt8OrQUInt8Type(const Delegate& delegate,
                                                   TfLiteContext* context,
                                                   const TfLiteTensor& tensor,
                                                   int tensor_index,
                                                   int node_index) {
    switch (tensor.type) {
      case kTfLiteInt8:
        if (delegate.support_signed_8bit_quantization()) {
          const auto* quantization_params =
              static_cast<const TfLiteAffineQuantization*>(
                  tensor.quantization.params);
          if (tensor.quantization.type != kTfLiteAffineQuantization ||
              quantization_params->quantized_dimension != 0 ||
              quantization_params->scale == nullptr ||
              quantization_params->scale->size != 1) {
            TF_LITE_MAYBE_KERNEL_LOG(
                context,
                "unsupported quantization type %d in tensor #%d in node #%d",
                tensor.quantization.type, tensor_index, node_index);
            return kTfLiteError;
          }
          return kTfLiteOk;
        }
        break;
      case kTfLiteUInt8:
        if (delegate.support_unsigned_8bit_quantization()) {
          const auto* quantization_params =
              static_cast<const TfLiteAffineQuantization*>(
                  tensor.quantization.params);
          if (tensor.quantization.type != kTfLiteAffineQuantization ||
              quantization_params->quantized_dimension != 0 ||
              quantization_params->scale == nullptr ||
              quantization_params->zero_point == nullptr ||
              quantization_params->scale->size != 1 ||
              quantization_params->zero_point->size != 1) {
            TF_LITE_MAYBE_KERNEL_LOG(
                context,
                "unsupported quantization type %d in tensor #%d in node #%d",
                tensor.quantization.type, tensor_index, node_index);
            return kTfLiteError;
          }
          return kTfLiteOk;
        }
        break;
      default:
        break;
    }

    TF_LITE_MAYBE_KERNEL_LOG(
        context, "unsupported type %s in tensor #%d in node #%d",
        TfLiteTypeGetName(tensor.type), tensor_index, node_index);
    return kTfLiteError;
  }

  static TfLiteStatus CheckTensorFloat32OrQUInt8Type(const Delegate& delegate,
                                                     TfLiteContext* context,
                                                     const TfLiteTensor& tensor,
                                                     int tensor_index,
                                                     int node_index) {
    switch (tensor.type) {
      case kTfLiteFloat32:
        return kTfLiteOk;
      case kTfLiteInt8:
        if (delegate.support_signed_8bit_quantization()) {
          const auto* quantization_params =
              static_cast<const TfLiteAffineQuantization*>(
                  tensor.quantization.params);
          if (tensor.quantization.type != kTfLiteAffineQuantization ||
              quantization_params->quantized_dimension != 0 ||
              quantization_params->scale == nullptr ||
              quantization_params->scale->size != 1) {
            TF_LITE_MAYBE_KERNEL_LOG(
                context,
                "unsupported quantization type %d in tensor #%d in node #%d",
                tensor.quantization.type, tensor_index, node_index);
            return kTfLiteError;
          }
          return kTfLiteOk;
        }
        break;
      case kTfLiteUInt8:
        if (delegate.support_unsigned_8bit_quantization()) {
          const auto* quantization_params =
              static_cast<const TfLiteAffineQuantization*>(
                  tensor.quantization.params);
          if (tensor.quantization.type != kTfLiteAffineQuantization ||
              quantization_params->quantized_dimension != 0 ||
              quantization_params->scale == nullptr ||
              quantization_params->zero_point == nullptr ||
              quantization_params->scale->size != 1 ||
              quantization_params->zero_point->size != 1) {
            TF_LITE_MAYBE_KERNEL_LOG(
                context,
                "unsupported quantization type %d in tensor #%d in node #%d",
                tensor.quantization.type, tensor_index, node_index);
            return kTfLiteError;
          }
          return kTfLiteOk;
        }
        break;
      default:
        break;
    }

    TF_LITE_MAYBE_KERNEL_LOG(
        context, "unsupported type %s in tensor #%d in node #%d",
        TfLiteTypeGetName(tensor.type), tensor_index, node_index);
    return kTfLiteError;
  }

  static TfLiteStatus CheckTensorFloat32OrQCInt8Type(
      const Delegate& delegate, TfLiteContext* context,
      const TfLiteTensor& tensor, int expected_quantized_dimension,
      int tensor_index, int node_index) {
    std::vector<size_t> tensor_dims(&tensor.dims->data[0],
                                    &tensor.dims->data[NumDimensions(&tensor)]);
    switch (tensor.type) {
      case kTfLiteFloat32:
        return kTfLiteOk;
      case kTfLiteInt8: {
        if (delegate.support_signed_8bit_quantization()) {
          if (tensor.quantization.type != kTfLiteAffineQuantization) {
            TF_LITE_MAYBE_KERNEL_LOG(
                context,
                "unsupported quantization type %d in tensor #%d in node #%d",
                tensor.quantization.type, tensor_index, node_index);
            return kTfLiteError;
          }
          const TfLiteAffineQuantization* quantization_params =
              static_cast<const TfLiteAffineQuantization*>(
                  tensor.quantization.params);
          if (quantization_params->scale == nullptr) {
            TF_LITE_MAYBE_KERNEL_LOG(context,
                                     "missing scale quantization parameters in "
                                     "tensor #%d in node #%d",
                                     tensor_index, node_index);
            return kTfLiteError;
          }
          if (quantization_params->scale->size > 1 &&
              quantization_params->quantized_dimension !=
                  expected_quantized_dimension) {
            TF_LITE_MAYBE_KERNEL_LOG(context,
                                     "unsupported quantized dimension %d in "
                                     "tensor #%d in node #%d",
                                     quantization_params->quantized_dimension,
                                     tensor_index, node_index);
            return kTfLiteError;
          }
          if (quantization_params->scale->size > 1) {
            if (xnn_validate_channelwise_quantized_tensor(
                    xnn_datatype_qcint8,
                    /*zero_point=*/quantization_params->zero_point->data[0],
                    quantization_params->scale->data, tensor_dims.size(),
                    /*channel_dim=*/quantization_params->quantized_dimension,
                    tensor_dims.data()) != xnn_status_success) {
              TF_LITE_MAYBE_KERNEL_LOG(
                  context,
                  "Channelwise quantized tensor #%d in node #%d has invalid "
                  "quantization parameters",
                  tensor_index, node_index);
              return kTfLiteError;
            }
          } else {
            if (xnn_validate_quantized_tensor(
                    xnn_datatype_qint8,
                    quantization_params->zero_point->data[0],
                    quantization_params->scale->data[0], tensor_dims.size(),
                    tensor_dims.data()) != xnn_status_success) {
              TF_LITE_MAYBE_KERNEL_LOG(context,
                                       "Quantized tensor #%d in node #%d has "
                                       "invalid quantization parameters",
                                       tensor_index, node_index);
              return kTfLiteError;
            }
          }
          return kTfLiteOk;
        }
        break;
      }
      case kTfLiteUInt8:
        if (delegate.support_unsigned_8bit_quantization()) {
          const auto* quantization_params =
              static_cast<const TfLiteAffineQuantization*>(
                  tensor.quantization.params);
          if (tensor.quantization.type != kTfLiteAffineQuantization ||
              quantization_params->quantized_dimension != 0 ||
              quantization_params->scale == nullptr ||
              quantization_params->zero_point == nullptr ||
              quantization_params->scale->size != 1 ||
              quantization_params->zero_point->size != 1) {
            TF_LITE_MAYBE_KERNEL_LOG(
                context,
                "unsupported quantization type %d in tensor #%d in node #%d",
                tensor.quantization.type, tensor_index, node_index);
            return kTfLiteError;
          }
          if (xnn_validate_quantized_tensor(
                  xnn_datatype_quint8, quantization_params->zero_point->data[0],
                  quantization_params->scale->data[0], tensor_dims.size(),
                  tensor_dims.data()) != xnn_status_success) {
            TF_LITE_MAYBE_KERNEL_LOG(context,
                                     "Quantized tensor #%d in node #%d has "
                                     "invalid quantization parameters",
                                     tensor_index, node_index);
            return kTfLiteError;
          }
          return kTfLiteOk;
        }
        break;
      default:
        break;
    }

    TF_LITE_MAYBE_KERNEL_LOG(
        context, "unsupported type %s in tensor #%d in node #%d",
        TfLiteTypeGetName(tensor.type), tensor_index, node_index);
    return kTfLiteError;
  }

  static TfLiteStatus CheckTensorFloat32OrQCInt4OrQCInt8Type(
      const Delegate& delegate, TfLiteContext* context,
      const TfLiteTensor& tensor, int expected_quantized_dimension,
      int tensor_index, int node_index) {
    switch (tensor.type) {
      case kTfLiteFloat32:
        return kTfLiteOk;
      case kTfLiteInt4:
      case kTfLiteInt8:
        if (delegate.support_signed_8bit_quantization() &&
            (kTfLiteInt8 == tensor.type || kTfLiteInt4 == tensor.type)) {
          if (tensor.quantization.type != kTfLiteAffineQuantization) {
            TF_LITE_MAYBE_KERNEL_LOG(
                context,
                "unsupported quantization type %d in tensor #%d in node #%d",
                tensor.quantization.type, tensor_index, node_index);
            return kTfLiteError;
          }
          const TfLiteAffineQuantization* quantization_params =
              static_cast<const TfLiteAffineQuantization*>(
                  tensor.quantization.params);
          if (quantization_params->scale == nullptr) {
            TF_LITE_MAYBE_KERNEL_LOG(context,
                                     "missing scale quantization parameters in "
                                     "tensor #%d in node #%d",
                                     tensor_index, node_index);
            return kTfLiteError;
          }
          if (quantization_params->scale->size > 1 &&
              quantization_params->quantized_dimension !=
                  expected_quantized_dimension) {
            TF_LITE_MAYBE_KERNEL_LOG(
                context,
                "unsupported quantized dimension %d in tensor #%d in node #%d",
                quantization_params->quantized_dimension, tensor_index,
                node_index);
            return kTfLiteError;
          }
          return kTfLiteOk;
        }
        break;
      case kTfLiteUInt8:
        if (delegate.support_unsigned_8bit_quantization()) {
          const auto* quantization_params =
              static_cast<const TfLiteAffineQuantization*>(
                  tensor.quantization.params);
          if (tensor.quantization.type != kTfLiteAffineQuantization ||
              quantization_params->quantized_dimension != 0 ||
              quantization_params->scale == nullptr ||
              quantization_params->zero_point == nullptr ||
              quantization_params->scale->size != 1 ||
              quantization_params->zero_point->size != 1) {
            TF_LITE_MAYBE_KERNEL_LOG(
                context,
                "unsupported quantization type %d in tensor #%d in node #%d",
                tensor.quantization.type, tensor_index, node_index);
            return kTfLiteError;
          }
          return kTfLiteOk;
        }
        break;
      default:
        break;
    }

    TF_LITE_MAYBE_KERNEL_LOG(
        context, "unsupported type %s in tensor #%d in node #%d",
        TfLiteTypeGetName(tensor.type), tensor_index, node_index);
    return kTfLiteError;
  }

  static TfLiteStatus CheckTensorFloat32OrQInt32Type(const Delegate& delegate,
                                                     TfLiteContext* context,
                                                     const TfLiteTensor& tensor,
                                                     int tensor_index,
                                                     int node_index) {
    switch (tensor.type) {
      case kTfLiteFloat32:
        return kTfLiteOk;
      case kTfLiteInt32:
        if (delegate.support_any_8bit_quantization()) {
          if (tensor.quantization.type != kTfLiteAffineQuantization ||
              static_cast<const TfLiteAffineQuantization*>(
                  tensor.quantization.params)
                      ->quantized_dimension != 0 ||
              static_cast<const TfLiteAffineQuantization*>(
                  tensor.quantization.params)
                      ->scale == nullptr ||
              static_cast<const TfLiteAffineQuantization*>(
                  tensor.quantization.params)
                      ->scale->size != 1) {
            TF_LITE_MAYBE_KERNEL_LOG(
                context,
                "unsupported quantization type %d in tensor #%d in node #%d",
                tensor.quantization.type, tensor_index, node_index);
            return kTfLiteError;
          }
          return kTfLiteOk;
        }
        break;
      default:
        break;
    }

    TF_LITE_MAYBE_KERNEL_LOG(
        context, "unsupported type %s in tensor #%d in node #%d",
        TfLiteTypeGetName(tensor.type), tensor_index, node_index);
    return kTfLiteError;
  }

  static TfLiteStatus CheckTensorFloat32OrQCInt32Type(
      const Delegate& delegate, TfLiteContext* context,
      const TfLiteTensor& tensor, int tensor_index, int node_index) {
    switch (tensor.type) {
      case kTfLiteFloat32:
        return kTfLiteOk;
      case kTfLiteInt32: {
        std::vector<size_t> tensor_dims(
            &tensor.dims->data[0], &tensor.dims->data[NumDimensions(&tensor)]);
        const TfLiteAffineQuantization* quantization_params =
            static_cast<const TfLiteAffineQuantization*>(
                tensor.quantization.params);
        if (delegate.support_signed_8bit_quantization()) {
          if (tensor.quantization.type != kTfLiteAffineQuantization ||
              quantization_params->quantized_dimension != 0) {
            TF_LITE_MAYBE_KERNEL_LOG(
                context,
                "unsupported quantization type %d in tensor #%d in node #%d",
                tensor.quantization.type, tensor_index, node_index);
            return kTfLiteError;
          }
          if (quantization_params->scale->size > 1) {
            if (xnn_validate_channelwise_quantized_tensor(
                    xnn_datatype_qcint32, /*zero_point=*/0,
                    quantization_params->scale->data, tensor_dims.size(),
                    /*channel_dim=*/0,
                    tensor_dims.data()) != xnn_status_success) {
              TF_LITE_MAYBE_KERNEL_LOG(
                  context,
                  "Channelwise quantized tensor #%d in node #%d has invalid "
                  "quantization parameters",
                  tensor_index, node_index);
              return kTfLiteError;
            }
          } else if (xnn_validate_quantized_tensor(
                         xnn_datatype_qint32,
                         quantization_params->zero_point->data[0],
                         quantization_params->scale->data[0],
                         tensor_dims.size(),
                         tensor_dims.data()) != xnn_status_success) {
            TF_LITE_MAYBE_KERNEL_LOG(context,
                                     "Quantized tensor #%d in node #%d has "
                                     "invalid quantization parameters",
                                     tensor_index, node_index);
            return kTfLiteError;
          }
          return kTfLiteOk;
        }
        break;
      }
      default:
        break;
    }
    TF_LITE_MAYBE_KERNEL_LOG(
        context, "unsupported type %s in tensor #%d in node #%d",
        TfLiteTypeGetName(tensor.type), tensor_index, node_index);
    return kTfLiteError;
  }

  static TfLiteStatus CheckTensorInt32Type(TfLiteContext* context,
                                           const TfLiteTensor& tensor,
                                           int tensor_index, int node_index) {
    switch (tensor.type) {
      case kTfLiteInt32:
        return kTfLiteOk;
      default:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported type %s in tensor #%d in node #%d",
            TfLiteTypeGetName(tensor.type), tensor_index, node_index);
    }
    return kTfLiteError;
  }

  static TfLiteStatus CheckTensorInt32OrInt64Type(TfLiteContext* context,
                                                  const TfLiteTensor& tensor,
                                                  int tensor_index,
                                                  int node_index) {
    switch (tensor.type) {
      case kTfLiteInt32:
      case kTfLiteInt64:
        return kTfLiteOk;
      default:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported type %s in tensor #%d in node #%d",
            TfLiteTypeGetName(tensor.type), tensor_index, node_index);
    }
    return kTfLiteError;
  }

  static TfLiteStatus CheckTensorShape(TfLiteContext* context,
                                       const TfLiteTensor& tensor,
                                       int min_num_dims, int max_num_dims,
                                       int tensor_index,
                                       BuiltinOperator op_type,
                                       int node_index) {
    if (min_num_dims == max_num_dims) {
      if (NumDimensions(&tensor) != min_num_dims) {
        TF_LITE_MAYBE_KERNEL_LOG(context,
                                 "unsupported number of shape dimensions (%d) "
                                 "in tensor #%d in %s node #%d: "
                                 "%d dimensions expected",
                                 NumDimensions(&tensor), tensor_index,
                                 EnumNameBuiltinOperator(op_type), node_index,
                                 min_num_dims);
        return kTfLiteError;
      }
    } else {
      if (NumDimensions(&tensor) < min_num_dims) {
        TF_LITE_MAYBE_KERNEL_LOG(context,
                                 "unsupported number of shape dimensions (%d) "
                                 "in tensor #%d in %s node #%d: "
                                 "at least %d dimensions expected",
                                 NumDimensions(&tensor), tensor_index,
                                 EnumNameBuiltinOperator(op_type), node_index,
                                 min_num_dims);
        return kTfLiteError;
      }
      if (NumDimensions(&tensor) > max_num_dims) {
        TF_LITE_MAYBE_KERNEL_LOG(context,
                                 "unsupported number of shape dimensions (%d) "
                                 "in tensor #%d in %s node #%d: "
                                 "at most %d dimensions expected",
                                 NumDimensions(&tensor), tensor_index,
                                 EnumNameBuiltinOperator(op_type), node_index,
                                 max_num_dims);
        return kTfLiteError;
      }
    }
    for (int i = 0; i < NumDimensions(&tensor); i++) {
      if (SizeOfDimension(&tensor, i) <= 0) {
        TF_LITE_MAYBE_KERNEL_LOG(context,
                                 "invalid num of elements (%d) in "
                                 "dimension #%d in tensor #%d in %s node #%d",
                                 SizeOfDimension(&tensor, i), i, tensor_index,
                                 EnumNameBuiltinOperator(op_type), node_index);
        return kTfLiteError;
      }
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorShape(TfLiteContext* context,
                                       const TfLiteTensor& tensor,
                                       int expected_num_dims, int tensor_index,
                                       BuiltinOperator op_type,
                                       int node_index) {
    return CheckTensorShape(context, tensor, expected_num_dims,
                            expected_num_dims, tensor_index, op_type,
                            node_index);
  }

  static TfLiteStatus CheckSlopeTensorShape(TfLiteContext* context,
                                            const TfLiteTensor& tensor,
                                            int tensor_index,
                                            BuiltinOperator op_type,
                                            int node_index) {
    if (NumDimensions(&tensor) < 1) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of shape dimensions (%d) in "
                               "tensor #%d in %s node #%d: "
                               "expected at least a 1D tensor",
                               NumDimensions(&tensor), tensor_index,
                               EnumNameBuiltinOperator(op_type), node_index);
      return kTfLiteError;
    }
    // Validate that all non-channel dimensions (if any) are exactly 1.
    for (int i = 0; i < NumDimensions(&tensor) - 1; i++) {
      if (SizeOfDimension(&tensor, i) != 1) {
        TF_LITE_MAYBE_KERNEL_LOG(
            context,
            "unexpected value %d of shape dimension #%d in "
            "tensor #%d in %s node #%d: "
            "expected 1 for non-channel dimensions",
            tensor.dims->data[i], i, tensor_index,
            EnumNameBuiltinOperator(op_type), node_index);
        return kTfLiteError;
      }
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckPaddingsTensorShape(TfLiteContext* context,
                                               const TfLiteTensor& tensor,
                                               int expected_rows,
                                               int tensor_index,
                                               int node_index) {
    if (NumDimensions(&tensor) != 2) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of shape dimensions (%d) in "
                               "padding tensor #%d in node #%d: "
                               "expected a 2D tensor",
                               NumDimensions(&tensor), tensor_index,
                               node_index);
      return kTfLiteError;
    }
    if (SizeOfDimension(&tensor, 0) != expected_rows) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of rows (%d) in "
                               "padding tensor #%d in node #%d: "
                               "%d rows expected",
                               NumDimensions(&tensor), tensor_index, node_index,
                               expected_rows);
      return kTfLiteError;
    }
    if (SizeOfDimension(&tensor, 1) != 2) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of columns (%d) in "
                               "padding tensor #%d in node #%d: "
                               "2 columns expected",
                               NumDimensions(&tensor), tensor_index,
                               node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckAxesTensorShape(TfLiteContext* context,
                                           const TfLiteTensor& tensor,
                                           int tensor_index, int node_index) {
    const int num_tensor_dims = NumDimensions(&tensor);
    if (num_tensor_dims > 1) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of shape dimensions (%d) in "
                               "axes tensor #%d in node #%d: "
                               "expected a 1D tensor",
                               num_tensor_dims, tensor_index, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckShapeTensorShape(TfLiteContext* context,
                                            const TfLiteTensor& tensor,
                                            bool squeeze_dims, int tensor_index,
                                            BuiltinOperator op_type,
                                            int node_index) {
    const int num_dims = NumDimensions(&tensor);
    if (num_dims != 1) {
      if (squeeze_dims) {
        for (int i = 0; i < num_dims - 1; i++) {
          if (tensor.dims->data[i] != 1) {
            TF_LITE_MAYBE_KERNEL_LOG(context,
                                     "unexpected non-unit (%d) shape "
                                     "dimension #%d in shape tensor "
                                     "#%d in %s node #%d: expected %d "
                                     "leading dimensions of the %dD "
                                     "tensor to be 1",
                                     tensor.dims->data[i], i, tensor_index,
                                     EnumNameBuiltinOperator(op_type),
                                     node_index, num_dims - 1, num_dims);
            return kTfLiteError;
          }
        }
      } else {
        TF_LITE_MAYBE_KERNEL_LOG(
            context,
            "unexpected number of shape dimensions (%d) in "
            "shape tensor #%d in %s node #%d: "
            "expected a 1D tensor",
            num_dims, tensor_index, EnumNameBuiltinOperator(op_type),
            node_index);
        return kTfLiteError;
      }
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorNonDynamicAllocation(
      const Delegate& delegate, TfLiteContext* context,
      const TfLiteTensor& tensor, int tensor_index, int node_index) {
    // TODO(b/149120844): remove checks once dynamic tensors are supported
    if (delegate.enable_subgraph_reshaping()) {
      return kTfLiteOk;
    }
    if (tensor.allocation_type == kTfLiteDynamic) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context,
          "invalid allocation type in tensor #%d in node #%d: "
          "expected non-dynamic tensor",
          tensor_index, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorStaticAllocation(TfLiteContext* context,
                                                  const TfLiteTensor& tensor,
                                                  int tensor_index,
                                                  BuiltinOperator op_type,
                                                  int node_index) {
    if (tensor.allocation_type != kTfLiteMmapRo ||
        tensor.data.raw_const == nullptr) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context,
          "invalid allocation type in tensor #%d in %s node #%d: "
          "expected static read-only tensor",
          tensor_index, EnumNameBuiltinOperator(op_type), node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorStaticOrPersistentRoAllocation(
      TfLiteContext* context, const TfLiteTensor& tensor, int tensor_index,
      int node_index) {
    if (tensor.allocation_type == kTfLiteMmapRo ||
        tensor.allocation_type == kTfLitePersistentRo ||
        tensor.data.raw_const == nullptr) {
      return kTfLiteOk;
    }
    TF_LITE_MAYBE_KERNEL_LOG(
        context,
        "invalid allocation type in tensor #%d in node #%d: "
        "expected static or persistent read-only tensor",
        tensor_index, node_index);
    return kTfLiteError;
  }

  static TfLiteStatus CheckTensorsDimensionMatch(
      TfLiteContext* context, const TfLiteTensor& input_tensor,
      const TfLiteTensor& output_tensor, int dimension_index, int node_index,
      const char* op_name) {
    if (SizeOfDimension(&input_tensor, dimension_index) !=
        SizeOfDimension(&output_tensor, dimension_index)) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context,
          "mismatch in shape dimension %d (%d != %d) in input and output "
          "tensors of %s operator #%d",
          dimension_index, SizeOfDimension(&input_tensor, dimension_index),
          SizeOfDimension(&output_tensor, dimension_index), op_name,
          node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static float GetTensorScaleOrDefault(const TfLiteTensor& tensor,
                                       float default_scale) {
    switch (tensor.type) {
      case kTfLiteInt8:
      case kTfLiteUInt8: {
        if (tensor.quantization.type != kTfLiteAffineQuantization) {
          return default_scale;
        }

        const auto* quantization_params =
            static_cast<const TfLiteAffineQuantization*>(
                tensor.quantization.params);
        if (quantization_params->quantized_dimension != 0 ||
            quantization_params->scale == nullptr ||
            quantization_params->scale->size != 1) {
          return default_scale;
        }

        return quantization_params->scale->data[0];
      }
      default:
        break;
    }
    return default_scale;
  }

  static TfLiteStatus CheckTensorsInputOutputScale(
      TfLiteContext* context, const TfLiteTensor& input_tensor,
      const TfLiteTensor& output_tensor, float scale_min, float scale_max,
      BuiltinOperator op_type, int node_index) {
    if (input_tensor.type != output_tensor.type) {
      // No validation needed
      return kTfLiteOk;
    }

    if (input_tensor.type == kTfLiteInt8 || input_tensor.type == kTfLiteUInt8) {
      const float input_scale = static_cast<const TfLiteAffineQuantization*>(
                                    input_tensor.quantization.params)
                                    ->scale->data[0];
      const float output_scale = static_cast<const TfLiteAffineQuantization*>(
                                     output_tensor.quantization.params)
                                     ->scale->data[0];

      const float input_output_scale = input_scale / output_scale;
      if (input_output_scale < scale_min || input_output_scale >= scale_max) {
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported input-to-output scale in %s node #%d",
            EnumNameBuiltinOperator(op_type), node_index);
        return kTfLiteError;
      }
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorsInputProductOutputScale(
      TfLiteContext* context, const TfLiteTensor& input1_tensor,
      const TfLiteTensor& input2_tensor, const TfLiteTensor& output_tensor,
      float scale_min, float scale_max, BuiltinOperator op_type,
      int node_index) {
    if (input1_tensor.type != input2_tensor.type ||
        input1_tensor.type != output_tensor.type) {
      // No validation needed
      return kTfLiteOk;
    }

    if (input1_tensor.type == kTfLiteInt8 ||
        input1_tensor.type == kTfLiteUInt8) {
      const float input1_scale = static_cast<const TfLiteAffineQuantization*>(
                                     input1_tensor.quantization.params)
                                     ->scale->data[0];
      const float input2_scale = static_cast<const TfLiteAffineQuantization*>(
                                     input2_tensor.quantization.params)
                                     ->scale->data[0];
      const float output_scale = static_cast<const TfLiteAffineQuantization*>(
                                     output_tensor.quantization.params)
                                     ->scale->data[0];

      const float product_scale = input1_scale * input2_scale;
      const float product_output_scale = product_scale / output_scale;
      if (product_output_scale < scale_min ||
          product_output_scale >= scale_max) {
        TF_LITE_MAYBE_KERNEL_LOG(
            context,
            "unsupported input-product-to-output scale in %s, node #%d",
            EnumNameBuiltinOperator(op_type), node_index);
        return kTfLiteError;
      }
    }
    return kTfLiteOk;
  }

  static TfLiteStatus VisitNode(
      xnn_subgraph_t subgraph, Delegate& delegate, TfLiteContext* context,
      TfLiteRegistration* registration, TfLiteNode* node, int node_index,
      const std::unordered_set<int>& quasi_static_tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    // TFLite context used for logging purposes. When we create a new node
    // (subgraph is non-null), logging context is the same as context, and
    // error messages are passed to TFLite. When we detect supported
    // operations (subgraph is null), logging context is null, and error
    // messages are suppressed.
#ifdef XNNPACK_DELEGATE_ENABLE_LOGGING
    TfLiteContext* logging_context = context;
#else
    TfLiteContext* logging_context = subgraph == nullptr ? nullptr : context;
#endif
    switch (registration->builtin_code) {
      case kTfLiteBuiltinAbs:
        return VisitAbsNode(subgraph, delegate, logging_context, node_index,
                            node, context->tensors, input_output_tensors);
      case kTfLiteBuiltinAdd: {
        const TfLiteAddParams* add_params =
            static_cast<const TfLiteAddParams*>(node->builtin_data);

        return VisitAddNode(subgraph, delegate, logging_context, node_index,
                            node, context->tensors, add_params,
                            input_output_tensors);
      }
      case kTfLiteBuiltinAssignVariable:
        return VisitAssignVariableNode(subgraph, delegate, logging_context,
                                       node_index, node, context->tensors,
                                       input_output_tensors);
      case kTfLiteBuiltinAveragePool2d: {
        const TfLitePoolParams* pool_params =
            static_cast<const TfLitePoolParams*>(node->builtin_data);

        return VisitAveragePool2DNode(subgraph, delegate, logging_context,
                                      node_index, node, context->tensors,
                                      pool_params, input_output_tensors);
      }
      case kTfLiteBuiltinBatchMatmul: {
        const TfLiteBatchMatMulParams* batchmatmul_params =
            static_cast<const TfLiteBatchMatMulParams*>(node->builtin_data);

        return VisitBatchMatMulNode(subgraph, delegate, logging_context,
                                    node_index, node, context->tensors,
                                    batchmatmul_params, input_output_tensors);
      }
      case kTfLiteBuiltinCeil:
        return VisitCeilNode(subgraph, delegate, logging_context, node_index,
                             node, context->tensors, input_output_tensors);
      case kTfLiteBuiltinConcatenation: {
        const TfLiteConcatenationParams* concat_params =
            static_cast<const TfLiteConcatenationParams*>(node->builtin_data);
        return VisitConcatenationNode(subgraph, delegate, logging_context,
                                      node_index, node, context->tensors,
                                      concat_params, input_output_tensors);
      }
      case kTfLiteBuiltinConv2d: {
        const TfLiteConvParams* conv_params =
            static_cast<const TfLiteConvParams*>(node->builtin_data);

        return VisitConv2DNode(subgraph, delegate, logging_context, node_index,
                               node, context->tensors, conv_params,
                               quasi_static_tensors, input_output_tensors);
      }
      case kTfLiteBuiltinDepthwiseConv2d: {
        const TfLiteDepthwiseConvParams* dwconv_params =
            static_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data);

        return VisitDepthwiseConv2DNode(subgraph, delegate, logging_context,
                                        node_index, node, context->tensors,
                                        dwconv_params, quasi_static_tensors,
                                        input_output_tensors);
      }
      case kTfLiteBuiltinDepthToSpace: {
        const TfLiteDepthToSpaceParams* depth_to_space_params =
            static_cast<const TfLiteDepthToSpaceParams*>(node->builtin_data);

        return VisitDepthToSpaceNode(
            subgraph, delegate, logging_context, node_index, node,
            context->tensors, depth_to_space_params, input_output_tensors);
      }
      case kTfLiteBuiltinDequantize:
        return VisitDequantizeNode(subgraph, delegate, logging_context,
                                   node_index, node, context->tensors,
                                   input_output_tensors);
      case kTfLiteBuiltinDiv: {
        const TfLiteDivParams* div_params =
            static_cast<const TfLiteDivParams*>(node->builtin_data);

        return VisitDivNode(subgraph, delegate, logging_context, node_index,
                            node, context->tensors, div_params,
                            input_output_tensors);
      }
      case kTfLiteBuiltinElu:
        return VisitEluNode(subgraph, delegate, logging_context, node_index,
                            node, context->tensors, input_output_tensors);
      case kTfLiteBuiltinFullyConnected: {
        // FullyConnected with sparse weight has version 8, which cannot be
        // delegated to XNNPack.
        if (registration->version == 8) {
          TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                                   "Unsupported version %d of FullyConnected.",
                                   registration->version);
          return kTfLiteError;
        }

        const TfLiteFullyConnectedParams* fc_params =
            static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

        return VisitFullyConnectedNode(subgraph, delegate, logging_context,
                                       node_index, node, context->tensors,
                                       fc_params, quasi_static_tensors,
                                       input_output_tensors);
      }
      case kTfLiteBuiltinFloor:
        return VisitFloorNode(subgraph, delegate, logging_context, node_index,
                              node, context->tensors, input_output_tensors);
      case kTfLiteBuiltinGelu: {
        const TfLiteGeluParams* gelu_params =
            static_cast<const TfLiteGeluParams*>(node->builtin_data);
        // Sorry, we don't do approximates here, only the real thing to full
        // accuracy.
        // TODO(b/338031720) - Add support for the tanh-based GELU
        // approximation.
        if (gelu_params->approximate) {
          TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                                   "Unsupported approximate Gelu.");
          return kTfLiteError;
        }
        return VisitGeluNode(subgraph, delegate, logging_context, node_index,
                             node, context->tensors, input_output_tensors);
      }
      case kTfLiteBuiltinHardSwish:
        return VisitHardSwishNode(subgraph, delegate, logging_context,
                                  node_index, node, context->tensors,
                                  input_output_tensors);
      case kTfLiteBuiltinLeakyRelu: {
        const TfLiteLeakyReluParams* leaky_relu_params =
            static_cast<const TfLiteLeakyReluParams*>(node->builtin_data);

        return VisitLeakyReluNode(subgraph, delegate, logging_context,
                                  node_index, node, context->tensors,
                                  leaky_relu_params, input_output_tensors);
      }
      case kTfLiteBuiltinLogistic:
        return VisitLogisticNode(subgraph, delegate, logging_context,
                                 node_index, node, context->tensors,
                                 input_output_tensors);
      case kTfLiteBuiltinMaxPool2d: {
        const TfLitePoolParams* pool_params =
            static_cast<const TfLitePoolParams*>(node->builtin_data);

        return VisitMaxPool2DNode(subgraph, delegate, logging_context,
                                  node_index, node, context->tensors,
                                  pool_params, input_output_tensors);
      }

      case kTfLiteBuiltinSum: {
        const TfLiteReducerParams* reducer_params =
            static_cast<const TfLiteReducerParams*>(node->builtin_data);
        return VisitSumNode(subgraph, delegate, logging_context, node_index,
                            node, context->tensors, reducer_params,
                            input_output_tensors);
      }
      case kTfLiteBuiltinMaximum:
        return VisitMaximumNode(subgraph, delegate, logging_context, node_index,
                                node, context->tensors, input_output_tensors);
      case kTfLiteBuiltinMean: {
        const TfLiteReducerParams* reducer_params =
            static_cast<const TfLiteReducerParams*>(node->builtin_data);

        return VisitMeanNode(subgraph, delegate, logging_context, node_index,
                             node, context->tensors, reducer_params,
                             input_output_tensors);
      }
      case kTfLiteBuiltinMinimum:
        return VisitMinimumNode(subgraph, delegate, logging_context, node_index,
                                node, context->tensors, input_output_tensors);
      case kTfLiteBuiltinMul: {
        const TfLiteMulParams* mul_params =
            static_cast<const TfLiteMulParams*>(node->builtin_data);

        return VisitMulNode(subgraph, delegate, logging_context, node_index,
                            node, context->tensors, mul_params,
                            input_output_tensors);
      }
      case kTfLiteBuiltinNeg:
        return VisitNegNode(subgraph, delegate, logging_context, node_index,
                            node, context->tensors, input_output_tensors);
      case kTfLiteBuiltinPad:
        return VisitPadNode(subgraph, delegate, logging_context, node_index,
                            node, context->tensors, input_output_tensors);
      case kTfLiteBuiltinPrelu:
        return VisitPreluNode(subgraph, delegate, logging_context, node_index,
                              node, context->tensors, quasi_static_tensors,
                              input_output_tensors);
      case kTfLiteBuiltinQuantize:
        return VisitQuantizeNode(subgraph, delegate, logging_context,
                                 node_index, node, context->tensors,
                                 input_output_tensors);
      case kTfLiteBuiltinReadVariable:
        return VisitReadVariableNode(subgraph, delegate, logging_context,
                                     node_index, node, context->tensors,
                                     input_output_tensors);
      case kTfLiteBuiltinRelu:
        return VisitReluNode(subgraph, delegate, logging_context, node_index,
                             node, context->tensors, 0.0f,
                             std::numeric_limits<float>::infinity(),
                             input_output_tensors);
      case kTfLiteBuiltinReluN1To1:
        return VisitReluNode(subgraph, delegate, logging_context, node_index,
                             node, context->tensors, -1.0f, 1.0f,
                             input_output_tensors);
      case kTfLiteBuiltinRelu6:
        return VisitReluNode(subgraph, delegate, logging_context, node_index,
                             node, context->tensors, 0.0f, 6.0f,
                             input_output_tensors);
      case kTfLiteBuiltinReshape: {
        const TfLiteReshapeParams* reshape_params =
            static_cast<const TfLiteReshapeParams*>(node->builtin_data);

        return VisitReshapeNode(subgraph, delegate, logging_context, node_index,
                                node, context->tensors, reshape_params,
                                input_output_tensors);
      }
      case kTfLiteBuiltinResizeBilinear: {
        const TfLiteResizeBilinearParams* resize_params =
            static_cast<const TfLiteResizeBilinearParams*>(node->builtin_data);

        return VisitResizeBilinearNode(subgraph, delegate, logging_context,
                                       node_index, node, context->tensors,
                                       resize_params, input_output_tensors);
      }
      case kTfLiteBuiltinRound:
        return VisitRoundNode(subgraph, delegate, logging_context, node_index,
                              node, context->tensors, input_output_tensors);
      case kTfLiteBuiltinRsqrt:
        return VisitRsqrtNode(subgraph, delegate, logging_context, node_index,
                              node, context->tensors, input_output_tensors);
      case kTfLiteBuiltinSlice:
        return VisitSliceNode(subgraph, delegate, logging_context, node_index,
                              node, context->tensors, input_output_tensors);
      case kTfLiteBuiltinSoftmax: {
        const TfLiteSoftmaxParams* softmax_params =
            static_cast<const TfLiteSoftmaxParams*>(node->builtin_data);

        return VisitSoftmaxNode(subgraph, delegate, logging_context, node_index,
                                node, context->tensors, softmax_params,
                                input_output_tensors);
      }
      case kTfLiteBuiltinSpaceToDepth: {
        const TfLiteSpaceToDepthParams* space_to_depth_params =
            static_cast<const TfLiteSpaceToDepthParams*>(node->builtin_data);

        return VisitSpaceToDepthNode(
            subgraph, delegate, logging_context, node_index, node,
            context->tensors, space_to_depth_params, input_output_tensors);
      }
      case kTfLiteBuiltinSplit: {
        const TfLiteSplitParams* split_params =
            static_cast<const TfLiteSplitParams*>(node->builtin_data);
        return VisitSplitNode(subgraph, delegate, logging_context, node_index,
                              node, context->tensors, split_params,
                              input_output_tensors);
      }
      case kTfLiteBuiltinSqrt:
        return VisitSqrtNode(subgraph, delegate, logging_context, node_index,
                             node, context->tensors, input_output_tensors);
      case kTfLiteBuiltinSquare:
        return VisitSquareNode(subgraph, delegate, logging_context, node_index,
                               node, context->tensors, input_output_tensors);
      case kTfLiteBuiltinSquaredDifference:
        return VisitSquaredDifferenceNode(subgraph, delegate, logging_context,
                                          node_index, node, context->tensors,
                                          input_output_tensors);
      case kTfLiteBuiltinStridedSlice: {
        const auto* params =
            static_cast<const TfLiteStridedSliceParams*>(node->builtin_data);
        return VisitStridedSliceNode(subgraph, delegate, logging_context,
                                     node_index, node, context->tensors, params,
                                     input_output_tensors);
      }
      case kTfLiteBuiltinSub: {
        const TfLiteSubParams* sub_params =
            static_cast<const TfLiteSubParams*>(node->builtin_data);

        return VisitSubNode(subgraph, delegate, logging_context, node_index,
                            node, context->tensors, sub_params,
                            input_output_tensors);
      }
      case kTfLiteBuiltinTanh:
        return VisitTanhNode(subgraph, delegate, logging_context, node_index,
                             node, context->tensors, input_output_tensors);
      case kTfLiteBuiltinTranspose: {
        return VisitTransposeNode(subgraph, delegate, logging_context,
                                  node_index, node, context->tensors,
                                  input_output_tensors);
      }
      case kTfLiteBuiltinTransposeConv: {
        const TfLiteTransposeConvParams* deconv_params =
            static_cast<const TfLiteTransposeConvParams*>(node->builtin_data);

        return VisitTransposeConvNode(subgraph, delegate, logging_context,
                                      node_index, node, context->tensors,
                                      deconv_params, quasi_static_tensors,
                                      input_output_tensors);
      }
      case kTfLiteBuiltinVarHandle:
        return VisitVarHandleNode(subgraph, delegate, logging_context,
                                  node_index, node);
      case kTfLiteBuiltinStablehloComposite: {
        const TfLiteStablehloCompositeParams* composite_params =
            static_cast<const TfLiteStablehloCompositeParams*>(
                node->builtin_data);
        if (strcmp(composite_params->name, kOdmlSDPA) == 0) {
          return VisitScaledDotAttentionCompositeNode(
              subgraph, delegate, context, node_index, node, context->tensors,
              composite_params->attributes, composite_params->attributes_size,
              input_output_tensors);
        } else {
#ifdef XNNPACK_DELEGATE_ENABLE_LOGGING
          TF_LITE_KERNEL_LOG(context,
                             "unsupported stablehlo.composite operator type "
                             "\"%s\" in node #%d",
                             composite_params->name, node_index);
#endif  // XNNPACK_DELEGATE_ENABLE_LOGGING
        }
        return kTfLiteError;
      }
      case kTfLiteBuiltinCustom: {
        if (strcmp(registration->custom_name, "Convolution2DTransposeBias") ==
            0) {
          TfLiteTransposeConvParams deconv_params = {kTfLitePaddingUnknown};
          SafeCopyCustomData(*node, &deconv_params);

          return VisitMediaPipeDeconvolutionNode(
              subgraph, delegate, context, node_index, node, context->tensors,
              &deconv_params, quasi_static_tensors, input_output_tensors);
        } else if (strcmp(registration->custom_name,
                          "MaxPoolingWithArgmax2D") == 0) {
          TfLitePoolParams pool_params = {kTfLitePaddingUnknown};
          SafeCopyCustomData(*node, &pool_params);

          return VisitMediaPipeMaxPoolingNode(
              subgraph, delegate, context, node_index, node, context->tensors,
              &pool_params, input_output_tensors);
        } else if (strcmp(registration->custom_name, "MaxUnpooling2D") == 0) {
          TfLitePoolParams pool_params = {kTfLitePaddingUnknown};
          SafeCopyCustomData(*node, &pool_params);

          return VisitMediaPipeUnpoolingNode(
              subgraph, delegate, context, node_index, node, context->tensors,
              &pool_params, input_output_tensors);
        } else if (strcmp(registration->custom_name, kOdmlSDPA) == 0) {
          return VisitScaledDotAttentionCompositeNode(
              subgraph, delegate, context, node_index, node, context->tensors,
              reinterpret_cast<const uint8_t*>(node->custom_initial_data),
              node->custom_initial_data_size, input_output_tensors);
        } else {
#ifdef XNNPACK_DELEGATE_ENABLE_LOGGING
          TF_LITE_KERNEL_LOG(
              context, "unsupported custom operator type \"%s\" in node #%d",
              registration->custom_name, node_index);
#endif  // XNNPACK_DELEGATE_ENABLE_LOGGING
        }
        return kTfLiteError;
      }
      default:
#ifdef XNNPACK_DELEGATE_ENABLE_LOGGING
        TF_LITE_KERNEL_LOG(context, "unsupported operator type %s in node #%d",
                           EnumNameBuiltinOperator(static_cast<BuiltinOperator>(
                               registration->builtin_code)),
                           node_index);
#endif  // XNNPACK_DELEGATE_ENABLE_LOGGING
        return kTfLiteError;
    }
  }

  static TfLiteStatus VisitAbsNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 1, 1, BuiltinOperator_ABS, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_abs(
          subgraph,
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_ABS),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitAddNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors, const TfLiteAddParams* add_params,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 2, 1, BuiltinOperator_ADD, node_index));

    const TfLiteTensor& input1_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input1_tensor,
                                       node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, input1_tensor, node->inputs->data[0],
        node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, input1_tensor, /*min_num_dims=*/0,
        /*max_num_dims=*/XNN_MAX_TENSOR_DIMS, node->inputs->data[0],
        BuiltinOperator_ADD, node_index));

    const TfLiteTensor& input2_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input2_tensor,
                                       node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, input2_tensor, node->inputs->data[1],
        node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, input2_tensor, /*min_num_dims=*/0,
        /*max_num_dims=*/XNN_MAX_TENSOR_DIMS, node->inputs->data[1],
        BuiltinOperator_ADD, node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (input1_tensor.type != input2_tensor.type ||
        input1_tensor.type != output_tensor.type) {
      TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                               "unsupported mixed types in ADD operator #%d",
                               node_index);
      return kTfLiteError;
    }
    const float scale_min = 1.0f / 1024.0f;
    const float scale_max = 256.0f;
    TF_LITE_ENSURE_STATUS(CheckTensorsInputOutputScale(
        logging_context, input1_tensor, output_tensor, scale_min, scale_max,
        BuiltinOperator_ADD, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorsInputOutputScale(
        logging_context, input2_tensor, output_tensor, scale_min, scale_max,
        BuiltinOperator_ADD, node_index));

    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = +std::numeric_limits<float>::infinity();
    if (add_params != nullptr) {
      TF_LITE_ENSURE_STATUS(ConvertActivationToOutputRange(
          logging_context, node_index, add_params->activation, &output_min,
          &output_max));
    }

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_add2(
          subgraph, output_min, output_max,
          /*input1_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*input2_id=*/input_output_tensors.at(node->inputs->data[1]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_ADD),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitAssignVariableNode(
      xnn_subgraph_t subgraph, Delegate& delegate,
      TfLiteContext* logging_context, int node_index, const TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    if (!delegate.support_variable_ops()) {
      return kTfLiteError;
    }
    if (subgraph == nullptr) {
      const int resource_tensor_id = node->inputs->data[0];
      return delegate.AssociateVariableWithTensor(
          resource_tensor_id, &tensors[node->inputs->data[1]], logging_context);
    } else {
      const xnn_status status = xnn_define_copy(
          subgraph, input_output_tensors.at(node->inputs->data[1]),
          input_output_tensors.at(node->inputs->data[0]), 0 /* flags */);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(
            logging_context, "failed to delegate %s node #%d",
            EnumNameBuiltinOperator(BuiltinOperator_ASSIGN_VARIABLE),
            node_index);
        return kTfLiteError;
      }
    }
    return kTfLiteOk;
  }

  static TfLiteStatus VisitAveragePool2DNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors, const TfLitePoolParams* pool_params,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 1, 1,
                                 BuiltinOperator_AVERAGE_POOL_2D, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    TF_LITE_ENSURE_STATUS(CheckPoolingParams(logging_context, pool_params,
                                             BuiltinOperator_AVERAGE_POOL_2D,
                                             node_index));

    uint32_t flags = 0;
    TF_LITE_ENSURE_STATUS(CalculatePadding(
        logging_context, pool_params->padding, &flags, node_index));

    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = +std::numeric_limits<float>::infinity();
    TF_LITE_ENSURE_STATUS(ConvertActivationToOutputRange(
        logging_context, node_index, pool_params->activation, &output_min,
        &output_max));

    if (subgraph != nullptr) {
      xnn_status status = xnn_status_success;
      if (pool_params->filter_height == 1 && pool_params->filter_width == 1) {
        status = xnn_define_clamp(
            subgraph, output_min, output_max,
            /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
            /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
            /*flags=*/0);
      } else {
        status = xnn_define_average_pooling_2d(
            subgraph,
            /*input_padding_top=*/0,
            /*input_padding_right=*/0,
            /*input_padding_bottom=*/0,
            /*input_padding_left=*/0,
            static_cast<uint32_t>(pool_params->filter_height),
            static_cast<uint32_t>(pool_params->filter_width),
            static_cast<uint32_t>(pool_params->stride_height),
            static_cast<uint32_t>(pool_params->stride_width), output_min,
            output_max,
            /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
            /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
            flags);
      }
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(
            logging_context, "failed to delegate %s node #%d",
            EnumNameBuiltinOperator(BuiltinOperator_AVERAGE_POOL_2D),
            node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitBatchMatMulNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors, const TfLiteBatchMatMulParams* params,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    // Check whether all required options are supported.
    if (params->adj_x) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "failed to delegate %s node #%d. adj_x is not supported",
          EnumNameBuiltinOperator(BuiltinOperator_BATCH_MATMUL), node_index);
      return kTfLiteError;
    }

    // Check the input tensor types.
    const TfLiteTensor& input_a = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_a, node->inputs->data[0], node_index));
    const TfLiteTensor& input_b = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQCInt8Type(
        delegate, logging_context, input_b,
        /*expected_quantized_dimension=*/params->adj_y
            ? NumDimensions(&input_b) - 2
            : NumDimensions(&input_b) - 1,
        node->inputs->data[1], node_index));

    // Check whether input_a will be quantized dynamically.
    const bool dynamically_quantized =
        (input_a.type == kTfLiteFloat32 && input_b.type == kTfLiteInt8);

    // Check the output tensor type.
    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    // Check the input tensor non-dynamic allocations.
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, input_a, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, input_b, node->inputs->data[1], node_index));

    // Check whether the dimensions are compatible.
    const int num_dims_a = NumDimensions(&input_a);
    if (num_dims_a < 2) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "failed to delegate %s node #%d. Unsupported number "
          "of dimensions %d for tensor #%d, must be at least 2",
          EnumNameBuiltinOperator(BuiltinOperator_BATCH_MATMUL), node_index,
          node->inputs->data[0], num_dims_a);
      return kTfLiteError;
    }
    const int num_dims_b = NumDimensions(&input_b);
    if (num_dims_b < 2) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "failed to delegate %s node #%d. Unsupported number "
          "of dimensions %d for tensor #%d, must be at least 2",
          EnumNameBuiltinOperator(BuiltinOperator_BATCH_MATMUL), node_index,
          node->inputs->data[1], num_dims_b);
      return kTfLiteError;
    }
    if (params->adj_x) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "failed to delegate %s node #%d. adj_x is not supported",
          EnumNameBuiltinOperator(BuiltinOperator_BATCH_MATMUL), node_index);
      return kTfLiteError;
    }

    // Create and attach the subgraph nodes.
    if (subgraph != nullptr) {
      const uint32_t flags = params->adj_y ? XNN_FLAG_TRANSPOSE_B : 0;

      // If we're using dynamic quantization, we first need to convert the first
      // input `A` from `float32` to `int8`, and set up the quantization
      // parameters of the already-quantized input `B`.
      if (dynamically_quantized) {
        // Compute some shapes and sizes.
        const int32_t n = params->adj_y
                              ? SizeOfDimension(&input_b, num_dims_b - 2)
                              : SizeOfDimension(&input_b, num_dims_b - 1);
        int32_t batch_size_b = 1;
        for (int i = 0; i < num_dims_b - 2; ++i) {
          batch_size_b *= SizeOfDimension(&input_b, i);
        }

        // Validate or create the quantization parameters for the per-channel
        // quantized input_b. Note that we currently only expect the `B` tensor
        // to be per-tensor quantized, and not per-channel (see b/332675940).
        TfLiteAffineQuantization* quant_params_b =
            reinterpret_cast<TfLiteAffineQuantization*>(
                input_b.quantization.params);
        if (quant_params_b->scale->size != batch_size_b * n) {
          if (quant_params_b->scale->size != 1) {
            TF_LITE_MAYBE_KERNEL_LOG(
                logging_context,
                "failed to delegate %s node #%d. unexpected number of "
                "quantizations scales (expected %d or 1, got %d)",
                EnumNameBuiltinOperator(BuiltinOperator_BATCH_MATMUL),
                node_index, batch_size_b * n, quant_params_b->scale->size);
            return kTfLiteError;
          }
          TfLiteFloatArrayFree(quant_params_b->scale);
          quant_params_b->scale = TfLiteFloatArrayCreate(batch_size_b * n);
          std::fill_n(quant_params_b->scale->data, batch_size_b * n,
                      input_b.params.scale);
          TfLiteIntArrayFree(quant_params_b->zero_point);
          quant_params_b->zero_point = TfLiteIntArrayCreate(batch_size_b * n);
          std::fill_n(quant_params_b->zero_point->data, batch_size_b * n,
                      input_b.params.zero_point);
          quant_params_b->quantized_dimension =
              params->adj_y ? num_dims_b - 2 : num_dims_b - 1;
        }

        // Create the quantized input_b.
        std::vector<size_t> dims_b(num_dims_b, 0);
        for (int i = 0; i < num_dims_b; ++i) {
          dims_b[i] = SizeOfDimension(&input_b, i);
        }
        const int32_t zero_point_value = quant_params_b->zero_point->data[0];
        uint32_t cq_input_b_id = XNN_INVALID_VALUE_ID;
        if (xnn_status status =
                xnn_define_channelwise_quantized_tensor_value_v2(
                    subgraph, xnn_datatype_qcint8, zero_point_value,
                    quant_params_b->scale->data, dims_b.size(),
                    /*channel_dim=*/
                    (params->adj_y ? num_dims_b - 2 : num_dims_b - 1),
                    dims_b.data(), GetTensorData<int8_t>(&input_b),
                    XNN_INVALID_VALUE_ID,
                    /*flags=*/0, &cq_input_b_id);
            status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(
              logging_context, "failed to update filter tensor %s node #%d",
              EnumNameBuiltinOperator(BuiltinOperator_BATCH_MATMUL),
              node_index);
          return kTfLiteError;
        }

        // Create the dynamically quantized input_a.
        uint32_t dq_input_a_id = XNN_INVALID_VALUE_ID;
        size_t dims_a[XNN_MAX_TENSOR_DIMS];
        for (int i = 0; i < num_dims_a; ++i) {
          dims_a[i] = SizeOfDimension(&input_a, i);
        }
        if (xnn_status status = xnn_define_dynamically_quantized_tensor_value(
                subgraph, xnn_datatype_qdint8, num_dims_a,
                /*num_nonbatch_dims=*/1, dims_a, XNN_INVALID_VALUE_ID,
                /*flags=*/0, &dq_input_a_id);
            status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(logging_context,
                             "failed to create XNNPACK Value for tensor %d",
                             -1);
          return kTfLiteError;
        }

        // Define the conversion op for the quantized input_a.
        if (xnn_status status = xnn_define_convert(
                subgraph,
                /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
                dq_input_a_id, /*flags=*/0);
            status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(
              logging_context, "failed to delegate %s node #%d",
              EnumNameBuiltinOperator(BuiltinOperator_BATCH_MATMUL),
              node_index);
          return kTfLiteError;
        }

        // Create the batch_matrix_multiply op.
        if (xnn_status status = xnn_define_batch_matrix_multiply(
                subgraph, dq_input_a_id, cq_input_b_id,
                input_output_tensors.at(node->outputs->data[0]), flags);
            status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(
              logging_context, "failed to delegate %s node #%d",
              EnumNameBuiltinOperator(BuiltinOperator_BATCH_MATMUL),
              node_index);
          return kTfLiteError;
        }

      } else {
        // No conversion of the inputs necessary, just send them on their way.
        if (xnn_status status = xnn_define_batch_matrix_multiply(
                subgraph, input_output_tensors.at(node->inputs->data[0]),
                input_output_tensors.at(node->inputs->data[1]),
                input_output_tensors.at(node->outputs->data[0]), flags);
            status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(
              logging_context, "failed to delegate %s node #%d",
              EnumNameBuiltinOperator(BuiltinOperator_BATCH_MATMUL),
              node_index);
          return kTfLiteError;
        }
      }
    }
    return kTfLiteOk;
  }

  static TfLiteStatus VisitCeilNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 1, 1, BuiltinOperator_CEIL, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_ceiling(
          subgraph,
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_CEIL),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitConcatenationNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const TfLiteConcatenationParams* concat_params,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 5, 1,
                                 BuiltinOperator_CONCATENATION, node_index));
    const int num_inputs = NumInputs(node);

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    // Check dimensions
    if (output_tensor.type == kTfLiteUInt8) {
      const int32_t zero_point =
          tensors[node->outputs->data[0]].params.zero_point;
      const float scale = tensors[node->outputs->data[0]].params.scale;
      for (int i = 0; i < num_inputs; i++) {
        if (tensors[node->inputs->data[i]].params.zero_point != zero_point) {
          TF_LITE_MAYBE_KERNEL_LOG(
              logging_context,
              "Mismatching quantization zero point across the %dth input "
              "(%" PRId32 ") and the output (%" PRId32
              ") for CONCATENATE operator #%d",
              i, tensors[node->inputs->data[i]].params.zero_point, zero_point,
              node_index);
          return kTfLiteError;
        }
        if (tensors[node->inputs->data[i]].params.scale != scale) {
          TF_LITE_MAYBE_KERNEL_LOG(
              logging_context,
              "Mismatching quantization scale across the %dth input (%f) "
              "and the output (%f) for CONCATENATE operator #%d",
              i, tensors[node->inputs->data[i]].params.scale, scale,
              node_index);
          return kTfLiteError;
        }
      }
    }

    for (int i = 0; i < num_inputs; i++) {
      const TfLiteTensor& input_tensor = tensors[node->inputs->data[i]];
      TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQUInt8Type(
          delegate, logging_context, input_tensor, node->inputs->data[i],
          node_index));
      TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
          delegate, logging_context, input_tensor, node->inputs->data[i],
          node_index));
    }

    if (subgraph != nullptr) {
      xnn_status status = xnn_status_invalid_parameter;
      int axis = concat_params->axis;
      if (num_inputs == 2) {
        status = xnn_define_concatenate2(
            subgraph, axis,
            /*input1_id=*/input_output_tensors.at(node->inputs->data[0]),
            /*input2_id=*/input_output_tensors.at(node->inputs->data[1]),
            /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
            /*flags=*/0);
      } else if (num_inputs == 3) {
        status = xnn_define_concatenate3(
            subgraph, axis,
            /*input1_id=*/input_output_tensors.at(node->inputs->data[0]),
            /*input2_id=*/input_output_tensors.at(node->inputs->data[1]),
            /*input3_id=*/input_output_tensors.at(node->inputs->data[2]),
            /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
            /*flags=*/0);
      } else if (num_inputs == 4) {
        status = xnn_define_concatenate4(
            subgraph, axis,
            /*input1_id=*/input_output_tensors.at(node->inputs->data[0]),
            /*input2_id=*/input_output_tensors.at(node->inputs->data[1]),
            /*input3_id=*/input_output_tensors.at(node->inputs->data[2]),
            /*input4_id=*/input_output_tensors.at(node->inputs->data[3]),
            /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
            /*flags=*/0);
      } else if (num_inputs == 5) {
        status = xnn_define_concatenate5(
            subgraph, axis,
            /*input1_id=*/input_output_tensors.at(node->inputs->data[0]),
            /*input2_id=*/input_output_tensors.at(node->inputs->data[1]),
            /*input3_id=*/input_output_tensors.at(node->inputs->data[2]),
            /*input4_id=*/input_output_tensors.at(node->inputs->data[3]),
            /*input5_id=*/input_output_tensors.at(node->inputs->data[4]),
            /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
            /*flags=*/0);
      }
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(
            logging_context, "failed to delegate %s node #%d",
            EnumNameBuiltinOperator(BuiltinOperator_CONCATENATION), node_index);
        return kTfLiteError;
      }
    }
    return kTfLiteOk;
  }

  static TfLiteStatus VisitConv2DNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors, const TfLiteConvParams* conv_params,
      const std::unordered_set<int>& quasi_static_tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckConvolutionParams(logging_context, conv_params, node_index));

    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 3, 1, BuiltinOperator_CONV_2D, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input_tensor,
                                       node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, input_tensor, 4, node->inputs->data[0],
        BuiltinOperator_CONV_2D, node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& filter_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQCInt8Type(
        delegate, logging_context, filter_tensor,
        /*expected_quantized_dimension=*/0, node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, filter_tensor, 4, node->inputs->data[1],
        BuiltinOperator_CONV_2D, node_index));
    if (quasi_static_tensors.count(node->inputs->data[1]) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, filter_tensor, node->inputs->data[1],
          BuiltinOperator_CONV_2D, node_index));
    }

    const int bias_tensor_id = node->inputs->data[2];
    if (bias_tensor_id < 0) {
      TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                               "unsupported CONV_2D node #%d without bias",
                               node_index);
      return kTfLiteError;
    }
    const TfLiteTensor& bias_tensor = tensors[bias_tensor_id];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQCInt32Type(delegate, logging_context, bias_tensor,
                                        node->inputs->data[2], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorShape(logging_context, bias_tensor, 1, node->inputs->data[2],
                         BuiltinOperator_CONV_2D, node_index));
    if (quasi_static_tensors.count(node->inputs->data[2]) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, bias_tensor, node->inputs->data[2],
          BuiltinOperator_CONV_2D, node_index));
    }

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, output_tensor, 4, node->outputs->data[0],
        BuiltinOperator_CONV_2D, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    bool dynamically_quantized = (delegate.enable_latest_operators() &&
                                  (input_tensor.type == kTfLiteFloat32 &&
                                   filter_tensor.type == kTfLiteInt8));
    if (input_tensor.type != output_tensor.type ||
        ((input_tensor.type != filter_tensor.type) && !dynamically_quantized)) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context, "unsupported mixed types in CONV_2D operator #%d",
          node_index);
      return kTfLiteError;
    }

    const int output_channels = SizeOfDimension(&filter_tensor, 0);
    const int kernel_height = SizeOfDimension(&filter_tensor, 1);
    const int kernel_width = SizeOfDimension(&filter_tensor, 2);
    const int input_channels = SizeOfDimension(&filter_tensor, 3);
    const int groups = SizeOfDimension(&input_tensor, 3) / input_channels;
    // Input tensor shape is not yet known.
    if (groups == 0) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "groups of zero is not supported by CONV_2D operator #%d",
          node_index);
      return kTfLiteError;
    }

    uint32_t flags;
    TF_LITE_ENSURE_STATUS(CalculatePadding(
        logging_context, conv_params->padding, &flags, node_index));

    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = +std::numeric_limits<float>::infinity();
    TF_LITE_ENSURE_STATUS(ConvertActivationToOutputRange(
        logging_context, node_index, conv_params->activation, &output_min,
        &output_max));

    if (subgraph != nullptr) {
      if (dynamically_quantized) {
        TfLiteAffineQuantization* filter_params =
            reinterpret_cast<TfLiteAffineQuantization*>(
                filter_tensor.quantization.params);
        if (filter_params->scale->size != output_channels) {
          TfLiteFloatArrayFree(filter_params->scale);
          filter_params->scale = TfLiteFloatArrayCreate(output_channels);
          for (int i = 0; i < output_channels; ++i) {
            filter_params->scale->data[i] = filter_tensor.params.scale;
          }
          TfLiteIntArrayFree(filter_params->zero_point);
          filter_params->zero_point = TfLiteIntArrayCreate(output_channels);
          for (int i = 0; i < output_channels; ++i) {
            filter_params->zero_point->data[i] =
                filter_tensor.params.zero_point;
          }
        }
        uint32_t dq_quantized_id = XNN_INVALID_VALUE_ID;
        std::vector<size_t> input_dims(
            &input_tensor.dims->data[0],
            &input_tensor.dims->data[NumDimensions(&input_tensor)]);
        xnn_status status = xnn_define_dynamically_quantized_tensor_value(
            subgraph, xnn_datatype_qdint8, input_dims.size(),
            /*num_nonbatch_dims=*/3, input_dims.data(), XNN_INVALID_VALUE_ID,
            /*flags=*/0, &dq_quantized_id);
        if (status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(logging_context,
                             "failed to create XNNPACK Value for tensor %d",
                             -1);
          return kTfLiteError;
        }
        status = xnn_define_convert(
            subgraph,
            /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
            dq_quantized_id, /*flags=*/0);
        if (status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                             EnumNameBuiltinOperator(BuiltinOperator_CONV_2D),
                             node_index);
          return kTfLiteError;
        }
        std::vector<size_t> filter_dims(
            &filter_tensor.dims->data[0],
            &filter_tensor.dims->data[NumDimensions(&filter_tensor)]);
        uint32_t kernel_id = XNN_INVALID_VALUE_ID;
        status = xnn_define_channelwise_quantized_tensor_value(
            subgraph, xnn_datatype_qcint8, filter_params->scale->data,
            filter_dims.size(), /*channel_dim=*/0, filter_dims.data(),
            GetTensorData<int8_t>(&filter_tensor), XNN_INVALID_VALUE_ID,
            /*flags=*/0, &kernel_id);
        if (status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(
              logging_context, "failed to update filter tensor %s node #%d",
              EnumNameBuiltinOperator(BuiltinOperator_CONV_2D), node_index);
          return kTfLiteError;
        }
        status = xnn_define_convolution_2d(
            subgraph,
            /*input_padding_top=*/0,
            /*input_padding_right=*/0,
            /*input_padding_bottom=*/0,
            /*input_padding_left=*/0, static_cast<uint32_t>(kernel_height),
            static_cast<uint32_t>(kernel_width),
            static_cast<uint32_t>(conv_params->stride_height),
            static_cast<uint32_t>(conv_params->stride_width),
            static_cast<uint32_t>(conv_params->dilation_height_factor),
            static_cast<uint32_t>(conv_params->dilation_width_factor), groups,
            static_cast<size_t>(input_channels),
            static_cast<size_t>(output_channels) / groups, output_min,
            output_max,
            /*input_id=*/dq_quantized_id,
            /*filter_id=*/kernel_id,
            /*bias_id=*/input_output_tensors.at(node->inputs->data[2]),
            /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
            flags);
        if (status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                             EnumNameBuiltinOperator(BuiltinOperator_CONV_2D),
                             node_index);
          return kTfLiteError;
        }
      } else {
        const xnn_status status = xnn_define_convolution_2d(
            subgraph,
            /*input_padding_top=*/0,
            /*input_padding_right=*/0,
            /*input_padding_bottom=*/0,
            /*input_padding_left=*/0, static_cast<uint32_t>(kernel_height),
            static_cast<uint32_t>(kernel_width),
            static_cast<uint32_t>(conv_params->stride_height),
            static_cast<uint32_t>(conv_params->stride_width),
            static_cast<uint32_t>(conv_params->dilation_height_factor),
            static_cast<uint32_t>(conv_params->dilation_width_factor), groups,
            static_cast<size_t>(input_channels),
            static_cast<size_t>(output_channels) / groups, output_min,
            output_max,
            /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
            /*filter_id=*/input_output_tensors.at(node->inputs->data[1]),
            /*bias_id=*/input_output_tensors.at(node->inputs->data[2]),
            /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
            flags);
        if (status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                             EnumNameBuiltinOperator(BuiltinOperator_CONV_2D),
                             node_index);
          return kTfLiteError;
        }
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitDepthwiseConv2DNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const TfLiteDepthwiseConvParams* dwconv_params,
      const std::unordered_set<int>& quasi_static_tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 3, 1, BuiltinOperator_DEPTHWISE_CONV_2D,
        node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input_tensor,
                                       node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, input_tensor, 4, node->inputs->data[0],
        BuiltinOperator_DEPTHWISE_CONV_2D, node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& filter_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQCInt8Type(
        delegate, logging_context, filter_tensor,
        /*expected_quantized_dimension=*/3, node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, filter_tensor, 4, node->inputs->data[1],
        BuiltinOperator_DEPTHWISE_CONV_2D, node_index));
    if (quasi_static_tensors.count(node->inputs->data[1]) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, filter_tensor, node->inputs->data[1],
          BuiltinOperator_DEPTHWISE_CONV_2D, node_index));
    }

    const int bias_tensor_id = node->inputs->data[2];
    if (bias_tensor_id < 0) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "unsupported DEPTHWISE_CONV_2D node #%d without bias", node_index);
      return kTfLiteError;
    }
    const TfLiteTensor& bias_tensor = tensors[bias_tensor_id];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQCInt32Type(delegate, logging_context, bias_tensor,
                                        node->inputs->data[2], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorShape(logging_context, bias_tensor, 1, node->inputs->data[2],
                         BuiltinOperator_DEPTHWISE_CONV_2D, node_index));
    if (quasi_static_tensors.count(node->inputs->data[2]) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, bias_tensor, node->inputs->data[2],
          BuiltinOperator_DEPTHWISE_CONV_2D, node_index));
    }

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, output_tensor, 4, node->outputs->data[0],
        BuiltinOperator_DEPTHWISE_CONV_2D, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (input_tensor.type != output_tensor.type ||
        input_tensor.type != filter_tensor.type) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "unsupported mixed types in DEPTHWISE_CONV_2D operator #%d",
          node_index);
      return kTfLiteError;
    }

    const int kernel_height = SizeOfDimension(&filter_tensor, 1);
    const int kernel_width = SizeOfDimension(&filter_tensor, 2);
    const int output_channels = SizeOfDimension(&filter_tensor, 3);

    TF_LITE_ENSURE_STATUS(CheckDepthwiseConvolutionParams(
        logging_context, dwconv_params, output_channels, node_index));

    uint32_t flags = 0;
    TF_LITE_ENSURE_STATUS(CalculatePadding(
        logging_context, dwconv_params->padding, &flags, node_index));

    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = +std::numeric_limits<float>::infinity();
    TF_LITE_ENSURE_STATUS(ConvertActivationToOutputRange(
        logging_context, node_index, dwconv_params->activation, &output_min,
        &output_max));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_depthwise_convolution_2d(
          subgraph,
          /*input_padding_top=*/0,
          /*input_padding_right=*/0,
          /*input_padding_bottom=*/0,
          /*input_padding_left=*/0, static_cast<uint32_t>(kernel_height),
          static_cast<uint32_t>(kernel_width),
          static_cast<uint32_t>(dwconv_params->stride_height),
          static_cast<uint32_t>(dwconv_params->stride_width),
          static_cast<uint32_t>(dwconv_params->dilation_height_factor),
          static_cast<uint32_t>(dwconv_params->dilation_width_factor),
          static_cast<uint32_t>(dwconv_params->depth_multiplier),
          /*input_channels=*/
          static_cast<uint32_t>(output_channels /
                                dwconv_params->depth_multiplier),
          output_min, output_max,
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*filter_id=*/input_output_tensors.at(node->inputs->data[1]),
          /*bias_id=*/input_output_tensors.at(node->inputs->data[2]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]), flags);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(
            logging_context, "failed to delegate %s node #%d",
            EnumNameBuiltinOperator(BuiltinOperator_DEPTHWISE_CONV_2D),
            node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitDepthToSpaceNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const TfLiteDepthToSpaceParams* depth_to_space_params,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 1, 1,
                                 BuiltinOperator_DEPTH_TO_SPACE, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input_tensor,
                                       node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (depth_to_space_params->block_size <= 1) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context, "invalid block size (%d) in DEPTH_TO_SPACE node #%d",
          depth_to_space_params->block_size, node_index);
      return kTfLiteError;
    }

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_depth_to_space(
          subgraph,
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*block_size=*/
          static_cast<uint32_t>(depth_to_space_params->block_size),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(
            logging_context, "failed to delegate %s node #%d",
            EnumNameBuiltinOperator(BuiltinOperator_DEPTH_TO_SPACE),
            node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitDequantizeNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 1, 1, BuiltinOperator_DEQUANTIZE, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorQInt8OrQUInt8Type(delegate, logging_context, input_tensor,
                                     node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, input_tensor, /*min_num_dims=*/0,
        /*max_num_dims=*/XNN_MAX_TENSOR_DIMS, node->inputs->data[0],
        BuiltinOperator_DEQUANTIZE, node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_convert(
          subgraph,
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_DEQUANTIZE),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitDivNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors, const TfLiteDivParams* div_params,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 2, 1, BuiltinOperator_DIV, node_index));

    const TfLiteTensor& input1_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input1_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, input1_tensor, node->inputs->data[0],
        node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, input1_tensor, /*min_num_dims=*/0,
        /*max_num_dims=*/XNN_MAX_TENSOR_DIMS, node->inputs->data[0],
        BuiltinOperator_DIV, node_index));

    const TfLiteTensor& input2_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input2_tensor, node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, input2_tensor, node->inputs->data[1],
        node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, input2_tensor, /*min_num_dims=*/0,
        /*max_num_dims=*/XNN_MAX_TENSOR_DIMS, node->inputs->data[1],
        BuiltinOperator_DIV, node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = +std::numeric_limits<float>::infinity();
    if (div_params != nullptr) {
      TF_LITE_ENSURE_STATUS(ConvertActivationToOutputRange(
          logging_context, node_index, div_params->activation, &output_min,
          &output_max));
    }

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_divide(
          subgraph, output_min, output_max,
          /*input1_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*input2_id=*/input_output_tensors.at(node->inputs->data[1]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_DIV),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitEluNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 1, 1, BuiltinOperator_ELU, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQInt8Type(delegate, logging_context, input_tensor,
                                      node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQInt8Type(delegate, logging_context, output_tensor,
                                      node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_elu(
          subgraph, /*alpha=*/1.0f,
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_ELU),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitFullyConnectedNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      TfLiteTensor* tensors, const TfLiteFullyConnectedParams* fc_params,
      const std::unordered_set<int>& quasi_static_tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckFullyConnectedParams(logging_context, fc_params, node_index));

    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 3, 1,
                                 BuiltinOperator_FULLY_CONNECTED, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input_tensor,
                                       node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& filter_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, filter_tensor, 2, node->inputs->data[1],
        BuiltinOperator_FULLY_CONNECTED, node_index));
    // Dynamic filter is supported, but only for FP32.
    if (delegate.support_dynamic_fully_connected_operator() &&
        filter_tensor.type == kTfLiteFloat32) {
      TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
          delegate, logging_context, filter_tensor, node->inputs->data[1],
          node_index));
    } else {
      TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQCInt4OrQCInt8Type(
          delegate, logging_context, filter_tensor,
          /*expected_quantized_dimension=*/0, node->inputs->data[1],
          node_index));
      if (quasi_static_tensors.count(node->inputs->data[1]) == 0) {
        TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
            logging_context, filter_tensor, node->inputs->data[1],
            BuiltinOperator_FULLY_CONNECTED, node_index));
      }
    }

    const int32_t output_channels = SizeOfDimension(&filter_tensor, 0);
    const int32_t input_channels = SizeOfDimension(&filter_tensor, 1);

    int bias_tensor_id = -1;
    if (node->inputs->size >= 3) {
      bias_tensor_id = node->inputs->data[2];
      if (bias_tensor_id >= 0) {
        const TfLiteTensor& bias_tensor = tensors[bias_tensor_id];
        // Dynamic bias is supported, but only for FP32.
        if (delegate.support_dynamic_fully_connected_operator() &&
            bias_tensor.type == kTfLiteFloat32) {
          TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
              delegate, logging_context, bias_tensor, node->inputs->data[2],
              node_index));
        } else {
          const int num_bias_elements = NumElements(&bias_tensor);
          if (num_bias_elements != output_channels) {
            TF_LITE_MAYBE_KERNEL_LOG(
                logging_context,
                "Fully Connected: Mismatch between number of bias elements %d "
                "and number of output channels %d at node %d",
                num_bias_elements, output_channels, node->inputs->data[0]);
            return kTfLiteError;
          }
          TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQCInt32Type(
              delegate, logging_context, bias_tensor, node->inputs->data[2],
              node_index));
          if (quasi_static_tensors.count(node->inputs->data[2]) == 0) {
            TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
                logging_context, bias_tensor, node->inputs->data[2],
                BuiltinOperator_FULLY_CONNECTED, node_index));
          }
        }
      }
    }

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    bool dynamically_quantized = (delegate.enable_latest_operators() &&
                                  (input_tensor.type == kTfLiteFloat32 &&
                                   (filter_tensor.type == kTfLiteInt4 ||
                                    filter_tensor.type == kTfLiteInt8)));
    if (input_tensor.type != output_tensor.type ||
        ((input_tensor.type != filter_tensor.type) && !dynamically_quantized)) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "unsupported mixed types in FULLY_CONNECTED operator #%d",
          node_index);
      return kTfLiteError;
    }

    if (NumDimensions(&input_tensor) == 0) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "unexpected number of shape dimensions %d in tensor #%d",
          NumDimensions(&input_tensor), node->inputs->data[0]);
      return kTfLiteError;
    }

    if (filter_tensor.type == kTfLiteInt4 && input_channels % 2 == 1) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "unsupported odd number of inputs channels (%d) in FULLY_CONNECTED"
          " operator #%d",
          input_channels, node_index);
      return kTfLiteError;
    }

    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = +std::numeric_limits<float>::infinity();
    TF_LITE_ENSURE_STATUS(ConvertActivationToOutputRange(
        logging_context, node_index, fc_params->activation, &output_min,
        &output_max));

    uint32_t dq_quantized_id = XNN_INVALID_VALUE_ID;
    if (subgraph != nullptr) {
      if (dynamically_quantized) {
        TfLiteAffineQuantization* filter_params =
            reinterpret_cast<TfLiteAffineQuantization*>(
                filter_tensor.quantization.params);
        xnn_datatype filter_datatype = GetXNNPackDatatype(
            logging_context, filter_tensor, node->inputs->data[1]);
        if (filter_datatype == xnn_datatype_qint8) {
          filter_datatype = xnn_datatype_qcint8;
          TfLiteFloatArrayFree(filter_params->scale);
          filter_params->scale = TfLiteFloatArrayCreate(output_channels);
          std::fill_n(filter_params->scale->data, output_channels,
                      filter_tensor.params.scale);
          TfLiteIntArrayFree(filter_params->zero_point);
          filter_params->zero_point = TfLiteIntArrayCreate(output_channels);
          for (int i = 0; i < output_channels; ++i) {
            filter_params->zero_point->data[i] =
                filter_tensor.params.zero_point;
          }
        }
        std::vector<size_t> input_dims(
            &input_tensor.dims->data[0],
            &input_tensor.dims->data[NumDimensions(&input_tensor)]);
        xnn_status status = xnn_define_dynamically_quantized_tensor_value(
            subgraph, xnn_datatype_qdint8, input_dims.size(),
            /*num_non_batch_dims=*/1, input_dims.data(), XNN_INVALID_VALUE_ID,
            /*flags=*/0, &dq_quantized_id);
        if (status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(logging_context,
                             "failed to create XNNPACK Value for tensor %d",
                             -1);
          return kTfLiteError;
        }
        // TODO(b/340399245) - Remove the flags parameter once we have a better
        // solution for delegating to `qp8_f32_qc4w` GEMM kernels.
        const size_t convert_flags =
            (filter_datatype == xnn_datatype_qcint4 &&
             filter_tensor.params.zero_point == 8)
                ? 0x00000080 /*XNN_FLAG_MAYBE_PACK_FOR_GEMM*/
                : 0;
        status = xnn_define_convert(
            subgraph,
            /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
            dq_quantized_id, convert_flags);
        if (status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(
              logging_context, "failed to delegate %s node #%d",
              EnumNameBuiltinOperator(BuiltinOperator_FULLY_CONNECTED),
              node_index);
          return kTfLiteError;
        }
        std::vector<size_t> filter_dims(
            &filter_tensor.dims->data[0],
            &filter_tensor.dims->data[NumDimensions(&filter_tensor)]);
        int32_t zero_point_value = filter_params->zero_point->data[0];
        uint32_t kernel_id = XNN_INVALID_VALUE_ID;
        status = xnn_define_channelwise_quantized_tensor_value_v2(
            subgraph, filter_datatype, zero_point_value,
            filter_params->scale->data, filter_dims.size(), /*channel_dim=*/0,
            filter_dims.data(), GetTensorData<int8_t>(&filter_tensor),
            XNN_INVALID_VALUE_ID,
            /*flags=*/0, &kernel_id);
        if (status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(
              logging_context, "failed to update filter tensor %s node #%d",
              EnumNameBuiltinOperator(BuiltinOperator_FULLY_CONNECTED),
              node_index);
          return kTfLiteError;
        }
        status = xnn_define_fully_connected(
            subgraph, output_min, output_max, dq_quantized_id, kernel_id,
            /*bias_id=*/bias_tensor_id >= 0
                ? input_output_tensors.at(bias_tensor_id)
                : XNN_INVALID_VALUE_ID,
            /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
            /*flags=*/fc_params->keep_num_dims
                ? 0
                : XNN_FLAG_TENSORFLOW_RESHAPE_2D);
        if (status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(
              logging_context, "failed to delegate %s node #%d",
              EnumNameBuiltinOperator(BuiltinOperator_FULLY_CONNECTED),
              node_index);
          return kTfLiteError;
        }
      } else {
        const xnn_status status = xnn_define_fully_connected(
            subgraph, output_min, output_max,
            /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
            /*filter_id=*/input_output_tensors.at(node->inputs->data[1]),
            /*bias_id=*/bias_tensor_id >= 0
                ? input_output_tensors.at(bias_tensor_id)
                : XNN_INVALID_VALUE_ID,
            /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
            /*flags=*/fc_params->keep_num_dims
                ? 0
                : XNN_FLAG_TENSORFLOW_RESHAPE_2D);
        if (status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(
              logging_context, "failed to delegate %s node #%d",
              EnumNameBuiltinOperator(BuiltinOperator_FULLY_CONNECTED),
              node_index);
          return kTfLiteError;
        }
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitFloorNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 1, 1, BuiltinOperator_FLOOR, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_floor(
          subgraph,
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_FLOOR),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitGeluNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 1, 1, BuiltinOperator_GELU, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_gelu(
          subgraph,
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_GELU),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitHardSwishNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 1, 1, BuiltinOperator_HARD_SWISH, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_hardswish(
          subgraph,
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_HARD_SWISH),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitLeakyReluNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const TfLiteLeakyReluParams* leaky_relu_params,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 1, 1, BuiltinOperator_LEAKY_RELU, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input_tensor,
                                       node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (!std::isnormal(leaky_relu_params->alpha) ||
        leaky_relu_params->alpha == 0.0f) {
      TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                               "unsupported alpha %g in LEAKY_RELU node #%d",
                               leaky_relu_params->alpha, node_index);
      return kTfLiteError;
    }

    const float input_scale =
        GetTensorScaleOrDefault(input_tensor, std::nanf(""));
    const float output_scale =
        GetTensorScaleOrDefault(output_tensor, std::nanf(""));
    if (std::isnormal(input_scale) && std::isnormal(output_scale)) {
      const float positive_scale = input_scale / output_scale;
      if (positive_scale < 1.0f / 256.0f || positive_scale > 128.0f) {
        TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                                 "unsupported positive input-to-output scale "
                                 "%g in LEAKY_RELU node #%d",
                                 positive_scale, node_index);
        return kTfLiteError;
      }

      const float negative_scale = positive_scale * leaky_relu_params->alpha;
      if (negative_scale < -127.99609375f || negative_scale > 128.0f ||
          std::fabs(negative_scale) < 1.0f / 256.0f) {
        TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                                 "unsupported negative input-to-output scale "
                                 "%g in LEAKY_RELU node #%d",
                                 negative_scale, node_index);
        return kTfLiteError;
      }
    }

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_leaky_relu(
          subgraph, leaky_relu_params->alpha,
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_LEAKY_RELU),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitLogisticNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 1, 1, BuiltinOperator_LOGISTIC, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input_tensor,
                                       node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_sigmoid(
          subgraph,
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_LOGISTIC),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMaxPool2DNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors, const TfLitePoolParams* pool_params,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 1, 1, BuiltinOperator_MAX_POOL_2D, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input_tensor,
                                       node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    TF_LITE_ENSURE_STATUS(CheckPoolingParams(
        logging_context, pool_params, BuiltinOperator_MAX_POOL_2D, node_index));

    uint32_t flags = 0;
    TF_LITE_ENSURE_STATUS(CalculatePadding(
        logging_context, pool_params->padding, &flags, node_index));

    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = +std::numeric_limits<float>::infinity();
    TF_LITE_ENSURE_STATUS(ConvertActivationToOutputRange(
        logging_context, node_index, pool_params->activation, &output_min,
        &output_max));

    if (subgraph != nullptr) {
      xnn_status status = xnn_status_success;
      if (pool_params->filter_height == 1 && pool_params->filter_width == 1) {
        status = xnn_define_clamp(
            subgraph, output_min, output_max,
            /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
            /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
            /*flags=*/0);
      } else {
        status = xnn_define_max_pooling_2d(
            subgraph,
            /*input_padding_top=*/0,
            /*input_padding_right=*/0,
            /*input_padding_bottom=*/0,
            /*input_padding_left=*/0,
            static_cast<uint32_t>(pool_params->filter_height),
            static_cast<uint32_t>(pool_params->filter_width),
            static_cast<uint32_t>(pool_params->stride_height),
            static_cast<uint32_t>(pool_params->stride_width),
            /*dilation_height=*/1, /*dilation_width=*/1, output_min, output_max,
            /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
            /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
            flags);
      }
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_MAX_POOL_2D),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitSumNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors, const TfLiteReducerParams* reducer_params,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 2, 1, BuiltinOperator_SUM, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, input_tensor, 4,
                                           node->inputs->data[0],
                                           BuiltinOperator_SUM, node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& axes_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorType(logging_context, axes_tensor,
                                          kTfLiteInt32, node->inputs->data[1],
                                          node_index));
    TF_LITE_ENSURE_STATUS(CheckAxesTensorShape(
        logging_context, axes_tensor, node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, axes_tensor, node->inputs->data[1],
        BuiltinOperator_SUM, node_index));

    const int32_t* axes_data =
        reinterpret_cast<const int32_t*>(axes_tensor.data.data);
    const int num_reduction_axes = NumElements(&axes_tensor);
    switch (num_reduction_axes) {
      case 1:
        if (axes_data[0] != 2) {
          TF_LITE_MAYBE_KERNEL_LOG(
              logging_context,
              "unsupported SUM reduction along non-spatial "
              "axis %d in node %d",
              axes_data[0], node_index);
          return kTfLiteError;
        }
        break;
      case 2:
        if (std::min(axes_data[0], axes_data[1]) != 1 ||
            std::max(axes_data[0], axes_data[1]) != 2) {
          TF_LITE_MAYBE_KERNEL_LOG(
              logging_context,
              "unsupported SUM reduction along non-spatial "
              "axes %d and %d in node %d",
              std::min(axes_data[0], axes_data[1]),
              std::max(axes_data[0], axes_data[1]), node_index);
          return kTfLiteError;
        }
        break;
      default:
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context,
            "unsupported SUM reduction along %d axes in node %d",
            SizeOfDimension(&axes_tensor, 0), node_index);
        return kTfLiteError;
    }

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       node->outputs->data[0], node_index));
    int expected_output_dims = 4;
    uint32_t flags = 0;
    if (!reducer_params->keep_dims) {
      expected_output_dims -= num_reduction_axes;
    } else {
      flags = XNN_FLAG_KEEP_DIMS;
    }
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, output_tensor, expected_output_dims,
        node->outputs->data[0], BuiltinOperator_SUM, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    const float output_min = -std::numeric_limits<float>::infinity();
    const float output_max = +std::numeric_limits<float>::infinity();

    if (subgraph != nullptr) {
      xnn_status status = xnn_status_success;
      switch (num_reduction_axes) {
        case 1:
          status = xnn_define_global_sum_pooling_1d(
              subgraph, output_min, output_max,
              /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
              /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
              flags);
          break;
        case 2:
          status = xnn_define_global_sum_pooling_2d(
              subgraph, output_min, output_max,
              /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
              /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
              flags);
          break;
        default:
          status = xnn_status_unsupported_parameter;
          break;
      }
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_SUM),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMaximumNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 2, 1, BuiltinOperator_MAXIMUM, node_index));

    const TfLiteTensor& input1_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input1_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, input1_tensor, node->inputs->data[0],
        node_index));

    const TfLiteTensor& input2_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input2_tensor, node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, input2_tensor, node->inputs->data[1],
        node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_maximum2(
          subgraph,
          /*input1_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*input2_id=*/input_output_tensors.at(node->inputs->data[1]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_MAXIMUM),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMeanNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors, const TfLiteReducerParams* reducer_params,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 2, 1, BuiltinOperator_MEAN, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input_tensor,
                                       node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& axes_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorType(logging_context, axes_tensor,
                                          kTfLiteInt32, node->inputs->data[1],
                                          node_index));
    TF_LITE_ENSURE_STATUS(CheckAxesTensorShape(
        logging_context, axes_tensor, node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, axes_tensor, node->inputs->data[1],
        BuiltinOperator_MEAN, node_index));

    const int32_t* axes_data =
        reinterpret_cast<const int32_t*>(axes_tensor.data.data);
    const int num_reduction_axes = NumElements(&axes_tensor);
    bool all_reductions_supported = false;
    bool use_legacy_path = false;
    if (input_tensor.type == kTfLiteFloat32) {
      all_reductions_supported = true;
      if (NumDimensions(&input_tensor) == 4) {
        use_legacy_path = true;
      }
    } else {
      TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, input_tensor, 4,
                                             node->inputs->data[0],
                                             BuiltinOperator_MEAN, node_index));
    }
    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       node->outputs->data[0], node_index));
    switch (num_reduction_axes) {
      case 1:
        if (axes_data[0] != 2) {
          if (all_reductions_supported) {
            use_legacy_path = false;
          } else {
            TF_LITE_MAYBE_KERNEL_LOG(
                logging_context,
                "unsupported MEAN reduction along non-spatial "
                "axis %d in node %d",
                axes_data[0], node_index);
            return kTfLiteError;
          }
        }
        break;
      case 2:
        if (std::min(axes_data[0], axes_data[1]) != 1 ||
            std::max(axes_data[0], axes_data[1]) != 2) {
          if (all_reductions_supported) {
            use_legacy_path = false;
          } else {
            TF_LITE_MAYBE_KERNEL_LOG(
                logging_context,
                "unsupported MEAN reduction along non-spatial "
                "axes %d and %d in node %d",
                std::min(axes_data[0], axes_data[1]),
                std::max(axes_data[0], axes_data[1]), node_index);
            return kTfLiteError;
          }
        }
        break;
      default:
        if (all_reductions_supported) {
          use_legacy_path = false;
        } else {
          TF_LITE_MAYBE_KERNEL_LOG(
              logging_context,
              "unsupported MEAN reduction along %d axes in node %d",
              SizeOfDimension(&axes_tensor, 0), node_index);
          return kTfLiteError;
        }
    }
    int expected_output_dims = 4;
    if (!reducer_params->keep_dims) {
      expected_output_dims -= num_reduction_axes;
    }
    if (NumDimensions(&output_tensor) != expected_output_dims) {
      if (all_reductions_supported) {
        use_legacy_path = false;
      } else {
        TF_LITE_ENSURE_STATUS(CheckTensorShape(
            logging_context, output_tensor, expected_output_dims,
            node->outputs->data[0], BuiltinOperator_MEAN, node_index));
      }
    }

    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (subgraph != nullptr) {
      uint32_t flags = reducer_params->keep_dims ? XNN_FLAG_KEEP_DIMS : 0;
      xnn_status status = xnn_status_success;
      if (all_reductions_supported && !use_legacy_path) {
        std::array<size_t, XNN_MAX_TENSOR_DIMS> reduction_axes;
        for (int i = 0; i < num_reduction_axes; ++i) {
          if (axes_data[i] < 0) {
            reduction_axes[i] = axes_data[i] + NumDimensions(&input_tensor);
          } else {
            reduction_axes[i] = axes_data[i];
          }
        }
        std::sort(&reduction_axes[0], &reduction_axes[num_reduction_axes]);
        status = xnn_define_static_mean(
            subgraph, num_reduction_axes, reduction_axes.data(),
            /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
            /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
            flags);
      } else {
        switch (num_reduction_axes) {
          case 1:
            status = xnn_define_global_average_pooling_1d(
                subgraph,
                /*output_min=*/-std::numeric_limits<float>::infinity(),
                /*output_max=*/+std::numeric_limits<float>::infinity(),
                /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
                /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
                flags);
            break;
          case 2:
            status = xnn_define_global_average_pooling_2d(
                subgraph,
                /*output_min=*/-std::numeric_limits<float>::infinity(),
                /*output_max=*/+std::numeric_limits<float>::infinity(),
                /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
                /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
                flags);
            break;
          default:
            break;
        }
      }
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_MEAN),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMediaPipeDeconvolutionNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const TfLiteTransposeConvParams* deconv_params,
      const std::unordered_set<int>& quasi_static_tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 3, 1, BuiltinOperator_CUSTOM, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, input_tensor, 4,
                                           node->inputs->data[0],
                                           BuiltinOperator_CUSTOM, node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& filter_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, filter_tensor, node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, filter_tensor, 4,
                                           node->inputs->data[1],
                                           BuiltinOperator_CUSTOM, node_index));
    if (quasi_static_tensors.count(node->inputs->data[1]) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, filter_tensor, node->inputs->data[1],
          BuiltinOperator_CUSTOM, node_index));
    }

    const TfLiteTensor& bias_tensor = tensors[node->inputs->data[2]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, bias_tensor, node->inputs->data[2], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, bias_tensor, 1,
                                           node->inputs->data[2],
                                           BuiltinOperator_CUSTOM, node_index));
    if (quasi_static_tensors.count(node->inputs->data[2]) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, bias_tensor, node->inputs->data[2],
          BuiltinOperator_CUSTOM, node_index));
    }

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_tensor, 4,
                                           node->outputs->data[0],
                                           BuiltinOperator_CUSTOM, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    const int* input_tensor_dims = input_tensor.dims->data;
    const int input_height = input_tensor_dims[1];
    const int input_width = input_tensor_dims[2];

    const int* output_tensor_dims = output_tensor.dims->data;
    const int output_height = output_tensor_dims[1];
    const int output_width = output_tensor_dims[2];

    const int output_channels = SizeOfDimension(&filter_tensor, 0);
    const int kernel_height = SizeOfDimension(&filter_tensor, 1);
    const int kernel_width = SizeOfDimension(&filter_tensor, 2);
    const int input_channels = SizeOfDimension(&filter_tensor, 3);

    TF_LITE_ENSURE_STATUS(CheckMediaPipeTransposedConvolutionParams(
        logging_context, deconv_params, node_index));

    int padding_top = 0;
    int padding_bottom = 0;
    int padding_left = 0;
    int padding_right = 0;
    int adjustment_height = 0;
    int adjustment_width = 0;
    TF_LITE_ENSURE_STATUS(CalculateTransposeConvPaddings(
        logging_context, deconv_params->padding, input_height, input_width,
        kernel_height, kernel_width, /*dilation_height=*/1,
        /*dilation_width=*/1, deconv_params->stride_height,
        deconv_params->stride_width, node_index, output_height, output_width,
        &padding_top, &padding_bottom, &padding_left, &padding_right,
        &adjustment_height, &adjustment_width));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_deconvolution_2d(
          subgraph,
          /*padding_top=*/padding_top,
          /*padding_right=*/padding_right,
          /*padding_bottom=*/padding_bottom,
          /*padding_left=*/padding_left,
          /*adjustment_height=*/adjustment_height,
          /*adjustment_width=*/adjustment_width,
          static_cast<uint32_t>(kernel_height),
          static_cast<uint32_t>(kernel_width),
          static_cast<uint32_t>(deconv_params->stride_height),
          static_cast<uint32_t>(deconv_params->stride_width),
          /*dilation_height=*/1,
          /*dilation_width=*/1,
          /*groups=*/1,
          /*group_input_channels=*/input_channels,
          /*group_output_channels=*/output_channels,
          /*output_min=*/-std::numeric_limits<float>::infinity(),
          /*output_max=*/+std::numeric_limits<float>::infinity(),
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*filter_id=*/input_output_tensors.at(node->inputs->data[1]),
          /*bias_id=*/input_output_tensors.at(node->inputs->data[2]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context,
                           "failed to delegate CUSTOM(%s) node #%d",
                           "Convolution2DTransposeBias", node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMediaPipeMaxPoolingNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors, const TfLitePoolParams* pool_params,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 1, 2, BuiltinOperator_CUSTOM, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, input_tensor, 4,
                                           node->inputs->data[0],
                                           BuiltinOperator_CUSTOM, node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& output_value_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32Type(logging_context, output_value_tensor,
                               node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_value_tensor,
                                           4, node->outputs->data[0],
                                           BuiltinOperator_CUSTOM, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_value_tensor, node->outputs->data[0],
        node_index));

    const TfLiteTensor& output_index_tensor = tensors[node->outputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_index_tensor,
                                           4, node->outputs->data[1],
                                           BuiltinOperator_CUSTOM, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_index_tensor, node->outputs->data[1],
        node_index));

    TF_LITE_ENSURE_STATUS(
        CheckMediaPipePoolParams(logging_context, pool_params, node_index));

    uint32_t flags = 0;
    TF_LITE_ENSURE_STATUS(CalculatePadding(
        logging_context, pool_params->padding, &flags, node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_argmax_pooling_2d(
          subgraph,
          /*input_padding_top=*/0,
          /*input_padding_right=*/0,
          /*input_padding_bottom=*/0,
          /*input_padding_left=*/0,
          static_cast<uint32_t>(pool_params->filter_height),
          static_cast<uint32_t>(pool_params->filter_width),
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_value_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*output_index_id=*/input_output_tensors.at(node->outputs->data[1]),
          flags);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context,
                           "failed to delegate CUSTOM(%s) node #%d",
                           "MaxPoolingWithArgmax2D", node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMediaPipeUnpoolingNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors, const TfLitePoolParams* pool_params,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 2, 1, BuiltinOperator_CUSTOM, node_index));

    const TfLiteTensor& input_value_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32Type(logging_context, input_value_tensor,
                               node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, input_value_tensor,
                                           4, node->inputs->data[0],
                                           BuiltinOperator_CUSTOM, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, input_value_tensor, node->inputs->data[0],
        node_index));

    const TfLiteTensor& input_index_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, input_index_tensor,
                                           4, node->inputs->data[1],
                                           BuiltinOperator_CUSTOM, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, input_index_tensor, node->inputs->data[1],
        node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_tensor, 4,
                                           node->outputs->data[0],
                                           BuiltinOperator_CUSTOM, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    TF_LITE_ENSURE_STATUS(
        CheckMediaPipePoolParams(logging_context, pool_params, node_index));

    uint32_t flags = 0;
    TF_LITE_ENSURE_STATUS(CalculatePadding(
        logging_context, pool_params->padding, &flags, node_index));
    if (flags != 0) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context, "invalid padding mode (%d) in node #%d",
          static_cast<int>(pool_params->padding), node_index);
    }

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_unpooling_2d(
          subgraph,
          /*padding_top=*/0,
          /*padding_right=*/0,
          /*padding_bottom=*/0,
          /*padding_left=*/0, static_cast<uint32_t>(pool_params->filter_height),
          static_cast<uint32_t>(pool_params->filter_width),
          /*input_value_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*input_index_id=*/input_output_tensors.at(node->inputs->data[1]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context,
                           "failed to delegate CUSTOM(%s) node #%d",
                           "MaxUnpooling2D", node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMinimumNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 2, 1, BuiltinOperator_MINIMUM, node_index));

    const TfLiteTensor& input1_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input1_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, input1_tensor, node->inputs->data[0],
        node_index));

    const TfLiteTensor& input2_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input2_tensor, node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, input2_tensor, node->inputs->data[1],
        node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_minimum2(
          subgraph,
          /*input1_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*input2_id=*/input_output_tensors.at(node->inputs->data[1]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_MINIMUM),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMulNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors, const TfLiteMulParams* mul_params,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 2, 1, BuiltinOperator_MUL, node_index));

    const TfLiteTensor& input1_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input1_tensor,
                                       node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, input1_tensor, node->inputs->data[0],
        node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, input1_tensor, /*min_num_dims=*/0,
        /*max_num_dims=*/XNN_MAX_TENSOR_DIMS, node->inputs->data[0],
        BuiltinOperator_MUL, node_index));

    const TfLiteTensor& input2_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input2_tensor,
                                       node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, input2_tensor, node->inputs->data[1],
        node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, input2_tensor, /*min_num_dims=*/0,
        /*max_num_dims=*/XNN_MAX_TENSOR_DIMS, node->inputs->data[1],
        BuiltinOperator_MUL, node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    const float scale_min = 1.0f / 65536.0f;
    const float scale_max = 256.0f;
    TF_LITE_ENSURE_STATUS(CheckTensorsInputProductOutputScale(
        logging_context, input1_tensor, input2_tensor, output_tensor, scale_min,
        scale_max, BuiltinOperator_MUL, node_index));

    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = +std::numeric_limits<float>::infinity();
    if (mul_params != nullptr) {
      TF_LITE_ENSURE_STATUS(ConvertActivationToOutputRange(
          logging_context, node_index, mul_params->activation, &output_min,
          &output_max));
    }

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_multiply2(
          subgraph, output_min, output_max,
          /*input1_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*input2_id=*/input_output_tensors.at(node->inputs->data[1]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_MUL),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitNegNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 1, 1, BuiltinOperator_NEG, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_negate(
          subgraph,
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_NEG),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitPadNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 2, 1, BuiltinOperator_PAD, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input_tensor,
                                       node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, input_tensor, 1, XNN_MAX_TENSOR_DIMS,
        node->inputs->data[0], BuiltinOperator_PAD, node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& paddings_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorType(logging_context, paddings_tensor,
                                          kTfLiteInt32, node->inputs->data[1],
                                          node_index));
    TF_LITE_ENSURE_STATUS(CheckPaddingsTensorShape(
        logging_context, paddings_tensor, NumDimensions(&input_tensor),
        node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, paddings_tensor, node->inputs->data[1],
        BuiltinOperator_PAD, node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, output_tensor, 1, XNN_MAX_TENSOR_DIMS,
        node->outputs->data[0], BuiltinOperator_PAD, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    const int32_t* paddings_data =
        reinterpret_cast<const int32_t*>(paddings_tensor.data.data);
    for (int i = 0; i < NumDimensions(&paddings_tensor); i++) {
      const int32_t pre_padding = paddings_data[i * 2 + 0];
      if (pre_padding < 0) {
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context,
            "invalid pre-padding %d for dimension #%d in node %d", pre_padding,
            i, node_index);
        return kTfLiteError;
      }

      const int32_t post_padding = paddings_data[i * 2 + 1];
      if (post_padding < 0) {
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context,
            "invalid post-padding %d for dimension #%d in node %d", pre_padding,
            i, node_index);
        return kTfLiteError;
      }
    }

    if (subgraph != nullptr) {
      std::array<size_t, XNN_MAX_TENSOR_DIMS> pre_paddings{};
      std::array<size_t, XNN_MAX_TENSOR_DIMS> post_paddings{};
      for (int i = 0; i < SizeOfDimension(&paddings_tensor, 0); i++) {
        pre_paddings[i] = static_cast<size_t>(paddings_data[i * 2 + 0]);
        post_paddings[i] = static_cast<size_t>(paddings_data[i * 2 + 1]);
      }

      const xnn_status status = xnn_define_static_constant_pad(
          subgraph, pre_paddings.data(), post_paddings.data(),
          /*padding_value=*/0.0f,
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_PAD),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitPreluNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_set<int>& quasi_static_tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 2, 1, BuiltinOperator_PRELU, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, input_tensor, 1, XNN_MAX_TENSOR_DIMS,
        node->inputs->data[0], BuiltinOperator_PRELU, node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& slope_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, slope_tensor, node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckSlopeTensorShape(
        logging_context, slope_tensor, node->inputs->data[1],
        BuiltinOperator_PRELU, node_index));
    if (quasi_static_tensors.count(node->inputs->data[1]) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, slope_tensor, node->inputs->data[1],
          BuiltinOperator_PRELU, node_index));
    }

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, output_tensor, 1, XNN_MAX_TENSOR_DIMS,
        node->outputs->data[0], BuiltinOperator_PRELU, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_prelu(
          subgraph,
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*slope_id=*/input_output_tensors.at(node->inputs->data[1]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_PRELU),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitQuantizeNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 1, 1, BuiltinOperator_QUANTIZE, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input_tensor,
                                       node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorQInt8OrQUInt8Type(delegate, logging_context, output_tensor,
                                     node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, input_tensor, /*min_num_dims=*/0,
        /*max_num_dims=*/XNN_MAX_TENSOR_DIMS, node->inputs->data[0],
        BuiltinOperator_QUANTIZE, node_index));

    const xnn_datatype input_datatype = GetXNNPackDatatype(
        logging_context, input_tensor, node->inputs->data[0]);
    const xnn_datatype output_datatype = GetXNNPackDatatype(
        logging_context, output_tensor, node->outputs->data[0]);
    bool supported_combination = false;
    switch (input_datatype) {
      case xnn_datatype_fp32:
        supported_combination = true;
        break;
      case xnn_datatype_qint8:
      case xnn_datatype_quint8:
        if (input_datatype == output_datatype) {
          const float input_scale =
              GetTensorScaleOrDefault(input_tensor, std::nanf(""));
          const float output_scale =
              GetTensorScaleOrDefault(output_tensor, std::nanf(""));
          const float input_output_scale = input_scale / output_scale;
          if (input_output_scale < 1.0f / 256.0f ||
              input_output_scale > 128.0f) {
            TF_LITE_MAYBE_KERNEL_LOG(
                logging_context,
                "unsupported input-to-output scale in QUANTIZE node #%d",
                node_index);
            return kTfLiteError;
          }
          supported_combination = true;
        }
        break;
      default:
        break;
    }
    if (!supported_combination) {
      TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                               "unsupported combination of input type (%s) and "
                               "output type (%s) in QUANTIZE node #%d",
                               TfLiteTypeGetName(input_tensor.type),
                               TfLiteTypeGetName(output_tensor.type),
                               node_index);
      return kTfLiteError;
    }

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_convert(
          subgraph,
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_QUANTIZE),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitReadVariableNode(
      xnn_subgraph_t subgraph, Delegate& delegate,
      TfLiteContext* logging_context, int node_index, const TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    if (!delegate.support_variable_ops()) {
      return kTfLiteError;
    }
    const int resource_tensor_id = node->inputs->data[0];
    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];

    if (subgraph == nullptr) {
      // This could be a scalar or unranked tensor, we don't support
      // unranked tensor so skip it.
      // TODO(b/245990811): try to support this, we can delay associating
      // dim and type with this tensor, assuming that another operation will
      // provide it, then check that we have dim and type later when
      // defining tensors.
      if (output_tensor.dims->size == 0) {
        return kTfLiteError;
      }
      return delegate.AssociateVariableWithTensor(
          resource_tensor_id, &tensors[node->outputs->data[0]],
          logging_context);
    } else {
      const xnn_status status = xnn_define_copy(
          subgraph, input_output_tensors.at(resource_tensor_id),
          input_output_tensors.at(output_tensor_id), 0 /* flags */);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(
            logging_context, "failed to delegate %s node #%d",
            EnumNameBuiltinOperator(BuiltinOperator_READ_VARIABLE), node_index);
        return kTfLiteError;
      }
    }
    return kTfLiteOk;
  }

  static TfLiteStatus VisitReluNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors, float output_min, float output_max,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 1, 1, BuiltinOperator_RELU, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_clamp(
          subgraph, output_min, output_max,
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_RELU),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitReshapeNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors, const TfLiteReshapeParams* reshape_params,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    switch (node->inputs->size) {
      case 1:
      case 2:
        break;
      default:
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context,
            "unexpected number of inputs (%d) in node #%d: "
            "either one or two inputs expected",
            node->inputs->size, node_index);
        return kTfLiteError;
    }
    if (node->outputs->size != 1) {
      TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                               "unexpected number of outputs (%d) in node "
                               "#%d: one output expected",
                               node->outputs->size, node_index);
      return kTfLiteError;
    }

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input_tensor,
                                       node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, input_tensor, /*min_num_dims=*/0,
        /*max_num_dims=*/XNN_MAX_TENSOR_DIMS, node->inputs->data[0],
        BuiltinOperator_RESHAPE, node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    std::array<size_t, XNN_MAX_TENSOR_DIMS> new_shape;
    int num_new_dimensions;
    if (node->inputs->size == 2) {
      const TfLiteTensor& shape_tensor = tensors[node->inputs->data[1]];
      TF_LITE_ENSURE_STATUS(CheckTensorType(logging_context, shape_tensor,
                                            kTfLiteInt32, node->inputs->data[1],
                                            node_index));
      TF_LITE_ENSURE_STATUS(CheckShapeTensorShape(
          logging_context, shape_tensor, /*squeeze_dims=*/true,
          node->inputs->data[1], BuiltinOperator_RESHAPE, node_index));
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, shape_tensor, node->inputs->data[1],
          BuiltinOperator_RESHAPE, node_index));
      num_new_dimensions = NumElements(&shape_tensor);
      for (int i = 0; i < num_new_dimensions; ++i) {
        if (shape_tensor.data.i32[i] == -1) {
          new_shape[i] = 0;
        } else {
          new_shape[i] = shape_tensor.data.i32[i];
        }
      }
    } else {
      num_new_dimensions = reshape_params->num_dimensions;
      for (int i = 0; i < num_new_dimensions; ++i) {
        if (reshape_params->shape[i] == -1) {
          new_shape[i] = 0;
        } else {
          new_shape[i] = reshape_params->shape[i];
        }
      }
    }

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, output_tensor, /*min_num_dims=*/0,
        /*max_num_dims=*/XNN_MAX_TENSOR_DIMS, node->outputs->data[0],
        BuiltinOperator_RESHAPE, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (output_tensor.type == kTfLiteUInt8 ||
        output_tensor.type == kTfLiteInt8) {
      if (input_tensor.params.zero_point != output_tensor.params.zero_point) {
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context,
            "Mismatching quantization zero point across the input "
            "(%" PRId32 ") and the output (%" PRId32
            ") for RESHAPE operator #%d",
            input_tensor.params.zero_point, output_tensor.params.zero_point,
            node_index);
        return kTfLiteError;
      }
      if (input_tensor.params.scale != output_tensor.params.scale) {
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context,
            "Mismatching quantization scale across the input (%f) "
            "and the output (%f) for RESHAPE operator #%d",
            input_tensor.params.scale, output_tensor.params.scale, node_index);
        return kTfLiteError;
      }
    }
    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_static_reshape(
          subgraph, num_new_dimensions, new_shape.data(),
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_RESHAPE),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitResizeBilinearNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const TfLiteResizeBilinearParams* resize_params,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 1,
                                 BuiltinOperator_RESIZE_BILINEAR, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input_tensor,
                                       node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, input_tensor, 4, node->inputs->data[0],
        BuiltinOperator_RESIZE_BILINEAR, node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& shape_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorType(logging_context, shape_tensor,
                                          kTfLiteInt32, node->inputs->data[1],
                                          node_index));
    TF_LITE_ENSURE_STATUS(CheckShapeTensorShape(
        logging_context, shape_tensor, /*squeeze_dims=*/false,
        node->inputs->data[1], BuiltinOperator_RESIZE_BILINEAR, node_index));
    if (SizeOfDimension(&shape_tensor, 0) != 2) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "unexpected number of dimensions %d in the output shape in node %d",
          SizeOfDimension(&shape_tensor, 0), node_index);
    }
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, shape_tensor, node->inputs->data[1],
        BuiltinOperator_RESIZE_BILINEAR, node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, output_tensor, 4, node->outputs->data[0],
        BuiltinOperator_RESIZE_BILINEAR, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    const int32_t* shape_data =
        reinterpret_cast<const int32_t*>(shape_tensor.data.data);
    for (int i = 0; i < NumDimensions(&shape_tensor); i++) {
      const int32_t dim = shape_data[i];
      if (dim <= 0) {
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context, "invalid output dimension #%d value %d in node %d",
            i, dim, node_index);
        return kTfLiteError;
      }
    }

    if (subgraph != nullptr) {
      uint32_t flags = 0;
      if (resize_params->align_corners) {
        flags |= XNN_FLAG_ALIGN_CORNERS;
      } else if (!resize_params->half_pixel_centers) {
        flags |= XNN_FLAG_TENSORFLOW_LEGACY_MODE;
      }
      const xnn_status status = xnn_define_static_resize_bilinear_2d(
          subgraph, static_cast<size_t>(shape_data[0]),
          static_cast<size_t>(shape_data[1]),
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]), flags);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(
            logging_context, "failed to delegate %s node #%d",
            EnumNameBuiltinOperator(BuiltinOperator_RESIZE_BILINEAR),
            node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitRoundNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 1, 1, BuiltinOperator_ROUND, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_bankers_rounding(
          subgraph,
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_ROUND),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitSliceNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    const int input_tensor_index = node->inputs->data[0];
    const int begin_tensor_index = node->inputs->data[1];
    const int size_tensor_index = node->inputs->data[2];
    const int output_tensor_index = node->outputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_index];
    const TfLiteTensor& begin_tensor = tensors[begin_tensor_index];
    const TfLiteTensor& size_tensor = tensors[size_tensor_index];
    const TfLiteTensor& output_tensor = tensors[output_tensor_index];

    TF_LITE_ENSURE_STATUS(CheckShapeTensorShape(
        logging_context, begin_tensor, /*squeeze_dims=*/false,
        begin_tensor_index, BuiltinOperator_SLICE, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, begin_tensor, begin_tensor_index,
        BuiltinOperator_SLICE, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorInt32OrInt64Type(
        logging_context, begin_tensor, begin_tensor_index, node_index));

    TF_LITE_ENSURE_STATUS(CheckShapeTensorShape(
        logging_context, size_tensor, /*squeeze_dims=*/false, size_tensor_index,
        BuiltinOperator_SLICE, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, size_tensor, size_tensor_index, BuiltinOperator_SLICE,
        node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorInt32OrInt64Type(
        logging_context, size_tensor, size_tensor_index, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorsDimensionMatch(
        logging_context, begin_tensor, size_tensor, 0, node_index, "SLICE"));

    const int num_dims = begin_tensor.dims->data[0];
    if (num_dims > XNN_MAX_TENSOR_DIMS) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "number of dimensions %d must be less than %d in SLICE node #%d",
          num_dims, XNN_MAX_TENSOR_DIMS, node_index);
    }
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input_tensor,
                                       input_tensor_index, node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        input_tensor_index, node_index));

    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       output_tensor_index, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, output_tensor_index,
        node_index));

    std::array<int64_t, XNN_MAX_TENSOR_DIMS> begin;
    std::array<int64_t, XNN_MAX_TENSOR_DIMS> size;
    CopyTensorDataInt32OrInt64(begin.data(), begin_tensor, num_dims);
    CopyTensorDataInt32OrInt64(size.data(), size_tensor, num_dims);

    for (size_t i = 0; i < num_dims; i++) {
      if (begin[i] < 0) {
        // TODO(b/329228576): Add support for negative begin.
        TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                                 "begin %" PRId64
                                 " must be greater than 0 in SLICE node #%d",
                                 begin[i], node_index);
      }
      if (size[i] <= 0) {
        // TODO(b/329228576): Add support for negative begin.
        TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                                 "size %" PRId64
                                 " must be positive in SLICE node #%d",
                                 size[i], node_index);
        return kTfLiteError;
      }
    }

    if (subgraph != nullptr) {
      // Convert to size_t.
      std::array<size_t, XNN_MAX_TENSOR_DIMS> offsets;
      std::copy(begin.begin(), begin.end(), offsets.begin());
      std::array<size_t, XNN_MAX_TENSOR_DIMS> sizes;
      std::copy(size.begin(), size.end(), sizes.begin());

      const xnn_status status = xnn_define_static_slice(
          subgraph, num_dims, offsets.data(), sizes.data(),
          input_output_tensors.at(node->inputs->data[0]),
          input_output_tensors.at(node->outputs->data[0]), /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_SLICE),
                           node_index);
        return kTfLiteError;
      }
    }
    return kTfLiteOk;
  }

  static TfLiteStatus VisitSoftmaxNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors, const TfLiteSoftmaxParams* params,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    if (params->beta != 1.0f) {
      if (logging_context != nullptr) {
        TF_LITE_KERNEL_LOG(logging_context,
                           "unsupported beta value %.7f in SOFTMAX node #%d",
                           params->beta, node_index);
      }
      return kTfLiteError;
    }

    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 1, 1, BuiltinOperator_SOFTMAX, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_softmax(
          subgraph,
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_SOFTMAX),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitSpaceToDepthNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const TfLiteSpaceToDepthParams* space_to_depth_params,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 1, 1,
                                 BuiltinOperator_SPACE_TO_DEPTH, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input_tensor,
                                       node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    const int block_size = space_to_depth_params->block_size;
    if (block_size <= 1) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "block size (%d) in SPACE_TO_DEPTH node #%d must be greater > 1",
          block_size, node_index);
      return kTfLiteError;
    }

    const int input_height = input_tensor.dims->data[1];
    const int input_width = input_tensor.dims->data[2];
    if (input_height % block_size != 0) {
      TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                               "SPACE_TO_DEPTH node #%d input height (%d) must "
                               "be divisible by block_size (%d).",
                               input_height, block_size, node_index);
      return kTfLiteError;
    }

    if (input_width % block_size != 0) {
      TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                               "SPACE_TO_DEPTH node #%d input width (%d) must "
                               "be divisible by block_size (%d).",
                               input_width, block_size, node_index);
      return kTfLiteError;
    }

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_space_to_depth_2d(
          subgraph, static_cast<uint32_t>(block_size),
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(
            logging_context, "failed to delegate %s node #%d",
            EnumNameBuiltinOperator(BuiltinOperator_SPACE_TO_DEPTH),
            node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitSplitNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors, const TfLiteSplitParams* split_params,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    const int num_outputs = NumOutputs(node);
    TF_LITE_ENSURE_EQ(logging_context, split_params->num_splits, num_outputs);
    TF_LITE_ENSURE_STATUS(CheckNumInputs(logging_context, node, 2,
                                         BuiltinOperator_SPLIT, node_index));
    TF_LITE_ENSURE_STATUS(CheckNumOutputs(logging_context, node, 2, 4,
                                          BuiltinOperator_SPLIT, node_index));

    const int split_dim_idx = node->inputs->data[0];
    const TfLiteTensor& split_dim_tensor = tensors[split_dim_idx];
    TF_LITE_ENSURE_STATUS(CheckTensorType(logging_context, split_dim_tensor,
                                          kTfLiteInt32, split_dim_idx,
                                          node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, split_dim_tensor, split_dim_idx, BuiltinOperator_SPLIT,
        node_index));

    const int input_idx = node->inputs->data[1];
    const TfLiteTensor& input_tensor = tensors[input_idx];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQUInt8Type(
        delegate, logging_context, input_tensor, input_idx, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, input_tensor, input_idx, node_index));

    int32_t split_dim = GetTensorData<int32_t>(&split_dim_tensor)[0];

    for (int i = 0; i < NumOutputs(node); i++) {
      const int output_idx = node->outputs->data[i];
      const TfLiteTensor& output_tensor = tensors[output_idx];

      TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQUInt8Type(
          delegate, logging_context, output_tensor, output_idx, node_index));
      TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
          delegate, logging_context, output_tensor, output_idx, node_index));
    }

    if (subgraph != nullptr) {
      xnn_status status = xnn_status_invalid_parameter;
      if (num_outputs == 2) {
        status = xnn_define_even_split2(
            subgraph, split_dim,
            /*input_id=*/input_output_tensors.at(input_idx),
            /*output1_id=*/input_output_tensors.at(node->outputs->data[0]),
            /*output2_id=*/input_output_tensors.at(node->outputs->data[1]),
            /*flags=*/0);
      } else if (num_outputs == 3) {
        status = xnn_define_even_split3(
            subgraph, split_dim,
            /*input_id=*/input_output_tensors.at(input_idx),
            /*output1_id=*/input_output_tensors.at(node->outputs->data[0]),
            /*output2_id=*/input_output_tensors.at(node->outputs->data[1]),
            /*output3_id=*/input_output_tensors.at(node->outputs->data[2]),
            /*flags=*/0);
      } else if (num_outputs == 4) {
        status = xnn_define_even_split4(
            subgraph, split_dim,
            /*input_id=*/input_output_tensors.at(input_idx),
            /*output1_id=*/input_output_tensors.at(node->outputs->data[0]),
            /*output2_id=*/input_output_tensors.at(node->outputs->data[1]),
            /*output3_id=*/input_output_tensors.at(node->outputs->data[2]),
            /*output4_id=*/input_output_tensors.at(node->outputs->data[3]),
            /*flags=*/0);
      }

      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_SPLIT),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitSquareNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 1, 1, BuiltinOperator_SQUARE, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, input_tensor, /*min_num_dims=*/0,
        /*max_num_dims=*/XNN_MAX_TENSOR_DIMS, node->inputs->data[0],
        BuiltinOperator_SQUARE, node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_square(
          subgraph,
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_SQUARE),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitTanhNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 1, 1, BuiltinOperator_TANH, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input_tensor,
                                       node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_tanh(
          subgraph,
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_TANH),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitTransposeNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 2, 1, BuiltinOperator_TRANSPOSE, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input_tensor,
                                       node->inputs->data[0], node_index));
    const TfLiteTensor& perm_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, perm_tensor, node->inputs->data[1],
        BuiltinOperator_TRANSPOSE, node_index));

    const int* perm_data = GetTensorData<int32_t>(&perm_tensor);

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));
    const int dims_count = NumElements(&perm_tensor);
    std::array<size_t, XNN_MAX_TENSOR_DIMS> perm;
    for (int i = 0; i < dims_count; ++i) {
      if (perm_data[i] < 0) {
        perm[i] = perm_data[i] + dims_count;
      } else {
        perm[i] = perm_data[i];
      }
    }

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_static_transpose(
          subgraph, dims_count, perm.data(),
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_TRANSPOSE),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitSqrtNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 1, 1, BuiltinOperator_SQRT, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_square_root(
          subgraph,
          /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_SQRT),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitRsqrtNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 1, 1, BuiltinOperator_RSQRT, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_reciprocal_square_root(
          subgraph, /*input_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_RSQRT),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitSquaredDifferenceNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 2, 1, BuiltinOperator_SQUARED_DIFFERENCE,
        node_index));

    const TfLiteTensor& input1_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input1_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, input1_tensor, node->inputs->data[0],
        node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, input1_tensor, /*min_num_dims=*/0,
        /*max_num_dims=*/XNN_MAX_TENSOR_DIMS, node->inputs->data[0],
        BuiltinOperator_SQUARED_DIFFERENCE, node_index));

    const TfLiteTensor& input2_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input2_tensor, node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, input2_tensor, node->inputs->data[1],
        node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, input2_tensor, /*min_num_dims=*/0,
        /*max_num_dims=*/XNN_MAX_TENSOR_DIMS, node->inputs->data[1],
        BuiltinOperator_SQUARED_DIFFERENCE, node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_squared_difference(
          subgraph,
          /*input1_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*input2_id=*/input_output_tensors.at(node->inputs->data[1]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(
            logging_context, "failed to delegate %s node #%d",
            EnumNameBuiltinOperator(BuiltinOperator_SQUARED_DIFFERENCE),
            node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitStridedSliceNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors, const TfLiteStridedSliceParams* params,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    // Only support strided slice with no ellipsis mask, no new axis mask, and
    // no shrink_axis-mask.
    if (params->ellipsis_mask != 0 || params->new_axis_mask != 0 ||
        params->shrink_axis_mask != 0) {
      return kTfLiteError;
    }

    const int stride_tensor_index = node->inputs->data[3];
    const TfLiteTensor& stride_tensor = tensors[stride_tensor_index];

    TF_LITE_ENSURE_STATUS(CheckShapeTensorShape(
        logging_context, stride_tensor, /*squeeze_dims=*/false,
        stride_tensor_index, BuiltinOperator_STRIDED_SLICE, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, stride_tensor, stride_tensor_index,
        BuiltinOperator_STRIDED_SLICE, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorInt32Type(
        logging_context, stride_tensor, stride_tensor_index, node_index));

    const int num_dims = stride_tensor.dims->data[0];
    if (num_dims > XNN_MAX_TENSOR_DIMS) {
      TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                               "number of dimensions %d must be less than %d "
                               "in STRIDED_SLICE node #%d",
                               num_dims, XNN_MAX_TENSOR_DIMS, node_index);
    }

    // Only support strides = 1.
    auto stride_data = GetTensorData<int32_t>(&stride_tensor);
    for (size_t i = 0; i < num_dims; i++) {
      if (stride_data[i] != 1) {
        TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                                 "stride at dimension %zu, %d, must be 1"
                                 "in STRIDED_SLICE node #%d",
                                 i, stride_data[i], node_index);
        return kTfLiteError;
      }
    }

    const int input_tensor_index = node->inputs->data[0];
    const int begin_tensor_index = node->inputs->data[1];
    const int end_tensor_index = node->inputs->data[2];
    const int output_tensor_index = node->outputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_index];
    const TfLiteTensor& begin_tensor = tensors[begin_tensor_index];
    const TfLiteTensor& end_tensor = tensors[end_tensor_index];
    const TfLiteTensor& output_tensor = tensors[output_tensor_index];

    TF_LITE_ENSURE_STATUS(CheckShapeTensorShape(
        logging_context, begin_tensor, /*squeeze_dims=*/false,
        begin_tensor_index, BuiltinOperator_STRIDED_SLICE, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, begin_tensor, begin_tensor_index,
        BuiltinOperator_STRIDED_SLICE, node_index));
    // TODO(b/246969669): TFLite only supports int32 begin ends and strides,
    // support int64 too when TFLite supports it as well.
    TF_LITE_ENSURE_STATUS(CheckTensorInt32Type(logging_context, begin_tensor,
                                               begin_tensor_index, node_index));

    TF_LITE_ENSURE_STATUS(CheckShapeTensorShape(
        logging_context, end_tensor, /*squeeze_dims=*/false, end_tensor_index,
        BuiltinOperator_STRIDED_SLICE, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, end_tensor, end_tensor_index,
        BuiltinOperator_STRIDED_SLICE, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorInt32Type(logging_context, end_tensor,
                                               end_tensor_index, node_index));

    const auto CheckParamTensorShape = [&](const TfLiteTensor& param_tensor,
                                           const char* param_tensor_name) {
      if (input_tensor.dims->size != GetTensorData<int32_t>(&param_tensor)[0]) {
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context,
            "%s shape (%d) must be equal to input shape (%d) "
            "in STRIDED_SLICE node #%d",
            param_tensor_name,
            reinterpret_cast<const int32_t*>(param_tensor.data.data)[0],
            input_tensor.dims->size, node_index);
        return kTfLiteError;
      }
      return kTfLiteOk;
    };

    TF_LITE_ENSURE_STATUS(CheckParamTensorShape(begin_tensor, "begin_tensor"));
    TF_LITE_ENSURE_STATUS(CheckParamTensorShape(end_tensor, "end_tensor"));
    TF_LITE_ENSURE_STATUS(
        CheckParamTensorShape(stride_tensor, "stride_tensor"));

    TF_LITE_ENSURE_STATUS(
        CheckTensorsDimensionMatch(logging_context, stride_tensor, begin_tensor,
                                   0, node_index, "STRIDED_SLICE"));
    TF_LITE_ENSURE_STATUS(
        CheckTensorsDimensionMatch(logging_context, begin_tensor, end_tensor, 0,
                                   node_index, "STRIDED_SLICE"));
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input_tensor,
                                       input_tensor_index, node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        input_tensor_index, node_index));

    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       output_tensor_index, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, output_tensor_index,
        node_index));

    auto begin_data = GetTensorData<int32_t>(&begin_tensor);
    auto end_data = GetTensorData<int32_t>(&end_tensor);
    auto input_shape = input_tensor.dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> begins;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> sizes;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> ends;
    for (size_t i = 0; i < num_dims; i++) {
      if (begin_data[i] < 0) {
        // TODO(b/329228576): Add support for negative begin.
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context,
            "begin %d must be greater than or equal to zero "
            "in STRIDED_SLICE node #%d",
            begin_data[i], node_index);
        return kTfLiteError;
      }
      begins[i] = begin_data[i] < 0 ? input_shape->data[i] + begin_data[i]
                                    : begin_data[i];
      if ((params->begin_mask & (1 << i)) != 0) {
        begins[i] = 0;
      }

      int actual_end_data = end_data[i];
      if (params->offset) {
        actual_end_data += begin_data[i];
      }
      // If end is negative, we count from the back, -1 is the last element.
      if (actual_end_data < 0) {
        // TODO(b/329228576): Add support for negative begin.
        TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                                 "end %d must be greater than or equal to zero "
                                 "in STRIDED_SLICE node #%d",
                                 end_data[i], node_index);
        return kTfLiteError;
      } else {
        ends[i] = actual_end_data;
      }

      if ((params->end_mask & (1 << i)) != 0) {
        // TODO(b/329228576): Add support for negative begin.
        TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                                 "non-zero end mask not supported "
                                 "in STRIDED_SLICE node #%d",
                                 end_data[i], node_index);
        return kTfLiteError;
      }

      if (begins[i] >= ends[i]) {
        TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                                 "begin index %zu must be less than end index "
                                 "%zu for STRIDED_SLICE node #%d",
                                 begins[i], ends[i], node_index);
      }

      sizes[i] = ends[i] - begins[i];
    }

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_static_slice(
          subgraph, num_dims, begins.data(), sizes.data(),
          input_output_tensors.at(input_tensor_index),
          input_output_tensors.at(output_tensor_index), /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(
            logging_context, "failed to delegate %s node #%d",
            EnumNameBuiltinOperator(BuiltinOperator_STRIDED_SLICE), node_index);
        return kTfLiteError;
      }
    }
    return kTfLiteOk;
  }

  static TfLiteStatus VisitScaledDotAttentionCompositeNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors, const uint8_t* buffer,
      const size_t buffer_size,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(buffer, buffer_size).AsMap();
    const float* const scale_ptr =
        flexbuffer_map["scale"].As<FloatPointer>().ptr;
    const float* const cap_ptr =
        flexbuffer_map["logit_cap"].As<FloatPointer>().ptr;
    return VisitDotAttentionNode(subgraph, delegate, logging_context,
                                 node_index, node, tensors, scale_ptr, cap_ptr,
                                 input_output_tensors);
  }

  static TfLiteStatus VisitDotAttentionNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors, const float* scale_param,
      const float* cap_param,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    const TfLiteTensor& query_proj = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, query_proj, node->inputs->data[0], node_index));

    const TfLiteTensor& key_proj = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, key_proj, node->inputs->data[1], node_index));

    const TfLiteTensor& value_proj = tensors[node->inputs->data[2]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, value_proj, node->inputs->data[2], node_index));

    const TfLiteTensor& atten_mask = tensors[node->inputs->data[3]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, atten_mask, node->inputs->data[3], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    // Head dimension match.
    TF_LITE_ENSURE_EQ(logging_context,
                      query_proj.dims->data[query_proj.dims->size - 1],
                      key_proj.dims->data[key_proj.dims->size - 1]);
    TF_LITE_ENSURE_EQ(logging_context,
                      query_proj.dims->data[query_proj.dims->size - 1],
                      value_proj.dims->data[value_proj.dims->size - 1]);
    // Max sequence length match.
    TF_LITE_ENSURE_EQ(logging_context, key_proj.dims->data[1],
                      atten_mask.dims->data[atten_mask.dims->size - 1]);
    TF_LITE_ENSURE_EQ(logging_context, value_proj.dims->data[1],
                      atten_mask.dims->data[atten_mask.dims->size - 1]);

    if (subgraph != nullptr) {
      // constants
      uint32_t query_proj_id = input_output_tensors.at(node->inputs->data[0]);
      uint32_t key_proj_id = input_output_tensors.at(node->inputs->data[1]);
      uint32_t value_proj_id = input_output_tensors.at(node->inputs->data[2]);
      uint32_t atten_mask_id = input_output_tensors.at(node->inputs->data[3]);
      uint32_t output_id = input_output_tensors.at(node->outputs->data[0]);
      float default_out_min = -std::numeric_limits<float>::infinity();
      float default_out_max = std::numeric_limits<float>::infinity();

      // Attention Type
      TF_LITE_ENSURE_EQ(logging_context,
                        query_proj.dims->data[2] % key_proj.dims->data[2], 0);
      bool is_mqa = (key_proj.dims->data[2] == 1);
      bool is_gqa =
          !is_mqa && (key_proj.dims->data[2] != query_proj.dims->data[2]);

      // Scale the query values
      const auto query_dim = query_proj.dims;
      TF_LITE_ENSURE_EQ(logging_context, query_dim->size, 4);
      float scale_const = 1.0f / sqrt(query_dim->data[3]);
      uint32_t scale_out_id = XNN_INVALID_VALUE_ID;
      if (scale_param != nullptr) {
        TF_LITE_ENSURE_EQ(
            logging_context, xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32, /*num_dims=*/0,
                                    /*dims=*/nullptr, scale_param,
                                    XNN_INVALID_VALUE_ID, 0, &scale_out_id));
      } else {
        // fallback, use default scale = 1 / sqrt(dim_per_head)
        uint32_t scale_orig_id = XNN_INVALID_VALUE_ID;
        TF_LITE_ENSURE_EQ(
            logging_context, xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32, /*num_dims=*/0,
                                    /*dims=*/nullptr, &query_proj.dims->data[3],
                                    XNN_INVALID_VALUE_ID, 0, &scale_orig_id));
        TF_LITE_ENSURE_EQ(
            logging_context, xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32, /*num_dims=*/0,
                                    /*dims=*/nullptr, nullptr,
                                    XNN_INVALID_VALUE_ID, 0, &scale_out_id));
        TF_LITE_ENSURE_EQ(
            logging_context, xnn_status_success,
            xnn_define_clamp(subgraph, scale_const, scale_const, scale_orig_id,
                             scale_out_id, /*flags=*/0));
      }
      uint32_t multiply_out_id = XNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_EQ(
          logging_context, xnn_status_success,
          xnn_define_tensor_value(subgraph, xnn_datatype_fp32, /*num_dims=*/0,
                                  /*dims=*/nullptr, nullptr,
                                  XNN_INVALID_VALUE_ID, 0, &multiply_out_id));
      TF_LITE_ENSURE_EQ(
          logging_context, xnn_status_success,
          xnn_define_multiply2(subgraph, default_out_min, default_out_max,
                               query_proj_id, scale_out_id, multiply_out_id,
                               /*flags=*/0));
      // Dot similarity
      // BTNH -> BNTH
      std::array<size_t, 4> permute_q = {0, 2, 1, 3};
      TF_LITE_ENSURE_EQ(logging_context, query_proj.dims->size,
                        permute_q.size());
      uint32_t permute_q_out_id = XNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_EQ(
          logging_context, xnn_status_success,
          xnn_define_tensor_value(subgraph, xnn_datatype_fp32, /*num_dims=*/0,
                                  /*dims=*/nullptr, nullptr,
                                  XNN_INVALID_VALUE_ID, 0, &permute_q_out_id));
      TF_LITE_ENSURE_EQ(logging_context, xnn_status_success,
                        xnn_define_static_transpose(
                            subgraph, permute_q.size(), permute_q.data(),
                            multiply_out_id, permute_q_out_id, /*flags=*/0));
      // BSNH -> BNSH
      std::array<size_t, 4> permute_k = {0, 2, 1, 3};
      TF_LITE_ENSURE_EQ(logging_context, key_proj.dims->size, permute_k.size());
      uint32_t permute_k_out_id = XNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_EQ(
          logging_context, xnn_status_success,
          xnn_define_tensor_value(subgraph, xnn_datatype_fp32, /*num_dims=*/0,
                                  /*dims=*/nullptr, nullptr,
                                  XNN_INVALID_VALUE_ID, 0, &permute_k_out_id));
      TF_LITE_ENSURE_EQ(logging_context, xnn_status_success,
                        xnn_define_static_transpose(
                            subgraph, permute_k.size(), permute_k.data(),
                            key_proj_id, permute_k_out_id, /*flags=*/0));
      // einsum(BNTH.BNSH -> BNTS)
      uint32_t fc_out_id = XNN_INVALID_VALUE_ID;
      if (!is_mqa) {
        // BatchMM (permute_q, permute_k)
        // [B, N, T, S] . [B, N, H, S]
        // output shape [query_proj_dim[0], query_proj_dim[2],
        // query_proj_dim[1], key_proj_dim[1]];
        TF_LITE_ENSURE_EQ(
            logging_context, xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32, /*num_dims=*/0,
                                    /*dims=*/nullptr, nullptr,
                                    XNN_INVALID_VALUE_ID, 0, &fc_out_id));
        if (is_gqa) {
          uint32_t q_reshape_id = XNN_INVALID_VALUE_ID;
          TF_LITE_ENSURE_EQ(
              logging_context, xnn_status_success,
              xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                      /*num_dims=*/0, /*dims=*/nullptr, nullptr,
                                      XNN_INVALID_VALUE_ID, 0, &q_reshape_id));
          uint32_t k_reshape_id = XNN_INVALID_VALUE_ID;
          TF_LITE_ENSURE_EQ(
              logging_context, xnn_status_success,
              xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                      /*num_dims=*/0, /*dims=*/nullptr, nullptr,
                                      XNN_INVALID_VALUE_ID, 0, &k_reshape_id));
          uint32_t bmm_reshape_id = XNN_INVALID_VALUE_ID;
          TF_LITE_ENSURE_EQ(
              logging_context, xnn_status_success,
              xnn_define_tensor_value(
                  subgraph, xnn_datatype_fp32, /*num_dims=*/0, /*dims=*/nullptr,
                  nullptr, XNN_INVALID_VALUE_ID, 0, &bmm_reshape_id));
          size_t num_query_groups = key_proj.dims->data[2];
          size_t head_per_query = query_proj.dims->data[2] / num_query_groups;
          std::array<size_t, 5> q_reshape_dims = {
              (size_t)query_proj.dims->data[0], num_query_groups,
              head_per_query, (size_t)query_proj.dims->data[1],
              (size_t)query_proj.dims->data[3]};
          std::array<size_t, 5> k_reshape_dims = {
              (size_t)key_proj.dims->data[0], num_query_groups, 1,
              (size_t)key_proj.dims->data[1], (size_t)key_proj.dims->data[3]};
          std::array<size_t, 4> bmm_reshape_dims = {
              (size_t)query_proj.dims->data[0],
              num_query_groups * head_per_query,
              (size_t)query_proj.dims->data[1], (size_t)key_proj.dims->data[1]};
          TF_LITE_ENSURE_EQ(
              logging_context, xnn_status_success,
              xnn_define_static_reshape(subgraph, q_reshape_dims.size(),
                                        q_reshape_dims.data(), permute_q_out_id,
                                        q_reshape_id, /*flags=*/0));
          TF_LITE_ENSURE_EQ(
              logging_context, xnn_status_success,
              xnn_define_static_reshape(subgraph, k_reshape_dims.size(),
                                        k_reshape_dims.data(), permute_k_out_id,
                                        k_reshape_id, /*flags=*/0));
          TF_LITE_ENSURE_EQ(
              logging_context, xnn_status_success,
              xnn_define_batch_matrix_multiply(subgraph, q_reshape_id,
                                               k_reshape_id, bmm_reshape_id,
                                               /*flags=*/XNN_FLAG_TRANSPOSE_B));
          TF_LITE_ENSURE_EQ(
              logging_context, xnn_status_success,
              xnn_define_static_reshape(subgraph, bmm_reshape_dims.size(),
                                        bmm_reshape_dims.data(), bmm_reshape_id,
                                        fc_out_id, /*flags=*/0));
        } else {
          TF_LITE_ENSURE_EQ(logging_context, xnn_status_success,
                            xnn_define_batch_matrix_multiply(
                                subgraph, permute_q_out_id, permute_k_out_id,
                                fc_out_id, /*flags=*/XNN_FLAG_TRANSPOSE_B));
        }
      } else {
        // FC (permute_q, permute_k)
        TFLITE_DCHECK(key_proj.dims->data[0] == 1);
        TFLITE_DCHECK(key_proj.dims->data[2] == 1);
        // squeezed_rhs shape: [S, H]
        std::array<size_t, 2> reshape_dims_k = {(size_t)key_proj.dims->data[1],
                                                (size_t)key_proj.dims->data[3]};
        uint32_t reshape_dims_k_out_id = XNN_INVALID_VALUE_ID;
        TF_LITE_ENSURE_EQ(
            logging_context, xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, /*num_dims=*/0, /*dims=*/nullptr,
                nullptr, XNN_INVALID_VALUE_ID, 0, &reshape_dims_k_out_id));
        TF_LITE_ENSURE_EQ(
            logging_context, xnn_status_success,
            xnn_define_static_reshape(subgraph, reshape_dims_k.size(),
                                      reshape_dims_k.data(), permute_k_out_id,
                                      reshape_dims_k_out_id, /*flags=*/0));
        // Output shape: [B, N, T, S]
        // FC: input = permuted_q, weight = reshaped_k, bias = nullptr,
        // params=(transpose=false)
        // assumes no sparse computation for now
        TF_LITE_ENSURE_EQ(
            logging_context, xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32, /*num_dims=*/0,
                                    /*dims=*/nullptr, nullptr,
                                    XNN_INVALID_VALUE_ID, 0, &fc_out_id));
        TF_LITE_ENSURE_EQ(logging_context, xnn_status_success,
                          xnn_define_fully_connected(
                              subgraph, default_out_min, default_out_max,
                              permute_q_out_id, reshape_dims_k_out_id,
                              XNN_INVALID_VALUE_ID, fc_out_id, /*flags=*/0));
      }
      if (cap_param != nullptr) {
        uint32_t cap_val_id = XNN_INVALID_VALUE_ID;
        TF_LITE_ENSURE_EQ(
            logging_context, xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32, /*num_dims=*/0,
                                    /*dims=*/nullptr, cap_param,
                                    XNN_INVALID_VALUE_ID, 0, &cap_val_id));
        uint32_t cap_div_out_id = XNN_INVALID_VALUE_ID;
        TF_LITE_ENSURE_EQ(
            logging_context, xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32, /*num_dims=*/0,
                                    /*dims=*/nullptr, nullptr,
                                    XNN_INVALID_VALUE_ID, 0, &cap_div_out_id));
        TF_LITE_ENSURE_EQ(
            logging_context, xnn_status_success,
            xnn_define_divide(subgraph, default_out_min, default_out_max,
                              fc_out_id, cap_val_id, cap_div_out_id,
                              /*flags=*/0));
        uint32_t cap_tanh_out_id = XNN_INVALID_VALUE_ID;
        TF_LITE_ENSURE_EQ(
            logging_context, xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32, /*num_dims=*/0,
                                    /*dims=*/nullptr, nullptr,
                                    XNN_INVALID_VALUE_ID, 0, &cap_tanh_out_id));
        TF_LITE_ENSURE_EQ(logging_context, xnn_status_success,
                          xnn_define_tanh(subgraph, cap_div_out_id,
                                          cap_tanh_out_id, /*flags=*/0));
        uint32_t cap_logits_id = XNN_INVALID_VALUE_ID;
        TF_LITE_ENSURE_EQ(
            logging_context, xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32, /*num_dims=*/0,
                                    /*dims=*/nullptr, nullptr,
                                    XNN_INVALID_VALUE_ID, 0, &cap_logits_id));
        TF_LITE_ENSURE_EQ(logging_context, xnn_status_success,
                          xnn_define_multiply2(subgraph, default_out_min,
                                               default_out_max, cap_tanh_out_id,
                                               cap_val_id, cap_logits_id, 0));
        fc_out_id = cap_logits_id;
      }
      // element_add atten_mask and matmul_out
      uint32_t padded_logits_id = XNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_EQ(
          logging_context, xnn_status_success,
          xnn_define_tensor_value(subgraph, xnn_datatype_fp32, /*num_dims=*/0,
                                  /*dims=*/nullptr, nullptr,
                                  XNN_INVALID_VALUE_ID, 0, &padded_logits_id));
      TF_LITE_ENSURE_EQ(
          logging_context, xnn_status_success,
          xnn_define_add2(subgraph, default_out_min, default_out_max,
                          atten_mask_id, fc_out_id, padded_logits_id,
                          /*flags=*/0));
      // softmax(padded_logits)
      uint32_t probs_id = XNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_EQ(
          logging_context, xnn_status_success,
          xnn_define_tensor_value(subgraph, xnn_datatype_fp32, /*num_dims=*/0,
                                  /*dims=*/nullptr, nullptr,
                                  XNN_INVALID_VALUE_ID, 0, &probs_id));
      TF_LITE_ENSURE_EQ(logging_context, xnn_status_success,
                        xnn_define_softmax(subgraph, padded_logits_id, probs_id,
                                           /*flags=*/0));
      // Permute(value_proj, {0, 2, 3, 1})
      std::array<size_t, 4> permute_v = {0, 2, 3, 1};
      TF_LITE_ENSURE_EQ(logging_context, value_proj.dims->size,
                        permute_v.size());
      uint32_t permute_v_out_id = XNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_EQ(
          logging_context, xnn_status_success,
          xnn_define_tensor_value(subgraph, xnn_datatype_fp32, /*num_dims=*/0,
                                  /*dims=*/nullptr, nullptr,
                                  XNN_INVALID_VALUE_ID, 0, &permute_v_out_id));
      TF_LITE_ENSURE_EQ(logging_context, xnn_status_success,
                        xnn_define_static_transpose(
                            subgraph, permute_v.size(), permute_v.data(),
                            value_proj_id, permute_v_out_id, /*flags=*/0));
      // Outcome
      // BNTS.BNHS -> BNTH
      uint32_t fc2_out_id = XNN_INVALID_VALUE_ID;
      if (!is_mqa) {
        // BatchMM (padded_logits, permute_v)
        // [B, N, T, S] . [B, N, H, S]
        // output shape [padded_logits_dims[0], padded_logits_dims[1],
        // padded_logits_dims[2], value_proj_dims[3]];
        TF_LITE_ENSURE_EQ(
            logging_context, xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32, /*num_dims=*/0,
                                    /*dims=*/nullptr, nullptr,
                                    XNN_INVALID_VALUE_ID, 0, &fc2_out_id));
        if (is_gqa) {
          uint32_t padded_logits_reshape_id = XNN_INVALID_VALUE_ID;
          TF_LITE_ENSURE_EQ(
              logging_context, xnn_status_success,
              xnn_define_tensor_value(
                  subgraph, xnn_datatype_fp32, /*num_dims=*/0, /*dims=*/nullptr,
                  nullptr, XNN_INVALID_VALUE_ID, 0, &padded_logits_reshape_id));
          uint32_t v_reshape_id = XNN_INVALID_VALUE_ID;
          TF_LITE_ENSURE_EQ(
              logging_context, xnn_status_success,
              xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                      /*num_dims=*/0, /*dims=*/nullptr, nullptr,
                                      XNN_INVALID_VALUE_ID, 0, &v_reshape_id));
          uint32_t bmm2_reshape_id = XNN_INVALID_VALUE_ID;
          TF_LITE_ENSURE_EQ(
              logging_context, xnn_status_success,
              xnn_define_tensor_value(
                  subgraph, xnn_datatype_fp32, /*num_dims=*/0, /*dims=*/nullptr,
                  nullptr, XNN_INVALID_VALUE_ID, 0, &bmm2_reshape_id));
          size_t num_query_groups = value_proj.dims->data[2];
          size_t head_per_query = query_proj.dims->data[2] / num_query_groups;
          std::array<size_t, 5> padded_logits_reshape_dims = {
              (size_t)query_proj.dims->data[0], num_query_groups,
              head_per_query, (size_t)query_proj.dims->data[1],
              (size_t)value_proj.dims->data[1]};
          std::array<size_t, 5> v_reshape_dims = {
              (size_t)value_proj.dims->data[0], num_query_groups, 1,
              (size_t)value_proj.dims->data[3],
              (size_t)value_proj.dims->data[1]};
          std::array<size_t, 4> bmm2_reshape_dims = {
              (size_t)query_proj.dims->data[0],
              num_query_groups * head_per_query,
              (size_t)query_proj.dims->data[1],
              (size_t)query_proj.dims->data[3]};
          TF_LITE_ENSURE_EQ(logging_context, xnn_status_success,
                            xnn_define_static_reshape(
                                subgraph, padded_logits_reshape_dims.size(),
                                padded_logits_reshape_dims.data(), probs_id,
                                padded_logits_reshape_id, /*flags=*/0));
          TF_LITE_ENSURE_EQ(
              logging_context, xnn_status_success,
              xnn_define_static_reshape(subgraph, v_reshape_dims.size(),
                                        v_reshape_dims.data(), permute_v_out_id,
                                        v_reshape_id, /*flags=*/0));
          TF_LITE_ENSURE_EQ(
              logging_context, xnn_status_success,
              xnn_define_batch_matrix_multiply(
                  subgraph, padded_logits_reshape_id, v_reshape_id,
                  bmm2_reshape_id, /*flags=*/XNN_FLAG_TRANSPOSE_B));
          TF_LITE_ENSURE_EQ(
              logging_context, xnn_status_success,
              xnn_define_static_reshape(
                  subgraph, bmm2_reshape_dims.size(), bmm2_reshape_dims.data(),
                  bmm2_reshape_id, fc2_out_id, /*flags=*/0));
        } else {
          TF_LITE_ENSURE_EQ(logging_context, xnn_status_success,
                            xnn_define_batch_matrix_multiply(
                                subgraph, probs_id, permute_v_out_id,
                                fc2_out_id, /*flags=*/XNN_FLAG_TRANSPOSE_B));
        }
      } else {
        // FC (padded_logits, permute_v)
        TFLITE_DCHECK(value_proj.dims->data[0] == 1);
        TFLITE_DCHECK(value_proj.dims->data[2] == 1);
        // squeezed_rhs shape: [S, H]
        std::array<size_t, 2> reshape_dims_v = {
            (size_t)value_proj.dims->data[3], (size_t)value_proj.dims->data[1]};
        uint32_t reshape_dims_v_out_id = XNN_INVALID_VALUE_ID;
        TF_LITE_ENSURE_EQ(
            logging_context, xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, /*num_dims=*/0, /*dims=*/nullptr,
                nullptr, XNN_INVALID_VALUE_ID, 0, &reshape_dims_v_out_id));
        TF_LITE_ENSURE_EQ(
            logging_context, xnn_status_success,
            xnn_define_static_reshape(subgraph, reshape_dims_v.size(),
                                      reshape_dims_v.data(), permute_v_out_id,
                                      reshape_dims_v_out_id, /*flags=*/0));
        // Output shape: [B, N, T, S]
        // FC: input = padded_logits, weight = reshaped_v, bias = nullptr,
        // params=(transpose=false)
        // assumes no sparse computation for now
        TF_LITE_ENSURE_EQ(
            logging_context, xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32, /*num_dims=*/0,
                                    /*dims=*/nullptr, nullptr,
                                    XNN_INVALID_VALUE_ID, 0, &fc2_out_id));
        TF_LITE_ENSURE_EQ(logging_context, xnn_status_success,
                          xnn_define_fully_connected(
                              subgraph, default_out_min, default_out_max,
                              probs_id, reshape_dims_v_out_id,
                              XNN_INVALID_VALUE_ID, fc2_out_id, /*flags=*/0));
      }
      // [B, N, T, H] -> BTNH
      // Permute(fc2_out_id, {0, 2, 1, 3}) -> output tensor
      std::array<size_t, 4> permute_fc = {0, 2, 1, 3};
      const xnn_status status = xnn_define_static_transpose(
          subgraph, permute_fc.size(), permute_fc.data(), fc2_out_id, output_id,
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           "odml.scaled_dot_product_attention", node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitSubNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors, const TfLiteSubParams* sub_params,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 2, 1, BuiltinOperator_SUB, node_index));

    const TfLiteTensor& input1_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input1_tensor,
                                       node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, input1_tensor, node->inputs->data[0],
        node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, input1_tensor, /*min_num_dims=*/0,
        /*max_num_dims=*/XNN_MAX_TENSOR_DIMS, node->inputs->data[0],
        BuiltinOperator_SUB, node_index));

    const TfLiteTensor& input2_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input2_tensor,
                                       node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, input2_tensor, node->inputs->data[1],
        node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(
        logging_context, input2_tensor, /*min_num_dims=*/0,
        /*max_num_dims=*/XNN_MAX_TENSOR_DIMS, node->inputs->data[1],
        BuiltinOperator_SUB, node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, node->outputs->data[0],
        node_index));

    const float scale_min = 1.0f / 1024.0f;
    const float scale_max = 256.0f;
    TF_LITE_ENSURE_STATUS(CheckTensorsInputOutputScale(
        logging_context, input1_tensor, output_tensor, scale_min, scale_max,
        BuiltinOperator_SUB, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorsInputOutputScale(
        logging_context, input2_tensor, output_tensor, scale_min, scale_max,
        BuiltinOperator_SUB, node_index));

    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = +std::numeric_limits<float>::infinity();
    if (sub_params != nullptr) {
      TF_LITE_ENSURE_STATUS(ConvertActivationToOutputRange(
          logging_context, node_index, sub_params->activation, &output_min,
          &output_max));
    }

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_subtract(
          subgraph, output_min, output_max,
          /*input1_id=*/input_output_tensors.at(node->inputs->data[0]),
          /*input2_id=*/input_output_tensors.at(node->inputs->data[1]),
          /*output_id=*/input_output_tensors.at(node->outputs->data[0]),
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate %s node #%d",
                           EnumNameBuiltinOperator(BuiltinOperator_SUB),
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitTransposeConvNode(
      xnn_subgraph_t subgraph, const Delegate& delegate,
      TfLiteContext* logging_context, int node_index, TfLiteNode* node,
      const TfLiteTensor* tensors,
      const TfLiteTransposeConvParams* deconv_params,
      const std::unordered_set<int>& quasi_static_tensors,
      const std::unordered_map<int, uint32_t>& input_output_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node,
                                 /*min_num_inputs=*/3, /*max_num_inputs=*/4,
                                 /*expected_num_outputs=*/1,
                                 BuiltinOperator_TRANSPOSE_CONV, node_index));
    const bool use_bias = node->inputs->size >= 4;

    const int output_shape_tensor_index = node->inputs->data[0];
    const TfLiteTensor& output_shape_tensor =
        tensors[output_shape_tensor_index];
    TF_LITE_ENSURE_STATUS(
        CheckTensorType(logging_context, output_shape_tensor, kTfLiteInt32,
                        output_shape_tensor_index, node_index));
    TF_LITE_ENSURE_STATUS(CheckShapeTensorShape(
        logging_context, output_shape_tensor, /*squeeze_dims=*/false,
        output_shape_tensor_index, BuiltinOperator_TRANSPOSE, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, output_shape_tensor, output_shape_tensor_index,
        BuiltinOperator_TRANSPOSE_CONV, node_index));
    const int output_shape_dims = SizeOfDimension(&output_shape_tensor, 0);
    if (output_shape_dims != 4) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "unsupported number of output shape dimensions (%d) in node #%d: "
          "4 dimensions expected",
          output_shape_dims, node_index);
      return kTfLiteError;
    }

    const int filter_tensor_index = node->inputs->data[1];
    const TfLiteTensor& filter_tensor = tensors[filter_tensor_index];
    TF_LITE_ENSURE_STATUS(
        CheckTensorShape(logging_context, filter_tensor, 4, filter_tensor_index,
                         BuiltinOperator_TRANSPOSE_CONV, node_index));
    if (quasi_static_tensors.count(filter_tensor_index) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, filter_tensor, filter_tensor_index,
          BuiltinOperator_TRANSPOSE_CONV, node_index));
    }

    const int input_tensor_index = node->inputs->data[2];
    const TfLiteTensor& input_tensor = tensors[input_tensor_index];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input_tensor,
                                       input_tensor_index, node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorShape(logging_context, input_tensor, 4, input_tensor_index,
                         BuiltinOperator_TRANSPOSE_CONV, node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(delegate, logging_context, input_tensor,
                                        input_tensor_index, node_index));

    bool dynamically_quantized = (input_tensor.type == kTfLiteFloat32 &&
                                  filter_tensor.type == kTfLiteInt8);
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQCInt8Type(
        delegate, logging_context, filter_tensor,
        /*expected_quantized_dimension=*/0, filter_tensor_index, node_index));

    uint32_t xnnpack_tensor_bias = XNN_INVALID_VALUE_ID;  // "No bias".
    if (use_bias) {
      const int bias_tensor_index = node->inputs->data[3];
      if (bias_tensor_index != kTfLiteOptionalTensor) {
        const TfLiteTensor& bias_tensor = tensors[bias_tensor_index];
        TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQCInt32Type(
            delegate, logging_context, bias_tensor, bias_tensor_index,
            node_index));
        TF_LITE_ENSURE_STATUS(
            CheckTensorShape(logging_context, bias_tensor, 1, bias_tensor_index,
                             BuiltinOperator_TRANSPOSE_CONV, node_index));
        if (quasi_static_tensors.count(bias_tensor_index) == 0) {
          TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
              logging_context, bias_tensor, bias_tensor_index,
              BuiltinOperator_TRANSPOSE_CONV, node_index));
        }
        if (subgraph != nullptr) {
          xnnpack_tensor_bias = input_output_tensors.at(bias_tensor_index);
        }
      }
    }

    const int output_tensor_index = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_index];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       output_tensor_index, node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorShape(logging_context, output_tensor, 4, output_tensor_index,
                         BuiltinOperator_TRANSPOSE_CONV, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        delegate, logging_context, output_tensor, output_tensor_index,
        node_index));

    const int* input_tensor_dims = input_tensor.dims->data;
    const int input_height = input_tensor_dims[1];
    const int input_width = input_tensor_dims[2];

    const int* filter_tensor_dims = filter_tensor.dims->data;
    const int output_channels = filter_tensor_dims[0];
    const int kernel_height = filter_tensor_dims[1];
    const int kernel_width = filter_tensor_dims[2];
    const int input_channels = filter_tensor_dims[3];

    const int32_t* output_shape = GetTensorData<int32_t>(&output_shape_tensor);
    const int output_height = output_shape[1];
    const int output_width = output_shape[2];
    const int output_tensor_channels = output_shape[3];
    if (output_channels != output_tensor_channels) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "transpose convolution kernel output channel dimension (%d) "
          "doesn't match output shape channel dimension (%d) in node #%d: "
          "4 dimensions expected",
          output_channels, output_tensor_channels, node_index);
      return kTfLiteError;
    }
    if (input_channels != input_tensor_dims[3]) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "transpose convolution kernel input channel dimension (%d) "
          "doesn't match filter input channel (%d) in node #%d",
          input_channels, input_tensor_dims[3], node_index);
      return kTfLiteError;
    }

    int padding_top = 0;
    int padding_bottom = 0;
    int padding_left = 0;
    int padding_right = 0;
    int adjustment_height = 0;
    int adjustment_width = 0;
    TF_LITE_ENSURE_STATUS(CalculateTransposeConvPaddings(
        logging_context, deconv_params->padding, input_height, input_width,
        kernel_height, kernel_width, /*dilation_height=*/1,
        /*dilation_width=*/1, deconv_params->stride_height,
        deconv_params->stride_width, node_index, output_height, output_width,
        &padding_top, &padding_bottom, &padding_left, &padding_right,
        &adjustment_height, &adjustment_width));

    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = +std::numeric_limits<float>::infinity();
    TF_LITE_ENSURE_STATUS(ConvertActivationToOutputRange(
        logging_context, node_index, deconv_params->activation, &output_min,
        &output_max));

    if (subgraph != nullptr) {
      if (dynamically_quantized) {
        TfLiteAffineQuantization* filter_params =
            reinterpret_cast<TfLiteAffineQuantization*>(
                filter_tensor.quantization.params);
        if (filter_params->scale->size != output_channels) {
          TfLiteFloatArrayFree(filter_params->scale);
          filter_params->scale = TfLiteFloatArrayCreate(output_channels);
          for (int i = 0; i < output_channels; ++i) {
            filter_params->scale->data[i] = filter_tensor.params.scale;
          }
          TfLiteIntArrayFree(filter_params->zero_point);
          filter_params->zero_point = TfLiteIntArrayCreate(output_channels);
          for (int i = 0; i < output_channels; ++i) {
            filter_params->zero_point->data[i] =
                filter_tensor.params.zero_point;
          }
        }
        uint32_t dq_quantized_id = XNN_INVALID_VALUE_ID;
        std::vector<size_t> input_dims(
            &input_tensor.dims->data[0],
            &input_tensor.dims->data[NumDimensions(&input_tensor)]);
        xnn_status status = xnn_define_dynamically_quantized_tensor_value(
            subgraph, xnn_datatype_qdint8, input_dims.size(),
            /*num_nonbatch_dims=*/3, input_dims.data(), XNN_INVALID_VALUE_ID,
            /*flags=*/0, &dq_quantized_id);
        if (status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(logging_context,
                             "failed to create XNNPACK Value for tensor %d",
                             -1);
          return kTfLiteError;
        }
        status = xnn_define_convert(
            subgraph,
            /*input_id=*/input_output_tensors.at(node->inputs->data[2]),
            dq_quantized_id, /*flags=*/0);
        if (status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(
              logging_context, "failed to delegate %s node #%d",
              EnumNameBuiltinOperator(BuiltinOperator_TRANSPOSE_CONV),
              node_index);
          return kTfLiteError;
        }
        std::vector<size_t> filter_dims(
            &filter_tensor.dims->data[0],
            &filter_tensor.dims->data[NumDimensions(&filter_tensor)]);
        uint32_t kernel_id = XNN_INVALID_VALUE_ID;
        status = xnn_define_channelwise_quantized_tensor_value(
            subgraph, xnn_datatype_qcint8, filter_params->scale->data,
            filter_dims.size(), /*channel_dim=*/0, filter_dims.data(),
            GetTensorData<int8_t>(&filter_tensor), XNN_INVALID_VALUE_ID,
            /*flags=*/0, &kernel_id);
        if (status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(
              logging_context, "failed to update filter tensor %s node #%d",
              EnumNameBuiltinOperator(BuiltinOperator_TRANSPOSE_CONV),
              node_index);
          return kTfLiteError;
        }
        status = xnn_define_deconvolution_2d(
            subgraph,
            /*padding_top=*/padding_top,
            /*padding_right=*/padding_right,
            /*padding_bottom=*/padding_bottom,
            /*padding_left=*/padding_left,
            /*adjustment_height=*/adjustment_height,
            /*adjustment_width=*/adjustment_width,
            static_cast<uint32_t>(kernel_height),
            static_cast<uint32_t>(kernel_width),
            static_cast<uint32_t>(deconv_params->stride_height),
            static_cast<uint32_t>(deconv_params->stride_width),
            /*dilation_height=*/1,
            /*dilation_width=*/1,
            /*groups=*/1,
            /*group_input_channels=*/input_channels,
            /*group_output_channels=*/output_channels,
            /*output_min=*/output_min,
            /*output_max=*/output_max,
            /*input_id=*/dq_quantized_id,
            /*filter_id=*/kernel_id,
            /*bias_id=*/xnnpack_tensor_bias,
            /*output_id=*/input_output_tensors.at(output_tensor_index),
            /*flags=*/0);
        if (status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(
              logging_context, "failed to delegate %s node #%d",
              EnumNameBuiltinOperator(BuiltinOperator_TRANSPOSE_CONV),
              node_index);
          return kTfLiteError;
        }
      } else {
        const xnn_status status = xnn_define_deconvolution_2d(
            subgraph,
            /*padding_top=*/padding_top,
            /*padding_right=*/padding_right,
            /*padding_bottom=*/padding_bottom,
            /*padding_left=*/padding_left,
            /*adjustment_height=*/adjustment_height,
            /*adjustment_width=*/adjustment_width,
            static_cast<uint32_t>(kernel_height),
            static_cast<uint32_t>(kernel_width),
            static_cast<uint32_t>(deconv_params->stride_height),
            static_cast<uint32_t>(deconv_params->stride_width),
            /*dilation_height=*/1,
            /*dilation_width=*/1,
            /*groups=*/1,
            /*group_input_channels=*/input_channels,
            /*group_output_channels=*/output_channels,
            /*output_min=*/output_min,
            /*output_max=*/output_max,
            /*input_id=*/input_output_tensors.at(input_tensor_index),
            /*filter_id=*/input_output_tensors.at(filter_tensor_index),
            /*bias_id=*/xnnpack_tensor_bias,
            /*output_id=*/input_output_tensors.at(output_tensor_index),
            /*flags=*/0);
        if (status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(
              logging_context, "failed to delegate %s node #%d",
              EnumNameBuiltinOperator(BuiltinOperator_TRANSPOSE_CONV),
              node_index);
          return kTfLiteError;
        }
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitVarHandleNode(xnn_subgraph_t subgraph,
                                         Delegate& delegate,
                                         TfLiteContext* logging_context,
                                         int node_index,
                                         const TfLiteNode* node) {
    if (!delegate.support_variable_ops()) {
      return kTfLiteError;
    }
    if (subgraph == nullptr) {
      const TfLiteVarHandleParams* params =
          static_cast<const TfLiteVarHandleParams*>(node->builtin_data);
      delegate.DefineVariable(params, node->outputs->data[0], logging_context,
                              node_index);
      // Always return error here because we don't know the type of this
      // variable yet, so we pretend that we can't handle this. Later, after
      // ReadVariable/AssignVariable tells us the data type, and we decide if
      // we can handle the datatype, we will update the nodes to delegate.
      return kTfLiteError;
    }
    // Nothing to do here when actually creating subgraph, as we don't
    // materialize any operators for this node.
    return kTfLiteOk;
  }

  inline bool NeedToSetUpVariableTensors() const {
    return has_variables_ && !variables_set_up_;
  }

  inline bool EnableSubgraphReshaping() const {
    return enable_subgraph_reshaping_;
  }

  inline Delegate* GetDelegate() const { return delegate_; }

 private:
  Subgraph(Delegate& delegate, xnn_runtime_t runtime,
           const std::unordered_set<int>& externals, std::vector<int>& inputs,
           std::vector<int>& outputs,
           std::unordered_map<int, uint32_t>& tflite_tensor_to_xnnpack)
      : runtime_(runtime, &xnn_delete_runtime) {
    for (int t : externals) {
      externals_[t] = nullptr;
    }
    tflite_tensor_to_xnnpack_ = tflite_tensor_to_xnnpack;
    inputs_ = inputs;
    outputs_ = outputs;
    has_variables_ = !delegate.GetAllVariableTensors().empty();
    enable_subgraph_reshaping_ = delegate.enable_subgraph_reshaping();
    delegate_ = &delegate;
  }

  // XNNPACK Runtime (subgraph + workspace) with smart-pointer for lifetime
  // management.
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> runtime_{
      nullptr, &xnn_delete_runtime};
  // Mapping from TFLite Tensor IDs for input/output tensors in the delegated
  // subgraph to their data locations.
  std::unordered_map<int, void*> externals_;
  // The input tensors to the XNNPack partition. Not all node input tensors
  // are consumed by XNNPack.
  std::vector<int> inputs_;
  // The output tensors to the XNNPack partition. Not all node output tensors
  // are consumed by XNNPack.
  std::vector<int> outputs_;
  // Mapping from TFLite Tensor IDs for tensors in the delegated subgraph to
  // the XNNPACK ID.
  std::unordered_map<int, uint32_t> tflite_tensor_to_xnnpack_;
  // Memory location to use for 0-size external tensors, as TFLite init their
  // data pointer to nullptr, and XNNPACK requires valid data pointers.
  char dummy_data_{0};
  // Persistent tensors need to be set up in all cases (even without external
  // inputs or outputs), but does not need to be set up again for further
  // invoke calls.
  bool has_variables_ = false;
  bool variables_set_up_ = false;
  bool enable_subgraph_reshaping_ = false;
  Delegate* delegate_;
};

TfLiteIntArray* Delegate::PrepareOpsToDelegate(TfLiteContext* context) {
  // Clear previous data, in case the delegate is reused without re-creation.
  static_unpacked_data_map_.clear();
  static_unpacked_data_.clear();
  static_unpack_nodes_.clear();
  static_sparse_weights_.clear();
  variable_holder_.ClearTensorIdToGlobalId();

  TfLiteIntArray* execution_plan = nullptr;
  if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context, "Unable to get graph execution plan.");
    return nullptr;
  }

  // Mapping for quasi-static (unpacked from static) tensor index to the node
  // index that produced it.
  std::unordered_map<int, int> quasi_static_tensors_producers;
  // Set of all quasi-static tensors in the execution plan.
  std::unordered_set<int> quasi_static_tensors;
  // Set of quasi-static tensors consumed by the delegated nodes.
  std::unordered_set<int> quasi_static_tensors_to_unpack;
  // Record all VarHandle nodes. At the point of visiting it, we don't know if
  // it can be delegated yet, because we don't know the type of the variable -
  // we rely on ReadVariable/AssignVariable to tell us the type. So the first
  // pass of VisitNode will associate the variable with a tensor, and after
  // the graph is walked once, we check all VarHandle nodes and decide if we
  // can handle them based on checking the global id of the variable tensor.
  // Maps VarHandle node index to local tensor id (i.e. output tensor id).
  std::unordered_map<int, int> variable_handles;

  TfLiteIntArray* nodes_to_delegate =
      TfLiteIntArrayCreate(execution_plan->size);
  nodes_to_delegate->size = 0;
  for (int i = 0; i < execution_plan->size; ++i) {
    const int node_index = execution_plan->data[i];

    // Check if TFLite nodes can be delegated to XNNPACK
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(context, node_index, &node,
                                        &registration) != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context,
                         "Unable to get node and registration for node %d.",
                         node_index);
      continue;  // Soft error (skip this node).
    }

    // Prepare to unpack FP16/INT8 tensors.
    if (registration->builtin_code == kTfLiteBuiltinDequantize &&
        node->inputs->size == 1 && node->outputs->size == 1) {
      const TfLiteTensor& input_tensor =
          context->tensors[node->inputs->data[0]];
      const TfLiteTensor& output_tensor =
          context->tensors[node->outputs->data[0]];

      bool is_supported_int8_tensor = input_tensor.type == kTfLiteInt8;
      if (is_supported_int8_tensor) {
        const auto* quant_params = static_cast<const TfLiteAffineQuantization*>(
            input_tensor.quantization.params);
        if (quant_params == nullptr) {
          is_supported_int8_tensor = false;
        }
      }
      if (input_tensor.sparsity == nullptr &&
          (input_tensor.allocation_type == kTfLiteMmapRo ||
           quasi_static_tensors.count(node->inputs->data[0]) != 0) &&
          (input_tensor.type == kTfLiteFloat16 || is_supported_int8_tensor) &&
          output_tensor.type == kTfLiteFloat32) {
        static_unpack_nodes_.insert(node_index);
        quasi_static_tensors_producers[node->outputs->data[0]] = node_index;
        quasi_static_tensors.insert(node->outputs->data[0]);

        if (input_tensor.allocation_type != kTfLiteMmapRo) {
          quasi_static_tensors_to_unpack.insert(node->inputs->data[0]);
        }

        // If dequantized input is sparse, so is its output
        if (static_sparse_weights_.count(node->inputs->data[0]) != 0) {
          static_sparse_weights_.insert(node->outputs->data[0]);
        }

        // Skip this node for now. If output of the node is consumed only by
        // delegated nodes, it will be added to nodes_to_delegate in the end.
        continue;
      }
    }

    // Prepare to unpack sparse tensors.
    // TODO(b/157729695): In the future, we also need to handle the case where
    // a sparse tensor is fed to a TFLite op directly, and no Densify() op is
    // inserted. For now this is not a problem because the Conv() op in tflite
    // can only consume dense tensors.
    if (registration->builtin_code == kTfLiteBuiltinDensify &&
        node->inputs->size == 1 && node->outputs->size == 1) {
      const TfLiteTensor& input_tensor =
          context->tensors[node->inputs->data[0]];
      const TfLiteTensor& output_tensor =
          context->tensors[node->outputs->data[0]];

      if (input_tensor.allocation_type == kTfLiteMmapRo &&
          input_tensor.sparsity != nullptr &&
          (input_tensor.type == kTfLiteFloat16 ||
           input_tensor.type == kTfLiteInt8 ||
           input_tensor.type == kTfLiteFloat32) &&
          output_tensor.type == input_tensor.type) {
        static_unpack_nodes_.insert(node_index);
        quasi_static_tensors_producers[node->outputs->data[0]] = node_index;
        quasi_static_tensors.insert(node->outputs->data[0]);
        static_sparse_weights_.insert(node->outputs->data[0]);

        // Skip this node for now. If output of the node is consumed only by
        // delegated nodes, it will be added to nodes_to_delegate in the end.
        continue;
      }
    }

    // Record node_index as we need it to check if node is delegated or not.
    if (registration->builtin_code == kTfLiteBuiltinVarHandle) {
      variable_handles[node_index] = node->outputs->data[0];
    }

    if (Subgraph::VisitNode(
            /*subgraph=*/nullptr, /*delegate=*/*this, context, registration,
            node, node_index, quasi_static_tensors,
            std::unordered_map<int, uint32_t>()) != kTfLiteOk) {
      // If a non-delegated node consumes output of a node that unpacks static
      // data, that node shouldn't be delegated.
      for (int j = 0; j < node->inputs->size; j++) {
        const auto it =
            quasi_static_tensors_producers.find(node->inputs->data[j]);
        if (it != quasi_static_tensors_producers.end()) {
          static_unpack_nodes_.erase(it->second);
        }
      }

      // Non-delegable node is not an error.
      continue;
    }

    for (int j = 0; j < node->inputs->size; j++) {
      if (quasi_static_tensors.count(node->inputs->data[j]) != 0) {
        quasi_static_tensors_to_unpack.insert(node->inputs->data[j]);
      }
    }

    nodes_to_delegate->data[nodes_to_delegate->size++] = node_index;
  }

  // Record which resource variables can be delegated.
  // A resource variable can be delegated if the global id can be found.
  for (const auto& n : variable_handles) {
    if (CanVariableBeDelegated(n.second)) {
      nodes_to_delegate->data[nodes_to_delegate->size++] = n.first;
    }
  }

  // Sort quasi-static tensors to be unpacked by the node index the produced
  // them. This ensures that in situations where quasi-static tensor is
  // produced from another quasi-static tensor, the tensors are unpacked in
  // the original execution plan order.
  std::vector<int> sorted_quasi_static_tensors_to_unpack(
      quasi_static_tensors_to_unpack.cbegin(),
      quasi_static_tensors_to_unpack.cend());
  std::sort(sorted_quasi_static_tensors_to_unpack.begin(),
            sorted_quasi_static_tensors_to_unpack.end(),
            [&quasi_static_tensors_producers](int t1, int t2) {
              return quasi_static_tensors_producers[t1] <
                     quasi_static_tensors_producers[t2];
            });

  // Unpack static data of all tensors
  for (int t : sorted_quasi_static_tensors_to_unpack) {
    const int producer_index = quasi_static_tensors_producers[t];
    // Check if TFLite nodes can be delegated to XNNPACK
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(context, producer_index, &node,
                                        &registration) != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context,
                         "Unable to get node and registration for node %d.",
                         producer_index);
      TfLiteIntArrayFree(nodes_to_delegate);
      return nullptr;  // Hard error.
    }

    if (node->inputs->size != 1) {
      TF_LITE_KERNEL_LOG(
          context, "unexpected number of inputs (%d) in %d node %d",
          node->inputs->size,
          static_cast<BuiltinOperator>(registration->builtin_code),
          producer_index);
      TfLiteIntArrayFree(nodes_to_delegate);
      return nullptr;  // Hard error.
    }

    if (node->outputs->size != 1) {
      TF_LITE_KERNEL_LOG(
          context, "unexpected number of outputs (%d) in %d node %d",
          node->outputs->size,
          static_cast<BuiltinOperator>(registration->builtin_code),
          producer_index);
      TfLiteIntArrayFree(nodes_to_delegate);
      return nullptr;  // Hard error.
    }

    const TfLiteTensor& input_tensor = context->tensors[node->inputs->data[0]];

    // Consider the case when the input to unpacking node is quasi-static.
    const auto static_unpacked_input_it_ =
        static_unpacked_data_map_.find(node->inputs->data[0]);
    if (static_unpacked_input_it_ == static_unpacked_data_map_.end()) {
      if (input_tensor.allocation_type != kTfLiteMmapRo) {
        TF_LITE_KERNEL_LOG(
            context,
            "unexpected allocation type (%d) in tensor %d in node %d (%d)",
            input_tensor.allocation_type, node->inputs->data[0], producer_index,
            registration->builtin_code);
        TfLiteIntArrayFree(nodes_to_delegate);
        return nullptr;  // Hard error.
      }
    }

    const TfLiteTensor& output_tensor = context->tensors[t];
    size_t tensor_elements = output_tensor.bytes;
    switch (output_tensor.type) {
      case kTfLiteFloat32:
        tensor_elements /= sizeof(float);
        break;
      case kTfLiteFloat16:
        tensor_elements /= sizeof(uint16_t);
        break;
      case kTfLiteInt8:
        tensor_elements /= sizeof(int8_t);
        break;
      default: {
        TF_LITE_KERNEL_LOG(context,
                           "unexpected datatype (%s) in tensor %d in node %d",
                           TfLiteTypeGetName(output_tensor.type),
                           node->outputs->data[0], producer_index);
        TfLiteIntArrayFree(nodes_to_delegate);
        return nullptr;  // Hard error.
      }
    }

    // Align to XNN_EXTRA_BYTES bytes
    while (static_unpacked_data_.size() % XNN_EXTRA_BYTES != 0) {
      static_unpacked_data_.push_back(0);
    }
    const size_t tensor_offset = static_unpacked_data_.size();
    static_unpacked_data_.resize(tensor_offset + context->tensors[t].bytes);

    char* unpacked_data = static_unpacked_data_.data() + tensor_offset;
    const char* packed_data =
        static_unpacked_input_it_ != static_unpacked_data_map_.end()
            ? static_unpacked_data_.data() + static_unpacked_input_it_->second
            : static_cast<const char*>(input_tensor.data.data);
    switch (registration->builtin_code) {
      case kTfLiteBuiltinDequantize: {
        // Such a condition has been checked when preparing to unpack
        // FP16/INT8 tensors.
        TFLITE_DCHECK(input_tensor.sparsity == nullptr);
        // Actual data unpacking
        switch (input_tensor.type) {
          case kTfLiteFloat16:
            DequantizeFloat16(reinterpret_cast<const uint16_t*>(packed_data),
                              reinterpret_cast<float*>(unpacked_data),
                              tensor_elements);
            break;
          case kTfLiteInt8: {
            TfLiteAffineQuantization* quant_params =
                static_cast<TfLiteAffineQuantization*>(
                    input_tensor.quantization.params);
            // Such conditions have been checked when preparing to unpack INT8
            // tensors.
            TFLITE_DCHECK(quant_params != nullptr);

            if (quant_params->scale->size == 1) {
              // Per-tensor quantization
              DequantizeInt8(reinterpret_cast<const int8_t*>(packed_data),
                             reinterpret_cast<float*>(unpacked_data),
                             GetTensorShape(&input_tensor),
                             input_tensor.params.zero_point,
                             input_tensor.params.scale);
            } else {
              // Per-channel quantization
              PerChannelDequantizeInt8(
                  reinterpret_cast<const int8_t*>(packed_data),
                  reinterpret_cast<float*>(unpacked_data),
                  GetTensorShape(&input_tensor), quant_params->zero_point->data,
                  quant_params->scale->data, quant_params->quantized_dimension);
            }
            break;
          }
          default:
            // This should not happen as we only allow FP16/INT8 input_tensor
            // when preparing the unpacking.
            TFLITE_DCHECK(false);
        }
        break;
      }
      case kTfLiteBuiltinDensify: {
        // Such a condition has been checked when preparing to unpack
        // FP16/INT8 tensors.
        TFLITE_DCHECK(input_tensor.sparsity != nullptr);
        const int dims_count = NumDimensions(&output_tensor);
        std::vector<int> vector_shape(dims_count);
        for (int i = 0; i < dims_count; i++) {
          vector_shape[i] = SizeOfDimension(&output_tensor, i);
        }

        switch (input_tensor.type) {
          case kTfLiteFloat32: {
            const size_t dense_size = context->tensors[t].bytes / sizeof(float);
            float* unpacked_fp32_data = reinterpret_cast<float*>(unpacked_data);
            tflite::internal::sparsity::FormatConverter<float> converter(
                vector_shape, *input_tensor.sparsity);
            converter.SparseToDense(
                static_cast<const float*>(input_tensor.data.data), dense_size,
                unpacked_fp32_data, context);
            break;
          }
          case kTfLiteFloat16: {
            const size_t dense_size =
                context->tensors[t].bytes / sizeof(Eigen::half);
            Eigen::half* unpacked_fp16_data =
                reinterpret_cast<Eigen::half*>(unpacked_data);
            tflite::internal::sparsity::FormatConverter<Eigen::half> converter(
                vector_shape, *input_tensor.sparsity);
            converter.SparseToDense(
                static_cast<const Eigen::half*>(input_tensor.data.data),
                dense_size, unpacked_fp16_data, context);
            break;
          }
          case kTfLiteInt8: {
            const size_t dense_size =
                context->tensors[t].bytes / sizeof(int8_t);
            int8_t* unpacked_int8_data =
                reinterpret_cast<int8_t*>(unpacked_data);
            tflite::internal::sparsity::FormatConverter<int8_t> converter(
                vector_shape, *input_tensor.sparsity);
            converter.SparseToDense(
                static_cast<const int8_t*>(input_tensor.data.data), dense_size,
                unpacked_int8_data, context);
            break;
          }
          default: {
            // This should not happen as we only allow FP16/INT8 input_tensor
            // when preparing the unpacking.
            TFLITE_DCHECK(false);
          }
        }
        break;
      }
      default:
        TF_LITE_KERNEL_LOG(context, "unexpected op registration %d at node %d",
                           registration->builtin_code, producer_index);
        TfLiteIntArrayFree(nodes_to_delegate);
        return nullptr;  // Hard error.
    }

    static_unpacked_data_map_[t] = tensor_offset;
  }

  // Now that the unpacking is done, we can update the weight cache mappings.
  //
  // We do it in a separate loop because `static_unpacked_data_` may need to
  // reallocate (and therefore invalidate the pointers) when it is grown.
  for (int t : sorted_quasi_static_tensors_to_unpack) {
    const int producer_index = quasi_static_tensors_producers[t];
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(context, producer_index, &node,
                                        &registration) != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context,
                         "Unable to get node and registration for node %d.",
                         producer_index);
      TfLiteIntArrayFree(nodes_to_delegate);
      return nullptr;  // Hard error.
    }
    const TfLiteTensor& input_tensor = context->tensors[node->inputs->data[0]];
    const auto tensor_offset = static_unpacked_data_map_[t];
    char* unpacked_data = static_unpacked_data_.data() + tensor_offset;
    const auto static_unpacked_input_it_ =
        static_unpacked_data_map_.find(node->inputs->data[0]);
    const char* packed_data =
        static_unpacked_input_it_ != static_unpacked_data_map_.end()
            ? static_unpacked_data_.data() + static_unpacked_input_it_->second
            : static_cast<const char*>(input_tensor.data.data);
    weight_cache_provider_.RemapDataBuffer(packed_data, unpacked_data);
  }

  // Add nodes that unpack static data consumed by delegated nodes.
  // Note: this is done purely to avoid the overhead of running these nodes
  // again in TFLite interpreter which would allocate memory for their
  // outputs. We mark them as delegated, but the delegate would simply ignore
  // these nodes as the static weights are already unpacked.
  for (int node_index : static_unpack_nodes_) {
    nodes_to_delegate->data[nodes_to_delegate->size++] = node_index;
  }
  std::sort(&nodes_to_delegate->data[0],
            &nodes_to_delegate->data[nodes_to_delegate->size]);

#ifdef XNNPACK_DELEGATE_TEST_MODE
  // In the test mode build (used by unit tests), XNNPACK delegate claims to
  // support all operators in the execution plan to disable fallback to the
  // default TensorFlow Lite kernels. Thus, if any of the ops in the model are
  // not supported by the delegate, they will cause a failure in
  // ::tflite::Interpreter::ModifyGraphWithDelegate, to be caught in the unit
  // tests.
  nodes_to_delegate->size = execution_plan->size;
  std::copy(&execution_plan->data[0],
            &execution_plan->data[execution_plan->size],
            &nodes_to_delegate->data[0]);
#endif

  return nodes_to_delegate;
}

void* SubgraphInit(TfLiteContext* context, const char* buffer, size_t length) {
  const TfLiteDelegateParams* params =
      reinterpret_cast<const TfLiteDelegateParams*>(buffer);

  return static_cast<void*>(Subgraph::Create(
      context, params,
      *static_cast<::tflite::xnnpack::Delegate*>(params->delegate->data_)));
}

TfLiteStatus SubgraphPrepare(TfLiteContext* context, TfLiteNode* node) {
  if (node->user_data == nullptr) {
    return kTfLiteError;
  }

  Subgraph* subgraph = static_cast<Subgraph*>(node->user_data);
  return static_cast<Subgraph*>(node->user_data)
      ->Prepare(context, node, subgraph->EnableSubgraphReshaping(),
                subgraph->GetDelegate());
}

TfLiteStatus SubgraphInvoke(TfLiteContext* context, TfLiteNode* node) {
  if (node->user_data == nullptr) {
    return kTfLiteError;
  }

  Subgraph* subgraph = static_cast<Subgraph*>(node->user_data);
  return static_cast<Subgraph*>(node->user_data)
      ->Invoke(context, subgraph->EnableSubgraphReshaping(),
               subgraph->GetDelegate());
}

void SubgraphFree(TfLiteContext* context, void* buffer) {
  if (buffer != nullptr) {
    delete static_cast<Subgraph*>(buffer);
  }
}

const TfLiteRegistration kSubgraphRegistration = {
    /*.init=*/SubgraphInit,
    /*.free=*/SubgraphFree,
    /*.prepare=*/SubgraphPrepare,
    /*.invoke=*/SubgraphInvoke,
    /*.profiling_string=*/nullptr,
    /*.builtin_code=*/0,
    /*.custom_name=*/"TfLiteXNNPackDelegate",
    /*.version=*/2,
};

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  TfLiteIntArray* ops_to_replace =
      static_cast<::tflite::xnnpack::Delegate*>(delegate->data_)
          ->PrepareOpsToDelegate(context);
  if (ops_to_replace == nullptr) {
    return kTfLiteError;
  }

  const TfLiteStatus status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, kSubgraphRegistration, ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}

}  // namespace
}  // namespace xnnpack
}  // namespace tflite

TfLiteXNNPackDelegateWeightsCache* TfLiteXNNPackDelegateWeightsCacheCreate() {
  xnn_status status = xnn_initialize(/*allocator=*/nullptr);
  if (status != xnn_status_success) {
    return nullptr;
  }

  xnn_weights_cache_t weights_cache = nullptr;
  if (xnn_create_weights_cache(&weights_cache) != xnn_status_success) {
    return nullptr;
  }
  return reinterpret_cast<TfLiteXNNPackDelegateWeightsCache*>(weights_cache);
}

TfLiteXNNPackDelegateWeightsCache*
TfLiteXNNPackDelegateWeightsCacheCreateWithSize(size_t size) {
  xnn_status status = xnn_initialize(/*allocator=*/nullptr);
  if (status != xnn_status_success) {
    return nullptr;
  }

  xnn_weights_cache_t weights_cache = nullptr;
  if (xnn_create_weights_cache_with_size(size, &weights_cache) !=
      xnn_status_success) {
    return nullptr;
  }
  return reinterpret_cast<TfLiteXNNPackDelegateWeightsCache*>(weights_cache);
}

bool TfLiteXNNPackDelegateWeightsCacheFinalizeSoft(
    TfLiteXNNPackDelegateWeightsCache* cache) {
  auto weights_cache = reinterpret_cast<xnn_weights_cache_t>(cache);
  xnn_status status = xnn_finalize_weights_cache(
      weights_cache, xnn_weights_cache_finalization_kind_soft);
  return status == xnn_status_success;
}

bool TfLiteXNNPackDelegateWeightsCacheFinalizeHard(
    TfLiteXNNPackDelegateWeightsCache* cache) {
  auto weights_cache = reinterpret_cast<xnn_weights_cache_t>(cache);
  xnn_status status = xnn_finalize_weights_cache(
      weights_cache, xnn_weights_cache_finalization_kind_hard);
  return status == xnn_status_success;
}

void TfLiteXNNPackDelegateWeightsCacheDelete(
    TfLiteXNNPackDelegateWeightsCache* cache) {
  if (cache == nullptr) {
    return;
  }
  auto weights_cache = reinterpret_cast<xnn_weights_cache_t>(cache);
  xnn_delete_weights_cache(weights_cache);
}

bool TfLiteXNNPackDelegateCanUseInMemoryWeightCacheProvider() {
  return tflite::xnnpack::InMemoryFileDescriptorAvailable();
}

const char* TfLiteXNNPackDelegateInMemoryFilePath() {
  return tflite::xnnpack::kInMemoryCachePath;
}

TfLiteXNNPackDelegateOptions TfLiteXNNPackDelegateOptionsDefault() {
  TfLiteXNNPackDelegateOptions options = {0};

  // Quantized inference is enabled by default on Web platform
#ifdef XNNPACK_DELEGATE_ENABLE_QS8
  options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_QS8;
#endif
#ifdef XNNPACK_DELEGATE_ENABLE_QU8
  options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_QU8;
#endif
#ifdef XNNPACK_DELEGATE_ENABLE_DYNAMIC_FULLY_CONNECTED
  options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_DYNAMIC_FULLY_CONNECTED;
#endif
#ifdef XNNPACK_DELEGATE_ENABLE_TRANSIENT_INDIRECTION_BUFFER
  options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_TRANSIENT_INDIRECTION_BUFFER;
#endif

  // Enable quantized inference for the delegate build used in unit tests.
  // Enable FULLY_CONNECTED operator with dynamic weights for the delegate build
  // used in unit tests.
#ifdef XNNPACK_DELEGATE_TEST_MODE
  options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_QS8;
  options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_QU8;
  options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_DYNAMIC_FULLY_CONNECTED;
#endif  // XNNPACK_DELEGATE_TEST_MODE

  return options;
}

TfLiteDelegate* TfLiteXNNPackDelegateCreate(
    const TfLiteXNNPackDelegateOptions* options) {
  return TfLiteXNNPackDelegateCreateWithThreadpool(options, nullptr);
}

TfLiteDelegate* TfLiteXNNPackDelegateCreateWithThreadpool(
    const TfLiteXNNPackDelegateOptions* options, TfLiteContext* context) {
  xnn_status status = xnn_initialize(/*allocator=*/nullptr);
  if (status != xnn_status_success) {
    return nullptr;
  }

  xnn_workspace_t workspace = nullptr;
  if (xnn_create_workspace(&workspace) != xnn_status_success) {
    return nullptr;
  }

  auto* xnnpack_delegate =
      new ::tflite::xnnpack::Delegate(options, workspace, context);
  return xnnpack_delegate ? xnnpack_delegate->tflite_delegate() : nullptr;
}

void* TfLiteXNNPackDelegateGetThreadPool(TfLiteDelegate* delegate) {
  if (delegate == nullptr) {
    return nullptr;
  }

  return static_cast<void*>(
      static_cast<::tflite::xnnpack::Delegate*>(delegate->data_)->threadpool());
}

const TfLiteXNNPackDelegateOptions* TfLiteXNNPackDelegateGetOptions(
    TfLiteDelegate* delegate) {
  if (delegate == nullptr) {
    return nullptr;
  }
  return &(static_cast<const tflite::xnnpack::Delegate*>(delegate->data_)
               ->options());
}

int TfLiteXNNPackDelegateGetFlags(TfLiteDelegate* delegate) {
  if (delegate == nullptr) {
    return 0;
  }

  auto* xnnpack_delegate =
      static_cast<::tflite::xnnpack::Delegate*>(delegate->data_);
  return xnnpack_delegate->options().flags;
}

void TfLiteXNNPackDelegateDelete(TfLiteDelegate* delegate) {
  if (delegate != nullptr) {
    ::tflite::xnnpack::Delegate* data =
        static_cast<::tflite::xnnpack::Delegate*>(delegate->data_);
    data->maybe_release_threadpool_ownership();
    delete data;
  }
}
