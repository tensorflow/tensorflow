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
#include "tensorflow/lite/core/interpreter_builder.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/allocation.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_utils.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/macros.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/internal/signature_def.h"
#include "tensorflow/lite/interpreter_options.h"
#include "tensorflow/lite/profiling/platform_profiler.h"
#include "tensorflow/lite/profiling/telemetry/c/telemetry_setting.h"
#include "tensorflow/lite/profiling/telemetry/c/telemetry_setting_internal.h"
#include "tensorflow/lite/schema/conversion_metadata_generated.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/shared_library.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/version.h"

// aligned_alloc is available (via cstdlib/stdlib.h) with C++17/C11.
// (introduced in stdc11 but realized in C++17)
#if __cplusplus >= 201703L && __STDC_VERSION__ >= 201112L
#if !defined(__ANDROID__) || __ANDROID_API__ >= 28
// Neither Apple nor Windows provide aligned_alloc.
#if !defined(__APPLE__) && !defined(_WIN32)
#define TFLITE_USE_STD_ALIGNED_ALLOC
#endif
#endif
#endif

// TODO(b/139446230): Move to portable platform header.
#if defined(__ANDROID__)
#define TFLITE_IS_MOBILE_PLATFORM
#endif  // defined(__ANDROID__)

#if defined(__APPLE__)
#include "TargetConditionals.h"
#if TARGET_IPHONE_SIMULATOR
#define TFLITE_IS_MOBILE_PLATFORM
#elif TARGET_OS_IPHONE
#define TFLITE_IS_MOBILE_PLATFORM
#endif
#endif  // defined(__APPLE__)

namespace tflite {

namespace {

constexpr char kConversionMetadataKey[] = "CONVERSION_METADATA";
constexpr char kTelemetryBuilderEventName[] = "InterpreterBuilder::operator()";

// Ensure that ErrorReporter is non-null.
ErrorReporter* ValidateErrorReporter(ErrorReporter* e) {
  return e ? e : DefaultErrorReporter();
}

template <typename T>
TfLiteStatus Copy(const T* data_ptr, TfLiteIntArray** arr) {
  if (data_ptr->values() == nullptr) {
    return kTfLiteError;
  }

  int size = data_ptr->values()->size();
  *arr = TfLiteIntArrayCreate(size);
  for (int i = 0; i < size; i++) {
    (*arr)->data[i] = static_cast<int>(data_ptr->values()->Get(i));
  }
  return kTfLiteOk;
}

TfLiteStatus ParseSparseIndexVector(const DimensionMetadata* src,
                                    TfLiteDimensionMetadata* tgt) {
  if (src->array_segments() == nullptr || src->array_indices() == nullptr) {
    return kTfLiteError;
  }
  TfLiteStatus status = kTfLiteOk;
  switch (src->array_segments_type()) {
    case SparseIndexVector_Int32Vector:
      status = Copy(src->array_segments_as_Int32Vector(), &tgt->array_segments);
      break;
    case SparseIndexVector_Uint16Vector:
      status =
          Copy(src->array_segments_as_Uint16Vector(), &tgt->array_segments);
      break;
    case SparseIndexVector_Uint8Vector:
      status = Copy(src->array_segments_as_Uint8Vector(), &tgt->array_segments);
      break;
    default:
      status = kTfLiteError;
      break;
  }
  if (status != kTfLiteOk) return status;

  switch (src->array_indices_type()) {
    case SparseIndexVector_Int32Vector:
      return Copy(src->array_indices_as_Int32Vector(), &tgt->array_indices);
    case SparseIndexVector_Uint16Vector:
      return Copy(src->array_indices_as_Uint16Vector(), &tgt->array_indices);
    case SparseIndexVector_Uint8Vector:
      return Copy(src->array_indices_as_Uint8Vector(), &tgt->array_indices);
    default:
      break;
  }
  return kTfLiteError;
}

// Helper that returns std::map that corresponds to vector of TensorMap.
std::map<std::string, uint32_t> GetMapFromTensorMap(
    const flatbuffers::Vector<flatbuffers::Offset<tflite::TensorMap>>*
        tensor_map) {
  if (!tensor_map) return {};
  std::map<std::string, uint32_t> result;
  for (const auto tensor : *tensor_map) {
    if (tensor != nullptr && tensor->name() != nullptr) {
      result[tensor->name()->c_str()] = tensor->tensor_index();
    }
  }
  return result;
}

inline bool ShouldCreateLazyDelegateProviders(int num_fp32_tensors) {
#if defined(XNNPACK_DELEGATE_ENABLE_QS8) || defined(XNNPACK_DELEGATE_ENABLE_QU8)
  return true;
#else
  return num_fp32_tensors > 0;
#endif
}

}  // namespace

constexpr const char* kEmptyTensorName = "";

// Using weak symbols to create a delegate allows automatic injection of the
// delegate simply by adding it as a dependency.
// For flex delegate, see also the strong override in
// lite/delegates/flex/delegate.cc.
TFLITE_ATTRIBUTE_WEAK Interpreter::TfLiteDelegatePtr AcquireFlexDelegate() {
  // TF_AcquireFlexDelegate isn't defined on Android, and the following block of
  // code would have no effect if TF_AcquireFlexDelegate isn't defined, so we
  // only enable that block for non-Android platforms.  Also, on Android 4.4
  // (Kitkat), the dlsym() implementation has a bug where dlsym() of an unknown
  // name will result in a SIGFPE, which would crash the process, so it's
  // important that on Android 4.4 we *don't* call SharedLibrary::GetSymbol
  // unless the symbol is sure to exist.
#if !defined(__ANDROID__)
  auto acquire_flex_delegate_func =
      reinterpret_cast<Interpreter::TfLiteDelegatePtr (*)()>(
          SharedLibrary::GetSymbol("TF_AcquireFlexDelegate"));
  if (acquire_flex_delegate_func) {
    return acquire_flex_delegate_func();
  }
#endif

#if !defined(TFLITE_IS_MOBILE_PLATFORM)
  // Load TF_AcquireFlexDelegate() from _pywrap_tensorflow_internal.so if it is
  // available.
#if defined(_WIN32)
  const wchar_t* filename_pywrap_tensorflow_internal =
      L"_pywrap_tensorflow_internal.pyd";
#elif defined(__APPLE__)
  const char* filename_pywrap_tensorflow_internal =
      "python/_pywrap_tensorflow_internal.so";
#else
  const char* filename_pywrap_tensorflow_internal =
      "_pywrap_tensorflow_internal.so";
#endif
  void* lib_tf_internal =
      SharedLibrary::LoadLibrary(filename_pywrap_tensorflow_internal);
#if defined(_WIN32)
  if (lib_tf_internal == nullptr) {
    lib_tf_internal = SharedLibrary::LoadLibrary(
        L"_pywrap_tensorflow_interpreter_wrapper.pyd");
  }
#endif
  if (lib_tf_internal) {
    acquire_flex_delegate_func =
        reinterpret_cast<Interpreter::TfLiteDelegatePtr (*)()>(
            SharedLibrary::GetLibrarySymbol(lib_tf_internal,
                                            "TF_AcquireFlexDelegate"));
    if (acquire_flex_delegate_func) {
      return acquire_flex_delegate_func();
    }
  }
#endif  // !defined(TFLITE_IS_MOBILE_PLATFORM)

  return Interpreter::TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

InterpreterBuilder::InterpreterBuilder(
    const FlatBufferModel& model, const OpResolver& op_resolver,
    const InterpreterOptions* options_experimental)
    : model_(model.GetModel()),
      op_resolver_(op_resolver),
      error_reporter_(ValidateErrorReporter(model.error_reporter())),
      metadata_(model.ReadAllMetadata()),
      allocation_(model.allocation()) {
  if (options_experimental) {
    options_ = *options_experimental;
  }
}

InterpreterBuilder::InterpreterBuilder(
    const ::tflite::Model* model, const OpResolver& op_resolver,
    ErrorReporter* error_reporter,
    const InterpreterOptions* options_experimental,
    const Allocation* allocation)
    : model_(model),
      op_resolver_(op_resolver),
      error_reporter_(ValidateErrorReporter(error_reporter)),
      metadata_(FlatBufferModel::ReadAllMetadata(model_)),
      allocation_(allocation) {
  if (options_experimental) {
    options_ = *options_experimental;
  }
}

InterpreterBuilder::~InterpreterBuilder() = default;

TfLiteStatus InterpreterBuilder::BuildLocalIndexToRegistrationMapping() {
  TfLiteStatus status = kTfLiteOk;
  // Reset state.
  flatbuffer_op_index_to_registration_.clear();
  unresolved_custom_ops_.clear();

  auto opcodes = model_->operator_codes();
  if (!opcodes) {
    return status;
  }
  int num_custom_ops = 0;
  for (const OperatorCode* opcode : *opcodes) {
    if (GetBuiltinCode(opcode) == BuiltinOperator_CUSTOM) {
      num_custom_ops++;
    }
  }
  unresolved_custom_ops_.reserve(num_custom_ops);
  for (const OperatorCode* opcode : *opcodes) {
    const TfLiteRegistration* registration = nullptr;
    status = GetRegistrationFromOpCode(opcode, op_resolver_, error_reporter_,
                                       &registration);
    if (status != kTfLiteOk) {
      if (GetBuiltinCode(opcode) != BuiltinOperator_CUSTOM) {
        return status;
      }
      // If it's an unresolved custom op, allow it for now. It might be resolved
      // by a delegate later.
      if (!opcode->custom_code()) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "Operator with CUSTOM builtin_code has no custom_code.\n");
        return status;
      }
      const auto* op_name = opcode->custom_code()->c_str();
      unresolved_custom_ops_.push_back(CreateUnresolvedCustomOp(op_name));
      registration = &unresolved_custom_ops_.back();
      has_flex_op_ |= IsFlexOp(op_name);
      status = kTfLiteOk;
    }
    flatbuffer_op_index_to_registration_.push_back(registration);
  }
  return status;
}

namespace {
template <class T>
std::vector<int> FlatBufferIntArrayToVector(T* flat_array) {
  // Initialize shape of tensors with null shape. Empty vectors are converted
  // to nullptr for models that are constructed via flatbuffers::Pack.
  if (flat_array == nullptr) {
    return {};
  }
  std::vector<int> ret(flat_array->size());
  for (int i = 0; i < flat_array->size(); i++) {
    ret[i] = flat_array->Get(i);
  }
  return ret;
}

// Used to determine how the op data parsing function creates its working space.
class MallocDataAllocator : public BuiltinDataAllocator {
 public:
  void* Allocate(size_t size, size_t alignment_hint) override {
#ifdef TFLITE_USE_STD_ALIGNED_ALLOC
    // Ensure that alignment is a power of two and a multiple of sizeof(void *)
    // and that size is an integral multiple of alignment.
    size_t used_alignment = std::max(alignment_hint, sizeof(void*));
    size_t used_size =
        ((size + used_alignment - 1) / used_alignment) * used_alignment;
    TFLITE_DCHECK(
        (used_alignment != 0) &&
        ((used_alignment & (used_alignment - 1)) == 0));  // is power-of-two
    return aligned_alloc(used_alignment, used_size);
#else
    return malloc(size);
#endif
  }
  void Deallocate(void* data) override { free(data); }
};

}  // namespace

TfLiteStatus InterpreterBuilder::ParseNodes(
    const flatbuffers::Vector<flatbuffers::Offset<Operator>>* operators,
    Subgraph* subgraph) {
  TfLiteStatus status = kTfLiteOk;

  // Reduce the number of redundant allocations
  subgraph->ReserveNodes(operators->size());

  for (int i = 0; i < operators->size(); ++i) {
    const auto* op = operators->Get(i);
    int index = op->opcode_index();
    if (index < 0 || index >= flatbuffer_op_index_to_registration_.size()) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Missing registration for opcode_index %d\n", index);
      status = kTfLiteError;
      continue;
    }

    const TfLiteRegistration* registration =
        flatbuffer_op_index_to_registration_[index];
    if (registration == nullptr) {
      TF_LITE_REPORT_ERROR(error_reporter_, "Skipping op for opcode_index %d\n",
                           index);
      status = kTfLiteError;
      continue;
    }

    BuiltinOperator op_type =
        static_cast<BuiltinOperator>(registration->builtin_code);

    if (op_type != BuiltinOperator_CUSTOM && op->custom_options()) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Found builtin operator %s with custom options.\n",
                           EnumNameBuiltinOperator(op_type));
    }

    void* builtin_data = nullptr;
    const char* init_data = nullptr;
    size_t init_data_size = 0;
    if (op_type == BuiltinOperator_CUSTOM) {
      if (op->custom_options()) {
        init_data = reinterpret_cast<const char*>(op->custom_options()->data());
        init_data_size = op->custom_options()->size();
      } else if (op->large_custom_options_offset() > 1 && allocation_) {
        if (op->large_custom_options_offset() +
                op->large_custom_options_size() >
            allocation_->bytes()) {
          TF_LITE_REPORT_ERROR(
              error_reporter_,
              "Custom Option Offset for opcode_index %d is out of bound\n",
              index);
          return kTfLiteError;
        }
        // If the custom op is storing payloads outside of flatbuffers
        init_data = reinterpret_cast<const char*>(allocation_->base()) +
                    op->large_custom_options_offset();
        init_data_size = op->large_custom_options_size();
      }
    } else {
      MallocDataAllocator malloc_allocator;
      TF_LITE_ENSURE_STATUS(ParseOpData(op, op_type, error_reporter_,
                                        &malloc_allocator, &builtin_data));
    }
    subgraph->AddNodeWithParameters(
        FlatBufferIntArrayToVector(op->inputs()),
        FlatBufferIntArrayToVector(op->outputs()),
        FlatBufferIntArrayToVector(op->intermediates()), init_data,
        init_data_size, builtin_data, registration);
  }

  return status;
}

TfLiteStatus InterpreterBuilder::ParseQuantization(
    const QuantizationParameters* src_quantization,
    TfLiteQuantization* quantization, const std::vector<int>& dims) {
  // Blockwise quantization.
  if (src_quantization && src_quantization->details_type() ==
                              QuantizationDetails_BlockwiseQuantization) {
    auto* src_quant = src_quantization->details_as_BlockwiseQuantization();
    quantization->type = kTfLiteBlockwiseQuantization;
    auto* blockwise_quantization =
        reinterpret_cast<TfLiteBlockwiseQuantization*>(
            malloc(sizeof(TfLiteBlockwiseQuantization)));
    blockwise_quantization->scale = src_quant->scales();
    blockwise_quantization->quantized_dimension = 0;
    blockwise_quantization->blocksize = src_quant->block_size();
    quantization->params = reinterpret_cast<void*>(blockwise_quantization);
    return kTfLiteOk;
  }
  quantization->type = kTfLiteNoQuantization;
  quantization->params = nullptr;
  if (!src_quantization || !src_quantization->scale() ||
      src_quantization->scale()->size() == 0) {
    return kTfLiteOk;
  }
  if (!src_quantization->zero_point()) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Quantization parameters has non-null scale but null zero_point.");
    return kTfLiteError;
  }

  // Ensure that the number of scales matches the number of zero_points.
  if (src_quantization->scale()->size() !=
      src_quantization->zero_point()->size()) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "QuantizationParam has %d zero_point values and %d scale values. "
        "Must "
        "have same number.",
        src_quantization->zero_point()->size(),
        src_quantization->scale()->size());
    return kTfLiteError;
  }

  const size_t num_scales = src_quantization->scale()->size();

  // Ensure that the quantization dimension is valid.
  if (src_quantization->quantized_dimension() < 0 ||
      (!dims.empty() &&
       src_quantization->quantized_dimension() >= dims.size())) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "quantized_dimension must be in range [0, %d). Was %d.", dims.size(),
        src_quantization->quantized_dimension());
    return kTfLiteError;
  }

  // Ensure that the number of scales is 1 for per-layer quantization, and
  // matches number of quantization dimensions for per-axis quantization.
  if (num_scales != 1 &&
      (!dims.empty() &&
       num_scales != dims[src_quantization->quantized_dimension()])) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "num_scales must be 1 for per-layer quantization, or %d for per-axis "
        "quantization, but got %d.",
        dims[src_quantization->quantized_dimension()], num_scales);
    return kTfLiteError;
  }

  // Affine-quantization.
  quantization->type = kTfLiteAffineQuantization;
  auto* affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  affine_quantization->scale = TfLiteFloatArrayCreate(num_scales);
  affine_quantization->zero_point = TfLiteIntArrayCreate(num_scales);
  for (size_t i = 0; i < num_scales; ++i) {
    affine_quantization->scale->data[i] = src_quantization->scale()->Get(i);
    affine_quantization->zero_point->data[i] =
        src_quantization->zero_point()->Get(i);
  }
  affine_quantization->quantized_dimension =
      src_quantization->quantized_dimension();
  quantization->params = reinterpret_cast<void*>(affine_quantization);
  return kTfLiteOk;
}

TfLiteStatus InterpreterBuilder::ParseSparsity(
    const SparsityParameters* src_sparsity, TfLiteSparsity** sparsity_ptr) {
  if (!src_sparsity) {
    return kTfLiteOk;
  }

  if (src_sparsity->traversal_order() == nullptr ||
      src_sparsity->dim_metadata() == nullptr) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Invalid sparsity parameter.");
    return kTfLiteError;
  }

  auto* sparsity =
      reinterpret_cast<TfLiteSparsity*>(malloc(sizeof(TfLiteSparsity)));
  memset(sparsity, 0, sizeof(TfLiteSparsity));
  *sparsity_ptr = sparsity;

  const size_t traversal_order_size = src_sparsity->traversal_order()->size();
  sparsity->traversal_order = TfLiteIntArrayCreate(traversal_order_size);
  for (int i = 0; i < traversal_order_size; i++) {
    sparsity->traversal_order->data[i] =
        src_sparsity->traversal_order()->Get(i);
  }

  if (src_sparsity->block_map()) {
    const size_t block_map_size = src_sparsity->block_map()->size();
    sparsity->block_map = TfLiteIntArrayCreate(block_map_size);
    for (int i = 0; i < block_map_size; i++) {
      sparsity->block_map->data[i] = src_sparsity->block_map()->Get(i);
    }
  }

  const size_t dim_metadata_size = src_sparsity->dim_metadata()->size();
  sparsity->dim_metadata_size = dim_metadata_size;
  sparsity->dim_metadata = reinterpret_cast<TfLiteDimensionMetadata*>(
      malloc(dim_metadata_size * sizeof(TfLiteDimensionMetadata)));
  memset(sparsity->dim_metadata, 0,
         dim_metadata_size * sizeof(TfLiteDimensionMetadata));

  for (int i = 0; i < dim_metadata_size; i++) {
    const auto* src_metadata = src_sparsity->dim_metadata()->Get(i);
    if (src_metadata->format() != DimensionType_DENSE &&
        src_metadata->format() != DimensionType_SPARSE_CSR) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "The %dth dimension has unknown type: %d.", i,
                           src_metadata->format());
      return kTfLiteError;
    }
    auto* tgt_metadata = &sparsity->dim_metadata[i];

    tgt_metadata->format =
        static_cast<TfLiteDimensionType>(src_metadata->format());

    if (tgt_metadata->format == kTfLiteDimDense) {
      tgt_metadata->dense_size = src_metadata->dense_size();
    } else {
      if (ParseSparseIndexVector(src_metadata, tgt_metadata) != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "The %dth sparse dimension has invalid parameters.", i);
        return kTfLiteError;
      }
    }
  }

  return kTfLiteOk;
}

TfLiteStatus InterpreterBuilder::ParseSignatureDefs(
    const flatbuffers::Vector<flatbuffers::Offset<SignatureDef>>*
        signature_def_list,
    Interpreter* interpreter) {
  if (signature_def_list == nullptr || signature_def_list->size() == 0) {
    return kTfLiteOk;
  }
  std::vector<internal::SignatureDef> signature_defs;
  signature_defs.reserve(signature_def_list->size());
  for (const auto fb_signature_def : *signature_def_list) {
    if (fb_signature_def == nullptr) {
      TF_LITE_REPORT_ERROR(error_reporter_, "NULL SignatureDef in the model.");
      return kTfLiteError;
    }
    if (fb_signature_def->signature_key() == nullptr) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Missing exported method name for SignatureDef");
      return kTfLiteError;
    }
    if (fb_signature_def->inputs() == nullptr) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "NULL SignatureDef inputs for exported method %s",
                           fb_signature_def->signature_key()->c_str());
      return kTfLiteError;
    }
    if (fb_signature_def->outputs() == nullptr) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "NULL SignatureDef outputs for exported method %s",
                           fb_signature_def->signature_key()->c_str());
      return kTfLiteError;
    }
    signature_defs.resize(signature_defs.size() + 1);
    auto& signature_def = signature_defs.back();
    signature_def.inputs = GetMapFromTensorMap(fb_signature_def->inputs());
    signature_def.outputs = GetMapFromTensorMap(fb_signature_def->outputs());
    signature_def.signature_key = fb_signature_def->signature_key()->c_str();
    signature_def.subgraph_index = fb_signature_def->subgraph_index();
  }
  interpreter->SetSignatureDef(std::move(signature_defs));
  return kTfLiteOk;
}

TfLiteStatus InterpreterBuilder::ParseTensors(
    const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
    const flatbuffers::Vector<flatbuffers::Offset<Tensor>>* tensors,
    Subgraph* subgraph, TfLiteTelemetrySubgraphInfo* subgraph_info) {
  TfLiteStatus status = kTfLiteOk;

  // A little helper to get the names of inputs and outputs. Note that they
  // must outlive the subgraph.
  auto get_name = [](const tflite::Tensor* t) -> const char* {
    auto name = t->name();
    if (name) return name->c_str();
    return kEmptyTensorName;
  };

  if (subgraph_info) {
    subgraph_info->quantizations.resize(tensors->size());
  }

  num_fp32_tensors_ = 0;
  for (int i = 0; i < tensors->size(); ++i) {
    const auto* tensor = tensors->Get(i);
    std::vector<int> dims = FlatBufferIntArrayToVector(tensor->shape());

    TfLiteType type;
    if (ConvertTensorType(tensor->type(), &type, error_reporter_) !=
        kTfLiteOk) {
      status = kTfLiteError;
      continue;
    }
    if (type == kTfLiteFloat32) {
      ++num_fp32_tensors_;
    }
    auto get_readonly_data = [&](const char** buffer_data,
                                 size_t* buffer_size) {
      // TODO(aselle): Check what happens if we have an unspecified size
      // constant.
      *buffer_data = nullptr;
      if (tensor->buffer() == 0) return kTfLiteOk;
      if (tensor->buffer() >= buffers->size()) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "Tensor %d specifies out of range buffer %d (only %d buffers).\n",
            i, tensor->buffer(), buffers->size());
        return kTfLiteError;
      }
      if (auto* buffer = (*buffers)[tensor->buffer()]) {
        auto offset = buffer->offset();
        if (auto* array = buffer->data()) {
          *buffer_size = array->size();
          *buffer_data = reinterpret_cast<const char*>(array->data());
          return kTfLiteOk;
        } else if (offset > 1 && allocation_) {
          if (offset + buffer->size() > allocation_->bytes()) {
            TF_LITE_REPORT_ERROR(
                error_reporter_,
                "Constant buffer %d specified an out of range offset.\n",
                tensor->buffer());
            return kTfLiteError;
          }
          *buffer_size = buffer->size();
          *buffer_data =
              reinterpret_cast<const char*>(allocation_->base()) + offset;
          return kTfLiteOk;
        }
      }
      return kTfLiteOk;
    };
    size_t buffer_size = 0;
    const char* buffer_ptr;
    TF_LITE_ENSURE_STATUS(get_readonly_data(&buffer_ptr, &buffer_size));

    const auto* src_quantization = tensor->quantization();
    TfLiteQuantization quantization{};
    if (ParseQuantization(src_quantization, &quantization, dims) != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Tensor %d has invalid quantization parameters.", i);
      status = kTfLiteError;
    }
    if (subgraph_info) subgraph_info->quantizations[i] = quantization;

    std::vector<int> dims_signature = {};
    if (tensor->shape_signature()) {
      dims_signature = FlatBufferIntArrayToVector(tensor->shape_signature());
    }

    bool is_variable = tensor->is_variable();
    if (buffer_ptr) {
      if (is_variable) {
        TF_LITE_REPORT_ERROR(error_reporter_,
                             "Tensor %d is a variable tensor with buffer. "
                             "It's not supported now.\n",
                             i);
        status = kTfLiteError;
      }

      // TODO(b/144999664): Only constant sparse tensor is supported now.
      const auto* src_sparsity = tensor->sparsity();
      TfLiteSparsity* sparsity = nullptr;
      if (ParseSparsity(src_sparsity, &sparsity) != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter_,
                             "Tensor %d has invalid sparsity parameters.", i);
        status = kTfLiteError;
      }

      if (subgraph->SetTensorParametersReadOnly(
              i, type, get_name(tensor), dims, quantization, buffer_ptr,
              buffer_size, allocation_, sparsity,
              /*buffer_identifier=*/tensor->buffer()) != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter_,
                             "Tensor %d is invalidly specified in schema.\n",
                             i);
        status = kTfLiteError;
      }
    } else {
      if (subgraph->SetTensorParametersReadWrite(
              i, type, get_name(tensor), dims, quantization, is_variable,
              dims_signature) != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter_,
                             "Tensor %d is invalidly specified in schema.\n",
                             i);
        status = kTfLiteError;
      }
    }
  }

  return status;
}

TfLiteStatus InterpreterBuilder::ApplyDelegates(Interpreter* interpreter) {
  // Apply Flex delegate if applicable.
  if (has_flex_op_) {
    if (Interpreter::TfLiteDelegatePtr flex_delegate = AcquireFlexDelegate()) {
      TF_LITE_ENSURE_STATUS(interpreter->ModifyGraphWithDelegateImpl(
          // Transfers ownership of flex_delegate to the interpreter.
          std::move(flex_delegate)));
    }
  }
  for (TfLiteDelegate* delegate : delegates_) {
    // Note that we DON'T transfer ownership of the delegate to the
    // interpreter. (Doing that would cause problems if operator() was invoked
    // twice.)
    TF_LITE_ENSURE_STATUS(interpreter->ModifyGraphWithDelegateImpl(delegate));
  }
  return kTfLiteOk;
}

TfLiteStatus InterpreterBuilder::SetNumThreads(int num_threads) {
  if (num_threads < -1) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "num_threads should be >= 0 or just -1 to let TFLite runtime set the "
        "value.");
    return kTfLiteError;
  }
  num_threads_ = num_threads;
  return kTfLiteOk;
}

TfLiteStatus InterpreterBuilder::operator()(
    std::unique_ptr<Interpreter>* interpreter, int num_threads) {
  TfLiteStatus status = SetNumThreads(num_threads);
  if (status != kTfLiteOk) {
    interpreter->reset();
    return status;
  }
  return (*this)(interpreter);
}

TfLiteStatus InterpreterBuilder::operator()(
    std::unique_ptr<Interpreter>* interpreter) {
  if (!interpreter) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Null output pointer passed to InterpreterBuilder.");
    return kTfLiteError;
  }

  // Safe exit by deleting partially created interpreter, to reduce verbosity
  // on error conditions. Use by return cleanup_on_error();
  auto cleanup_and_error = [&interpreter]() {
    interpreter->reset();
    return kTfLiteError;
  };

  if (!model_) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Null pointer passed in as model.");
    return cleanup_and_error();
  }

  if (model_->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model_->version(), TFLITE_SCHEMA_VERSION);
    return cleanup_and_error();
  }

  if (BuildLocalIndexToRegistrationMapping() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Registration failed.\n");
    return cleanup_and_error();
  }

  // Flatbuffer model schemas define a list of opcodes independent of the
  // graph. We first map those to registrations. This reduces string lookups
  // for custom ops since we only do it once per custom op rather than once
  // per custom op invocation in the model graph. Construct interpreter with
  // correct number of tensors and operators.
  auto* subgraphs = model_->subgraphs();
  auto* buffers = model_->buffers();

  if (subgraphs->size() == 0) {
    TF_LITE_REPORT_ERROR(error_reporter_, "No subgraph in the model.\n");
    return cleanup_and_error();
  }

  if (!buffers) {
    TF_LITE_REPORT_ERROR(error_reporter_, "No buffers in the model.\n");
    return cleanup_and_error();
  }

  *interpreter = std::make_unique<Interpreter>(error_reporter_);
  if (subgraphs->size() > 1) {
    (*interpreter)->AddSubgraphs(subgraphs->size() - 1);
  }

  // Set num threads after all the subgraphs are added.
  (*interpreter)->SetNumThreads(num_threads_);

  // Set Interpreter options
  (*interpreter)->ApplyOptionsImpl(&options_);

  (*interpreter)
      ->SetProfilerImpl(tflite::profiling::MaybeCreatePlatformProfiler());

  bool telemetry_registered = telemetry_profiler_ != nullptr;
  std::unique_ptr<TfLiteTelemetryInterpreterSettings> telemetry_settings;
  if (telemetry_registered) {
    (*interpreter)->AddProfiler(std::move(telemetry_profiler_));
    telemetry_settings = std::make_unique<TfLiteTelemetryInterpreterSettings>();
    telemetry_settings->subgraph_infos.resize(subgraphs->size());
  }

  for (int subgraph_index = 0; subgraph_index < subgraphs->size();
       ++subgraph_index) {
    const tflite::SubGraph* subgraph = (*subgraphs)[subgraph_index];
    tflite::Subgraph* modified_subgraph =
        (*interpreter)->subgraph(subgraph_index);
    modified_subgraph->allocation_ = allocation_;
    auto* subgraph_info =
        telemetry_registered
            ? &telemetry_settings->subgraph_infos[subgraph_index]
            : nullptr;
    auto operators = subgraph->operators();
    auto tensors = subgraph->tensors();
    if (!tensors) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Did not get tensors in subgraph %d.\n",
                           subgraph_index);
      return cleanup_and_error();
    }
    if (modified_subgraph->AddTensors(tensors->size()) != kTfLiteOk) {
      return cleanup_and_error();
    }
    // Parse inputs/outputs
    modified_subgraph->SetInputs(
        FlatBufferIntArrayToVector(subgraph->inputs()));
    modified_subgraph->SetOutputs(
        FlatBufferIntArrayToVector(subgraph->outputs()));

    // Finally setup nodes and tensors
    // Parse tensors before nodes as ParseNodes checks input tensors for the
    // nodes.
    if (ParseTensors(buffers, tensors, modified_subgraph, subgraph_info) !=
        kTfLiteOk)
      return cleanup_and_error();
    if (operators && ParseNodes(operators, modified_subgraph) != kTfLiteOk)
      return cleanup_and_error();

    std::vector<int> variables;
    for (int i = 0; i < modified_subgraph->tensors_size(); ++i) {
      auto* tensor = modified_subgraph->tensor(i);
      if (tensor->is_variable) {
        variables.push_back(i);
      }
    }
    modified_subgraph->SetVariables(std::move(variables));
    if (subgraph->name()) {
      modified_subgraph->SetName(subgraph->name()->c_str());
    }
  }

  if (ParseSignatureDefs(model_->signature_defs(), interpreter->get()) !=
      kTfLiteOk) {
    return cleanup_and_error();
  }

  if (options_.GetUseSignatureTensorNames()) {
    for (auto& signature_def : (*interpreter)->signature_defs_) {
      auto* subgraph = (*interpreter)->subgraph(signature_def.subgraph_index);
      for (auto& [name, tensor_index] : signature_def.inputs) {
        auto tensor = subgraph->tensor(tensor_index);
        tensor->name = name.c_str();
      }
      for (auto& [name, tensor_index] : signature_def.outputs) {
        auto tensor = subgraph->tensor(tensor_index);
        tensor->name = name.c_str();
      }
    }
  }

  if ((*interpreter)->SetMetadata(metadata_) != kTfLiteOk) {
    return cleanup_and_error();
  }

  if (ShouldCreateLazyDelegateProviders(num_fp32_tensors_)) {
    (*interpreter)->lazy_delegate_providers_ =
        op_resolver_.GetDelegateCreators();
  }

  if (telemetry_registered) {
    ParseConversionMetadata(telemetry_settings.get());
    (*interpreter)->SetTelemetrySettings(std::move(telemetry_settings));
    // Reports model and interpreter settings if telemetry is applied.
    (*interpreter)->ReportTelemetrySettings(kTelemetryBuilderEventName);
  }

  TfLiteStatus status = ApplyDelegates(interpreter->get());
  if (status != kTfLiteOk) {
    interpreter->reset();
  }

  // Apply Interpreter options again for dynamic allocation.
  if (options_.GetDynamicAllocationForLargeTensors()) {
    (*interpreter)->ApplyOptionsImpl(&options_);
  }

  return status;
}

void InterpreterBuilder::AddDelegate(TfLiteDelegate* delegate) {
  if (delegate == nullptr) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Null delegate.");
  } else {
    delegates_.push_back(delegate);
  }
}

void InterpreterBuilder::AddDelegate(
    TfLiteOpaqueDelegateStruct* opaque_delegate) {
  // The following cast is safe only because this code is part of the TF Lite
  // runtime code.  Apps using TF Lite should not rely on
  // TfLiteOpaqueDelegateStruct and TfLiteDelegate being equivalent.
  AddDelegate(reinterpret_cast<TfLiteDelegate*>(opaque_delegate));
}

void InterpreterBuilder::ParseConversionMetadata(
    TfLiteTelemetryInterpreterSettings* settings) {
  if (settings == nullptr) return;
  auto it = metadata_.find(kConversionMetadataKey);
  if (it == metadata_.end()) {
    // No conversion metadata embeded.
    return;
  }
  auto* conversion_meta = GetConversionMetadata(it->second.data());
  if (conversion_meta == nullptr || conversion_meta->options() == nullptr) {
    // Empty conversion metadata.
    return;
  }
  settings->conversion_metadata =
      std::make_unique<TfLiteTelemetryConversionMetadata>();
  settings->conversion_metadata->model_optimization_modes =
      FlatBufferIntArrayToVector(
          conversion_meta->options()->model_optimization_modes());
}

}  // namespace tflite
