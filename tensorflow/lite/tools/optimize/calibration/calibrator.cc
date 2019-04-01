/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/optimize/calibration/calibrator.h"

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/optimize/calibration/calibration_common.h"
#include "tensorflow/lite/tools/optimize/calibration/calibration_logger.h"
#include "tensorflow/lite/tools/optimize/calibration/calibration_reader.h"
#include "tensorflow/lite/tools/optimize/calibration/logging_op_resolver.h"
#include "tensorflow/lite/tools/optimize/calibration/node_info_delegate.h"

namespace tflite {
namespace optimize {
namespace calibration {

namespace {

// Calibrator is used to hold information that can be accessed during kernel
// invocations.
// TfLite kernel invocations are C functions and cannot look at the global
// structure of the graph. Calibrator allows the kernel invoke functions to
// access the global structure of graph and know which node is currently being
// executed. This also allows us to write a simple kernel invoke wrapper
// (see LoggingEval) that can work for most builtin ops.
class Calibrator {
 public:
  Calibrator(const std::unordered_map<const TfLiteNode*, OperatorInfo>&
                 node_ptr_opinfo_map,
             std::unique_ptr<LoggingOpResolver> logging_op_resolver)
      : node_ptr_opinfo_map_(node_ptr_opinfo_map),
        logging_op_resolver_(std::move(logging_op_resolver)) {
    logger_ = absl::make_unique<Logger>();
  }

  // Returns the wrapped kernel invoke function |TfLiteRegistration.invoke|.
  KernelEvalFuncPtr GetKernelInvoke(const TfLiteNode* node) const;

  // Gets the instance of logger associated with the current context.
  Logger* GetLogger() const { return logger_.get(); }

  // Gets the operator information about the given TfLiteNode.
  const OperatorInfo& GetOpInfo(const TfLiteNode* node) const {
    return node_ptr_opinfo_map_.at(node);
  }

 private:
  std::unordered_map<const TfLiteNode*, OperatorInfo> node_ptr_opinfo_map_;
  std::unique_ptr<LoggingOpResolver> logging_op_resolver_;
  const std::unordered_map<int, OperatorInfo> index_opinfo_;
  std::unique_ptr<Logger> logger_;
};

KernelEvalFuncPtr Calibrator::GetKernelInvoke(const TfLiteNode* node) const {
  auto op_info = node_ptr_opinfo_map_.at(node);
  return logging_op_resolver_->GetWrappedKernelInvoke(op_info.builtin_op_code,
                                                      1);
}

// A registry of |Calibrator| objects per |TfLiteContext|.
// This global registry is needed to access |Calibrator| objects in the kernel
// invoke functions i.e. |TfLiteRegistration.invoke|.
// Kernel invoke functions are C functions that have limited access to
// |TfLiteContext|. Kernel invoke functions don't have access to global state of
// graph. That means during a kernel invocation, the function cannot know which
// node it was invoked for. E.g. in case of a model with |Conv| op at two
// locations, there is no easy way for the Conv.invoke function to disambiguate
// the calls.
//
// For calibration we solve this problem by creating a map of calibrators
// per |TfLiteContext|. This map is |GlobalCalibrationRegistry|.
//
// This registry is then accessed using a global getter function:
// |GetCalibratorRegistry|.
// E.g.
// TfLiteStatus SomeKernelInvokeFn(TfLiteContext* context, TfLiteNode* node) {
//   .... code ....
//   auto registry = GetCalibratorRegistry();
//   auto calibrator = registry->GetCalibrator(context);
//   ..... code ....
//  }
//
// This way the kernel invoke functions can get the access to the Calibrator
// object associated with the |TfLiteContext|.
class GlobalCalibratorRegistry {
 public:
  // Get the |Calibrator| associated with given context, returns null if no
  // calibrator is associated with the given context.
  Calibrator* GetCalibrator(const TfLiteContext* context) const {
    if (calibrator_registry_.find(context) == calibrator_registry_.cend()) {
      return nullptr;
    }
    return calibrator_registry_.at(context).get();
  }

  // Removes the association between calibrator and context.
  // Note: This deletes the calibrator as well.
  void RemoveCalibrator(const TfLiteContext* context) {
    calibrator_registry_.erase(context);
  }

  // Creates an instance of |Calibrator|.
  // Registry owns the |Calibrator| object which can be deleted by calling
  // |RemoveCalibrator|.
  TfLiteStatus CreateCalibrator(
      const TfLiteContext* context,
      const std::unordered_map<const TfLiteNode*, OperatorInfo>& node_to_opinfo,
      std::unique_ptr<LoggingOpResolver> logging_op_resolver,
      Calibrator** calibrator_ptr, ErrorReporter* reporter) {
    if (calibrator_registry_.find(context) != calibrator_registry_.cend()) {
      reporter->Report(
          "Failed to create calibrator, context already registered.");
      return kTfLiteError;
    }
    std::unique_ptr<Calibrator> calibrator = absl::make_unique<Calibrator>(
        node_to_opinfo, std::move(logging_op_resolver));
    calibrator_registry_[context] = std::move(calibrator);
    *calibrator_ptr = calibrator_registry_.at(context).get();
    return kTfLiteOk;
  }

 private:
  std::unordered_map<const TfLiteContext*, std::unique_ptr<Calibrator>>
      calibrator_registry_;
};

GlobalCalibratorRegistry* GetCalibratorRegistry() {
  static GlobalCalibratorRegistry* registry = new GlobalCalibratorRegistry();
  return registry;
}

// A wrapper implementation for |TfLiteRegistration.invoke| that logs inputs,
// invokes the wrapped implementation and then logs the outputs.
TfLiteStatus LoggingEval(TfLiteContext* context, TfLiteNode* node) {
  Calibrator* calibrator = GetCalibratorRegistry()->GetCalibrator(context);

  if (!calibrator) {
    context->ReportError(context, "No calibrator found for context.");
    return kTfLiteError;
  }

  auto kernel_invoke = calibrator->GetKernelInvoke(node);
  auto logger = calibrator->GetLogger();
  auto op_info = calibrator->GetOpInfo(node);

  for (int i : op_info.loggable_inputs) {
    auto tensor = context->tensors[i];
    TF_LITE_ENSURE_STATUS(
        logger->LogTensorValue(i, tensor.data.f, tensor.bytes / sizeof(float)));
  }

  auto status = kernel_invoke(context, node);
  // TODO(shashishekhar): An intermediate tensor in graph will get logged twice
  // once as an input and second time as output. This doesn't change the min max
  // values but is inefficient.
  // Using moving average will also break this.

  for (int i : op_info.loggable_outputs) {
    auto tensor = context->tensors[i];
    TF_LITE_ENSURE_STATUS(
        logger->LogTensorValue(i, tensor.data.f, tensor.bytes / sizeof(float)));
  }

  return status;
}

// Returns the loggable tensors. Not all inputs and outputs need to be logged.
// For example, const weight tensors which have buffers associated with them
// don't need to be logged.
std::vector<int> GetLoggableTensorIndices(
    const std::vector<int>& tensor_indices,
    const flatbuffers::Vector<flatbuffers::Offset<Tensor>>* tensors,
    const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* tensor_buffers) {
  std::vector<int> loggable;
  for (auto tensor_index : tensor_indices) {
    auto tensor = tensors->Get(tensor_index);
    auto buffer_index = tensor->buffer();
    const bool has_no_buffer =
        (tensor_buffers->Get(buffer_index) == nullptr) ||
        (tensor_buffers->Get(buffer_index)->data() == nullptr) ||
        (tensor_buffers->Get(buffer_index)->data()->size() == 0);
    if (has_no_buffer && tensor->type() == tflite::TensorType_FLOAT32) {
      loggable.push_back(tensor_index);
    }
  }
  return loggable;
}

// Creates a mapping between the static model graph and the runtime TfLiteNode*
// nodes in the graph for the given context.
// This is done by querying the TfLiteContext for node and registrations using
// the |NodeInfoDelegateObserver|.
TfLiteStatus GetNodeOpInfoMapAndContext(
    const std::unordered_map<int, OperatorInfo>& node_to_opinfo,
    tflite::Interpreter* const interpreter,
    std::unordered_map<const TfLiteNode*, OperatorInfo>* node_ptr_opinfo_map,
    const TfLiteContext** context) {
  NodeInfoDelegateObserver delegate_observer(node_to_opinfo,
                                             node_ptr_opinfo_map);
  NodeInfoDelegateParams delegate_params;
  delegate_params.delegate_observer = &delegate_observer;
  TfLiteDelegate logging_delegate = CreateNodeInfoDelegate(&delegate_params);

  auto modify_status = interpreter->ModifyGraphWithDelegate(&logging_delegate);
  if (modify_status != kTfLiteOk) {
    return kTfLiteError;
  }
  *context = delegate_observer.GetContext();
  return kTfLiteOk;
}

string GetOpName(const tflite::OperatorCode& opcode) {
  if (opcode.custom_code() != nullptr) {
    return opcode.custom_code()->str();
  }
  return tflite::EnumNamesBuiltinOperator()[opcode.builtin_code()];
}

// A |CalibrationReader| that owns the Calibrator.
class Reader : public CalibrationReader {
 public:
  Reader(const TfLiteContext* context, const Logger* logger)
      : CalibrationReader(logger), context_(context) {}

  ~Reader() override { GetCalibratorRegistry()->RemoveCalibrator(context_); }

 private:
  const TfLiteContext* context_;
};

}  // namespace

TfLiteStatus BuildLoggingInterpreter(
    const FlatBufferModel& model, const OpResolver& op_resolver,
    std::unique_ptr<Interpreter>* interpreter,
    std::unique_ptr<CalibrationReader>* calibration_reader) {
  auto tflite_model = model.GetModel();
  auto subgraphs = tflite_model->subgraphs();
  auto tensor_buffers = tflite_model->buffers();

  if (subgraphs->size() != 1) {
    model.error_reporter()->Report(
        "Only models with a single subgraph are supported, model had %d "
        "subgraphs",
        subgraphs->size());
    return kTfLiteError;
  }

  // Populate the node index to operator info map.
  // We want to collect this information so we can use it during runtime to
  // log details of which inputs and outputs.
  // At runtime TFLite kernel invoke functions can only look into their
  // own node in the graph (TFLiteNode*) and some limited context information.
  auto primary_subgraph = subgraphs->Get(0);
  auto operator_codes = tflite_model->operator_codes();
  auto operators = primary_subgraph->operators();
  auto tensors = primary_subgraph->tensors();
  std::unordered_map<int, OperatorInfo> node_to_opinfo;
  BuiltinOpsSet op_and_versions;

  for (size_t i = 0; i < operators->size(); i++) {
    OperatorInfo op_info;
    op_info.node_index = i;
    auto op = operators->Get(i);
    auto operator_code = operator_codes->Get(op->opcode_index());
    op_info.builtin_op_code = operator_code->builtin_code();
    op_info.name = GetOpName(*operator_code);
    op_info.is_custom_op = operator_code->custom_code() != nullptr;

    auto op_inputs = op->inputs();
    auto op_outputs = op->outputs();
    op_info.inputs = std::vector<int>(op_inputs->begin(), op_inputs->end());
    op_info.outputs = std::vector<int>(op_outputs->begin(), op_outputs->end());
    op_info.loggable_inputs =
        GetLoggableTensorIndices(op_info.inputs, tensors, tensor_buffers);
    op_info.loggable_outputs =
        GetLoggableTensorIndices(op_info.outputs, tensors, tensor_buffers);
    if (!op_info.is_custom_op) {
      op_info.registration = op_resolver.FindOp(operator_code->builtin_code(),
                                                operator_code->version());
    } else {
      op_info.registration =
          op_resolver.FindOp(op_info.name.c_str(), operator_code->version());
    }
    node_to_opinfo[i] = op_info;
    op_and_versions.insert({op_info.builtin_op_code, operator_code->version()});
  }

  // Prepare the logging op resolver to use |LoggingEval| for kernel
  // invocations.
  auto logging_op_resolver = absl::make_unique<LoggingOpResolver>(
      op_and_versions, op_resolver, LoggingEval);
  tflite::InterpreterBuilder(model, *logging_op_resolver)(interpreter);

  if (!(*interpreter)) {
    model.error_reporter()->Report("Failed to construct interpreter");
    return kTfLiteError;
  }

  // Compute the mapping between runtime and static graph structure, i.e.
  // (TfLiteContext, TfLiteNode) -> OperatorInfo
  std::unordered_map<const TfLiteNode*, OperatorInfo> node_ptr_opinfo_map;
  const TfLiteContext* context = nullptr;
  GetNodeOpInfoMapAndContext(node_to_opinfo, interpreter->get(),
                             &node_ptr_opinfo_map, &context);

  Calibrator* calibrator = nullptr;
  // Register a calibrator object for the context. This can be accessed
  // during invocations by the logging kernels.
  TF_LITE_ENSURE_STATUS(GetCalibratorRegistry()->CreateCalibrator(
      context, node_ptr_opinfo_map, std::move(logging_op_resolver), &calibrator,
      model.error_reporter()));
  *calibration_reader = std::unique_ptr<CalibrationReader>(
      new Reader(context, calibrator->GetLogger()));

  return kTfLiteOk;
}

}  // namespace calibration
}  // namespace optimize
}  // namespace tflite
