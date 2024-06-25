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
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/tools/optimize/calibration/builtin_logging_ops/lstm.h"
#include "tensorflow/lite/tools/optimize/calibration/calibration_common.h"
#include "tensorflow/lite/tools/optimize/calibration/calibration_logger.h"
#include "tensorflow/lite/tools/optimize/calibration/calibration_reader.h"
#include "tensorflow/lite/tools/optimize/calibration/custom_logging_ops/lstm.h"
#include "tensorflow/lite/tools/optimize/calibration/logging_op.h"
#include "tensorflow/lite/tools/optimize/calibration/logging_op_resolver.h"

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
             std::unique_ptr<LoggingOpResolver> logging_op_resolver,
             ErrorReporter* error_reporter)
      : node_ptr_opinfo_map_(node_ptr_opinfo_map),
        logging_op_resolver_(std::move(logging_op_resolver)),
        error_reporter_(error_reporter) {
    logger_ = std::make_unique<Logger>();
  }

  // Returns the wrapped kernel invoke function |TfLiteRegistration.invoke|.
  KernelEvalFuncPtr GetKernelInvoke(const TfLiteNode* node) const;

  // Gets the instance of logger associated with the current context.
  Logger* GetLogger() const { return logger_.get(); }

  // Gets the error reporter.
  ErrorReporter* GetErrorReporter() const { return error_reporter_; }

  // Gets the operator information about the given TfLiteNode.
  const OperatorInfo& GetOpInfo(const TfLiteNode* node) const {
    return node_ptr_opinfo_map_.at(node);
  }

  std::vector<const TfLiteNode*> GetNodesUnderCalibration() {
    std::vector<const TfLiteNode*> nodes;
    nodes.reserve(node_ptr_opinfo_map_.size());
    for (const auto& entry : node_ptr_opinfo_map_) {
      nodes.push_back(entry.first);
    }
    return nodes;
  }

 private:
  std::unordered_map<const TfLiteNode*, OperatorInfo> node_ptr_opinfo_map_;
  std::unique_ptr<LoggingOpResolver> logging_op_resolver_;
  const std::unordered_map<int, OperatorInfo> index_opinfo_;
  std::unique_ptr<Logger> logger_;
  ErrorReporter* error_reporter_;
};

KernelEvalFuncPtr Calibrator::GetKernelInvoke(const TfLiteNode* node) const {
  auto op_info = node_ptr_opinfo_map_.at(node);
  if (op_info.is_custom_op) {
    return logging_op_resolver_->GetWrappedKernelInvoke(op_info.name.c_str(),
                                                        op_info.version);
  }
  return logging_op_resolver_->GetWrappedKernelInvoke(op_info.builtin_op_code,
                                                      op_info.version);
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
  Calibrator* GetCalibrator(const TfLiteNode* node) const {
    if (node_to_calibrator_.find(node) == node_to_calibrator_.cend()) {
      return nullptr;
    }
    return node_to_calibrator_.at(node);
  }

  // Removes the association between calibrator and context.
  // Note: This deletes the calibrator as well.
  void RemoveCalibrator(const TfLiteContext* context) {
    Calibrator* calibrator = calibrator_registry_.at(context).get();
    auto nodes = calibrator->GetNodesUnderCalibration();
    for (auto node : nodes) {
      node_to_calibrator_.erase(node);
    }
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
    auto calibrator = std::make_unique<Calibrator>(
        node_to_opinfo, std::move(logging_op_resolver), reporter);
    calibrator_registry_[context] = std::move(calibrator);
    *calibrator_ptr = calibrator_registry_.at(context).get();
    for (const auto& entry : node_to_opinfo) {
      node_to_calibrator_[entry.first] = *calibrator_ptr;
    }
    return kTfLiteOk;
  }

 private:
  absl::flat_hash_map<const TfLiteContext*, std::unique_ptr<Calibrator>>
      calibrator_registry_;
  absl::flat_hash_map<const TfLiteNode*, Calibrator*> node_to_calibrator_;
};

GlobalCalibratorRegistry* GetCalibratorRegistry() {
  static GlobalCalibratorRegistry* registry = new GlobalCalibratorRegistry();
  return registry;
}

// Get the logging kernel if there are any.
// TODO(jianlijianli): extend this to support multiple recipe for the same
// model.
logging_kernel_func_ptr GetLoggingEvalFunc(TfLiteContext* context,
                                           TfLiteNode* node,
                                           int builtin_op_code) {
  switch (builtin_op_code) {
    case BuiltinOperator_LSTM: {
      if (node->intermediates->size == 12) {
        return tflite::optimize::calibration::custom::lstm_logging_kernel;
      }
      return tflite::optimize::calibration::builtin::lstm_logging_kernel;
    }
    case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM:
      return tflite::optimize::calibration::builtin::
          unidirectional_sequence_lstm_logging_kernel;
    default:
      return nullptr;
  }
}

// A wrapper implementation for |TfLiteRegistration.invoke| that logs inputs,
// invokes the wrapped implementation and then logs the outputs.
TfLiteStatus LoggingEval(TfLiteContext* context, TfLiteNode* node) {
  Calibrator* calibrator = GetCalibratorRegistry()->GetCalibrator(node);

  if (!calibrator) {
    TF_LITE_KERNEL_LOG(context, "No calibrator found for context.");
    return kTfLiteError;
  }

  auto kernel_invoke = calibrator->GetKernelInvoke(node);
  auto logger = calibrator->GetLogger();
  auto op_info = calibrator->GetOpInfo(node);
  auto error_reporter = calibrator->GetErrorReporter();

  for (int i : op_info.loggable_inputs) {
    auto tensor = context->tensors[i];
    TF_LITE_ENSURE_STATUS(
        logger->LogTensorValue(op_info.subgraph_index, i, tensor.data.f,
                               tensor.bytes / sizeof(float), error_reporter));
  }
  auto builtin_op_code = calibrator->GetOpInfo(node).builtin_op_code;
  auto kernel_invoke_intermediate =
      GetLoggingEvalFunc(context, node, builtin_op_code);
  if (kernel_invoke_intermediate == nullptr) {
    TF_LITE_ENSURE_STATUS(kernel_invoke(context, node));
  } else {
    TF_LITE_ENSURE_STATUS(
        kernel_invoke_intermediate(context, op_info.subgraph_index, node,
                                   calibrator->GetLogger(), error_reporter));
  }

  // TODO(shashishekhar): An intermediate tensor in graph will get logged twice
  // once as an input and second time as output. This doesn't change the min max
  // values but is inefficient.
  // Using moving average will also break this.

  // Log input again to make sure the state tensors are captured after lstm
  // cell.
  for (int i : op_info.loggable_inputs) {
    auto tensor = context->tensors[i];
    TF_LITE_ENSURE_STATUS(
        logger->LogTensorValue(op_info.subgraph_index, i, tensor.data.f,
                               tensor.bytes / sizeof(float), error_reporter));
  }

  for (int i : op_info.loggable_outputs) {
    auto tensor = context->tensors[i];
    TF_LITE_ENSURE_STATUS(
        logger->LogTensorValue(op_info.subgraph_index, i, tensor.data.f,
                               tensor.bytes / sizeof(float), error_reporter));
  }

  return kTfLiteOk;
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
    if (tensor_index == kTfLiteOptionalTensor) {
      continue;
    }
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
    const absl::flat_hash_map<std::tuple<int, int>, OperatorInfo>&
        node_to_opinfo,
    tflite::Interpreter* const interpreter,
    std::unordered_map<const TfLiteNode*, OperatorInfo>* node_ptr_opinfo_map,
    TfLiteContext** context) {
  *context = interpreter->primary_subgraph().context();

  // Since we only consider the primary subgraph while populating
  // node_to_opinfo, do the same here.
  // Because Flex delegate can merge multiple op nodes into one Delegate node if
  // they are located in a row, the size of the execution plan can be lesser
  // than the size of the graph's op nodes.
  TF_LITE_ENSURE(*context,
                 interpreter->execution_plan().size() <= node_to_opinfo.size());
  for (const auto& entry : node_to_opinfo) {
    auto op_info = entry.second;
    int subgraph_index, op_index;
    std::tie(subgraph_index, op_index) = entry.first;
    const auto* node_and_reg =
        interpreter->node_and_registration(subgraph_index, op_index);
    op_info.registration = &node_and_reg->second;
    node_ptr_opinfo_map->insert({&node_and_reg->first, op_info});
  }
  return kTfLiteOk;
}

string GetOpName(const tflite::OperatorCode& opcode) {
  if (opcode.custom_code() != nullptr) {
    return opcode.custom_code()->str();
  }
  return tflite::EnumNamesBuiltinOperator()[GetBuiltinCode(&opcode)];
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

bool HasInputs(BuiltinOperator code) {
  switch (code) {
    case BuiltinOperator_CALL_ONCE:
    case BuiltinOperator_VAR_HANDLE:
    // Custom ops, including Flex ops, might not have inputs.
    case BuiltinOperator_CUSTOM:
      return false;
    default:
      return true;
  }
}

bool HasOutputs(BuiltinOperator code) {
  switch (code) {
    case BuiltinOperator_ASSIGN_VARIABLE:
    case BuiltinOperator_CALL_ONCE:
    // Custom ops, including Flex ops, might not have outputs.
    case BuiltinOperator_CUSTOM:
      return false;
    default:
      return true;
  }
}

}  // namespace

TfLiteStatus BuildLoggingInterpreter(
    const FlatBufferModel& model, const OpResolver& op_resolver,
    std::unique_ptr<Interpreter>* interpreter,
    std::unique_ptr<CalibrationReader>* calibration_reader) {
  return BuildLoggingInterpreter(model.GetModel(), model.error_reporter(),
                                 op_resolver, interpreter, calibration_reader,
                                 model.allocation());
}

TfLiteStatus BuildLoggingInterpreter(
    const tflite::Model* tflite_model, ErrorReporter* error_reporter,
    const OpResolver& op_resolver, std::unique_ptr<Interpreter>* interpreter,
    std::unique_ptr<CalibrationReader>* calibration_reader,
    const Allocation* allocation) {
  if (error_reporter == nullptr) {
    // Make sure error_reporter is valid.
    error_reporter = DefaultErrorReporter();
  }
  auto subgraphs = tflite_model->subgraphs();
  auto tensor_buffers = tflite_model->buffers();

  // Populate the node index to operator info map.
  // We want to collect this information so we can use it during runtime to
  // log details of which inputs and outputs.
  // At runtime TFLite kernel invoke functions can only look into their
  // own node in the graph (TFLiteNode*) and some limited context information.
  absl::flat_hash_map<std::tuple<int, int>, OperatorInfo> node_to_opinfo;
  BuiltinOpsSet builtin_op_and_versions;
  CustomOpsSet custom_op_and_versions;

  for (size_t subgraph_index = 0; subgraph_index < subgraphs->size();
       subgraph_index++) {
    auto subgraph = subgraphs->Get(subgraph_index);
    auto operator_codes = tflite_model->operator_codes();
    auto operators = subgraph->operators();
    auto tensors = subgraph->tensors();
    if (!operators) {
      continue;
    }

    for (size_t i = 0; i < operators->size(); i++) {
      OperatorInfo op_info;
      op_info.subgraph_index = subgraph_index;
      op_info.node_index = i;
      auto op = operators->Get(i);
      auto operator_code = operator_codes->Get(op->opcode_index());
      op_info.builtin_op_code = GetBuiltinCode(operator_code);
      op_info.name = GetOpName(*operator_code);
      op_info.is_custom_op = operator_code->custom_code() != nullptr;
      op_info.version = operator_code->version();

      auto op_inputs = op->inputs();
      auto op_outputs = op->outputs();
      if (op_inputs) {
        op_info.inputs = std::vector<int>(op_inputs->begin(), op_inputs->end());
      } else if (HasInputs(op_info.builtin_op_code)) {
        TFLITE_LOG(TFLITE_LOG_WARNING, "Op %s missing inputs",
                   op_info.name.c_str());
      }
      if (op_outputs) {
        op_info.outputs =
            std::vector<int>(op_outputs->begin(), op_outputs->end());
      } else if (HasOutputs(op_info.builtin_op_code)) {
        TFLITE_LOG(TFLITE_LOG_WARNING, "Op %s missing outputs",
                   op_info.name.c_str());
      }
      op_info.loggable_inputs =
          GetLoggableTensorIndices(op_info.inputs, tensors, tensor_buffers);
      op_info.loggable_outputs =
          GetLoggableTensorIndices(op_info.outputs, tensors, tensor_buffers);
      if (op_info.is_custom_op) {
        op_info.registration =
            op_resolver.FindOp(op_info.name.c_str(), operator_code->version());
        custom_op_and_versions.insert(
            {op_info.name.c_str(), operator_code->version()});
      } else {
        op_info.registration = op_resolver.FindOp(GetBuiltinCode(operator_code),
                                                  operator_code->version());
        builtin_op_and_versions.insert(
            {op_info.builtin_op_code, operator_code->version()});
      }
      std::tuple<int, int> key{subgraph_index, i};
      node_to_opinfo[key] = op_info;
    }
  }

  // Prepare the logging op resolver to use |LoggingEval| for kernel
  // invocations.
  auto logging_op_resolver = std::make_unique<LoggingOpResolver>(
      builtin_op_and_versions, custom_op_and_versions, op_resolver, LoggingEval,
      error_reporter);
  tflite::InterpreterBuilder(tflite_model, *logging_op_resolver, error_reporter,
                             /*options_experimental=*/nullptr,
                             allocation)(interpreter);

  if (!(*interpreter)) {
    error_reporter->Report("Failed to construct interpreter");
    return kTfLiteError;
  }

  // Compute the mapping between runtime and static graph structure, i.e.
  // (TfLiteContext, TfLiteNode) -> OperatorInfo
  std::unordered_map<const TfLiteNode*, OperatorInfo> node_ptr_opinfo_map;
  TfLiteContext* context = nullptr;
  TF_LITE_ENSURE_STATUS(GetNodeOpInfoMapAndContext(
      node_to_opinfo, interpreter->get(), &node_ptr_opinfo_map, &context));

  Calibrator* calibrator = nullptr;
  // Register a calibrator object for the context. This can be accessed
  // during invocations by the logging kernels.
  TF_LITE_ENSURE_STATUS(GetCalibratorRegistry()->CreateCalibrator(
      context, node_ptr_opinfo_map, std::move(logging_op_resolver), &calibrator,
      error_reporter));
  *calibration_reader = std::unique_ptr<CalibrationReader>(
      new Reader(context, calibrator->GetLogger()));

  return kTfLiteOk;
}

}  // namespace calibration
}  // namespace optimize
}  // namespace tflite
