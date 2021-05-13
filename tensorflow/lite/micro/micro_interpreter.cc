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
#include "tensorflow/lite/micro/micro_interpreter.h"

#include <cstdarg>
#include <cstddef>
#include <cstdint>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/tensor_utils.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"

namespace tflite {

MicroInterpreter::MicroInterpreter(const Model* model,
                                   const MicroOpResolver& op_resolver,
                                   uint8_t* tensor_arena,
                                   size_t tensor_arena_size,
                                   ErrorReporter* error_reporter,
                                   MicroProfiler* profiler)
    : model_(model),
      op_resolver_(op_resolver),
      error_reporter_(error_reporter),
      allocator_(*MicroAllocator::Create(tensor_arena, tensor_arena_size,
                                         error_reporter)),

      graph_(&context_, model, &allocator_),
      tensors_allocated_(false),
      initialization_status_(kTfLiteError),
      input_tensors_(nullptr),
      output_tensors_(nullptr) {
  Init(profiler);
}

MicroInterpreter::MicroInterpreter(const Model* model,
                                   const MicroOpResolver& op_resolver,
                                   MicroAllocator* allocator,
                                   ErrorReporter* error_reporter,
                                   MicroProfiler* profiler)
    : model_(model),
      op_resolver_(op_resolver),
      error_reporter_(error_reporter),
      allocator_(*allocator),
      graph_(&context_, model, allocator),
      tensors_allocated_(false),
      initialization_status_(kTfLiteError),
      input_tensors_(nullptr),
      output_tensors_(nullptr) {
  Init(profiler);
}

MicroInterpreter::~MicroInterpreter() {
  if (graph_.GetAllocations() != nullptr) {
    graph_.FreeSubgraphs();
  }
}

void MicroInterpreter::Init(MicroProfiler* profiler) {
  context_.impl_ = static_cast<void*>(this);
  context_.ReportError = ReportOpError;
  context_.GetTensor = GetTensor;
  context_.ReportError = ReportOpError;
  context_.GetTensor = GetTensor;
  context_.GetEvalTensor = GetEvalTensor;
  context_.profiler = profiler;

  initialization_status_ = kTfLiteOk;
}

TfLiteStatus MicroInterpreter::PrepareNodeAndRegistrationDataFromFlatbuffer() {
  for (int subgraph_idx = 0; subgraph_idx < graph_.NumSubgraphs();
       subgraph_idx++) {
    const SubGraph* subgraph = model_->subgraphs()->Get(subgraph_idx);
    TFLITE_DCHECK(subgraph != nullptr);

    auto* opcodes = model_->operator_codes();
    BuiltinDataAllocator* builtin_data_allocator =
        allocator_.GetBuiltinDataAllocator();
    for (size_t i = 0; i < subgraph->operators()->size(); ++i) {
      const auto* op = subgraph->operators()->Get(i);
      const size_t index = op->opcode_index();
      if (index >= opcodes->size()) {
        MicroPrintf("Missing registration for opcode_index %d\n", index);
        return kTfLiteError;
      }
      const auto* opcode = opcodes->Get(index);
      TfLiteStatus status =
          GetRegistrationFromOpCode(opcode, op_resolver_, error_reporter_,
                                    &(graph_.GetAllocations()[subgraph_idx]
                                          .node_and_registrations[i]
                                          .registration));
      if (status != kTfLiteOk) {
        MicroPrintf("Failed to get registration from op code %s\n ",
                    EnumNameBuiltinOperator(GetBuiltinCode(opcode)));
        return status;
      }
      const auto* registration = graph_.GetAllocations()[subgraph_idx]
                                     .node_and_registrations[i]
                                     .registration;
      if (registration == nullptr) {
        MicroPrintf("Skipping op for opcode_index %d\n", index);
        return kTfLiteError;
      }
      BuiltinOperator op_type =
          static_cast<BuiltinOperator>(registration->builtin_code);

      const char* custom_data = nullptr;
      size_t custom_data_size = 0;
      unsigned char* builtin_data = nullptr;

      if (op_type == BuiltinOperator_CUSTOM) {
        // Custom Ops may or may not have a non-null custom_options field.
        if (op->custom_options() != nullptr) {
          custom_data =
              reinterpret_cast<const char*>(op->custom_options()->data());
          custom_data_size = op->custom_options()->size();
        }
      } else {
        if (op->custom_options() != nullptr) {
          MicroPrintf(
              "Unsupported behavior: found builtin operator %s with custom "
              "options.\n",
              EnumNameBuiltinOperator(op_type));
          return kTfLiteError;
        }

        MicroOpResolver::BuiltinParseFunction parser =
            op_resolver_.GetOpDataParser(op_type);
        if (parser == nullptr) {
          MicroPrintf("Did not find a parser for %s",
                      EnumNameBuiltinOperator(op_type));

          return kTfLiteError;
        }
        TF_LITE_ENSURE_STATUS(parser(op, error_reporter_,
                                     builtin_data_allocator,
                                     (void**)(&builtin_data)));
      }

      TfLiteIntArray* inputs_array;
      TF_LITE_ENSURE_STATUS(allocator_.FlatBufferVectorToTfLiteTypeArray(
          op->inputs(), &inputs_array));

      TfLiteIntArray* outputs_array;
      TF_LITE_ENSURE_STATUS(allocator_.FlatBufferVectorToTfLiteTypeArray(
          op->outputs(), &outputs_array));

      TfLiteNode* node = &(
          graph_.GetAllocations()[subgraph_idx].node_and_registrations[i].node);
      *node = {};
      node->inputs = inputs_array;
      node->outputs = outputs_array;
      node->builtin_data = reinterpret_cast<void*>(builtin_data);
      node->custom_initial_data = custom_data;
      node->custom_initial_data_size = custom_data_size;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus MicroInterpreter::AllocateTensors() {
  SubgraphAllocations* allocations = allocator_.StartModelAllocation(model_);

  if (allocations == nullptr) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Failed starting model allocation.\n");
    initialization_status_ = kTfLiteError;
    return kTfLiteError;
  }

  graph_.SetSubgraphAllocations(allocations);

  TF_LITE_ENSURE_STATUS(PrepareNodeAndRegistrationDataFromFlatbuffer());

  // Only allow AllocatePersistentBuffer in Init stage.
  context_.AllocatePersistentBuffer = AllocatePersistentBuffer;
  context_.RequestScratchBufferInArena = nullptr;
  context_.GetScratchBuffer = nullptr;
  context_.GetExecutionPlan = GetGraph;
  graph_.InitSubgraphs();

  // Both AllocatePersistentBuffer and RequestScratchBufferInArena is
  // available in Prepare stage.
  context_.RequestScratchBufferInArena = RequestScratchBufferInArena;
  graph_.PrepareSubgraphs();

  // Prepare is done, we're ready for Invoke. Memory allocation is no longer
  // allowed. Kernels can only fetch scratch buffers via GetScratchBuffer.
  context_.AllocatePersistentBuffer = nullptr;
  context_.RequestScratchBufferInArena = nullptr;
  context_.GetScratchBuffer = GetScratchBuffer;

  TF_LITE_ENSURE_OK(&context_, allocator_.FinishModelAllocation(
                                   model_, graph_.GetAllocations(),
                                   &scratch_buffer_handles_));

  // TODO(b/162311891): Drop these allocations when the interpreter supports
  // handling buffers from TfLiteEvalTensor.
  input_tensors_ =
      reinterpret_cast<TfLiteTensor**>(allocator_.AllocatePersistentBuffer(
          sizeof(TfLiteTensor*) * inputs_size()));
  if (input_tensors_ == nullptr) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Failed to allocate memory for context->input_tensors_, "
        "%d bytes required",
        sizeof(TfLiteTensor*) * inputs_size());
    return kTfLiteError;
  }

  for (size_t i = 0; i < inputs_size(); ++i) {
    input_tensors_[i] = allocator_.AllocatePersistentTfLiteTensor(
        model_, graph_.GetAllocations(), inputs().Get(i), 0);
    if (input_tensors_[i] == nullptr) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Failed to initialize input tensor %d", i);
      return kTfLiteError;
    }
  }

  // TODO(b/162311891): Drop these allocations when the interpreter supports
  // handling buffers from TfLiteEvalTensor.
  output_tensors_ =
      reinterpret_cast<TfLiteTensor**>(allocator_.AllocatePersistentBuffer(
          sizeof(TfLiteTensor*) * outputs_size()));
  if (output_tensors_ == nullptr) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Failed to allocate memory for context->output_tensors_, "
        "%d bytes required",
        sizeof(TfLiteTensor*) * outputs_size());
    return kTfLiteError;
  }

  for (size_t i = 0; i < outputs_size(); ++i) {
    output_tensors_[i] = allocator_.AllocatePersistentTfLiteTensor(
        model_, graph_.GetAllocations(), outputs().Get(i), 0);
    if (output_tensors_[i] == nullptr) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Failed to initialize output tensor %d", i);
      return kTfLiteError;
    }
  }

  TF_LITE_ENSURE_STATUS(ResetVariableTensors());

  tensors_allocated_ = true;
  return kTfLiteOk;
}

TfLiteStatus MicroInterpreter::Invoke() {
  if (initialization_status_ != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Invoke() called after initialization failed\n");
    return kTfLiteError;
  }

  // Ensure tensors are allocated before the interpreter is invoked to avoid
  // difficult to debug segfaults.
  if (!tensors_allocated_) {
    TF_LITE_ENSURE_OK(&context_, AllocateTensors());
  }
  return graph_.InvokeSubgraph(0);
}

TfLiteTensor* MicroInterpreter::input(size_t index) {
  const size_t length = inputs_size();
  if (index >= length) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Input index %d out of range (length is %d)", index,
                         length);
    return nullptr;
  }
  return input_tensors_[index];
}

TfLiteTensor* MicroInterpreter::output(size_t index) {
  const size_t length = outputs_size();
  if (index >= length) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Output index %d out of range (length is %d)", index,
                         length);
    return nullptr;
  }
  return output_tensors_[index];
}

TfLiteStatus MicroInterpreter::ResetVariableTensors() {
  return graph_.ResetVariableTensors();
}

void* MicroInterpreter::AllocatePersistentBuffer(TfLiteContext* ctx,
                                                 size_t bytes) {
  return reinterpret_cast<MicroInterpreter*>(ctx->impl_)
      ->allocator_.AllocatePersistentBuffer(bytes);
}

TfLiteStatus MicroInterpreter::RequestScratchBufferInArena(TfLiteContext* ctx,
                                                           size_t bytes,
                                                           int* buffer_idx) {
  MicroInterpreter* interpreter =
      reinterpret_cast<MicroInterpreter*>(ctx->impl_);
  return interpreter->allocator_.RequestScratchBufferInArena(
      bytes, interpreter->graph_.GetCurrentSubgraphIndex(), buffer_idx);
}

void* MicroInterpreter::GetScratchBuffer(TfLiteContext* ctx, int buffer_idx) {
  MicroInterpreter* interpreter =
      reinterpret_cast<MicroInterpreter*>(ctx->impl_);
  ScratchBufferHandle* handle =
      interpreter->scratch_buffer_handles_ + buffer_idx;
  return handle->data;
}

void MicroInterpreter::ReportOpError(struct TfLiteContext* context,
                                     const char* format, ...) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
  MicroInterpreter* interpreter =
      static_cast<MicroInterpreter*>(context->impl_);
  va_list args;
  va_start(args, format);
  TF_LITE_REPORT_ERROR(interpreter->error_reporter_, format, args);
  va_end(args);
#endif
}

TfLiteTensor* MicroInterpreter::GetTensor(const struct TfLiteContext* context,
                                          int tensor_idx) {
  MicroInterpreter* interpreter =
      static_cast<MicroInterpreter*>(context->impl_);
  return interpreter->allocator_.AllocateTempTfLiteTensor(
      interpreter->model_, interpreter->graph_.GetAllocations(), tensor_idx,
      interpreter->get_subgraph_index());
}

TfLiteEvalTensor* MicroInterpreter::GetEvalTensor(
    const struct TfLiteContext* context, int tensor_idx) {
  MicroInterpreter* interpreter =
      reinterpret_cast<MicroInterpreter*>(context->impl_);
  return &interpreter->graph_
              .GetAllocations()[interpreter->get_subgraph_index()]
              .tensors[tensor_idx];
}

TfLiteStatus MicroInterpreter::GetGraph(struct TfLiteContext* context,
                                        TfLiteIntArray** args) {
  MicroInterpreter* interpreter =
      reinterpret_cast<MicroInterpreter*>(context->impl_);
  *args = reinterpret_cast<TfLiteIntArray*>(&interpreter->graph_);
  return kTfLiteOk;
}

}  // namespace tflite
