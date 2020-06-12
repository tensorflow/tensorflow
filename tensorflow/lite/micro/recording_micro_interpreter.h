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

#ifndef TENSORFLOW_LITE_MICRO_RECORDING_MICRO_INTERPRETER_H_
#define TENSORFLOW_LITE_MICRO_RECORDING_MICRO_INTERPRETER_H_

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"

namespace tflite {

// Utility subclass that enables internal recordings of the MicroInterpreter.
// This class should be used to audit and analyze memory arena usage for a given
// model and interpreter.
//
// After construction and the first Invoke() or AllocateTensors() call - the
// memory usage is recorded and available through the GetMicroAllocator()
// function. See RecordingMicroAlloctor for more details on what is currently
// recorded from arena allocations.
//
// It is recommended for users to increase the tensor arena size by at least 1kb
// to ensure enough additional memory is available for internal recordings.
class RecordingMicroInterpreter : public MicroInterpreter {
 public:
  RecordingMicroInterpreter(const Model* model,
                            const MicroOpResolver* op_resolver,
                            uint8_t* tensor_arena, size_t tensor_arena_size,
                            ErrorReporter* error_reporter)
      : MicroInterpreter(model, op_resolver,
                         RecordingMicroAllocator::Create(
                             tensor_arena, tensor_arena_size, error_reporter),
                         error_reporter),
        recording_micro_allocator_(
            static_cast<const RecordingMicroAllocator&>(allocator())) {}

  const RecordingMicroAllocator& GetMicroAllocator() const {
    return recording_micro_allocator_;
  }

 private:
  const RecordingMicroAllocator& recording_micro_allocator_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_RECORDING_MICRO_INTERPRETER_H_
