// Copyright (c) 2020, XMOS Ltd, All rights reserved

#include "tensorflow/lite/micro/kernels/xcore/xcore_interpreter.h"

namespace tflite {
namespace micro {
namespace xcore {
XCoreInterpreter::XCoreInterpreter(const tflite::Model* model,
                                   const tflite::MicroOpResolver& resolver,
                                   uint8_t* arena, size_t arena_size,
                                   tflite::ErrorReporter* reporter,
                                   bool use_current_thread,
                                   tflite::Profiler* profiler)
    : tflite::MicroInterpreter(model, resolver, arena, arena_size, reporter,
                               profiler),
      dispatcher_(reporter, use_current_thread) {
  SetDispatcher(&dispatcher_);
}

XCoreInterpreter::XCoreInterpreter(const tflite::Model* model,
                                   const tflite::MicroOpResolver& resolver,
                                   tflite::MicroAllocator* allocator,
                                   tflite::ErrorReporter* reporter,
                                   bool use_current_thread,
                                   tflite::Profiler* profiler)
    : tflite::MicroInterpreter(model, resolver, allocator, reporter, profiler),
      dispatcher_(reporter, use_current_thread) {
  SetDispatcher(&dispatcher_);
}

}  // namespace xcore
}  // namespace micro
}  // namespace tflite
