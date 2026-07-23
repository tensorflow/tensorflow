/* Copyright 2026 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TENSORFLOW_LITE_CORE_C_ALLOCATOR_INTERNAL_H_
#define TENSORFLOW_LITE_CORE_C_ALLOCATOR_INTERNAL_H_

#include "tensorflow/lite/core/c/common.h"

namespace tflite {
namespace internal {

TfLiteAllocator* GetCurrentTfLiteAllocator();
TfLiteAllocator* SetCurrentTfLiteAllocator(TfLiteAllocator* allocator);

class ScopedTfLiteAllocator {
 public:
  explicit ScopedTfLiteAllocator(TfLiteAllocator* allocator)
      : previous_(SetCurrentTfLiteAllocator(allocator)) {}

  ~ScopedTfLiteAllocator() { SetCurrentTfLiteAllocator(previous_); }

  ScopedTfLiteAllocator(const ScopedTfLiteAllocator&) = delete;
  ScopedTfLiteAllocator& operator=(const ScopedTfLiteAllocator&) = delete;

 private:
  TfLiteAllocator* previous_;
};

}  // namespace internal
}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_C_ALLOCATOR_INTERNAL_H_
