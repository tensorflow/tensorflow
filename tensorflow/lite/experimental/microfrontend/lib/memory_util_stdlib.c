/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This version of the allocation utilities uses standard malloc/free
// implementations for the memory required by the frontend.

#include <stdlib.h>

#include "tensorflow/lite/experimental/microfrontend/lib/memory_util.h"

void* microfrontend_alloc(size_t size) { return malloc(size); }

void microfrontend_free(void* ptr) { free(ptr); }
