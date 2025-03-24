// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_DISPATCH_OP_SCHEMA_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_DISPATCH_OP_SCHEMA_H_

#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"

// Utilities for working with the dispatch op custom options buffer. These
// functions leverage the flexbuffer api under the hood which allows for inplace
// updates.

namespace litert::internal {

// Schema representing the custom options data for dispatch ops. Primarly used
// to for tracking location of bytecode.
struct DispatchOpOptions {
  // The size of the bytecode for the dispatch op.
  size_t bytecode_size;

  // The offset of the bytecode for the dispatch op relative to the start of the
  // model file.
  size_t bytecode_offset;

  // Name of specific dispatch op or entry point to be called in a shared
  // bytecode module.
  std::string name;
};

// Get a serialized representation of the dispatch op options. These should
// be stored directly in the custom options of the dispatch op.
OwningBufferRef<uint8_t> MakeDispatchOpOptions(DispatchOpOptions options);

// Update the dispatch op options in the given buffer with the given options.
// The buffer should be the custom options buffer of the dispatch op. Fails if
// the passed values would resize the buffer.
bool UpdateDispatchOpOptionsInPlace(DispatchOpOptions options,
                                    MutableBufferRef<uint8_t> buffer);

// Get the dispatch op options from the given buffer. The buffer should be the
// custom options buffer of the dispatch op.
DispatchOpOptions GetDispatchOpOptions(BufferRef<uint8_t> buffer);

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_DISPATCH_OP_SCHEMA_H_
