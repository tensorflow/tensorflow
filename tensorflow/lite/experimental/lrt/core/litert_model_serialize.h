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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_LITERT_MODEL_SERIALIZE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_LITERT_MODEL_SERIALIZE_H_

#include <stddef.h>

#include "tensorflow/lite/experimental/lrt/c/litert_common.h"
#include "tensorflow/lite/experimental/lrt/c/litert_model.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Shared "custom_code" for all dispatch ops.
static const char kLiteRtDispatchOpCustomCode[] = "NPU_OP";

// Template for build tag to add to the model, encodes context about how the
// model was generated.
static const char kLiteRtBuildTagTpl[] =
    "soc_man:%s,soc_model:%s,serialization_strategy:%s";

// Metadata key to lookup the build tag.
static const char kLiteRtBuildTagKey[] = "LiteRtStamp";

// Serializaton strategy ID for adding raw byte code directly to the metadata.
static const char kLiteRtMetadataSerializationStrategy[] = "METADATA";

// Serialization strategy ID for appending byte code to the end of the file.
static const char kLiteRtAppendSerializationStrategy[] = "APPEND";

// Tag that prefixes the placeholder string below.
static const char kLiteRtAppendedByteCodePrefix[] = "<npu_byte_code>";

// NPU bytecode information for the append strategy. Placeholder
// for post-processing step, [<offset_str>,<size_str>] padded to fixed length.
static const char kLiteRtAppendedByteCodePlaceholder[] =
    "[**********,**********]";

// Metadata key for any NPU bytecode information.
static const char kLiteRtMetadataByteCodeKey[] = "LiteRtNpuByteCode";

//===----------------------------------------------------------------------===//
//
//                                                     << BYTE CODE PACKING >>
//
// Strategies for packaging LiteRtCompilerPlugin compilation output with the
// flatbuffer. These are different short-term approaches used for testing and/or
// development.
//
//                                                              < STRATEGIES >
//
// All serialization strategies add 2 metadata buffers to the model. The first
// is a build stamp, which indicates the make/model as well as the serialization
// strategy targeted during plugin appliction. The second contains information
// about the NPU bytecode, which may be a location to find it, or the raw data
// itself.
//
// "METADATA" strategy
//
// Adds the raw NPU bytecode directly in the flatbuffer in a standard metadata
// buffer.
//
// This is intented for use in testing as it may bloat the flatbuffer size.
// Packing the byte code in this way allows it to be rendered by existing tflite
// tooling.
//
// "APPEND" strategy
//
// Appends compiled byte code to the end of the flatbuffer. This avoids cases
// where embedding byte code directly would break 2gb flatbuffer limit.
// Offset into the file and where the byte code starts and size is stored in
// metadata.
//
// The actual value of the offset is written into serialized flatbuffer
// as a post processing step. This function populates the offset with a fixed
// size placeholder for size_t(s) which may be left padded by some filler
// characters.
//
//===----------------------------------------------------------------------===//

// Adds NPU bytecode and build tag to metadata.
// Registers the "custom_code".
LiteRtStatus LiteRtModelAddByteCodeMetadata(LiteRtModel model,
                                            const char* soc_manufacturer,
                                            const char* soc_model,
                                            const void* byte_code,
                                            size_t byte_code_size);

// Preps the model for future post processing step. A
// string with parts parseable as size_t (offset, size) is set in the metadata.
// A future step will find the prefix of this string and
// replace the size_t portions with the actual offset and size
// post-serializaiton. This post-process step cannot not change the length of
// the string, and therefore the result may be left-padded with filler
// characters. Also populates build tag and registers "custom_code".
LiteRtStatus LiteRtModelPrepareForByteCodeAppend(LiteRtModel model,
                                                 const char* soc_manufacturer,
                                                 const char* soc_model);

#ifdef __cplusplus
}

#include "tensorflow/lite/experimental/lrt/cc/litert_support.h"
#include "tensorflow/lite/experimental/lrt/core/util/buffer_ref.h"

namespace litert::internal {

// Completes the post-processing step for the "APPEND" strategy.
// Updates the byte code offset/size placeholders in "serialized" model without
// changing the size of the model. If the string representation of size/offset
// is less than length of the placeholder string, result will contain
// string will be left-padded with filler characters. Fails if it is greater
// than the placeholder string length or it cannot find the placeholder.
LiteRtStatus FinishByteCodeAppend(MutableBufferRef<uint8_t> serialized_model,
                                  size_t byte_code_size);

// See the "append" byte code packing strategy. Small utility to parse the
// offset from original file and size encoded in the "metadata_buffer".
LiteRtResult<std::pair<size_t, size_t>> ParseByteCodeOffsetFromMetadata(
    BufferRef<uint8_t> metadata_buffer);

}  // namespace litert::internal

#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_LITERT_MODEL_SERIALIZE_H_
