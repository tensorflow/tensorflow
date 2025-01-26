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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TOOLS_DUMP_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TOOLS_DUMP_H_

#include <iostream>
#include <istream>
#include <ostream>

#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/compiler/plugin/compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"

namespace litert::internal {

//
// LiteRt IR
//

// Dump details about the given LiteRtOpT to the given stream.
void Dump(const LiteRtOpT& op, std::ostream& out = std::cerr);

// Dump details about the given LiteRtSubgraphT to the given stream.
void Dump(const LiteRtSubgraphT& subgraph, std::ostream& out = std::cerr);

// Dump details about the given LiteRtTensorT to the given stream.
void Dump(const LiteRtTensorT& tensor, std::ostream& out = std::cerr);

// Dump details about the given LiteRtOpCode to the given stream.
void Dump(LiteRtOpCode code, std::ostream& out = std::cerr);

// Dump details about the given LiteRtElementType to the given stream.
void Dump(LiteRtElementType type, std::ostream& out = std::cerr);

// Dump details about the given LiteRtRankedTensorType to the given stream.
void Dump(const LiteRtRankedTensorType& type, std::ostream& out = std::cerr);

// Dump details about the given LiteRtModel to the given stream.
void Dump(const LiteRtModelT& model, std::ostream& out = std::cerr);

// Dump details about the given quantization params.
void Dump(Quantization quantization, std::ostream& out = std::cerr);

// Dump details about options
void DumpOptions(const LiteRtOpT& op, std::ostream& out = std::cerr);

//
// Library Utilities
//

// Dumps details about the loaded LiteRtCompilerPlugin library.
void Dump(const CompilerPlugin& plugin, std::ostream& out = std::cerr);

// Dumps details about the dynamic library (see "dlinfo").
void DumpDLL(void* lib_handle, std::ostream& out = std::cerr);

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TOOLS_DUMP_H_
