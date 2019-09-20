/*
 *
 * Copyright 2015, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef GRPC_INTERNAL_COMPILER_CPP_GENERATOR_H
#define GRPC_INTERNAL_COMPILER_CPP_GENERATOR_H

// cpp_generator.h/.cc do not directly depend on GRPC/ProtoBuf, such that they
// can be used to generate code for other serialization systems, such as
// FlatBuffers.

#include <memory>
#include <vector>

#include "src/compiler/config.h"
#include "src/compiler/schema_interface.h"

#ifndef GRPC_CUSTOM_STRING
#include <string>
#define GRPC_CUSTOM_STRING std::string
#endif

namespace grpc {

typedef GRPC_CUSTOM_STRING string;

}  // namespace grpc

namespace grpc_cpp_generator {

// Contains all the parameters that are parsed from the command line.
struct Parameters {
  // Puts the service into a namespace
  grpc::string services_namespace;
  // Use system includes (<>) or local includes ("")
  bool use_system_headers;
  // Prefix to any grpc include
  grpc::string grpc_search_path;
  // Generate GMOCK code to facilitate unit testing.
  bool generate_mock_code;
};

// Return the prologue of the generated header file.
grpc::string GetHeaderPrologue(grpc_generator::File *file,
                               const Parameters &params);

// Return the includes needed for generated header file.
grpc::string GetHeaderIncludes(grpc_generator::File *file,
                               const Parameters &params);

// Return the includes needed for generated source file.
grpc::string GetSourceIncludes(grpc_generator::File *file,
                               const Parameters &params);

// Return the epilogue of the generated header file.
grpc::string GetHeaderEpilogue(grpc_generator::File *file,
                               const Parameters &params);

// Return the prologue of the generated source file.
grpc::string GetSourcePrologue(grpc_generator::File *file,
                               const Parameters &params);

// Return the services for generated header file.
grpc::string GetHeaderServices(grpc_generator::File *file,
                               const Parameters &params);

// Return the services for generated source file.
grpc::string GetSourceServices(grpc_generator::File *file,
                               const Parameters &params);

// Return the epilogue of the generated source file.
grpc::string GetSourceEpilogue(grpc_generator::File *file,
                               const Parameters &params);

// Return the prologue of the generated mock file.
grpc::string GetMockPrologue(grpc_generator::File *file,
                             const Parameters &params);

// Return the includes needed for generated mock file.
grpc::string GetMockIncludes(grpc_generator::File *file,
                             const Parameters &params);

// Return the services for generated mock file.
grpc::string GetMockServices(grpc_generator::File *file,
                             const Parameters &params);

// Return the epilogue of generated mock file.
grpc::string GetMockEpilogue(grpc_generator::File *file,
                             const Parameters &params);

// Return the prologue of the generated mock file.
grpc::string GetMockPrologue(grpc_generator::File *file,
                             const Parameters &params);

// Return the includes needed for generated mock file.
grpc::string GetMockIncludes(grpc_generator::File *file,
                             const Parameters &params);

// Return the services for generated mock file.
grpc::string GetMockServices(grpc_generator::File *file,
                             const Parameters &params);

// Return the epilogue of generated mock file.
grpc::string GetMockEpilogue(grpc_generator::File *file,
                             const Parameters &params);

}  // namespace grpc_cpp_generator

#endif  // GRPC_INTERNAL_COMPILER_CPP_GENERATOR_H
