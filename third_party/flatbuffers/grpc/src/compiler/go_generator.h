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

#ifndef GRPC_INTERNAL_COMPILER_GO_GENERATOR_H
#define GRPC_INTERNAL_COMPILER_GO_GENERATOR_H

//go generator is used to generate GRPC code for serialization system, such as flatbuffers
#include <memory>
#include <vector>

#include "src/compiler/schema_interface.h"

namespace grpc_go_generator {

struct Parameters {
  //Defines the custom parameter types for methods
  //eg: flatbuffers uses flatbuffers.Builder as input for the client and output for the server
  grpc::string custom_method_io_type;

  //Package name for the service
  grpc::string package_name;

  //Prefix for RPC Calls
  grpc::string service_prefix;
};

// Return the source of the generated service file.
grpc::string GenerateServiceSource(grpc_generator::File *file,
                                   const grpc_generator::Service *service,
                                   grpc_go_generator::Parameters *parameters);

}

#endif  // GRPC_INTERNAL_COMPILER_GO_GENERATOR_H
