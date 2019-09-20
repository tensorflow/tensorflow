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

#ifndef GRPC_INTERNAL_COMPILER_SCHEMA_INTERFACE_H
#define GRPC_INTERNAL_COMPILER_SCHEMA_INTERFACE_H

#include "src/compiler/config.h"

#include <memory>
#include <vector>

#ifndef GRPC_CUSTOM_STRING
#  include <string>
#  define GRPC_CUSTOM_STRING std::string
#endif

namespace grpc {

typedef GRPC_CUSTOM_STRING string;

}  // namespace grpc

namespace grpc_generator {

// A common interface for objects having comments in the source.
// Return formatted comments to be inserted in generated code.
struct CommentHolder {
  virtual ~CommentHolder() {}
  virtual grpc::string GetLeadingComments(const grpc::string prefix) const = 0;
  virtual grpc::string GetTrailingComments(const grpc::string prefix) const = 0;
  virtual std::vector<grpc::string> GetAllComments() const = 0;
};

// An abstract interface representing a method.
struct Method : public CommentHolder {
  virtual ~Method() {}

  virtual grpc::string name() const = 0;

  virtual grpc::string input_type_name() const = 0;
  virtual grpc::string output_type_name() const = 0;

  virtual bool get_module_and_message_path_input(
      grpc::string *str, grpc::string generator_file_name,
      bool generate_in_pb2_grpc, grpc::string import_prefix) const = 0;
  virtual bool get_module_and_message_path_output(
      grpc::string *str, grpc::string generator_file_name,
      bool generate_in_pb2_grpc, grpc::string import_prefix) const = 0;

  virtual grpc::string get_input_type_name() const = 0;
  virtual grpc::string get_output_type_name() const = 0;
  virtual bool NoStreaming() const = 0;
  virtual bool ClientStreaming() const = 0;
  virtual bool ServerStreaming() const = 0;
  virtual bool BidiStreaming() const = 0;
};

// An abstract interface representing a service.
struct Service : public CommentHolder {
  virtual ~Service() {}

  virtual grpc::string name() const = 0;

  virtual int method_count() const = 0;
  virtual std::unique_ptr<const Method> method(int i) const = 0;
};

struct Printer {
  virtual ~Printer() {}

  virtual void Print(const std::map<grpc::string, grpc::string> &vars,
                     const char *template_string) = 0;
  virtual void Print(const char *string) = 0;
  virtual void Indent() = 0;
  virtual void Outdent() = 0;
};

// An interface that allows the source generated to be output using various
// libraries/idls/serializers.
struct File : public CommentHolder {
  virtual ~File() {}

  virtual grpc::string filename() const = 0;
  virtual grpc::string filename_without_ext() const = 0;
  virtual grpc::string package() const = 0;
  virtual std::vector<grpc::string> package_parts() const = 0;
  virtual grpc::string additional_headers() const = 0;

  virtual int service_count() const = 0;
  virtual std::unique_ptr<const Service> service(int i) const = 0;

  virtual std::unique_ptr<Printer> CreatePrinter(grpc::string *str) const = 0;
};
}  // namespace grpc_generator

#endif  // GRPC_INTERNAL_COMPILER_SCHEMA_INTERFACE_H
