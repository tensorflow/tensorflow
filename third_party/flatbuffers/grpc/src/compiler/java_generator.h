/*
 * Copyright 2016 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NET_GRPC_COMPILER_JAVA_GENERATOR_H_
#define NET_GRPC_COMPILER_JAVA_GENERATOR_H_

#include <stdlib.h>  // for abort()
#include <iostream>
#include <map>
#include <string>

#include "src/compiler/schema_interface.h"

class LogMessageVoidify {
 public:
  LogMessageVoidify() {}
  // This has to be an operator with a precedence lower than << but
  // higher than ?:
  void operator&(std::ostream&) {}
};

class LogHelper {
  std::ostream* os_;

 public:
  LogHelper(std::ostream* os) : os_(os) {}
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning( \
    disable : 4722)  // the flow of control terminates in a destructor
  // (needed to compile ~LogHelper where destructor emits abort intentionally -
  // inherited from grpc/java code generator).
#endif
  ~LogHelper() {
    *os_ << std::endl;
    ::abort();
  }
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
  std::ostream& get_os() const { return *os_; }
};

// Abort the program after logging the mesage if the given condition is not
// true. Otherwise, do nothing.
#define GRPC_CODEGEN_CHECK(x)                                            \
  (x) ? (void)0                                                          \
      : LogMessageVoidify() & LogHelper(&std::cerr).get_os()             \
                                  << "CHECK FAILED: " << __FILE__ << ":" \
                                  << __LINE__ << ": "

// Abort the program after logging the mesage.
#define GRPC_CODEGEN_FAIL GRPC_CODEGEN_CHECK(false)

namespace grpc_java_generator {
struct Parameters {
  //        //Defines the custom parameter types for methods
  //        //eg: flatbuffers uses flatbuffers.Builder as input for the client
  //        and output for the server grpc::string custom_method_io_type;

  // Package name for the service
  grpc::string package_name;
};

// Return the source of the generated service file.
grpc::string GenerateServiceSource(grpc_generator::File* file,
                                   const grpc_generator::Service* service,
                                   grpc_java_generator::Parameters* parameters);

}  // namespace grpc_java_generator

#endif  // NET_GRPC_COMPILER_JAVA_GENERATOR_H_
