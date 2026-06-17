/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PUBLIC_SESSION_OPTIONS_H_
#define TENSORFLOW_CORE_PUBLIC_SESSION_OPTIONS_H_

#include <string>

#include "tensorflow/core/protobuf/config.pb.h"

namespace tsl {
class Env;
}  // namespace tsl
namespace tensorflow {

/// Configuration information for a Session.
struct SessionOptions {
  /// The environment to use.
  tsl::Env* env;

  /// \brief The TensorFlow runtime to connect to.
  ///
  /// If 'target' is empty or unspecified, the local TensorFlow runtime
  /// implementation will be used.  Otherwise, the TensorFlow engine
  /// defined by 'target' will be used to perform all computations.
  ///
  /// "target" can be either a single entry or a comma separated list
  /// of entries. Each entry is a resolvable address of the
  /// following format:
  ///   local
  ///   ip:port
  ///   host:port
  ///   ... other system-specific formats to identify tasks and jobs ...
  ///
  /// NOTE: at the moment 'local' maps to an in-process service-based
  /// runtime.
  ///
  /// Upon creation, a single session affines itself to one of the
  /// remote processes, with possible load balancing choices when the
  /// "target" resolves to a list of possible processes.
  ///
  /// If the session disconnects from the remote process during its
  /// lifetime, session calls may fail immediately.
  std::string target;

  /// Configuration options.
  ConfigProto config;

  SessionOptions();
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_PUBLIC_SESSION_OPTIONS_H_
