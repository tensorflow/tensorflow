#ifndef TENSORFLOW_PUBLIC_SESSION_OPTIONS_H_
#define TENSORFLOW_PUBLIC_SESSION_OPTIONS_H_

#include <string>
#include "tensorflow/core/framework/config.pb.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

class Env;

/// Configuration information for a Session.
struct SessionOptions {
  /// The environment to use.
  Env* env;

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
  string target;

  /// Configuration options.
  ConfigProto config;

  SessionOptions();
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_PUBLIC_SESSION_OPTIONS_H_
