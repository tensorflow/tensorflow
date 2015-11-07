#ifndef TENSORFLOW_COMMON_RUNTIME_SESSION_FACTORY_H_
#define TENSORFLOW_COMMON_RUNTIME_SESSION_FACTORY_H_

#include <string>

#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {

class Session;
class SessionOptions;

class SessionFactory {
 public:
  virtual Session* NewSession(const SessionOptions& options) = 0;
  virtual ~SessionFactory() {}
  static void Register(const string& runtime_type, SessionFactory* factory);
  static SessionFactory* GetFactory(const string& runtime_type);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_SESSION_FACTORY_H_
