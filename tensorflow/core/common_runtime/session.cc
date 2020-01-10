#include <string>

#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

namespace {
Status GetFactory(const SessionOptions& options, SessionFactory** ret) {
  string runtime_type = "LOCAL_SESSION";
  if (!options.target.empty()) {
    // Use the service based session.
    runtime_type = "REMOTE_SESSION";
  }
  *ret = SessionFactory::GetFactory(runtime_type);
  if (!*ret) {
    return errors::NotFound("Could not find session factory for ",
                            runtime_type);
  }
  return Status::OK();
}
}  // end namespace

Session* NewSession(const SessionOptions& options) {
  SessionFactory* factory;
  Status s = GetFactory(options, &factory);
  if (!s.ok()) {
    LOG(ERROR) << s;
    return nullptr;
  }
  return factory->NewSession(options);
}

Status NewSession(const SessionOptions& options, Session** out_session) {
  SessionFactory* factory;
  Status s = GetFactory(options, &factory);
  if (!s.ok()) {
    *out_session = nullptr;
    LOG(ERROR) << s;
    return s;
  }
  *out_session = factory->NewSession(options);
  if (!*out_session) {
    return errors::Internal("Failed to create session.");
  }
  return Status::OK();
}

}  // namespace tensorflow
