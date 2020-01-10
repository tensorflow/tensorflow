#include "tensorflow/core/common_runtime/session_factory.h"

#include <unordered_map>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
namespace tensorflow {
namespace {

static mutex* get_session_factory_lock() {
  static mutex session_factory_lock;
  return &session_factory_lock;
}

typedef std::unordered_map<string, SessionFactory*> SessionFactories;
SessionFactories* session_factories() {
  static SessionFactories* factories = new SessionFactories;
  return factories;
}

}  // namespace

void SessionFactory::Register(const string& runtime_type,
                              SessionFactory* factory) {
  mutex_lock l(*get_session_factory_lock());
  if (!session_factories()->insert({runtime_type, factory}).second) {
    LOG(ERROR) << "Two session factories are being registered "
               << "under" << runtime_type;
  }
}

SessionFactory* SessionFactory::GetFactory(const string& runtime_type) {
  mutex_lock l(*get_session_factory_lock());  // could use reader lock
  auto it = session_factories()->find(runtime_type);
  if (it == session_factories()->end()) {
    return nullptr;
  }
  return it->second;
}

}  // namespace tensorflow
