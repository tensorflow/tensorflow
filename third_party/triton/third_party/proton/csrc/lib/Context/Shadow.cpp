#include "Context/Shadow.h"

#include <stdexcept>
#include <thread>

namespace proton {

void ShadowContextSource::initializeThreadContext() {
  if (!mainContextStack) {
    mainContextStack = &threadContextStack[this];
    threadContextInitialized[this] = false;
  }
  if (!threadContextInitialized[this]) {
    threadContextStack[this] = *mainContextStack;
    threadContextInitialized[this] = true;
  }
}

void ShadowContextSource::enterScope(const Scope &scope) {
  initializeThreadContext();
  threadContextStack[this].push_back(scope);
}

std::vector<Context> ShadowContextSource::getContextsImpl() {
  initializeThreadContext();
  return threadContextStack[this];
}

size_t ShadowContextSource::getDepth() {
  initializeThreadContext();
  return threadContextStack[this].size();
}

void ShadowContextSource::exitScope(const Scope &scope) {
  if (threadContextStack[this].empty()) {
    throw std::runtime_error("Context stack is empty");
  }
  if (threadContextStack[this].back() != scope) {
    throw std::runtime_error("Context stack is not balanced");
  }
  threadContextStack[this].pop_back();
}

/*static*/ thread_local std::map<ShadowContextSource *, bool>
    ShadowContextSource::threadContextInitialized;

/*static*/ thread_local std::map<ShadowContextSource *, std::vector<Context>>
    ShadowContextSource::threadContextStack;

} // namespace proton
