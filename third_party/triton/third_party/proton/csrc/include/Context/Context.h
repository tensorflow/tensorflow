#ifndef PROTON_CONTEXT_CONTEXT_H_
#define PROTON_CONTEXT_CONTEXT_H_

#include <atomic>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace proton {

/// A context is a named object.
struct Context {
  std::string name{};

  Context() = default;
  Context(const std::string &name) : name(name) {}
  virtual ~Context() = default;

  bool operator==(const Context &other) const { return name == other.name; }
  bool operator!=(const Context &other) const { return !(*this == other); }
  bool operator<(const Context &other) const { return name < other.name; }
  bool operator>(const Context &other) const { return name > other.name; }
  bool operator<=(const Context &other) const { return !(*this > other); }
  bool operator>=(const Context &other) const { return !(*this < other); }
};

/// A context source is an object that can provide a list of contexts.
class ContextSource {
public:
  ContextSource() = default;
  virtual ~ContextSource() = default;

  std::vector<Context> getContexts() {
    auto contexts = getContextsImpl();
    if (state.has_value()) {
      contexts.push_back(state.value());
    }
    return contexts;
  }

  void setState(std::optional<Context> state) { ContextSource::state = state; }

  virtual size_t getDepth() = 0;

protected:
  virtual std::vector<Context> getContextsImpl() = 0;
  static thread_local std::optional<Context> state;
};

/// A scope is a context with a unique identifier.
struct Scope : public Context {
  const static size_t DummyScopeId = std::numeric_limits<size_t>::max();
  static std::atomic<size_t> scopeIdCounter;

  static size_t getNewScopeId() { return scopeIdCounter++; }

  size_t scopeId{};

  explicit Scope(size_t scopeId) : Context(), scopeId(scopeId) {}

  explicit Scope(const std::string &name) : Context(name) {
    scopeId = getNewScopeId();
  }

  Scope(size_t scopeId, const std::string &name)
      : scopeId(scopeId), Context(name) {}

  Scope() : Scope(DummyScopeId, "") {}

  bool operator==(const Scope &other) const {
    return scopeId == other.scopeId && name == other.name;
  }
  bool operator!=(const Scope &other) const { return !(*this == other); }
  bool operator<(const Scope &other) const {
    return scopeId < other.scopeId || name < other.name;
  }
  bool operator>(const Scope &other) const {
    return scopeId > other.scopeId || name > other.name;
  }
  bool operator<=(const Scope &other) const { return !(*this > other); }
  bool operator>=(const Scope &other) const { return !(*this < other); }
};

/// A scope interface allows to instrument handles before and after a scope.
/// Scopes can be nested.
class ScopeInterface {
public:
  ScopeInterface() = default;
  virtual ~ScopeInterface() = default;
  virtual void enterScope(const Scope &scope) = 0;
  virtual void exitScope(const Scope &scope) = 0;
};

/// An op interface allows to instrument handles before and after an operation,
/// which cannot be nested.
class OpInterface {
public:
  OpInterface() = default;
  virtual ~OpInterface() = default;
  void enterOp(const Scope &scope) {
    if (isOpInProgress()) {
      return;
    }
    startOp(scope);
    setOpInProgress(true);
  }
  void exitOp(const Scope &scope) {
    if (!isOpInProgress()) {
      return;
    }
    stopOp(scope);
    setOpInProgress(false);
  }

protected:
  virtual void startOp(const Scope &scope) = 0;
  virtual void stopOp(const Scope &scope) = 0;
  virtual bool isOpInProgress() = 0;
  virtual void setOpInProgress(bool value) = 0;
};

class ThreadLocalOpInterface : public OpInterface {
public:
  using OpInterface::OpInterface;

protected:
  bool isOpInProgress() override final { return opInProgress[this]; }
  void setOpInProgress(bool value) override final {
    opInProgress[this] = value;
    if (opInProgress.size() > MAX_CACHE_OBJECTS && !value)
      opInProgress.erase(this);
  }

private:
  inline static const int MAX_CACHE_OBJECTS = 10;
  static thread_local std::map<ThreadLocalOpInterface *, bool> opInProgress;
};

} // namespace proton

#endif // PROTON_CONTEXT_CONTEXT_H_
