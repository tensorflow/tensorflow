#ifndef PROTON_SESSION_SESSION_H_
#define PROTON_SESSION_SESSION_H_

#include "Context/Context.h"
#include "Data/Metric.h"
#include "Utility/Singleton.h"
#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace proton {

class Profiler;
class Data;
enum class OutputFormat;

/// A session is a collection of profiler, context source, and data objects.
/// There could be multiple sessions in the system, each can correspond to a
/// different duration, or the same duration but with different configurations.
class Session {
public:
  ~Session() = default;

  void activate();

  void deactivate();

  void finalize(OutputFormat outputFormat);

  size_t getContextDepth();

private:
  Session(size_t id, const std::string &path, Profiler *profiler,
          std::unique_ptr<ContextSource> contextSource,
          std::unique_ptr<Data> data)
      : id(id), path(path), profiler(profiler),
        contextSource(std::move(contextSource)), data(std::move(data)) {}

  template <typename T> std::vector<T *> getInterfaces() {
    std::vector<T *> interfaces;
    // There's an implicit order between contextSource and profiler/data. The
    // latter two rely on the contextSource to obtain the context, so we need to
    // add the contextSource first.
    if (auto interface = dynamic_cast<T *>(contextSource.get())) {
      interfaces.push_back(interface);
    }
    if (auto interface = dynamic_cast<T *>(profiler)) {
      interfaces.push_back(interface);
    }
    if (auto interface = dynamic_cast<T *>(data.get())) {
      interfaces.push_back(interface);
    }
    return interfaces;
  }

  const std::string path{};
  size_t id{};
  Profiler *profiler{};
  std::unique_ptr<ContextSource> contextSource{};
  std::unique_ptr<Data> data{};

  friend class SessionManager;
};

/// A session manager is responsible for managing the lifecycle of sessions.
/// There's a single and unique session manager in the system.
class SessionManager : public Singleton<SessionManager> {
public:
  SessionManager() = default;
  ~SessionManager() = default;

  size_t addSession(const std::string &path, const std::string &profilerName,
                    const std::string &profilerPath,
                    const std::string &contextSourceName,
                    const std::string &dataName);

  void finalizeSession(size_t sessionId, OutputFormat outputFormat);

  void finalizeAllSessions(OutputFormat outputFormat);

  void activateSession(size_t sessionId);

  void activateAllSessions();

  void deactivateSession(size_t sessionId);

  void deactivateAllSessions();

  size_t getContextDepth(size_t sessionId);

  void enterScope(const Scope &scope);

  void exitScope(const Scope &scope);

  void enterOp(const Scope &scope);

  void exitOp(const Scope &scope);

  void addMetrics(size_t scopeId,
                  const std::map<std::string, MetricValueType> &metrics);

  void setState(std::optional<Context> context);

private:
  std::unique_ptr<Session> makeSession(size_t id, const std::string &path,
                                       const std::string &profilerName,
                                       const std::string &profilerPath,
                                       const std::string &contextSourceName,
                                       const std::string &dataName);

  void activateSessionImpl(size_t sessionId);

  void deActivateSessionImpl(size_t sessionId);

  size_t getSessionId(const std::string &path) { return sessionPaths[path]; }

  bool hasSession(const std::string &path) {
    return sessionPaths.find(path) != sessionPaths.end();
  }

  bool hasSession(size_t sessionId) {
    return sessions.find(sessionId) != sessions.end();
  }

  void removeSession(size_t sessionId);

  template <typename Interface, typename Counter, bool isRegistering>
  void updateInterfaceCount(size_t sessionId, Counter &interfaceCounts) {
    auto interfaces = sessions[sessionId]->getInterfaces<Interface>();
    for (auto *interface : interfaces) {
      auto it = std::find_if(
          interfaceCounts.begin(), interfaceCounts.end(),
          [interface](const auto &pair) { return pair.first == interface; });

      if (it != interfaceCounts.end()) {
        if constexpr (isRegistering) {
          ++it->second;
        } else {
          --it->second;
          if (it->second == 0) {
            interfaceCounts.erase(it);
          }
        }
      } else if constexpr (isRegistering) {
        interfaceCounts.emplace_back(interface, 1);
      }
    }
  }

  template <typename Interface, typename Counter>
  void registerInterface(size_t sessionId, Counter &interfaceCounts) {
    updateInterfaceCount<Interface, Counter, true>(sessionId, interfaceCounts);
  }

  template <typename Interface, typename Counter>
  void unregisterInterface(size_t sessionId, Counter &interfaceCounts) {
    updateInterfaceCount<Interface, Counter, false>(sessionId, interfaceCounts);
  }

  mutable std::mutex mutex;

  size_t nextSessionId{};
  // path -> session id
  std::map<std::string, size_t> sessionPaths;
  // session id -> active
  std::map<size_t, bool> sessionActive;
  // session id -> session
  std::map<size_t, std::unique_ptr<Session>> sessions;
  // {scope, active count}
  std::vector<std::pair<ScopeInterface *, size_t>> scopeInterfaceCounts;
  // {op, active count}
  std::vector<std::pair<OpInterface *, size_t>> opInterfaceCounts;
  // {context source, active count}
  std::vector<std::pair<ContextSource *, size_t>> contextSourceCounts;
};

} // namespace proton

#endif // PROTON_SESSION_H_
