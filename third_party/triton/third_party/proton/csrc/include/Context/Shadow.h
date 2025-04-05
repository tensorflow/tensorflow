#ifndef PROTON_CONTEXT_SHADOW_H_
#define PROTON_CONTEXT_SHADOW_H_

#include "Context.h"
#include <vector>

namespace proton {

/// ShadowContextSource is designed to:
///
///   - Maintain a main context stack for the main thread.
///   - Provide thread-local context stacks for individual threads.
///   - Allow threads to inherit and shadow the main context stack with their
///     own user-defined scopes.
///
/// This implementation is suited for use cases like PyTorch, where:
///
///   - The main thread initializes the main context stack during session setup.
///   - The backward phase spawns multiple CPU threads.
class ShadowContextSource : public ContextSource, public ScopeInterface {
public:
  ShadowContextSource() = default;

  void enterScope(const Scope &scope) override;

  void exitScope(const Scope &scope) override;

  size_t getDepth() override;

private:
  std::vector<Context> getContextsImpl() override;

  void initializeThreadContext();

  std::vector<Context> *mainContextStack{};
  static thread_local std::map<ShadowContextSource *, bool>
      threadContextInitialized;
  static thread_local std::map<ShadowContextSource *, std::vector<Context>>
      threadContextStack;
};

} // namespace proton

#endif // PROTON_CONTEXT_CONTEXT_H_
