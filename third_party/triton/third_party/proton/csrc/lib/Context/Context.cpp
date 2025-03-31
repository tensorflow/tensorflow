#include "Context/Context.h"

namespace proton {

/*static*/ thread_local std::optional<Context> ContextSource::state =
    std::nullopt;

std::atomic<size_t> Scope::scopeIdCounter{1};

/*static*/ thread_local std::map<ThreadLocalOpInterface *, bool>
    ThreadLocalOpInterface::opInProgress;

} // namespace proton
