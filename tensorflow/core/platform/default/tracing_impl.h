#ifndef TENSORFLOW_PLATFORM_DEFAULT_TRACING_IMPL_H_
#define TENSORFLOW_PLATFORM_DEFAULT_TRACING_IMPL_H_

// Stub implementations of tracing functionality.

#include "tensorflow/core/public/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/tracing.h"

namespace tensorflow {
namespace port {

// Definitions that do nothing for platforms that don't have underlying thread
// tracing support.
#define TRACELITERAL(a) \
  do {                  \
  } while (0)
#define TRACESTRING(s) \
  do {                 \
  } while (0)
#define TRACEPRINTF(format, ...) \
  do {                           \
  } while (0)

inline uint64 Tracing::UniqueId() { return random::New64(); }
inline bool Tracing::IsActive() { return false; }
inline void Tracing::RegisterCurrentThread(const char* name) {}

// Posts an atomic threadscape event with the supplied category and arg.
inline void Tracing::RecordEvent(EventCategory category, uint64 arg) {
  // TODO(opensource): Implement
}

inline Tracing::ScopedActivity::ScopedActivity(EventCategory category,
                                               uint64 arg)
    : enabled_(false), region_id_(category_id_[category]) {}

inline Tracing::ScopedActivity::~ScopedActivity() {}

}  // namespace port
}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_DEFAULT_TRACING_IMPL_H_
