#ifndef PROTON_PROFILER_ROCTRACER_PROFILER_H_
#define PROTON_PROFILER_ROCTRACER_PROFILER_H_

#include "Profiler/GPUProfiler.h"

namespace proton {

class RoctracerProfiler : public GPUProfiler<RoctracerProfiler> {
public:
  RoctracerProfiler();
  virtual ~RoctracerProfiler();

private:
  struct RoctracerProfilerPimpl;
};

} // namespace proton

#endif // PROTON_PROFILER_ROCTRACER_PROFILER_H_
