#ifndef PROTON_PROFILER_CUPTI_PROFILER_H_
#define PROTON_PROFILER_CUPTI_PROFILER_H_

#include "Profiler/GPUProfiler.h"

namespace proton {

class CuptiProfiler : public GPUProfiler<CuptiProfiler> {
public:
  CuptiProfiler();
  virtual ~CuptiProfiler();

private:
  struct CuptiProfilerPimpl;
};

} // namespace proton

#endif // PROTON_PROFILER_CUPTI_PROFILER_H_
