#ifndef PROTON_PROFILER_GPU_PROFILER_H_
#define PROTON_PROFILER_GPU_PROFILER_H_

#include "Context/Context.h"
#include "Profiler.h"
#include "Utility/Atomic.h"
#include "Utility/Map.h"
#include "Utility/Set.h"

#include <atomic>
#include <deque>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace proton {

// Singleton<ConcreteProfilerT>: Each concrete GPU profiler, e.g.,
// CuptiProfiler, should be a singleton.
template <typename ConcreteProfilerT>
class GPUProfiler : public Profiler,
                    public ThreadLocalOpInterface,
                    public Singleton<ConcreteProfilerT> {
public:
  GPUProfiler() = default;
  virtual ~GPUProfiler() = default;

  using CorrIdToExternIdMap =
      ThreadSafeMap<uint64_t,
                    std::pair<size_t, size_t>, /*<extern_id, num_kernels>*/
                    std::unordered_map<uint64_t, std::pair<size_t, size_t>>>;
  using ApiExternIdSet = ThreadSafeSet<size_t, std::unordered_set<size_t>>;

  ConcreteProfilerT &enablePCSampling() {
    pcSamplingEnabled = true;
    return dynamic_cast<ConcreteProfilerT &>(*this);
  }
  ConcreteProfilerT &disablePCSampling() {
    pcSamplingEnabled = false;
    return dynamic_cast<ConcreteProfilerT &>(*this);
  }
  bool isPCSamplingEnabled() const { return pcSamplingEnabled; }

  ConcreteProfilerT &setLibPath(const std::string &libPath) {
    pImpl->setLibPath(libPath);
    return dynamic_cast<ConcreteProfilerT &>(*this);
  }

protected:
  // OpInterface
  void startOp(const Scope &scope) override {
    this->correlation.pushExternId(scope.scopeId);
    for (auto data : getDataSet())
      data->addOp(scope.scopeId, scope.name);
  }
  void stopOp(const Scope &scope) override { this->correlation.popExternId(); }

  // Profiler
  virtual void doStart() override { pImpl->doStart(); }
  virtual void doFlush() override { pImpl->doFlush(); }
  virtual void doStop() override { pImpl->doStop(); }

  struct ThreadState {
    ConcreteProfilerT &profiler;
    size_t scopeId{Scope::DummyScopeId};

    ThreadState(ConcreteProfilerT &profiler) : profiler(profiler) {}

    void enterOp() {
      if (profiler.isOpInProgress())
        return;
      scopeId = Scope::getNewScopeId();
      profiler.enterOp(Scope(scopeId));
      profiler.correlation.apiExternIds.insert(scopeId);
    }

    void exitOp() {
      if (!profiler.isOpInProgress())
        return;
      profiler.exitOp(Scope(scopeId));
    }
  };

  struct Correlation {
    std::atomic<uint64_t> maxSubmittedCorrelationId{0};
    std::atomic<uint64_t> maxCompletedCorrelationId{0};
    // Mapping from a native profiler correlation id to an external id.
    CorrIdToExternIdMap corrIdToExternId;
    // A set of kernels triggered by GPU runtime APIs (e.g., torch
    // kernels) other than Triton.
    // It stores a subset of external ids in corrIdToExternId.
    ApiExternIdSet apiExternIds;
    static thread_local std::deque<size_t> externIdQueue;

    Correlation() = default;

    void submit(const uint64_t correlationId) {
      atomicMax(maxSubmittedCorrelationId, correlationId);
    }

    void complete(const uint64_t correlationId) {
      atomicMax(maxCompletedCorrelationId, correlationId);
    }

    void pushExternId(size_t externId) { externIdQueue.push_back(externId); }

    void popExternId() { externIdQueue.pop_front(); }

    // Correlate the correlationId with the last externId
    void correlate(uint64_t correlationId, size_t numInstances = 1) {
      if (externIdQueue.empty())
        return;
      corrIdToExternId[correlationId] = {externIdQueue.back(), numInstances};
    }

    template <typename FlushFnT>
    void flush(uint64_t maxRetries, uint64_t sleepMs, FlushFnT &&flushFn) {
      flushFn();
      auto submittedId = maxSubmittedCorrelationId.load();
      auto completedId = maxCompletedCorrelationId.load();
      auto retries = maxRetries;
      while ((completedId < submittedId) && retries > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(sleepMs));
        flushFn();
        completedId = maxCompletedCorrelationId.load();
        --retries;
      }
    }
  };

  static thread_local ThreadState threadState;
  Correlation correlation;

  // Use the pimpl idiom to hide the implementation details. This lets us avoid
  // including the cupti header from this header. The cupti header and the
  // equivalent header from AMD define conflicting macros, so we want to use
  // those headers only within cpp files.
  class GPUProfilerPimplInterface {
  public:
    GPUProfilerPimplInterface(ConcreteProfilerT &profiler)
        : profiler(profiler) {}
    virtual ~GPUProfilerPimplInterface() = default;

    virtual void setLibPath(const std::string &libPath) = 0;
    virtual void doStart() = 0;
    virtual void doFlush() = 0;
    virtual void doStop() = 0;

  protected:
    ConcreteProfilerT &profiler;
  };
  std::unique_ptr<GPUProfilerPimplInterface> pImpl;

  bool pcSamplingEnabled{false};
};

} // namespace proton

#endif // PROTON_PROFILER_GPU_PROFILER_H_
