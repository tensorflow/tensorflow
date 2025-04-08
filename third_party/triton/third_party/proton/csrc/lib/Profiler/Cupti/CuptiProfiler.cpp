#include "Profiler/Cupti/CuptiProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Driver/Device.h"
#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/CuptiApi.h"
#include "Profiler/Cupti/CuptiPCSampling.h"
#include "Utility/Map.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace proton {

template <>
thread_local GPUProfiler<CuptiProfiler>::ThreadState
    GPUProfiler<CuptiProfiler>::threadState(CuptiProfiler::instance());

template <>
thread_local std::deque<size_t>
    GPUProfiler<CuptiProfiler>::Correlation::externIdQueue{};

namespace {

std::shared_ptr<Metric> convertActivityToMetric(CUpti_Activity *activity) {
  std::shared_ptr<Metric> metric;
  switch (activity->kind) {
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    auto *kernel = reinterpret_cast<CUpti_ActivityKernel5 *>(activity);
    if (kernel->start < kernel->end) {
      metric = std::make_shared<KernelMetric>(
          static_cast<uint64_t>(kernel->start),
          static_cast<uint64_t>(kernel->end), 1,
          static_cast<uint64_t>(kernel->deviceId),
          static_cast<uint64_t>(DeviceType::CUDA));
    } // else: not a valid kernel activity
    break;
  }
  default:
    break;
  }
  return metric;
}

uint32_t
processActivityKernel(CuptiProfiler::CorrIdToExternIdMap &corrIdToExternId,
                      CuptiProfiler::ApiExternIdSet &apiExternIds,
                      std::set<Data *> &dataSet, CUpti_Activity *activity) {
  // Support CUDA >= 11.0
  auto *kernel = reinterpret_cast<CUpti_ActivityKernel5 *>(activity);
  auto correlationId = kernel->correlationId;
  if (/*Not a valid context*/ !corrIdToExternId.contain(correlationId))
    return correlationId;
  auto [parentId, numInstances] = corrIdToExternId.at(correlationId);
  if (kernel->graphId == 0) {
    // Non-graph kernels
    for (auto *data : dataSet) {
      auto scopeId = parentId;
      if (apiExternIds.contain(scopeId)) {
        // It's triggered by a CUDA op but not triton op
        scopeId = data->addOp(parentId, kernel->name);
      }
      data->addMetric(scopeId, convertActivityToMetric(activity));
    }
  } else {
    // Graph kernels
    // A single graph launch can trigger multiple kernels.
    // Our solution is to construct the following maps:
    // --- Application threads ---
    // 1. graphId -> numKernels
    // 2. graphExecId -> graphId
    // --- CUPTI thread ---
    // 3. corrId -> numKernels
    for (auto *data : dataSet) {
      auto externId = data->addOp(parentId, kernel->name);
      data->addMetric(externId, convertActivityToMetric(activity));
    }
  }
  apiExternIds.erase(parentId);
  --numInstances;
  if (numInstances == 0) {
    corrIdToExternId.erase(correlationId);
  } else {
    corrIdToExternId[correlationId].second = numInstances;
  }
  return correlationId;
}

uint32_t processActivity(CuptiProfiler::CorrIdToExternIdMap &corrIdToExternId,
                         CuptiProfiler::ApiExternIdSet &apiExternIds,
                         std::set<Data *> &dataSet, CUpti_Activity *activity) {
  auto correlationId = 0;
  switch (activity->kind) {
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    correlationId = processActivityKernel(corrIdToExternId, apiExternIds,
                                          dataSet, activity);
    break;
  }
  default:
    break;
  }
  return correlationId;
}

void setRuntimeCallbacks(CUpti_SubscriberHandle subscriber, bool enable) {
#define CALLBACK_ENABLE(id)                                                    \
  cupti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,       \
                              CUPTI_CB_DOMAIN_RUNTIME_API, id)

  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000);
  CALLBACK_ENABLE(
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000);
  CALLBACK_ENABLE(
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000);

#undef CALLBACK_ENABLE
}

void setDriverCallbacks(CUpti_SubscriberHandle subscriber, bool enable) {
#define CALLBACK_ENABLE(id)                                                    \
  cupti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,       \
                              CUPTI_CB_DOMAIN_DRIVER_API, id)

  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunch);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz);
#undef CALLBACK_ENABLE
}

void setGraphCallbacks(CUpti_SubscriberHandle subscriber, bool enable) {

#define CALLBACK_ENABLE(id)                                                    \
  cupti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,       \
                              CUPTI_CB_DOMAIN_RESOURCE, id)

  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING);
#undef CALLBACK_ENABLE
}

void setResourceCallbacks(CUpti_SubscriberHandle subscriber, bool enable) {
#define CALLBACK_ENABLE(id)                                                    \
  cupti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,       \
                              CUPTI_CB_DOMAIN_RESOURCE, id)

  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_MODULE_LOADED);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_CONTEXT_CREATED);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING);
#undef CALLBACK_ENABLE
}

bool isDriverAPILaunch(CUpti_CallbackId cbId) {
  return cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunch ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz;
}

} // namespace

struct CuptiProfiler::CuptiProfilerPimpl
    : public GPUProfiler<CuptiProfiler>::GPUProfilerPimplInterface {
  CuptiProfilerPimpl(CuptiProfiler &profiler)
      : GPUProfiler<CuptiProfiler>::GPUProfilerPimplInterface(profiler) {}
  virtual ~CuptiProfilerPimpl() = default;

  void setLibPath(const std::string &libPath) override {
    cupti::setLibPath(libPath);
  }
  void doStart() override;
  void doFlush() override;
  void doStop() override;

  static void allocBuffer(uint8_t **buffer, size_t *bufferSize,
                          size_t *maxNumRecords);
  static void completeBuffer(CUcontext context, uint32_t streamId,
                             uint8_t *buffer, size_t size, size_t validSize);
  static void callbackFn(void *userData, CUpti_CallbackDomain domain,
                         CUpti_CallbackId cbId, const void *cbData);

  static constexpr size_t AlignSize = 8;
  static constexpr size_t BufferSize = 64 * 1024 * 1024;
  static constexpr size_t AttributeSize = sizeof(size_t);

  CUpti_SubscriberHandle subscriber{};
  CuptiPCSampling pcSampling;

  ThreadSafeMap<uint32_t, size_t, std::unordered_map<uint32_t, size_t>>
      graphIdToNumInstances;
  ThreadSafeMap<uint32_t, uint32_t, std::unordered_map<uint32_t, uint32_t>>
      graphExecIdToGraphId;
};

void CuptiProfiler::CuptiProfilerPimpl::allocBuffer(uint8_t **buffer,
                                                    size_t *bufferSize,
                                                    size_t *maxNumRecords) {
  *buffer = static_cast<uint8_t *>(aligned_alloc(AlignSize, BufferSize));
  if (*buffer == nullptr) {
    throw std::runtime_error("[PROTON] aligned_alloc failed");
  }
  *bufferSize = BufferSize;
  *maxNumRecords = 0;
}

void CuptiProfiler::CuptiProfilerPimpl::completeBuffer(CUcontext ctx,
                                                       uint32_t streamId,
                                                       uint8_t *buffer,
                                                       size_t size,
                                                       size_t validSize) {
  CuptiProfiler &profiler = threadState.profiler;
  auto dataSet = profiler.getDataSet();
  uint32_t maxCorrelationId = 0;
  CUptiResult status;
  CUpti_Activity *activity = nullptr;
  do {
    status = cupti::activityGetNextRecord<false>(buffer, validSize, &activity);
    if (status == CUPTI_SUCCESS) {
      auto correlationId =
          processActivity(profiler.correlation.corrIdToExternId,
                          profiler.correlation.apiExternIds, dataSet, activity);
      maxCorrelationId = std::max(maxCorrelationId, correlationId);
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    } else {
      throw std::runtime_error("[PROTON] cupti::activityGetNextRecord failed");
    }
  } while (true);

  std::free(buffer);

  profiler.correlation.complete(maxCorrelationId);
}

void CuptiProfiler::CuptiProfilerPimpl::callbackFn(void *userData,
                                                   CUpti_CallbackDomain domain,
                                                   CUpti_CallbackId cbId,
                                                   const void *cbData) {
  CuptiProfiler &profiler = threadState.profiler;
  if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
    auto *resourceData =
        static_cast<CUpti_ResourceData *>(const_cast<void *>(cbData));
    auto *pImpl = dynamic_cast<CuptiProfilerPimpl *>(profiler.pImpl.get());
    if (cbId == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
      auto *moduleResource = static_cast<CUpti_ModuleResourceData *>(
          resourceData->resourceDescriptor);
      if (profiler.isPCSamplingEnabled()) {
        pImpl->pcSampling.loadModule(moduleResource->pCubin,
                                     moduleResource->cubinSize);
      }
    } else if (cbId == CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING) {
      auto *moduleResource = static_cast<CUpti_ModuleResourceData *>(
          resourceData->resourceDescriptor);
      if (profiler.isPCSamplingEnabled()) {
        pImpl->pcSampling.unloadModule(moduleResource->pCubin,
                                       moduleResource->cubinSize);
      }
    } else if (cbId == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
      if (profiler.isPCSamplingEnabled()) {
        pImpl->pcSampling.initialize(resourceData->context);
      }
    } else if (cbId == CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING) {
      if (profiler.isPCSamplingEnabled()) {
        pImpl->pcSampling.finalize(resourceData->context);
      }
    } else {
      auto *graphData =
          static_cast<CUpti_GraphData *>(resourceData->resourceDescriptor);
      uint32_t graphId = 0;
      uint32_t graphExecId = 0;
      if (graphData->graph)
        cupti::getGraphId<true>(graphData->graph, &graphId);
      if (graphData->graphExec)
        cupti::getGraphExecId<true>(graphData->graphExec, &graphExecId);
      if (cbId == CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED ||
          cbId == CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED) {
        if (!pImpl->graphIdToNumInstances.contain(graphId))
          pImpl->graphIdToNumInstances[graphId] = 1;
        else
          pImpl->graphIdToNumInstances[graphId]++;
      } else if (cbId == CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING) {
        pImpl->graphIdToNumInstances[graphId]--;
      } else if (cbId == CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED) {
        pImpl->graphExecIdToGraphId[graphExecId] = graphId;
      } else if (cbId == CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING) {
        pImpl->graphExecIdToGraphId.erase(graphExecId);
      } else if (cbId == CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING) {
        pImpl->graphIdToNumInstances.erase(graphId);
      }
    }
  } else {
    const CUpti_CallbackData *callbackData =
        static_cast<const CUpti_CallbackData *>(cbData);
    auto *pImpl = dynamic_cast<CuptiProfilerPimpl *>(profiler.pImpl.get());
    if (callbackData->callbackSite == CUPTI_API_ENTER) {
      threadState.enterOp();
      size_t numInstances = 1;
      if (cbId == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch ||
          cbId == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz) {
        auto graphExec = static_cast<const cuGraphLaunch_params *>(
                             callbackData->functionParams)
                             ->hGraph;
        uint32_t graphExecId = 0;
        cupti::getGraphExecId<true>(graphExec, &graphExecId);
        numInstances = std::numeric_limits<size_t>::max();
        auto findGraph = false;
        if (pImpl->graphExecIdToGraphId.contain(graphExecId)) {
          auto graphId = pImpl->graphExecIdToGraphId[graphExecId];
          if (pImpl->graphIdToNumInstances.contain(graphId)) {
            numInstances = pImpl->graphIdToNumInstances[graphId];
            findGraph = true;
          }
        }
        if (!findGraph)
          std::cerr << "[PROTON] Cannot find graph for graphExecId: "
                    << graphExecId
                    << ", and t may cause memory leak. To avoid this problem, "
                       "please start profiling before the graph is created."
                    << std::endl;
      }
      profiler.correlation.correlate(callbackData->correlationId, numInstances);
      if (profiler.isPCSamplingEnabled() && isDriverAPILaunch(cbId)) {
        pImpl->pcSampling.start(callbackData->context);
      }
    } else if (callbackData->callbackSite == CUPTI_API_EXIT) {
      if (profiler.isPCSamplingEnabled() && isDriverAPILaunch(cbId)) {
        // XXX: Conservatively stop every GPU kernel for now
        auto scopeId = profiler.correlation.externIdQueue.back();
        pImpl->pcSampling.stop(
            callbackData->context, scopeId,
            profiler.correlation.apiExternIds.contain(scopeId));
      }
      threadState.exitOp();
      profiler.correlation.submit(callbackData->correlationId);
    }
  }
}

void CuptiProfiler::CuptiProfilerPimpl::doStart() {
  cupti::subscribe<true>(&subscriber, callbackFn, nullptr);
  if (profiler.isPCSamplingEnabled()) {
    setResourceCallbacks(subscriber, /*enable=*/true);
    // Continuous PC sampling is not compatible with concurrent kernel profiling
    cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_KERNEL);
  } else {
    cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  }
  cupti::activityRegisterCallbacks<true>(allocBuffer, completeBuffer);
  setGraphCallbacks(subscriber, /*enable=*/true);
  setRuntimeCallbacks(subscriber, /*enable=*/true);
  setDriverCallbacks(subscriber, /*enable=*/true);
}

void CuptiProfiler::CuptiProfilerPimpl::doFlush() {
  // cuptiActivityFlushAll returns the activity records associated with all
  // contexts/streams.
  // This is a blocking call but it doesn’t issue any CUDA synchronization calls
  // implicitly thus it’s not guaranteed that all activities are completed on
  // the underlying devices.
  // We do an "opportunistic" synchronization here to try to ensure that all
  // activities are completed on the current context.
  // If the current context is not set, we don't do any synchronization.
  CUcontext cuContext = nullptr;
  cuda::ctxGetCurrent<false>(&cuContext);
  if (cuContext) {
    cuda::ctxSynchronize<true>();
  }
  profiler.correlation.flush(
      /*maxRetries=*/100, /*sleepMs=*/10,
      /*flush=*/[]() {
        cupti::activityFlushAll<true>(
            /*flag=*/0);
      });
  // CUPTI_ACTIVITY_FLAG_FLUSH_FORCED is used to ensure that even incomplete
  // activities are flushed so that the next profiling session can start with
  // new activities.
  cupti::activityFlushAll<true>(/*flag=*/CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
}

void CuptiProfiler::CuptiProfilerPimpl::doStop() {
  if (profiler.isPCSamplingEnabled()) {
    profiler.disablePCSampling();
    CUcontext cuContext = nullptr;
    cuda::ctxGetCurrent<false>(&cuContext);
    if (cuContext)
      pcSampling.finalize(cuContext);
    setResourceCallbacks(subscriber, /*enable=*/false);
    cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_KERNEL);
  } else {
    cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  }
  setGraphCallbacks(subscriber, /*enable=*/false);
  setRuntimeCallbacks(subscriber, /*enable=*/false);
  setDriverCallbacks(subscriber, /*enable=*/false);
  cupti::unsubscribe<true>(subscriber);
  cupti::finalize<true>();
}

CuptiProfiler::CuptiProfiler() {
  pImpl = std::make_unique<CuptiProfilerPimpl>(*this);
}

CuptiProfiler::~CuptiProfiler() = default;

} // namespace proton
