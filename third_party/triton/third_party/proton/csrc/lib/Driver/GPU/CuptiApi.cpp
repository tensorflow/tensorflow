#include "Driver/GPU/CuptiApi.h"
#include "Driver/Device.h"
#include "Driver/Dispatch.h"

namespace proton {

namespace cupti {

struct ExternLibCupti : public ExternLibBase {
  using RetType = CUptiResult;
  static constexpr const char *name = "libcupti.so";
  static inline std::string defaultDir = "";
  static constexpr RetType success = CUPTI_SUCCESS;
  static void *lib;
};

void *ExternLibCupti::lib = nullptr;

DEFINE_DISPATCH(ExternLibCupti, getVersion, cuptiGetVersion, uint32_t *);

DEFINE_DISPATCH(ExternLibCupti, getContextId, cuptiGetContextId, CUcontext,
                uint32_t *);

DEFINE_DISPATCH(ExternLibCupti, activityRegisterCallbacks,
                cuptiActivityRegisterCallbacks,
                CUpti_BuffersCallbackRequestFunc,
                CUpti_BuffersCallbackCompleteFunc)

DEFINE_DISPATCH(ExternLibCupti, subscribe, cuptiSubscribe,
                CUpti_SubscriberHandle *, CUpti_CallbackFunc, void *)

DEFINE_DISPATCH(ExternLibCupti, enableDomain, cuptiEnableDomain, uint32_t,
                CUpti_SubscriberHandle, CUpti_CallbackDomain)

DEFINE_DISPATCH(ExternLibCupti, enableCallback, cuptiEnableCallback, uint32_t,
                CUpti_SubscriberHandle, CUpti_CallbackDomain, CUpti_CallbackId);

DEFINE_DISPATCH(ExternLibCupti, activityEnable, cuptiActivityEnable,
                CUpti_ActivityKind)

DEFINE_DISPATCH(ExternLibCupti, activityDisable, cuptiActivityDisable,
                CUpti_ActivityKind)

DEFINE_DISPATCH(ExternLibCupti, activityEnableContext,
                cuptiActivityEnableContext, CUcontext, CUpti_ActivityKind)

DEFINE_DISPATCH(ExternLibCupti, activityDisableContext,
                cuptiActivityDisableContext, CUcontext, CUpti_ActivityKind)

DEFINE_DISPATCH(ExternLibCupti, activityFlushAll, cuptiActivityFlushAll,
                uint32_t)

DEFINE_DISPATCH(ExternLibCupti, activityGetNextRecord,
                cuptiActivityGetNextRecord, uint8_t *, size_t,
                CUpti_Activity **)

DEFINE_DISPATCH(ExternLibCupti, activityPushExternalCorrelationId,
                cuptiActivityPushExternalCorrelationId,
                CUpti_ExternalCorrelationKind, uint64_t)

DEFINE_DISPATCH(ExternLibCupti, activityPopExternalCorrelationId,
                cuptiActivityPopExternalCorrelationId,
                CUpti_ExternalCorrelationKind, uint64_t *)

DEFINE_DISPATCH(ExternLibCupti, activitySetAttribute, cuptiActivitySetAttribute,
                CUpti_ActivityAttribute, size_t *, void *)

DEFINE_DISPATCH(ExternLibCupti, unsubscribe, cuptiUnsubscribe,
                CUpti_SubscriberHandle)

DEFINE_DISPATCH(ExternLibCupti, finalize, cuptiFinalize)

DEFINE_DISPATCH(ExternLibCupti, getGraphExecId, cuptiGetGraphExecId,
                CUgraphExec, uint32_t *);

DEFINE_DISPATCH(ExternLibCupti, getGraphId, cuptiGetGraphId, CUgraph,
                uint32_t *);

DEFINE_DISPATCH(ExternLibCupti, getCubinCrc, cuptiGetCubinCrc,
                CUpti_GetCubinCrcParams *);

DEFINE_DISPATCH(ExternLibCupti, getSassToSourceCorrelation,
                cuptiGetSassToSourceCorrelation,
                CUpti_GetSassToSourceCorrelationParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingGetNumStallReasons,
                cuptiPCSamplingGetNumStallReasons,
                CUpti_PCSamplingGetNumStallReasonsParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingGetStallReasons,
                cuptiPCSamplingGetStallReasons,
                CUpti_PCSamplingGetStallReasonsParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingSetConfigurationAttribute,
                cuptiPCSamplingSetConfigurationAttribute,
                CUpti_PCSamplingConfigurationInfoParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingEnable, cuptiPCSamplingEnable,
                CUpti_PCSamplingEnableParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingDisable, cuptiPCSamplingDisable,
                CUpti_PCSamplingDisableParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingGetData, cuptiPCSamplingGetData,
                CUpti_PCSamplingGetDataParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingStart, cuptiPCSamplingStart,
                CUpti_PCSamplingStartParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingStop, cuptiPCSamplingStop,
                CUpti_PCSamplingStopParams *);

void setLibPath(const std::string &path) { ExternLibCupti::defaultDir = path; }

} // namespace cupti

} // namespace proton
