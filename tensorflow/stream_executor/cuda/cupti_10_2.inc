// Auto-generated, do not edit.

extern "C" {

CUptiResult CUPTIAPI cuptiGetResultString(CUptiResult result,
                                          const char **str) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUptiResult, const char **);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiGetResultString");
  if (!func_ptr) {
    if (str) {
      *str = "CUPTI could not be loaded or symbol could not be found.";
    }
    return GetSymbolNotFoundError();
  }
  return func_ptr(result, str);
}

CUptiResult CUPTIAPI cuptiGetVersion(uint32_t *version) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(uint32_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiGetVersion");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(version);
}

CUptiResult CUPTIAPI cuptiSupportedDomains(size_t *domainCount,
                                           CUpti_DomainTable *domainTable) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(size_t *, CUpti_DomainTable *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiSupportedDomains");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(domainCount, domainTable);
}

CUptiResult CUPTIAPI cuptiSubscribe(CUpti_SubscriberHandle *subscriber,
                                    CUpti_CallbackFunc callback,
                                    void *userdata) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_SubscriberHandle *,
                                          CUpti_CallbackFunc, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiSubscribe");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(subscriber, callback, userdata);
}

CUptiResult CUPTIAPI cuptiUnsubscribe(CUpti_SubscriberHandle subscriber) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_SubscriberHandle);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiUnsubscribe");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(subscriber);
}

CUptiResult CUPTIAPI cuptiGetCallbackState(uint32_t *enable,
                                           CUpti_SubscriberHandle subscriber,
                                           CUpti_CallbackDomain domain,
                                           CUpti_CallbackId cbid) {
  using FuncPtr =
      CUptiResult(CUPTIAPI *)(uint32_t *, CUpti_SubscriberHandle,
                              CUpti_CallbackDomain, CUpti_CallbackId);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiGetCallbackState");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(enable, subscriber, domain, cbid);
}

CUptiResult CUPTIAPI cuptiEnableCallback(uint32_t enable,
                                         CUpti_SubscriberHandle subscriber,
                                         CUpti_CallbackDomain domain,
                                         CUpti_CallbackId cbid) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(
      uint32_t, CUpti_SubscriberHandle, CUpti_CallbackDomain, CUpti_CallbackId);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEnableCallback");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(enable, subscriber, domain, cbid);
}

CUptiResult CUPTIAPI cuptiEnableDomain(uint32_t enable,
                                       CUpti_SubscriberHandle subscriber,
                                       CUpti_CallbackDomain domain) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(uint32_t, CUpti_SubscriberHandle,
                                          CUpti_CallbackDomain);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEnableDomain");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(enable, subscriber, domain);
}

CUptiResult CUPTIAPI cuptiEnableAllDomains(uint32_t enable,
                                           CUpti_SubscriberHandle subscriber) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(uint32_t, CUpti_SubscriberHandle);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEnableAllDomains");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(enable, subscriber);
}

CUptiResult CUPTIAPI cuptiGetCallbackName(CUpti_CallbackDomain domain,
                                          uint32_t cbid, const char **name) {
  using FuncPtr =
      CUptiResult(CUPTIAPI *)(CUpti_CallbackDomain, uint32_t, const char **);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiGetCallbackName");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(domain, cbid, name);
}

CUptiResult CUPTIAPI
cuptiSetEventCollectionMode(CUcontext context, CUpti_EventCollectionMode mode) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUcontext, CUpti_EventCollectionMode);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiSetEventCollectionMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(context, mode);
}

CUptiResult CUPTIAPI cuptiDeviceGetAttribute(CUdevice device,
                                             CUpti_DeviceAttribute attrib,
                                             size_t *valueSize, void *value) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUdevice, CUpti_DeviceAttribute,
                                          size_t *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiDeviceGetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(device, attrib, valueSize, value);
}

CUptiResult CUPTIAPI cuptiDeviceGetTimestamp(CUcontext context,
                                             uint64_t *timestamp) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUcontext, uint64_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiDeviceGetTimestamp");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(context, timestamp);
}

CUptiResult CUPTIAPI cuptiDeviceGetNumEventDomains(CUdevice device,
                                                   uint32_t *numDomains) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUdevice, uint32_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiDeviceGetNumEventDomains");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(device, numDomains);
}

CUptiResult CUPTIAPI cuptiDeviceEnumEventDomains(
    CUdevice device, size_t *arraySizeBytes, CUpti_EventDomainID *domainArray) {
  using FuncPtr =
      CUptiResult(CUPTIAPI *)(CUdevice, size_t *, CUpti_EventDomainID *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiDeviceEnumEventDomains");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(device, arraySizeBytes, domainArray);
}

CUptiResult CUPTIAPI cuptiDeviceGetEventDomainAttribute(
    CUdevice device, CUpti_EventDomainID eventDomain,
    CUpti_EventDomainAttribute attrib, size_t *valueSize, void *value) {
  using FuncPtr =
      CUptiResult(CUPTIAPI *)(CUdevice, CUpti_EventDomainID,
                              CUpti_EventDomainAttribute, size_t *, void *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cuptiDeviceGetEventDomainAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(device, eventDomain, attrib, valueSize, value);
}

CUptiResult CUPTIAPI cuptiGetNumEventDomains(uint32_t *numDomains) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(uint32_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiGetNumEventDomains");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(numDomains);
}

CUptiResult CUPTIAPI cuptiEnumEventDomains(size_t *arraySizeBytes,
                                           CUpti_EventDomainID *domainArray) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(size_t *, CUpti_EventDomainID *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEnumEventDomains");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(arraySizeBytes, domainArray);
}

CUptiResult CUPTIAPI cuptiEventDomainGetAttribute(
    CUpti_EventDomainID eventDomain, CUpti_EventDomainAttribute attrib,
    size_t *valueSize, void *value) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(
      CUpti_EventDomainID, CUpti_EventDomainAttribute, size_t *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEventDomainGetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(eventDomain, attrib, valueSize, value);
}

CUptiResult CUPTIAPI cuptiEventDomainGetNumEvents(
    CUpti_EventDomainID eventDomain, uint32_t *numEvents) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_EventDomainID, uint32_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEventDomainGetNumEvents");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(eventDomain, numEvents);
}

CUptiResult CUPTIAPI cuptiEventDomainEnumEvents(CUpti_EventDomainID eventDomain,
                                                size_t *arraySizeBytes,
                                                CUpti_EventID *eventArray) {
  using FuncPtr =
      CUptiResult(CUPTIAPI *)(CUpti_EventDomainID, size_t *, CUpti_EventID *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEventDomainEnumEvents");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(eventDomain, arraySizeBytes, eventArray);
}

CUptiResult CUPTIAPI cuptiEventGetAttribute(CUpti_EventID event,
                                            CUpti_EventAttribute attrib,
                                            size_t *valueSize, void *value) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_EventID, CUpti_EventAttribute,
                                          size_t *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEventGetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(event, attrib, valueSize, value);
}

CUptiResult CUPTIAPI cuptiEventGetIdFromName(CUdevice device,
                                             const char *eventName,
                                             CUpti_EventID *event) {
  using FuncPtr =
      CUptiResult(CUPTIAPI *)(CUdevice, const char *, CUpti_EventID *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEventGetIdFromName");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(device, eventName, event);
}

CUptiResult CUPTIAPI cuptiEventGroupCreate(CUcontext context,
                                           CUpti_EventGroup *eventGroup,
                                           uint32_t flags) {
  using FuncPtr =
      CUptiResult(CUPTIAPI *)(CUcontext, CUpti_EventGroup *, uint32_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEventGroupCreate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(context, eventGroup, flags);
}

CUptiResult CUPTIAPI cuptiEventGroupDestroy(CUpti_EventGroup eventGroup) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_EventGroup);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEventGroupDestroy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(eventGroup);
}

CUptiResult CUPTIAPI cuptiEventGroupGetAttribute(
    CUpti_EventGroup eventGroup, CUpti_EventGroupAttribute attrib,
    size_t *valueSize, void *value) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(
      CUpti_EventGroup, CUpti_EventGroupAttribute, size_t *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEventGroupGetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(eventGroup, attrib, valueSize, value);
}

CUptiResult CUPTIAPI cuptiEventGroupSetAttribute(
    CUpti_EventGroup eventGroup, CUpti_EventGroupAttribute attrib,
    size_t valueSize, void *value) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(
      CUpti_EventGroup, CUpti_EventGroupAttribute, size_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEventGroupSetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(eventGroup, attrib, valueSize, value);
}

CUptiResult CUPTIAPI cuptiEventGroupAddEvent(CUpti_EventGroup eventGroup,
                                             CUpti_EventID event) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_EventGroup, CUpti_EventID);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEventGroupAddEvent");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(eventGroup, event);
}

CUptiResult CUPTIAPI cuptiEventGroupRemoveEvent(CUpti_EventGroup eventGroup,
                                                CUpti_EventID event) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_EventGroup, CUpti_EventID);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEventGroupRemoveEvent");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(eventGroup, event);
}

CUptiResult CUPTIAPI
cuptiEventGroupRemoveAllEvents(CUpti_EventGroup eventGroup) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_EventGroup);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEventGroupRemoveAllEvents");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(eventGroup);
}

CUptiResult CUPTIAPI
cuptiEventGroupResetAllEvents(CUpti_EventGroup eventGroup) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_EventGroup);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEventGroupResetAllEvents");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(eventGroup);
}

CUptiResult CUPTIAPI cuptiEventGroupEnable(CUpti_EventGroup eventGroup) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_EventGroup);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEventGroupEnable");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(eventGroup);
}

CUptiResult CUPTIAPI cuptiEventGroupDisable(CUpti_EventGroup eventGroup) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_EventGroup);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEventGroupDisable");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(eventGroup);
}

CUptiResult CUPTIAPI cuptiEventGroupReadEvent(CUpti_EventGroup eventGroup,
                                              CUpti_ReadEventFlags flags,
                                              CUpti_EventID event,
                                              size_t *eventValueBufferSizeBytes,
                                              uint64_t *eventValueBuffer) {
  using FuncPtr =
      CUptiResult(CUPTIAPI *)(CUpti_EventGroup, CUpti_ReadEventFlags,
                              CUpti_EventID, size_t *, uint64_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEventGroupReadEvent");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(eventGroup, flags, event, eventValueBufferSizeBytes,
                  eventValueBuffer);
}

CUptiResult CUPTIAPI cuptiEventGroupReadAllEvents(
    CUpti_EventGroup eventGroup, CUpti_ReadEventFlags flags,
    size_t *eventValueBufferSizeBytes, uint64_t *eventValueBuffer,
    size_t *eventIdArraySizeBytes, CUpti_EventID *eventIdArray,
    size_t *numEventIdsRead) {
  using FuncPtr =
      CUptiResult(CUPTIAPI *)(CUpti_EventGroup, CUpti_ReadEventFlags, size_t *,
                              uint64_t *, size_t *, CUpti_EventID *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEventGroupReadAllEvents");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(eventGroup, flags, eventValueBufferSizeBytes,
                  eventValueBuffer, eventIdArraySizeBytes, eventIdArray,
                  numEventIdsRead);
}

CUptiResult CUPTIAPI cuptiEventGroupSetsCreate(
    CUcontext context, size_t eventIdArraySizeBytes,
    CUpti_EventID *eventIdArray, CUpti_EventGroupSets **eventGroupPasses) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUcontext, size_t, CUpti_EventID *,
                                          CUpti_EventGroupSets **);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEventGroupSetsCreate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(context, eventIdArraySizeBytes, eventIdArray,
                  eventGroupPasses);
}

CUptiResult CUPTIAPI
cuptiEventGroupSetsDestroy(CUpti_EventGroupSets *eventGroupSets) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_EventGroupSets *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEventGroupSetsDestroy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(eventGroupSets);
}

CUptiResult CUPTIAPI
cuptiEventGroupSetEnable(CUpti_EventGroupSet *eventGroupSet) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_EventGroupSet *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEventGroupSetEnable");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(eventGroupSet);
}

CUptiResult CUPTIAPI
cuptiEventGroupSetDisable(CUpti_EventGroupSet *eventGroupSet) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_EventGroupSet *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEventGroupSetDisable");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(eventGroupSet);
}

CUptiResult CUPTIAPI cuptiEnableKernelReplayMode(CUcontext context) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUcontext);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEnableKernelReplayMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(context);
}

CUptiResult CUPTIAPI cuptiDisableKernelReplayMode(CUcontext context) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUcontext);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiDisableKernelReplayMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(context);
}

CUptiResult CUPTIAPI cuptiKernelReplaySubscribeUpdate(
    CUpti_KernelReplayUpdateFunc updateFunc, void *customData) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_KernelReplayUpdateFunc, void *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cuptiKernelReplaySubscribeUpdate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(updateFunc, customData);
}

CUptiResult CUPTIAPI cuptiGetNumMetrics(uint32_t *numMetrics) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(uint32_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiGetNumMetrics");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(numMetrics);
}

CUptiResult CUPTIAPI cuptiEnumMetrics(size_t *arraySizeBytes,
                                      CUpti_MetricID *metricArray) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(size_t *, CUpti_MetricID *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiEnumMetrics");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(arraySizeBytes, metricArray);
}

CUptiResult CUPTIAPI cuptiDeviceGetNumMetrics(CUdevice device,
                                              uint32_t *numMetrics) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUdevice, uint32_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiDeviceGetNumMetrics");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(device, numMetrics);
}

CUptiResult CUPTIAPI cuptiDeviceEnumMetrics(CUdevice device,
                                            size_t *arraySizeBytes,
                                            CUpti_MetricID *metricArray) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUdevice, size_t *, CUpti_MetricID *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiDeviceEnumMetrics");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(device, arraySizeBytes, metricArray);
}

CUptiResult CUPTIAPI cuptiMetricGetAttribute(CUpti_MetricID metric,
                                             CUpti_MetricAttribute attrib,
                                             size_t *valueSize, void *value) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_MetricID, CUpti_MetricAttribute,
                                          size_t *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiMetricGetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(metric, attrib, valueSize, value);
}

CUptiResult CUPTIAPI cuptiMetricGetIdFromName(CUdevice device,
                                              const char *metricName,
                                              CUpti_MetricID *metric) {
  using FuncPtr =
      CUptiResult(CUPTIAPI *)(CUdevice, const char *, CUpti_MetricID *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiMetricGetIdFromName");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(device, metricName, metric);
}

CUptiResult CUPTIAPI cuptiMetricGetNumEvents(CUpti_MetricID metric,
                                             uint32_t *numEvents) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_MetricID, uint32_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiMetricGetNumEvents");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(metric, numEvents);
}

CUptiResult CUPTIAPI cuptiMetricEnumEvents(CUpti_MetricID metric,
                                           size_t *eventIdArraySizeBytes,
                                           CUpti_EventID *eventIdArray) {
  using FuncPtr =
      CUptiResult(CUPTIAPI *)(CUpti_MetricID, size_t *, CUpti_EventID *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiMetricEnumEvents");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(metric, eventIdArraySizeBytes, eventIdArray);
}

CUptiResult CUPTIAPI cuptiMetricGetNumProperties(CUpti_MetricID metric,
                                                 uint32_t *numProp) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_MetricID, uint32_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiMetricGetNumProperties");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(metric, numProp);
}

CUptiResult CUPTIAPI
cuptiMetricEnumProperties(CUpti_MetricID metric, size_t *propIdArraySizeBytes,
                          CUpti_MetricPropertyID *propIdArray) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_MetricID, size_t *,
                                          CUpti_MetricPropertyID *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiMetricEnumProperties");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(metric, propIdArraySizeBytes, propIdArray);
}

CUptiResult CUPTIAPI
cuptiMetricGetRequiredEventGroupSets(CUcontext context, CUpti_MetricID metric,
                                     CUpti_EventGroupSets **eventGroupSets) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUcontext, CUpti_MetricID,
                                          CUpti_EventGroupSets **);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cuptiMetricGetRequiredEventGroupSets");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(context, metric, eventGroupSets);
}

CUptiResult CUPTIAPI cuptiMetricCreateEventGroupSets(
    CUcontext context, size_t metricIdArraySizeBytes,
    CUpti_MetricID *metricIdArray, CUpti_EventGroupSets **eventGroupPasses) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUcontext, size_t, CUpti_MetricID *,
                                          CUpti_EventGroupSets **);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiMetricCreateEventGroupSets");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(context, metricIdArraySizeBytes, metricIdArray,
                  eventGroupPasses);
}

CUptiResult CUPTIAPI cuptiMetricGetValue(CUdevice device, CUpti_MetricID metric,
                                         size_t eventIdArraySizeBytes,
                                         CUpti_EventID *eventIdArray,
                                         size_t eventValueArraySizeBytes,
                                         uint64_t *eventValueArray,
                                         uint64_t timeDuration,
                                         CUpti_MetricValue *metricValue) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUdevice, CUpti_MetricID, size_t,
                                          CUpti_EventID *, size_t, uint64_t *,
                                          uint64_t, CUpti_MetricValue *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiMetricGetValue");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(device, metric, eventIdArraySizeBytes, eventIdArray,
                  eventValueArraySizeBytes, eventValueArray, timeDuration,
                  metricValue);
}

CUptiResult CUPTIAPI cuptiMetricGetValue2(
    CUpti_MetricID metric, size_t eventIdArraySizeBytes,
    CUpti_EventID *eventIdArray, size_t eventValueArraySizeBytes,
    uint64_t *eventValueArray, size_t propIdArraySizeBytes,
    CUpti_MetricPropertyID *propIdArray, size_t propValueArraySizeBytes,
    uint64_t *propValueArray, CUpti_MetricValue *metricValue) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(
      CUpti_MetricID, size_t, CUpti_EventID *, size_t, uint64_t *, size_t,
      CUpti_MetricPropertyID *, size_t, uint64_t *, CUpti_MetricValue *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiMetricGetValue2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(metric, eventIdArraySizeBytes, eventIdArray,
                  eventValueArraySizeBytes, eventValueArray,
                  propIdArraySizeBytes, propIdArray, propValueArraySizeBytes,
                  propValueArray, metricValue);
}

CUptiResult CUPTIAPI cuptiGetTimestamp(uint64_t *timestamp) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(uint64_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiGetTimestamp");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(timestamp);
}

CUptiResult CUPTIAPI cuptiGetContextId(CUcontext context, uint32_t *contextId) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUcontext, uint32_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiGetContextId");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(context, contextId);
}

CUptiResult CUPTIAPI cuptiGetStreamId(CUcontext context, CUstream stream,
                                      uint32_t *streamId) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUcontext, CUstream, uint32_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiGetStreamId");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(context, stream, streamId);
}

CUptiResult CUPTIAPI cuptiGetStreamIdEx(CUcontext context, CUstream stream,
                                        uint8_t perThreadStream,
                                        uint32_t *streamId) {
  using FuncPtr =
      CUptiResult(CUPTIAPI *)(CUcontext, CUstream, uint8_t, uint32_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiGetStreamIdEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(context, stream, perThreadStream, streamId);
}

CUptiResult CUPTIAPI cuptiGetDeviceId(CUcontext context, uint32_t *deviceId) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUcontext, uint32_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiGetDeviceId");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(context, deviceId);
}

CUptiResult CUPTIAPI cuptiActivityEnable(CUpti_ActivityKind kind) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_ActivityKind);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiActivityEnable");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(kind);
}

CUptiResult CUPTIAPI cuptiActivityDisable(CUpti_ActivityKind kind) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_ActivityKind);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiActivityDisable");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(kind);
}

CUptiResult CUPTIAPI cuptiActivityEnableContext(CUcontext context,
                                                CUpti_ActivityKind kind) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUcontext, CUpti_ActivityKind);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiActivityEnableContext");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(context, kind);
}

CUptiResult CUPTIAPI cuptiActivityDisableContext(CUcontext context,
                                                 CUpti_ActivityKind kind) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUcontext, CUpti_ActivityKind);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiActivityDisableContext");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(context, kind);
}

CUptiResult CUPTIAPI cuptiActivityGetNumDroppedRecords(CUcontext context,
                                                       uint32_t streamId,
                                                       size_t *dropped) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUcontext, uint32_t, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cuptiActivityGetNumDroppedRecords");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(context, streamId, dropped);
}

CUptiResult CUPTIAPI cuptiActivityGetNextRecord(uint8_t *buffer,
                                                size_t validBufferSizeBytes,
                                                CUpti_Activity **record) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(uint8_t *, size_t, CUpti_Activity **);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiActivityGetNextRecord");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(buffer, validBufferSizeBytes, record);
}

CUptiResult CUPTIAPI cuptiActivityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc funcBufferRequested,
    CUpti_BuffersCallbackCompleteFunc funcBufferCompleted) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_BuffersCallbackRequestFunc,
                                          CUpti_BuffersCallbackCompleteFunc);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiActivityRegisterCallbacks");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(funcBufferRequested, funcBufferCompleted);
}

CUptiResult CUPTIAPI cuptiActivityFlush(CUcontext context, uint32_t streamId,
                                        uint32_t flag) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUcontext, uint32_t, uint32_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiActivityFlush");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(context, streamId, flag);
}

CUptiResult CUPTIAPI cuptiActivityFlushAll(uint32_t flag) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(uint32_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiActivityFlushAll");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(flag);
}

CUptiResult CUPTIAPI cuptiActivityGetAttribute(CUpti_ActivityAttribute attr,
                                               size_t *valueSize, void *value) {
  using FuncPtr =
      CUptiResult(CUPTIAPI *)(CUpti_ActivityAttribute, size_t *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiActivityGetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(attr, valueSize, value);
}

CUptiResult CUPTIAPI cuptiActivitySetAttribute(CUpti_ActivityAttribute attr,
                                               size_t *valueSize, void *value) {
  using FuncPtr =
      CUptiResult(CUPTIAPI *)(CUpti_ActivityAttribute, size_t *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiActivitySetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(attr, valueSize, value);
}

CUptiResult CUPTIAPI cuptiActivityConfigureUnifiedMemoryCounter(
    CUpti_ActivityUnifiedMemoryCounterConfig *config, uint32_t count) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(
      CUpti_ActivityUnifiedMemoryCounterConfig *, uint32_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cuptiActivityConfigureUnifiedMemoryCounter");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(config, count);
}

CUptiResult CUPTIAPI
cuptiGetAutoBoostState(CUcontext context, CUpti_ActivityAutoBoostState *state) {
  using FuncPtr =
      CUptiResult(CUPTIAPI *)(CUcontext, CUpti_ActivityAutoBoostState *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiGetAutoBoostState");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(context, state);
}

CUptiResult CUPTIAPI cuptiActivityConfigurePCSampling(
    CUcontext ctx, CUpti_ActivityPCSamplingConfig *config) {
  using FuncPtr =
      CUptiResult(CUPTIAPI *)(CUcontext, CUpti_ActivityPCSamplingConfig *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cuptiActivityConfigurePCSampling");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(ctx, config);
}

CUptiResult CUPTIAPI cuptiGetLastError(void) {
  using FuncPtr = CUptiResult(CUPTIAPI *)();
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiGetLastError");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr();
}

CUptiResult CUPTIAPI cuptiSetThreadIdType(CUpti_ActivityThreadIdType type) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_ActivityThreadIdType);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiSetThreadIdType");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(type);
}

CUptiResult CUPTIAPI cuptiGetThreadIdType(CUpti_ActivityThreadIdType *type) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUpti_ActivityThreadIdType *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiGetThreadIdType");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(type);
}

CUptiResult CUPTIAPI cuptiComputeCapabilitySupported(int major, int minor,
                                                     int *support) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiComputeCapabilitySupported");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(major, minor, support);
}

CUptiResult CUPTIAPI cuptiDeviceSupported(CUdevice dev, int *support) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(CUdevice, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiDeviceSupported");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dev, support);
}

CUptiResult CUPTIAPI cuptiFinalize(void) {
  using FuncPtr = CUptiResult(CUPTIAPI *)();
  static auto func_ptr = LoadSymbol<FuncPtr>("cuptiFinalize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr();
}

CUptiResult CUPTIAPI cuptiActivityPushExternalCorrelationId(
    CUpti_ExternalCorrelationKind kind, uint64_t id) {
  using FuncPtr =
      CUptiResult(CUPTIAPI *)(CUpti_ExternalCorrelationKind, uint64_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cuptiActivityPushExternalCorrelationId");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(kind, id);
}

CUptiResult CUPTIAPI cuptiActivityPopExternalCorrelationId(
    CUpti_ExternalCorrelationKind kind, uint64_t *lastId) {
  using FuncPtr =
      CUptiResult(CUPTIAPI *)(CUpti_ExternalCorrelationKind, uint64_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cuptiActivityPopExternalCorrelationId");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(kind, lastId);
}

CUptiResult CUPTIAPI cuptiActivityEnableLatencyTimestamps(uint8_t enable) {
  using FuncPtr = CUptiResult(CUPTIAPI *)(uint8_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cuptiActivityEnableLatencyTimestamps");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(enable);
}

}  // extern "C"
