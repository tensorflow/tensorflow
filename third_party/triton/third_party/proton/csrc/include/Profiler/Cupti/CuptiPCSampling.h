#ifndef PROTON_PROFILER_CUPTI_PC_SAMPLING_H_
#define PROTON_PROFILER_CUPTI_PC_SAMPLING_H_

#include "CuptiProfiler.h"
#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/CuptiApi.h"
#include "Utility/Map.h"
#include "Utility/Singleton.h"
#include <atomic>
#include <mutex>

namespace proton {

struct CubinData {
  size_t cubinCrc;
  const char *cubin;
  size_t cubinSize;

  struct LineInfoKey {
    uint32_t functionIndex;
    uint64_t pcOffset;

    bool operator<(const LineInfoKey &other) const {
      return functionIndex < other.functionIndex ||
             (functionIndex == other.functionIndex &&
              pcOffset < other.pcOffset);
    }
  };

  struct LineInfoValue {
    uint32_t lineNumber{};
    const std::string functionName{};
    const std::string dirName{};
    const std::string fileName{};

    LineInfoValue() = default;

    LineInfoValue(uint32_t lineNumber, const std::string &functionName,
                  const std::string &dirName, const std::string &fileName)
        : lineNumber(lineNumber), functionName(functionName), dirName(dirName),
          fileName(fileName) {}
  };

  std::map<LineInfoKey, LineInfoValue> lineInfo;
};

struct ConfigureData {
  ConfigureData() = default;

  ~ConfigureData() {
    if (stallReasonNames) {
      for (size_t i = 0; i < numStallReasons; i++) {
        if (stallReasonNames[i])
          std::free(stallReasonNames[i]);
      }
      std::free(stallReasonNames);
    }
    if (stallReasonIndices)
      std::free(stallReasonIndices);
    if (pcSamplingData.pPcData) {
      for (size_t i = 0; i < numValidStallReasons; ++i) {
        std::free(pcSamplingData.pPcData[i].stallReason);
      }
      std::free(pcSamplingData.pPcData);
    }
  }

  void initialize(CUcontext context);

  CUpti_PCSamplingConfigurationInfo configureStallReasons();
  CUpti_PCSamplingConfigurationInfo configureSamplingPeriod();
  CUpti_PCSamplingConfigurationInfo configureSamplingBuffer();
  CUpti_PCSamplingConfigurationInfo configureScratchBuffer();
  CUpti_PCSamplingConfigurationInfo configureHardwareBufferSize();
  CUpti_PCSamplingConfigurationInfo configureStartStopControl();
  CUpti_PCSamplingConfigurationInfo configureCollectionMode();

  // The amount of data reserved on the GPU
  static constexpr size_t HardwareBufferSize = 128 * 1024 * 1024;
  // The amount of data copied from the hardware buffer each time
  static constexpr size_t ScratchBufferSize = 16 * 1024 * 1024;
  // The number of PCs copied from the scratch buffer each time
  static constexpr size_t DataBufferPCCount = 1024;
  // The sampling period in cycles = 2^frequency
  static constexpr uint32_t DefaultFrequency = 10;

  CUcontext context{};
  uint32_t contextId;
  uint32_t numStallReasons{};
  uint32_t numValidStallReasons{};
  char **stallReasonNames{};
  uint32_t *stallReasonIndices{};
  std::map<size_t, size_t> stallReasonIndexToMetricIndex{};
  std::set<size_t> notIssuedStallReasonIndices{};
  CUpti_PCSamplingData pcSamplingData{};
  // The memory storing configuration information has to be kept alive during
  // the profiling session
  std::vector<CUpti_PCSamplingConfigurationInfo> configurationInfos;
};

class CuptiPCSampling : public Singleton<CuptiPCSampling> {

public:
  CuptiPCSampling() = default;
  virtual ~CuptiPCSampling() = default;

  void initialize(CUcontext context);

  void start(CUcontext context);

  void stop(CUcontext context, uint64_t externId, bool isAPI);

  void finalize(CUcontext context);

  void loadModule(const char *cubin, size_t cubinSize);

  void unloadModule(const char *cubin, size_t cubinSize);

private:
  ConfigureData *getConfigureData(uint32_t contextId);

  CubinData *getCubinData(uint64_t cubinCrc);

  void processPCSamplingData(ConfigureData *configureData, uint64_t externId,
                             bool isAPI);

  ThreadSafeMap<uint32_t, ConfigureData> contextIdToConfigureData;
  // In case the same cubin is loaded multiple times, we need to keep track of
  // all of them
  ThreadSafeMap<size_t, std::pair<CubinData, /*count=*/size_t>>
      cubinCrcToCubinData;
  ThreadSafeSet<uint32_t> contextInitialized;

  std::atomic<bool> pcSamplingStarted{false};
  std::mutex pcSamplingMutex{};
  std::mutex contextMutex{};
};

} // namespace proton

#endif // PROTON_PROFILER_CUPTI_PC_SAMPLING_H_
