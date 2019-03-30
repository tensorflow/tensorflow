/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_TOOLS_EVALUATION_EVALUATION_STAGE_H_
#define TENSORFLOW_LITE_TOOLS_EVALUATION_EVALUATION_STAGE_H_

#include <functional>
#include <map>
#include <regex>  // NOLINT
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

namespace tflite {
namespace evaluation {

class EvaluationStage;

typedef std::function<std::unique_ptr<EvaluationStage>(
    const EvaluationStageConfig&)>
    FactoryFunc;

// Superclass for a single stage of an EvaluationPipeline.
// Provides basic functionality for construction and accessing
// initializers/inputs/outputs.
// Every subclass of EvaluationStage will define its own behavior by specifying
// appropriate accessor TAGs and implementing the Init, Run and Close methods.
class EvaluationStage {
 public:
  // Initializes an EvaluationStage. Returns false if initialization failed,
  // true otherwise.
  // Should be called only once, before any call to Run().
  // object_map should contain {initializer name : object pointer} mappings
  // required for initialization.
  //
  // NOTE: EvaluationStage will not take ownership of any elements of
  // object_map.
  bool Init(absl::flat_hash_map<std::string, void*>& object_map);

  // An individual run of the EvaluationStage. Returns false if there was a
  // failure, true otherwise.
  // Init() should be called before any calls to run().
  // Inputs are acquired from and outputs are written to the incoming
  // object_map, using appropriate TAGs.
  //
  // NOTE: The EvaluationStage should maintain ownership of outputs it
  // populates into object_map. Ownership of inputs must be maintained
  // elsewhere.
  virtual bool Run(absl::flat_hash_map<std::string, void*>& object_map) = 0;

  // Returns the latest metrics based on all Run() calls made so far.
  virtual EvaluationStageMetrics LatestMetrics() = 0;

  // The canonical way to instantiate EvaluationStages.
  // Remember to call <classname>_ENABLE() first.
  static std::unique_ptr<EvaluationStage> Create(
      const EvaluationStageConfig& config) {
    if (!config.has_specification() ||
        !config.specification().has_process_class()) {
      LOG(ERROR) << "Process specification not present in config: "
                 << config.name();
      return nullptr;
    }
    auto& factory_ptr =
        (*GetFactoryMapPtr())[config.specification().process_class()];
    if (!factory_ptr) return nullptr;
    return factory_ptr(config);
  }

  // Used by DEFINE_REGISTRATION.
  // This method takes ownership of factory.
  // Should only be used via DEFINE_REGISTRATION macro.
  static void RegisterStage(const ProcessClass& process_class,
                            FactoryFunc class_factory) {
    (*GetFactoryMapPtr())[process_class] = std::move(class_factory);
  }

  virtual ~EvaluationStage() = default;

 protected:
  // Constructs an EvaluationStage.
  // Each subclass constructor must invoke this constructor.
  //
  // NOTE: Do NOT use constructors to obtain new EvaluationStages. Use
  // EvaluationStage::Create instead.
  explicit EvaluationStage(const EvaluationStageConfig& config)
      : config_(config) {}

  // Class-specific initialization, to be overridden by EvaluationStage
  // sub-classes. Gets called in EvaluationStage::Init().
  //
  // NOTE: This object should not take ownership of any elements of object_map.
  virtual bool DoInit(absl::flat_hash_map<std::string, void*>& object_map) = 0;

  // The three following functions return the initializer/input/output TAGs used
  // by an EvaluationStage. These should be mapped to meaningful names in the
  // EvaluationStageConfig, and to required objects during calls to Init/Run.
  // Format for TAGs: [A-Z0-9_]+ (Uppercase letters, numbers, "_")
  // Refer docs in tflite.evaluation.EvaluationStageConfig for more information.

  // Returns the expected initializer TAGs.
  virtual std::vector<std::string> GetInitializerTags() = 0;

  // Returns the expected input TAGs.
  virtual std::vector<std::string> GetInputTags() = 0;

  // Returns the expected output TAGs.
  virtual std::vector<std::string> GetOutputTags() = 0;

  // Populates a pointer to the object corresponding to provided TAG.
  // Returns true if success, false otherwise.
  // object_map contain a {name : object pointer} mapping, with the
  // name being mapped to the expected TAG in the EvaluationStageConfig.
  // NOTE: object pointer must be non-NULL.
  template <class T>
  bool GetObjectFromTag(const std::string& tag,
                        absl::flat_hash_map<std::string, void*>& object_map,
                        T** object_ptr) {
    *object_ptr = nullptr;
    // Find name corresponding to TAG.
    auto mapping_iter = tags_to_names_map_.find(tag);
    if (mapping_iter == tags_to_names_map_.end()) {
      LOG(ERROR) << "Unexpected TAG to GetObjectFromTag: " << tag;
      return false;
    }
    const std::string& expected_name = mapping_iter->second;

    // Find object from name.
    auto object_iter = object_map.find(expected_name);
    if (object_iter == object_map.end()) {
      LOG(ERROR) << "Could not find object for name: " << expected_name;
      return false;
    }
    if (!object_iter->second) {
      LOG(ERROR) << "Found null pointer for name: " << expected_name;
      return false;
    }
    *object_ptr = static_cast<T*>(object_iter->second);
    return true;
  }

  // Maps the appropriate name to a given object in object_map. The name is
  // derived from mappings provided in the EvaluationStageConfig.
  // Returns false if tag is invalid, true otherwise.
  //
  // NOTE: The EvaluationStage must maintain ownership of object for the
  // lifetime of object_map
  bool AssignObjectToTag(const std::string& tag, void* object_ptr,
                         absl::flat_hash_map<std::string, void*>& object_map) {
    // Find name corresponding to TAG.
    auto mapping_iter = tags_to_names_map_.find(tag);
    if (mapping_iter == tags_to_names_map_.end()) {
      LOG(ERROR) << "Unexpected TAG to AssignObjectToTag: " << tag;
      return false;
    }
    const std::string& expected_name = mapping_iter->second;

    object_map[expected_name] = object_ptr;
    return true;
  }

  EvaluationStageConfig config_;

 private:
  // Verifies that all TAGs from expected_tags are present in
  // tag_to_name_mappings, and then populates tags_to_names_map_ with the
  // appropriate entries. Returns false in case any TAG/mapping is invalid, true
  // otherwise.
  // expected_tags should be a list of TAG-strings.
  // tag_to_name_mappings should be RepeatedPtrField of strings mapping TAGs to
  // names in the form "SOME_TAG:some_name".
  bool ProcessExpectedTags(const std::vector<std::string>& expected_tags,
                           std::vector<std::string>& tag_to_name_mappings);

  static std::map<ProcessClass, FactoryFunc>* GetFactoryMapPtr() {
    return process_class_to_factory_map_;
  }

  // Used by factories.
  static std::map<ProcessClass, FactoryFunc>* process_class_to_factory_map_;

  // Maps expected TAGs to their names as defined by the EvaluationStageConfig.
  absl::flat_hash_map<std::string, std::string> tags_to_names_map_;

  // To ensure correct formatting in the config.
  const std::regex kTagNameMappingPattern{"^([A-Z0-9_]+):([a-z0-9_]+)$",
                                          std::regex::optimize};

  // To ensure correct formatting in TAG names.
  const std::regex kTagPattern{"^[A-Z0-9_]+$", std::regex::optimize};
};

// Add this to headers of new EvaluationStages.
#define DECLARE_FACTORY(classname) void classname##_ENABLE();

// Add this to implementation files of new EvaluationStages.
// Call <stage_name>_ENABLE() before using EvaluationStage::Create for the
// class.
#define DEFINE_FACTORY(classname, processclass)                                \
  void classname##_ENABLE() {                                                  \
    FactoryFunc classname##Factory = [](const EvaluationStageConfig& config) { \
      return absl::make_unique<classname>(config);                             \
    };                                                                         \
    EvaluationStage::RegisterStage(processclass, classname##Factory);          \
  }

// Use this to assign a non-nullptr pointer to tag in object_map.
#define ASSIGN_OBJECT(tag, ptr, object_map)       \
  if (!AssignObjectToTag(tag, ptr, object_map)) { \
    return false;                                 \
  }

// Use this to obtain pointers to required object.
// Will return false if name corresponding to tag is not found, or if the
// pointer found is nullptr.
#define GET_OBJECT(tag, object_map, location)         \
  if (!GetObjectFromTag(tag, object_map, location)) { \
    return false;                                     \
  }

}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_EVALUATION_STAGE_H_
