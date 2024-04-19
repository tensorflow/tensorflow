/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/metrics/types_util.h"

#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "tensorflow/lite/python/metrics/converter_error_data.pb.h"

namespace mlir {
namespace TFL {
namespace {

// Extracts information from mlir::FileLineColLoc to the proto message
// tflite::metrics::ConverterErrorData::FileLoc.
void ExtractFileLine(const FileLineColLoc& loc,
                     tflite::metrics::ConverterErrorData::FileLoc* fileline) {
  fileline->set_filename(loc.getFilename().str());
  fileline->set_line(loc.getLine());
  fileline->set_column(loc.getColumn());
}

// Defines a child class of Location to access its protected members.
class LocationExtractor : public Location {
 public:
  explicit LocationExtractor(const Location& loc) : Location(loc) {}

  void Extract(tflite::metrics::ConverterErrorData* error_data) {
    using tflite::metrics::ConverterErrorData;
    auto mutable_location = error_data->mutable_location();

    llvm::TypeSwitch<LocationAttr>(impl)
        .Case<OpaqueLoc>([&](OpaqueLoc loc) {
          LocationExtractor(loc.getFallbackLocation()).Extract(error_data);
        })
        .Case<UnknownLoc>([&](UnknownLoc loc) {
          mutable_location->set_type(ConverterErrorData::UNKNOWNLOC);
        })
        .Case<FileLineColLoc>([&](FileLineColLoc loc) {
          if (!mutable_location->has_type()) {
            mutable_location->set_type(ConverterErrorData::CALLSITELOC);
          }
          auto new_call = mutable_location->mutable_call()->Add();
          ExtractFileLine(loc, new_call->mutable_source());
        })
        .Case<NameLoc>([&](NameLoc loc) {
          if (!mutable_location->has_type()) {
            mutable_location->set_type(ConverterErrorData::NAMELOC);
          }

          auto new_call = mutable_location->mutable_call()->Add();
          new_call->set_name(loc.getName().str());
          // Add child as the source location.
          auto child_loc = loc.getChildLoc();
          if (child_loc.isa<FileLineColLoc>()) {
            auto typed_child_loc = child_loc.dyn_cast<FileLineColLoc>();
            ExtractFileLine(typed_child_loc, new_call->mutable_source());
          }
        })
        .Case<CallSiteLoc>([&](CallSiteLoc loc) {
          mutable_location->set_type(ConverterErrorData::CALLSITELOC);
          LocationExtractor(loc.getCallee()).Extract(error_data);
          LocationExtractor(loc.getCaller()).Extract(error_data);
        })
        .Case<FusedLoc>([&](FusedLoc loc) {
          auto locations = loc.getLocations();
          size_t num_locs = locations.size();
          // Skip the first location if it stores information for propagating
          // op_type metadata.
          if (num_locs > 0) {
            if (auto name_loc = locations[0].dyn_cast<mlir::NameLoc>()) {
              if (name_loc.getName().strref().ends_with(":")) {
                if (num_locs == 2) {
                  return LocationExtractor(locations[1]).Extract(error_data);
                } else if (num_locs > 2) {
                  locations = {locations.begin() + 1, locations.end()};
                }
              }
            }
          }

          mutable_location->set_type(ConverterErrorData::FUSEDLOC);
          llvm::interleave(
              locations,
              [&](Location l) { LocationExtractor(l).Extract(error_data); },
              [&]() {});
        });
  }
};
}  // namespace

tflite::metrics::ConverterErrorData NewConverterErrorData(
    const std ::string& pass_name, const std::string& error_message,
    tflite::metrics::ConverterErrorData::ErrorCode error_code,
    const std::string& op_name, const Location& location) {
  using tflite::metrics::ConverterErrorData;
  ConverterErrorData error;
  if (!pass_name.empty()) {
    error.set_subcomponent(pass_name);
  }

  if (!error_message.empty()) {
    error.set_error_message(error_message);
  }

  if (!op_name.empty()) {
    error.mutable_operator_()->set_name(op_name);
  }

  error.set_error_code(error_code);
  LocationExtractor(location).Extract(&error);
  return error;
}

}  // namespace TFL
}  // namespace mlir
