// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <android/log.h>

#include "android_native_app_glue.h"
#include "animal_generated.h" // Includes "flatbuffers/flatbuffers.h".

void android_main(android_app *) {
  flatbuffers::FlatBufferBuilder builder;
  auto name = builder.CreateString("Dog");
  auto sound = builder.CreateString("Bark");
  auto animal_buffer = sample::CreateAnimal(builder, name, sound);
  builder.Finish(animal_buffer);

  // We now have a FlatBuffer that can be stored on disk or sent over a network.

  // ...Code to store on disk or send over a network goes here...

  // Instead, we're going to access it immediately, as if we just recieved this.

  auto animal = sample::GetAnimal(builder.GetBufferPointer());

  assert(animal->name()->str() == "Dog");
  assert(animal->sound()->str() == "Bark");
  (void)animal; // To silence "Unused Variable" warnings.

  __android_log_print(ANDROID_LOG_INFO, "FlatBufferSample",
      "FlatBuffer successfully created and verified.");
}
