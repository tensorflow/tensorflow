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
// This file is empty to ensure that a specialized implementation of
// micro_time.h is used (instead of the default implementation from
// tensorflow/lite/micro/micro_time.cc).
//
// The actual target-specific implementation of micro_time.h is in
// system_setup.cc since that allows us to consolidate all the target-specific
// specializations into one source file.
//
//
