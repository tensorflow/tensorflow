#!/usr/bin/env bash
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e

export LAUNCHPAD_CHROME=${LAUNCHPAD_CHROME:-$(which chromium-browser)}

cd tensorflow/tensorboard

# Install all js dependencies (tooling via npm, frontend assets via bower)
npm install
npm install bower gulp
./node_modules/bower/bin/bower install

# Compile the frontend code
./node_modules/gulp/bin/gulp.js compile.all

# Run wct in headless chrome using xvfb
xvfb-run ./node_modules/web-component-tester/bin/wct --skip-plugin=sauce

