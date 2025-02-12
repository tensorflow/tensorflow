#!/bin/bash
# Copyright 2022 Google LLC All Rights Reserved.
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

# Generates a handy index.html with a bunch of Kokoro links for GitHub
# presubmits.
# Usage: generate_index_html.sh /path/to/output/index.html

tee "$1" <<EOF
<html>
<head>
<title>#$KOKORO_GITHUB_PULL_REQUEST_NUMBER_xla | $(basename "$KOKORO_JOB_NAME")</title>
</head>
<body>
<h1>XLA Job Logs and Links</h1>
<h2>Job Details</h2>
<ul>
<li>Job name: $KOKORO_JOB_NAME</li>
<li>Job pool: $KOKORO_JOB_POOL</li>
<li>Job ID: $KOKORO_BUILD_ID</li>
<li>Current HEAD Piper Changelist (may be empty): cl/${KOKORO_PIPER_CHANGELIST:-not available}</li>
<li>Pull Request Number: $KOKORO_GITHUB_PULL_REQUEST_NUMBER_xla</li>
<li>Pull Request Link: <a href="$KOKORO_GITHUB_PULL_REQUEST_URL_xla">$KOKORO_GITHUB_PULL_REQUEST_URL_xla</a></li>
<li>Commit: $KOKORO_GIT_COMMIT_xla</li>
</ul>
<h2>Googlers-Only Links</h2>
<ul>
<li><a href="http://sponge2/$KOKORO_BUILD_ID">Sponge2</a></li>
<li><a href="http://sponge/target:$KOKORO_JOB_NAME">Sponge - recent jobs</a></li>
</ul>
<h2>Non-Googler Links</h2>
<ul>
<li><a href="https://source.cloud.google.com/results/invocations/$KOKORO_BUILD_ID">ResultStore</a></li>
</ul>
</body></html>
EOF
