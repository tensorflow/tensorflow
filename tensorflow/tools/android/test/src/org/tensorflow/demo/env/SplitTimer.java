/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.demo.env;

import android.os.SystemClock;

/**
 * A simple utility timer for measuring CPU time and wall-clock splits.
 */
public class SplitTimer {
  private final Logger logger;

  private long lastWallTime;
  private long lastCpuTime;

  public SplitTimer(final String name) {
    logger = new Logger(name);
    newSplit();
  }

  public void newSplit() {
    lastWallTime = SystemClock.uptimeMillis();
    lastCpuTime = SystemClock.currentThreadTimeMillis();
  }

  public void endSplit(final String splitName) {
    final long currWallTime = SystemClock.uptimeMillis();
    final long currCpuTime = SystemClock.currentThreadTimeMillis();

    logger.i(
        "%s: cpu=%dms wall=%dms",
        splitName, currCpuTime - lastCpuTime, currWallTime - lastWallTime);

    lastWallTime = currWallTime;
    lastCpuTime = currCpuTime;
  }
}
