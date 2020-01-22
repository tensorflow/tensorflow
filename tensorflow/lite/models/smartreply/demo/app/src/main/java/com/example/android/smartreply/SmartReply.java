/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

package com.example.android.smartreply;

import android.support.annotation.Keep;

/**
 * SmartReply contains predicted message, and confidence.
 *
 * <p>NOTE: this class used by JNI, class name and constructor should not be obfuscated.
 */
@Keep
public class SmartReply {

  private final String text;
  private final float score;

  @Keep
  public SmartReply(String text, float score) {
    this.text = text;
    this.score = score;
  }

  public String getText() {
    return text;
  }

  public float getScore() {
    return score;
  }
}
