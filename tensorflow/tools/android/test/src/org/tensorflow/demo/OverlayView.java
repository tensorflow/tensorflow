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

package org.tensorflow.demo;

import android.content.Context;
import android.graphics.Canvas;
import android.util.AttributeSet;
import android.view.View;
import java.util.LinkedList;
import java.util.List;

/**
 * A simple View providing a render callback to other classes.
 */
public class OverlayView extends View {
  private final List<DrawCallback> callbacks = new LinkedList<DrawCallback>();

  public OverlayView(final Context context, final AttributeSet attrs) {
    super(context, attrs);
  }

  /**
   * Interface defining the callback for client classes.
   */
  public interface DrawCallback {
    public void drawCallback(final Canvas canvas);
  }

  public void addCallback(final DrawCallback callback) {
    callbacks.add(callback);
  }

  @Override
  public synchronized void draw(final Canvas canvas) {
    for (final DrawCallback callback : callbacks) {
      callback.drawCallback(canvas);
    }
  }
}
