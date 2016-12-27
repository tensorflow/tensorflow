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

package org.tensorflow.util;

import java.util.concurrent.atomic.AtomicIntegerFieldUpdater;

/**
 * An abstract reference-counted object.
 */
public abstract class AbstractRefCounted implements RefCounted {

  /**
   * Invoked when the reference count reaches zero.
   *
   * Subclasses should implememt this method to deallocate underlying native objects.
   */
  protected abstract void deallocate();

  @Override
  public int refCount() {
        return refCnt;
    }

  @Override
  public RefCounted ref() {
    for (; ; ) {
      int current = this.refCnt;
      if (current < 1) {
        throw new IllegalStateException("ref() cannot be called after refCount() reaches zero");
      }
      if (refCntUpdater.compareAndSet(this, current, current + 1)) {
        break;
      }
    }

    return this;
  }

  @Override
  public boolean unref() {
        for(;;) {
            int current = this.refCnt;
            if(current < 1) {
                throw new IllegalStateException("unref() cannot be called after refCount() reaches zero");
            }
            if (refCntUpdater.compareAndSet(this, current, current - 1)) {
                if (current == 1) {
                    deallocate();
                    return true;
                }
                return false;
            }
        }
  }

  private static final AtomicIntegerFieldUpdater<AbstractRefCounted> refCntUpdater =
          AtomicIntegerFieldUpdater.newUpdater(AbstractRefCounted.class, "refCnt");
  private volatile int refCnt = 1;
}
