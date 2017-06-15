/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/**
 * A class that allows marking promises as cancelled.
 *
 * This can be useful to, e.g., prevent old network requests from
 * stomping new ones and writing bad data.
 *
 * Usage:
 *
 *     const canceller = new Canceller();
 *     let myPromise: Promise<Foo> = getPromise();
 *     myPromise.then(canceller.cancellable(({value, cancelled} => {
 *       if (cancelled) {
 *         console.warn("Don't make promises you can't keep >:-{");
 *       }
 *       console.log("Enjoy your value:", value);
 *     }));
 *
 *     // If `myPromise` is resolved now, then `cancelled` will be `false`.
 *     canceller.cancelAll();
 *     // If `myPromise` is resolved now, then `cancelled` will be `true`.
 */
export class Canceller {
  /**
   * How many times has `cancelAll` been called?
   */
  private cancellationCount = 0;

  /**
   * Create a cancellable task. This returns a new function that, when
   * invoked, will pass its argument to the provided function as well as
   * a `cancelled` argument. This argument will be `false` unless and
   * until `cancelAll` is invoked after the creation of this task.
   */
  public cancellable<T, U>(f: (result: {value: T, cancelled: boolean}) => U):
      (T) => U {
    const originalCancellationCount = this.cancellationCount;
    return (value) => {
      const cancelled = this.cancellationCount !== originalCancellationCount;
      return f({value, cancelled});
    };
  }

  /**
   * Mark all outstanding tasks as cancelled. Tasks not yet created will
   * not be affected.
   */
  public cancelAll(): void {
    this.cancellationCount++;
  }
}
