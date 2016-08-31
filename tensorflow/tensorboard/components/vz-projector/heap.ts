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

/** Min key heap. */
export type HeapItem<T> = {
  key: number,
  value: T
};

/**
 * Min-heap data structure. Provides O(1) for peek, returning the smallest key.
 */
// TODO(jart): Rename to Heap and use Comparator.
export class MinHeap<T> {
  private arr: HeapItem<T>[] = [];

  /** Push an element with the provided key. */
  push(key: number, value: T): void {
    this.arr.push({key, value});
    this.bubbleUp(this.arr.length - 1);
  }

  /** Pop the element with the smallest key. */
  pop(): HeapItem<T> {
    if (this.arr.length === 0) {
      throw new Error('pop() called on empty binary heap');
    }
    let item = this.arr[0];
    let last = this.arr.length - 1;
    this.arr[0] = this.arr[last];
    this.arr.pop();
    if (last > 0) {
      this.bubbleDown(0);
    }
    return item;
  };

  /** Returns, but doesn't remove the element with the smallest key */
  peek(): HeapItem<T> { return this.arr[0]; }

  /**
   * Pops the element with the smallest key and at the same time
   * adds the newly provided element. This is faster than calling
   * pop() and push() separately.
   */
  popPush(key: number, value: T): HeapItem<T> {
    if (this.arr.length === 0) {
      throw new Error('pop() called on empty binary heap');
    }
    let item = this.arr[0];
    this.arr[0] = {key, value};
    if (this.arr.length > 0) {
      this.bubbleDown(0);
    }
    return item;
  }

  /** Returns the number of elements in the heap. */
  size(): number { return this.arr.length; }

  /** Returns all the items in the heap. */
  items(): HeapItem<T>[] { return this.arr; }

  private swap(a: number, b: number) {
    let temp = this.arr[a];
    this.arr[a] = this.arr[b];
    this.arr[b] = temp;
  }

  private bubbleDown(pos: number) {
    let left = (pos << 1) + 1;
    let right = left + 1;
    let largest = pos;
    if (left < this.arr.length && this.arr[left].key < this.arr[largest].key) {
      largest = left;
    }
    if (right < this.arr.length &&
        this.arr[right].key < this.arr[largest].key) {
      largest = right;
    }
    if (largest != pos) {
      this.swap(largest, pos);
      this.bubbleDown(largest);
    }
  }

  private bubbleUp(pos: number) {
    if (pos <= 0) {
      return;
    }
    let parent = ((pos - 1) >> 1);
    if (this.arr[pos].key < this.arr[parent].key) {
      this.swap(pos, parent);
      this.bubbleUp(parent);
    }
  }
}

/** List that keeps the K elements with the smallest keys. */
export class KMin<T> {
  private k: number;
  private maxHeap = new MinHeap<T>();

  /** Constructs a new k-min data structure with the provided k. */
  constructor(k: number) { this.k = k; }

  /** Adds an element to the list. */
  add(key: number, value: T) {
    if (this.maxHeap.size() < this.k) {
      this.maxHeap.push(-key, value);
      return;
    }
    let largest = this.maxHeap.peek();
    // If the new element is smaller, replace the largest with the new element.
    if (key < -largest.key) {
      this.maxHeap.popPush(-key, value);
    }
  }

  /** Returns the k items with the smallest keys. */
  getMinKItems(): T[] {
    let items = this.maxHeap.items();
    items.sort((a, b) => b.key - a.key);
    return items.map(a => a.value);
  }

  /** Returns the size of the list. */
  getSize(): number { return this.maxHeap.size(); }

  /** Returns the largest key in the list. */
  getLargestKey(): number {
    return this.maxHeap.size() == 0 ? null : -this.maxHeap.peek().key;
  }
}
