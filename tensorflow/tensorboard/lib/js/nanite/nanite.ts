/* Copyright 2015 Google Inc. All Rights Reserved.

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
module TF.Nanite {

interface PolyListener {
  element: any;
  path: string;
}

  /**
   * A Store is a wrapper around a value that may change over time.
   *
   * Entities can subscribe to updates from the store by attaching a callback
   * using store.out(cb).
   *
   * Stores support upward and downard binding to Polymer elements using
   * bindToPolymer and bindFromPolymer. The polymer binding methods behave
   * like regular Polymer data bindings, and support typechecking via a type
   * sentinel.
   *
   * A store can be mapped over, producing a new store.
   *
   *  Note that "undefined" is not a valid value for inclusion in the store,
   * and behavior for stores containing "undefined" is ... undefined.
   */
  export class Store<T> {
    private _value: T;
    private polyListeners: PolyListener[];
    private callbacks: ((val?: T) => void)[];

    constructor(value?: T) {
      this._value = value;
      this.polyListeners = [];
      this.callbacks = [];
    }
    /**
     * bindToPolymer supports binding a value from a store to a Polymer element;
     * it is equivalent to a one-way downward databinding.
     *
     * Example: If you want to effect the following binding:
     * <foo-element id="fooId" the-property="[[value]]">
     *
     * You can use the following Javascript code:
     * var fooElement = document.getElementById("fooId");
     * // or document.createElement("foo-element")
     * var valueStore = new Store();
     * valueStore.bindToPolymer(fooElement, "theProperty");
     *
     * Or, better yet, you can use Typescript and get type safety!
     * This assumes that FooElement was defined with an interface that
     * specifies theProperty as a FooType - perhaps because the element was
     * created via polymer-ts.
     *
     * var fooElement = <FooElement> document.getElementById("fooId");
     * // or <FooElement> document.createElement("foo-element");
     * var valueStore = new Store<FooType>();
     * valueStore.bindToPolymer(fooElement, "theProperty", fooElement.theProperty);
     *
     * The third argument is a type sentinel - it isn't actually used by the
     * function, but because Typescript can check the type information,
     * it will throw a compiler error if theProperty isn't a valid property
     * on FooElements, or if the type signature is incorrect.
     *
     * bindToPolymer returns a function that, when called, destroys the binding.
     */
    public bindToPolymer(element: polymer.Base & Element, path: string, sentinel: T) {
      var l = {element: element, path: path};
      this.polyListeners.push(l);
      if (this._value !== undefined) {
        element.set(path, this._value);
      }
      return () => {
        this.polyListeners = this.polyListeners.filter(l => l.element !== element);
      };
    }

    /**
     * bindFromPolymer supports upward binding a value from a Polymer element
     * to a store. It implements "automatic" or 2-way binding from Polymer,
     * but without the 2-way aspect - values only flow up.
     *
     * As in Polymer, the property must have "notify: true" set for upward
     * binding to work.
     *
     * This function is conceptually very similar to bindToPolymer; see
     * more extensive docs (with usage examples) at bindToPolymer.
     *
     * bindFromPolymer returns a function that, when called, destroys the binding.
     */
    public bindFromPolymer(element: polymer.Base & Element, path: string, sentinel: T) {
      var dashCaseName = (<any> Polymer).CaseMap.camelToDashCase(path);
      var eventName = dashCaseName + (<any> Polymer).Base._EVENT_CHANGED;
      var listener = (x: any) => {
        this.set(x.detail.value);
      };
      element.addEventListener(eventName, listener);
      var currentValue = element[path];
      if (currentValue !== undefined) {
        this.set(currentValue);
      }

      return () => element.removeEventListener(eventName, listener);
    }

    /**
     * Attach a callback to the store.
     * Whenever the value in the store changes, the callback is called with
     * the new value. Callbacks can not currently be deregistered.
     */
    public out(cb: (val?: T) => void): Function {
      this.callbacks.push(cb);
      if (this.value() !== undefined) {
        cb(this.value());
      }
      return () => {
        this.callbacks = this.callbacks.filter((c) => c !== cb);
      };
    }

    /**
     * Map a function over the store, creating a new store.
     * Example: Suppose that this store contains a "mode" string. If
     * mode is "bar" or "zoink", then "canFoo" should be true; otherwise,
     * "canFoo" is false. Then:
     * var canFoo = mode.map(x => x === "bar" || x === "zoink");
     */
    public map<X>(f: (val: T) => X): Store<X> {
      return map1<T, X>(f, this);
    }

    /** Set the value in the store; update listeners */
    public set(value: T): this {
      if (value === undefined) {
        throw new Error("undefined is not a valid store value.");
      }
      this._value = value;
      this.polyListeners.forEach((l) => {
        l.element.set(l.path, this._value);
      });
      this.callbacks.forEach((cb) => cb(this._value));
      return this;
    }

    /** Get the value currently in the store. */
    public value(): T {
      return this._value;
    }
  }

  function applyIfAllDefined(f: Function, stores: Store<any>[]): Function {
    return function() {
      var vals = stores.map((store) => store.value());
      if (vals.every((x) => x !== undefined)) {
        return f.apply(null, vals);
      } else {
        return undefined;
      }
    };
  }

  function _map(fun: Function, stores: Store<any>[]): Store<any> {
    var f = applyIfAllDefined(fun, stores);
    var newStore = new Store(f());
    var update = () => {
      var result = f();
      if (result !== undefined) {
        newStore.set(result);
      }
    };
    stores.forEach((s) => s.out(update));
    return newStore;
  }

  /**
   * Map a function of type A->X over a store of type A; produces a store of
   * type X. For example, if "canFoo" should be true when mode is one of "zoink"
   * or "zod", then
   * var canFooStore = modeStore.map((x) => x === "zoink" || x === "zod");
   */
  export function map1<A, X>(f: ((a: A) => X), as: Store<A>): Store<X> {
    return _map(f, [as]);
  }

  /**
   * Map a function of type A->B->X over stores of type A and B;
   * produces a store of type X
   */
  export function map2<A, B, X>(f: (a: A, b: B) => X, as: Store<A>, bs: Store<B>): Store<X> {
    return _map(f, [as, bs]);
  }

  /**
   * Map a function of type A->B->C->X over stores of type A, B, and C;
   * produces a store of type X
   */
  export function map3<A, B, C, X>(
      f: (a: A, b: B, c: C) => X,
      as: Store<A>, bs: Store<B>, cs: Store<C>): Store<X> {
    return _map(f, [as, bs, cs]);
  }

}
