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
var assert = chai.assert;
declare function fixture(id: string): void;

module TF.Nanite {
  window.HTMLImports.whenReady(function() {
    Polymer({
      is: "test-element",
      properties: {
        upward: {type: Number, notify: true},
        downward: {type: Number},
      }
    });
  });
  describe("nanite", () => {
    it("value retrieval works", () => {
      var store = new Store<number>();
      assert.isUndefined(store.value());
      store.set(3);
      assert.equal(store.value(), 3);
    });

    it("store value can be set in constructor", () => {
      var store = new Store<number>(5);
      assert.equal(store.value(), 5);
    });

    it("store callback called immediately if value is set", () => {
      var store = new Store<number>(5);
      var signal: number;
      var update = (x: number) => signal = x;
      store.out(update);
      assert.equal(signal, 5);
    });

    it("store callbacks called on value change", () => {
      var signal: number;
      var update = (x: number) => signal = x;
      var store = new Store<number>();
      store.out(update);
      store.set(4);
      assert.equal(signal, 4);
    });

    it("map store sets immediately if value set on source store", () => {
      var source = new Store<number>(99);
      var dest = source.map((x: number) => x + 1);
      assert.equal(dest.value(), 100);
    });

    it("map store updates when source updates", () => {
      var source = new Store<number>(0);
      var dest = source.map((x: number) => x + 1);
      source.set(41);
      assert.equal(dest.value(), 42);
    });

    it("store callbacks can be removed", () => {
      var source = new Store<number>(99);
      var signal1: number;
      var signal2: number;
      var update1 = (x: number) => signal1 = x;
      var update2 = (x: number) => signal2 = x;
      var unbind1 = source.out(update1);
      var unbind2 = source.out(update2);
      assert.equal(signal1, 99);
      assert.equal(signal2, 99);
      source.set(98);
      assert.equal(signal1, 98);
      assert.equal(signal2, 98);
      unbind2();
      source.set(909);
      assert.equal(signal1, 909);
      assert.equal(signal2, 98);
      unbind2(); // just make sure calling it several times doesn't break things
      source.set(100);
      assert.equal(signal1, 100);
      assert.equal(signal2, 98);
    });

    it("map3 works as expected", () => {
      var s1 = new Store<string>("foo");
      var s2 = new Store<string>("bar");
      var s3 = new Store<string>();
      var combine = function(a: string, b: string, c: string) {
        return a + "," + b + "," + c;
      };
      var sx = map3(combine, s1, s2, s3);
      assert.isUndefined(sx.value());
      s3.set("zod");
      assert.equal(sx.value(), "foo,bar,zod");
    });

    describe("Nanite polymer tests", function() {
      var testElement;
      var store;
      interface TestElement {
        upward: number;
        downward: number;
      }
      beforeEach(function() {
        testElement = fixture("testElementFixture");
        store = new TF.Nanite.Store();
        assert.isUndefined(testElement.downward);
        assert.isUndefined(testElement.upward);
        assert.isUndefined(store.value());
      });

      it("downward bind works - basic", function() {
        store.bindToPolymer(testElement, "downward", testElement.downward);
        store.set(5);
        assert.equal(testElement.downward, 5);
        store.set(6);
        assert.equal(testElement.downward, 6);
      });

      it("downward bind works - initialization", function() {
        store.set(99);
        store.bindToPolymer(testElement, "downward", testElement.downward);
        assert.equal(testElement.downward, 99);
      });

      it("upward bind works", function() {
        store.bindFromPolymer(testElement, "upward", testElement.upward);
        testElement.upward = 42;
        assert.equal(store.value(), 42);
      });

      it("upward bind works - initialization", function() {
        testElement.upward = 909;
        store.bindFromPolymer(testElement, "upward", testElement.upward);
        assert.equal(store.value(), 909);
      });

      it("upward bind can be unbound", function() {
        testElement.upward = 909;
        var unbind = store.bindFromPolymer(testElement, "upward", testElement.upward);
        assert.equal(store.value(), 909);
        unbind();
        testElement.upward = 404;
        assert.equal(store.value(), 909);
      });

      it("downward bind can be unbound", function() {
        var unbind = store.bindToPolymer(testElement, "downward", testElement.downward);
        store.set(200);
        assert.equal(store.value(), testElement.downward);
        unbind();
        store.set(204);
        assert.notEqual(store.value(), testElement.downward);
      });
    });
  });
}
