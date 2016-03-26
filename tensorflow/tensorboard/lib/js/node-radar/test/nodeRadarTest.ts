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

module TF {
  let assert = chai.assert;

  describe("NodeRadar", function() {
    let container;
    let r: NodeRadar<number>;
    function assertScansEqual<T>(expected: ScanResponse<T>, actual: ScanResponse<T>) {
      assert.deepEqual(expected.visible, actual.visible, "visible");
      assert.deepEqual(expected.almost, actual.almost, "almost");
      assert.deepEqual(expected.hidden, actual.hidden, "hidden");
    }

    beforeEach(function() {
      let root = document.body;
      container = document.createElement("div");
      container.classList.add("container");
      container.style.border = "solid 4px cyan";
      container.style.position = "fixed";
      container.style.height = "200px";
      container.style.width = "calc(100% - 160px)";
      container.style.margin = "250px 80px";
      root.appendChild(container);
      let tempNode;
      for (let i = 0; i < 100; i++) {
        tempNode = document.createElement("div");
        tempNode.classList.add("node");
        tempNode.style.height = "50px";
        tempNode.style.border = "dotted 4px grey";
        tempNode.textContent = i;
        root.appendChild(tempNode);
      }
      r = new NodeRadar<number>(container);
    });

    afterEach(function() {
      [].forEach.call(document.querySelectorAll("div"), function(n) {
        n.parentNode.removeChild(n);
      });
    });

    it("adds and removes nodes", function() {
      let node = document.querySelector(".node");
      function count() {
        let s = r.scan();
        return s.visible.length + s.almost.length + s.hidden.length;
      }
      assert.equal(count(), 0, "no nodes exist at start");
      r.add(node, 1);
      assert.equal(count(), 1, "one node exists after adding it");
      r.add(node, 2);
      assert.equal(count(), 2, "then two nodes exist");
      // removing non-existent node should have no effect
      r.remove(3);
      assert.equal(count(), 2, "removing non-existent node did nothing");
      r.remove(1);
      assert.equal(count(), 1, "removing a node worked");
      r.remove(1);
      assert.equal(count(), 1, "removing a node again did nothing");
      r.remove(2);
      assert.equal(count(), 0, "all nodes were removed");
    });

    it("throws an error if value is reused", function() {
      let node = document.querySelector(".node");
      r.add(node, 1);
      assert.throws(function() {
        r.add(node, 1);
      }, "Values in NodeRadar must be unique");
      let f = function() { return 3; };
      let r2 = new NodeRadar<Function>(container);
      r2.add(node, f);
      assert.throws(function() {
        r2.add(node, f);
      }, "Values in NodeRadar must be unique");
    });

    it("scans correctly", function() {
      window.scrollTo(0, 0);
      let nodes = document.querySelectorAll(".node");
      Array.prototype.slice.call(nodes).forEach((n, i) => r.add(n, i));

      let actual = r.scan();
      let expected = {
        visible: [4, 5, 6, 7],
        almost: [0, 1, 2, 3, 8, 9, 10, 11],
        hidden: _.range(12, 100),
      };
      assertScansEqual(expected, actual);
    });
    it("scans correctly after scrolling", function() {
      let nodes = document.querySelectorAll(".node");
      Array.prototype.slice.call(nodes).forEach((n, i) => r.add(n, i));
      window.scrollTo(0, 45);
      let actual = r.scan();
      let expected = {
        visible: [5, 6, 7, 8],
        almost: [1, 2, 3, 4, 9, 10, 11, 12],
        hidden: [0].concat(_.range(13, 100)),
      };
      assertScansEqual(expected, actual);
    });
    it("scans correctly after size change", function() {
      let nodes = document.querySelectorAll(".node");
      Array.prototype.slice.call(nodes).forEach((n, i) => r.add(n, i));
      container.style.margin = "0px 0px";
      container.style.height = "400px";
      let actual = r.scan();
      let expected = {
        visible: _.range(0, 8),
        almost: _.range(8, 15),
        hidden: _.range(15, 100),
      };
      assertScansEqual(expected, actual);
    });

  });

}
