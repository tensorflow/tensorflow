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
/// <reference path="../../../../typings/tsd.d.ts" />

module TF {
  let assert = chai.assert;
  let expect = chai.expect;

  describe("NodeRadar", function() {
    var container, r: NodeRadar;

    beforeEach(function() {
      var root = document.body;
      container = document.createElement("div");
      container.classList.add("container");
      container.style.border = "solid 4px cyan";
      container.style.position = "fixed";
      container.style.height = "200px";
      container.style.width = "calc(100% - 160px)";
      container.style.margin = "250px 80px";
      root.appendChild(container);
      var tempNode;
      for (let i = 0; i < 100; i++) {
        tempNode = document.createElement("div");
        tempNode.classList.add("node");
        tempNode.style.height = "50px";
        tempNode.style.border = "dotted 4px grey";
        tempNode.textContent = i;
        root.appendChild(tempNode);
      }
      r = new NodeRadar(container);
    });

    afterEach(function() {
      [].forEach.call(document.querySelectorAll("div"), function(n) {
        n.parentNode.removeChild(n);
      });
    });

    it("adds a html element", function() {
      var node = document.querySelector(".node");
      r.add(node, null);
      assert.equal(r.getNodes().length, 1);
      assert.equal(r.getNodes()[0].visibility.node, node);
    });

    it("adds a nodelist", function() {
      var nodes = document.querySelectorAll(".node"); // returns a NodeList
      r.add(nodes, null);
      assert.equal(r.getNodes().length, nodes.length);
    });

    it("adds an html collection", function() {
      var nodes = document.body.children; // returns an HTMLCollection
      r.add(nodes, null);
      assert.equal(r.getNodes().length, nodes.length);
    });

    it("removes a node", function() {
      var node = document.querySelector(".node");
      r.add(node, null);
      assert.equal(r.getNodes().length, 1);
      r.remove(node);
      assert.equal(r.getNodes().length, 0);
    });

    it("throws an error if it can't find a node", function() {
      var node = document.querySelector(".node");
      r.add(node, null);
      expect(() => {
        r.checkVisibility(document.createElement("div"));
      }).to.throw("Couldn't find node to check visibility.");
    });

    it("scans correctly on startup", function() {
      window.scrollTo(0, 0);
      var nodes = document.querySelectorAll(".node");
      r.add(nodes, null);
      var zero = r.checkVisibility(nodes[0]),
          one = r.checkVisibility(nodes[1]),
          four = r.checkVisibility(nodes[4]),
          five = r.checkVisibility(nodes[5]),
          seven = r.checkVisibility(nodes[7]),
          eight = r.checkVisibility(nodes[8]),
          eleven = r.checkVisibility(nodes[11]);

      assert.equal(zero.almost, false);
      assert.equal(zero.partial, false);
      assert.equal(zero.full, false);

      assert.equal(one.almost, true);
      assert.equal(one.partial, false);
      assert.equal(one.full, false);

      assert.equal(four.almost, true);
      assert.equal(four.partial, true);
      assert.equal(four.full, false);

      assert.equal(five.almost, true);
      assert.equal(five.partial, true);
      assert.equal(five.full, true);

      assert.equal(seven.almost, true);
      assert.equal(seven.partial, true);
      assert.equal(seven.full, false);

      assert.equal(eight.almost, true);
      assert.equal(eight.partial, false);
      assert.equal(eight.full, false);

      assert.equal(eleven.almost, false);
      assert.equal(eleven.partial, false);
      assert.equal(eleven.full, false);
    });

    it("scans correctly after scrolling", function() {
      window.scrollTo(0, 0);
      var nodes = document.querySelectorAll(".node");
      r.add(nodes, null);
      var zero = r.checkVisibility(nodes[0]),
          one = r.checkVisibility(nodes[1]),
          four = r.checkVisibility(nodes[4]),
          five = r.checkVisibility(nodes[5]),
          seven = r.checkVisibility(nodes[7]),
          eight = r.checkVisibility(nodes[8]),
          eleven = r.checkVisibility(nodes[11]);

      window.scrollTo(0, 45);
      r.scan();

      assert.equal(zero.almost, false);
      assert.equal(zero.partial, false);
      assert.equal(zero.full, false);

      assert.equal(one.almost, false);
      assert.equal(one.partial, false);
      assert.equal(one.full, false);

      assert.equal(four.almost, true);
      assert.equal(four.partial, false);
      assert.equal(four.full, false);

      assert.equal(five.almost, true);
      assert.equal(five.partial, true);
      assert.equal(five.full, false);

      assert.equal(seven.almost, true);
      assert.equal(seven.partial, true);
      assert.equal(seven.full, true);

      assert.equal(eight.almost, true);
      assert.equal(eight.partial, true);
      assert.equal(eight.full, false);

      assert.equal(eleven.almost, true);
      assert.equal(eleven.partial, false);
      assert.equal(eleven.full, false);
    });

    it("executes callbacks when visibility changes", function() {
      window.scrollTo(0, 0);
      var node = document.querySelectorAll(".node")[5];
      var called;
      r.add(node, () => { called = true; });
      called = false;
      window.scrollTo(0, 60);
      r.scan();
      assert.equal(called, true);
    });

    it("does not execute callbacks when visibility does not change", function() {
      window.scrollTo(0, 0);
      var node = document.querySelectorAll(".node")[5];
      var called;
      r.add(node, () => { called = true; });
      called = false;
      window.scrollTo(0, 2);
      r.scan();
      assert.equal(called, false);
    });

  });

}
