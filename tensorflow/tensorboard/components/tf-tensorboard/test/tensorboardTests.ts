describe("end-to-end test", () => {
  let assert = chai.assert;
  window.HTMLImports.whenReady(() => {
    let tabs: polymer.Base = <any> d3.select("#tabs").node();
    let tb = d3.select("tf-tensorboard");
    let tabNames: string[] = [];
    // Find all the tabs.
    d3.selectAll("#tabs paper-tab").each(function(d, i) {
      tabNames.push(this.dataset.mode);
    });

    // In case we move the tabs to a different place in the DOM
    // and the query selector above becomes incorrect.
    assert.isTrue(tabNames.length >= 4, "at least four tabs were found");

    function testTab(tabIndex: number) {
      it(`selecting ${tabNames[tabIndex]} tab`, done => {
        // Every dashboard emits a rendered event when it is done rendering.
        tb.on("rendered", () => done());
        tabs.set("selected", tabIndex);
      });
    }
    // Listen for when the default tab has rendered and test other tabs after.
    tb.on("rendered", () => {
      // The default tab already rendered. Test everything else.
      // If a bug happened while rendering the default tab, the test would
      // have failed. Re-selecting the default tab and listening for
      // "rendered" event won't work since the content is not re-stamped.
      let selected = +tabs.get("selected");
      for (let i = 0; i < tabNames.length; i++) {
        if (i !== selected) {
          testTab(i);
        }
      }
    });
  });
});
