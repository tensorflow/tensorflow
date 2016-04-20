describe('end-to-end test', () => {
  let assert = chai.assert;
  window.HTMLImports.whenReady(() => {
    let tb = d3.select('tf-tensorboard');
    var tabs = (<any>tb.node()).$.tabs;

    function testTab(tabIndex: number) {
      it(`selecting ${TF.TensorBoard.TABS[tabIndex]} tab`, done => {
        // Every dashboard emits a rendered event when it is done rendering.
        tb.on('rendered', () => done());
        tabs.set('selected', tabIndex);
      });
    }
    // Listen for when the default tab has rendered and test other tabs after.
    tb.on('rendered', () => {
      // The default tab already rendered. Test everything else.
      // If a bug happened while rendering the default tab, the test would
      // have failed. Re-selecting the default tab and listening for
      // "rendered" event won't work since the content is not re-stamped.
      let selected = +tabs.get('selected');
      for (let i = 0; i < TF.TensorBoard.TABS.length; i++) {
        if (i !== selected) {
          testTab(i);
        }
      }
    });
  });
});
