describe('fast tab switch', () => {
  let assert = chai.assert;
  window.HTMLImports.whenReady(() => {
    let tb = d3.select('tf-tensorboard');
    var tabs = (<any>tb.node()).$.tabs;

    // This test will select the events tab. Once the events tab
    // renders, will select the graph tab, and immediately select
    // the images tab wihout waiting for the graph tab to finish
    // rendering. Finally, it finishes when the images tab
    // has rendered and no errors were thrown.
    let eventsTabIndex = TF.TensorBoard.TABS.indexOf('events');
    let imagesTabIndex = TF.TensorBoard.TABS.indexOf('images');
    let graphTabIndex = TF.TensorBoard.TABS.indexOf('graphs');

    // Listen for when the events tab rendered.
    tb.on('rendered', () => {
      it('switching to graph tab and immediately to images', done => {
        // Select the graph tab.
        tabs.set('selected', graphTabIndex);
        // Interrupt graph rendering by immediately selecting the images tab
        // and finish when the images tab has rendered.
        tb.on('rendered', () => done());
        tabs.set('selected', imagesTabIndex);
      });
    });
    // Select the events tab.
    tabs.set('selected', eventsTabIndex);
  });
});
