describe('tf-tensorboard tests', () => {
  window.HTMLImports.whenReady(() => {
    let assert = chai.assert;
    let demoRouter = TF.Backend.router('data', true);
    function makeTensorBoard() {
      let tensorboard: any = fixture('testElementFixture');
      tensorboard.router = demoRouter;
      return tensorboard;
    }
    var tensorboard;
    beforeEach(function() { tensorboard = makeTensorBoard(); });

    it('specified tabs are correct', function(done) {
      setTimeout(function() {
        var tabs = tensorboard.$.tabs.getElementsByTagName('paper-tab');
        var tabMode = Array.prototype.map.call(tabs, (x) => x.dataMode);
        assert.deepEqual(tabMode, TF.TensorBoard.TABS, 'mode is correct');
        var tabText =
            Array.prototype.map.call(tabs, (x) => x.innerText.toLowerCase());
        assert.deepEqual(tabText, TF.TensorBoard.TABS, 'text is correct');
        done();
      });
    });

    describe('non-graph tabs: reloading the selected dashboard', function() {
      TF.TensorBoard.TABS.forEach((name, tabIndex) => {
        if (name === 'graphs') {
          return;
        }
        it(`${name}: calling reload reloads dashboard`, function(done) {
          tensorboard.$.tabs.set('selected', tabIndex);
          d3.select(tensorboard).on('rendered', function() {
            var called = false;
            tensorboard.selectedDashboard().reload = function() {
              called = true;
            };
            tensorboard.reload();
            assert.isFalse(
                tensorboard.$$('#reload-button').disabled,
                'reload button not disabled');
            assert.isTrue(called, `reload was called`);
            done();
          });
        });
      });
    });

    it('reload is disabled for graph dashboard', function(done) {
      var idx = TF.TensorBoard.TABS.indexOf('graphs');
      assert.notEqual(idx, -1, 'graphs was found');
      tensorboard.$.tabs.set('selected', idx);
      setTimeout(
          function() {  // async so that the queued tab change will happen
            var called = false;
            tensorboard.selectedDashboard().reload = function() {
              called = true;
            };
            tensorboard.reload();
            assert.isTrue(
                tensorboard.$$('#reload-button').disabled,
                'reload button disabled');
            assert.isFalse(called, `reload was not called`);
            done();
          });
    });
  });
});
