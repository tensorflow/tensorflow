/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
describe('tf-tensorboard tests', () => {
  window.HTMLImports.whenReady(() => {
    let assert = chai.assert;
    let tensorboard: any;
    beforeEach(function() {
      tensorboard = fixture('tensorboardFixture');
      tensorboard.demoDir = 'data';
      tensorboard.autoReloadEnabled = false;
    });

    it('specified tabs are correct', function(done) {
      setTimeout(function() {
        let tabs = tensorboard.$.tabs.getElementsByTagName('paper-tab');
        let tabMode = Array.prototype.map.call(tabs, (x) => x.dataMode);
        assert.deepEqual(tabMode, TF.Globals.TABS, 'mode is correct');
        let tabText =
            Array.prototype.map.call(tabs, (x) => x.innerText.toLowerCase());
        assert.deepEqual(tabText, TF.Globals.TABS, 'text is correct');
        done();
      });
    });

    it('respects router manually provided', function() {
      let router = TF.Backend.router('data', true);
      tensorboard.router = router;
      tensorboard.demoDir = null;
      assert.equal(tensorboard._backend.router, router);
    });

    it('renders injected content', function() {
      let injected = tensorboard.querySelector('#inject-me');
      assert.isNotNull(injected);
    });

    describe('non-graph tabs: reloading the selected dashboard', function() {
      TF.Globals.TABS.forEach((name, tabIndex) => {
        if (name === 'graphs') {
          return;
        }
        it(`${name}: calling reload reloads dashboard`, function(done) {
          tensorboard.$.tabs.set('selected', tabIndex);
          setTimeout(function() {
            let called = false;
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
      let idx = TF.Globals.TABS.indexOf('graphs');
      assert.notEqual(idx, -1, 'graphs was found');
      tensorboard.$.tabs.set('selected', idx);
      setTimeout(
          function() {  // async so that the queued tab change will happen
            let called = false;
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

    describe('top right global icons', function() {
      it('Clicking the reload button will call reload', function() {
        let called = false;
        tensorboard.reload = function() { called = true; };
        tensorboard.$$('#reload-button').click();
        assert.isTrue(called);
      });

      it('settings pane is hidden', function() {
        assert.equal(tensorboard.$.settings.style['display'], 'none');
      });

      it('settings icon button opens the settings pane', function(done) {
        tensorboard.$$('#settings-button').click();
        // This test is a little hacky since we depend on polymer's
        // async behavior, which is difficult to predict.

        // keep checking until the panel is visible. error with a timeout if it
        // is broken.
        function verify() {
          if (tensorboard.$.settings.style['display'] !== 'none') {
            done();
          } else {
            setTimeout(verify, 3);  // wait and see if it becomes true
          }
        }
        verify();
      });

      it('Autoreload checkbox toggle works', function() {
        let checkbox = tensorboard.$$('#auto-reload-checkbox');
        assert.equal(checkbox.checked, tensorboard.autoReloadEnabled);
        let oldValue = checkbox.checked;
        checkbox.click();
        assert.notEqual(oldValue, checkbox.checked);
        assert.equal(checkbox.checked, tensorboard.autoReloadEnabled);
      });

      it('Autoreload checkbox contains correct interval info', function() {
        let checkbox = tensorboard.$$('#auto-reload-checkbox');
        let timeInSeconds = tensorboard.autoReloadIntervalSecs + 's';
        assert.include(checkbox.innerText, timeInSeconds);
      });
    });
  });
});
