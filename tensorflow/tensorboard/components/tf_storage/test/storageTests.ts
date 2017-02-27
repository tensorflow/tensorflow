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
/* tslint:disable:no-namespace */
module TF.URIStorage {
  describe('URIStorage', function() {
    it('get/setString', function() {
      setString('key_a', 'hello');
      setString('key_b', 'there');
      chai.assert.equal('hello', getString('key_a'));
      chai.assert.equal('there', getString('key_b'));
      chai.assert.equal(null, getString('key_c'));
    });

    it('get/setNumber', function() {
      setNumber('key_a', 12);
      setNumber('key_b', 3.4);
      chai.assert.equal(12, getNumber('key_a'));
      chai.assert.equal(3.4, getNumber('key_b'));
      chai.assert.equal(null, getNumber('key_c'));
    });

    it('get/setObject', function() {
      let obj = {'foo': 2.3, 'bar': 'barstr'};
      setObject('key_a', obj);
      chai.assert.deepEqual(obj, getObject('key_a'));
    });

    it('get/setWeirdValues', function() {
      setNumber('key_a', NaN);
      chai.assert.deepEqual(NaN, getNumber('key_a'));

      setNumber('key_a', +Infinity);
      chai.assert.equal(+Infinity, getNumber('key_a'));

      setNumber('key_a', -Infinity);
      chai.assert.equal(-Infinity, getNumber('key_a'));

      setNumber('key_a', 1 / 3);
      chai.assert.equal(1 / 3, getNumber('key_a'));

      setNumber('key_a', -0);
      chai.assert.equal(-0, getNumber('key_a'));
    });

    it('set/getTab', function() {
      setString(TAB, TF.Globals.TABS[0]);
      chai.assert.equal(TF.Globals.TABS[0], getString(TAB));
    });
  });
}
