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
import {TAB, getString, getNumber, getObject, setString, setNumber, setObject} from '../storage';
import {TABS} from '../../tf-globals/globals';

/* tslint:disable:no-namespace */
describe('URIStorage', () => {
  it('get/setString', () => {
    setString('key_a', 'hello', false);
    setString('key_b', 'there', false);
    chai.assert.equal('hello', getString('key_a', false));
    chai.assert.equal('there', getString('key_b', false));
    chai.assert.equal(null, getString('key_c', false));
  });

  it('get/setNumber', () => {
    setNumber('key_a', 12, false);
    setNumber('key_b', 3.4, false);
    chai.assert.equal(12, getNumber('key_a', false));
    chai.assert.equal(3.4, getNumber('key_b', false));
    chai.assert.equal(null, getNumber('key_c', false));
  });

  it('get/setObject', () => {
    const obj = {'foo': 2.3, 'bar': 'barstr'};
    setObject('key_a', obj, false);
    chai.assert.deepEqual(obj, getObject('key_a', false));
  });

  it('get/setWeirdValues', () => {
    setNumber('key_a', NaN, false);
    chai.assert.deepEqual(NaN, getNumber('key_a', false));

    setNumber('key_a', +Infinity, false);
    chai.assert.equal(+Infinity, getNumber('key_a', false));

    setNumber('key_a', -Infinity, false);
    chai.assert.equal(-Infinity, getNumber('key_a', false));

    setNumber('key_a', 1 / 3, false);
    chai.assert.equal(1 / 3, getNumber('key_a', false));

    setNumber('key_a', -0, false);
    chai.assert.equal(-0, getNumber('key_a', false));
  });

  it('set/getTab', () => {
    setString(TAB, TABS[0], false);
    chai.assert.equal(TABS[0], getString(TAB, false));
  });
});

