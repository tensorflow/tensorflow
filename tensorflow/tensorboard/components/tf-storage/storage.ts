/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

/* tslint:disable:no-namespace variable-name */
/**
 * The Storage Module provides storage for URL parameters, and an API for
 * getting and setting TensorBoard's stateful URI.
 *
 * It generates URI components like: events&runPrefix=train*
 * which TensorBoard uses after like localhost:8000/#events&runPrefix=train*
 * to store state in the URI.
 */
module TF.URIStorage {
  type StringDict = {[key: string]: string};

  /**
   * A key that users cannot use, since TensorBoard uses this to store info
   * about the active tab.
   */
  export let TAB = '__tab__';

  /**
   * The name of the property for users to set on a Polymer component
   * in order for its stored properties to be stored in the URI unambiguously.
   * (No need to set this if you want mutliple instances of the component to
   * share URI state)
   *
   * Example:
   * <my-component disambiguator="0"></my-component>
   *
   * The disambiguator should be set to any unique value so that multiple
   * instances of the component can store properties in URI storage.
   *
   * Because it's hard to dereference this variable in HTML property bindings,
   * it is NOT safe to change the disambiguator string without find+replace
   * across the codebase.
   */
  export let DISAMBIGUATOR = 'disambiguator';

  /**
   * Return a boolean stored in the URI, given a corresponding key.
   * Undefined if not found.
   */
  export function getBoolean(key: string): boolean {
    let items = _componentToDict(_readComponent());
    let item = items[key];
    return item === 'true' ? true : item === 'false' ? false : undefined;
  }

  /**
   * Store a boolean in the URI, with a corresponding key.
   */
  export function setBoolean(key: string, value: boolean) {
    let items = _componentToDict(_readComponent());
    items[key] = value.toString();
    _writeComponent(_dictToComponent(items));
  }

  /**
   * Return a string stored in the URI, given a corresponding key.
   * Undefined if not found.
   */
  export function getString(key: string): string {
    let items = _componentToDict(_readComponent());
    return items[key];
  }

  /**
   * Store a string in the URI, with a corresponding key.
   */
  export function setString(key: string, value: string) {
    let items = _componentToDict(_readComponent());
    items[key] = value;
    _writeComponent(_dictToComponent(items));
  }

  /**
   * Return a number stored in the URI, given a corresponding key.
   * Undefined if not found.
   */
  export function getNumber(key: string): number {
    let items = _componentToDict(_readComponent());
    return items[key] === undefined ? undefined : +items[key];
  }

  /**
   * Store a number in the URI, with a corresponding key.
   */
  export function setNumber(key: string, value: number) {
    let items = _componentToDict(_readComponent());
    items[key] = '' + value;
    _writeComponent(_dictToComponent(items));
  }

  /**
   * Return an object stored in the URI, given a corresponding key.
   * Undefined if not found.
   */
  export function getObject(key: string): Object {
    let items = _componentToDict(_readComponent());
    return items[key] === undefined ? undefined : JSON.parse(atob(items[key]));
  }

  /**
   * Store an object in the URI, with a corresponding key.
   */
  export function setObject(key: string, value: Object) {
    let items = _componentToDict(_readComponent());
    items[key] = btoa(JSON.stringify(value));
    _writeComponent(_dictToComponent(items));
  }

  /**
   * Get a unique storage name for a (Polymer component, propertyName) tuple.
   *
   * DISAMBIGUATOR must be set on the component, if other components use the
   * same propertyName.
   */
  export function getURIStorageName(
      component: Object, propertyName: string): string {
    let d = component[DISAMBIGUATOR];
    let components = d == null ? [propertyName] : [d, propertyName];
    return components.join('.');
  }

  /**
   * Return a function that:
   * (1) Initializes a Polymer boolean property with a default value, if its
   *     value is not already set
   * (2) Sets up listener that updates Polymer property on hash change.
   */
  export function getBooleanInitializer(
      propertyName: string, defaultVal: boolean): Function {
    return _getInitializer(getBoolean, propertyName, defaultVal);
  }

  /**
   * Return a function that:
   * (1) Initializes a Polymer string property with a default value, if its
   *     value is not already set
   * (2) Sets up listener that updates Polymer property on hash change.
   */
  export function getStringInitializer(
      propertyName: string, defaultVal: string): Function {
    return _getInitializer(getString, propertyName, defaultVal);
  }

  /**
   * Return a function that:
   * (1) Initializes a Polymer number property with a default value, if its
   *     value is not already set
   * (2) Sets up listener that updates Polymer property on hash change.
   */
  export function getNumberInitializer(
      propertyName: string, defaultVal: number): Function {
    return _getInitializer(getNumber, propertyName, defaultVal);
  }

  /**
   * Return a function that:
   * (1) Initializes a Polymer Object property with a default value, if its
   *     value is not already set
   * (2) Sets up listener that updates Polymer property on hash change.
   *
   * Generates a deep clone of the defaultVal to avoid mutation issues.
   */
  export function getObjectInitializer(
      propertyName: string, defaultVal: Object): Function {
    let clone = _.cloneDeep(defaultVal);
    return _getInitializer(getObject, propertyName, clone);
  }

  /**
   * Return a function that updates URIStorage when a string property changes.
   */
  export function getBooleanObserver(
      propertyName: string, defaultVal: boolean): Function {
    return _getObserver(getBoolean, setBoolean, propertyName, defaultVal);
  }

  /**
   * Return a function that updates URIStorage when a string property changes.
   */
  export function getStringObserver(
      propertyName: string, defaultVal: string): Function {
    return _getObserver(getString, setString, propertyName, defaultVal);
  }

  /**
   * Return a function that updates URIStorage when a number property changes.
   */
  export function getNumberObserver(
      propertyName: string, defaultVal: number): Function {
    return _getObserver(getNumber, setNumber, propertyName, defaultVal);
  }

  /**
   * Return a function that updates URIStorage when an object property changes.
   * Generates a deep clone of the defaultVal to avoid mutation issues.
   */
  export function getObjectObserver(
      propertyName: string, defaultVal: Object): Function {
    let clone = _.cloneDeep(defaultVal);
    return _getObserver(getObject, setObject, propertyName, clone);
  }

  /**
   * Read component from URI (e.g. returns "events&runPrefix=train*").
   */
  function _readComponent(): string {
    return TF.Globals.USE_HASH ? window.location.hash.slice(1) :
                                 TF.Globals.FAKE_HASH;
  }

  /**
   * Write component to URI.
   */
  function _writeComponent(component: string) {
    if (TF.Globals.USE_HASH) {
      window.location.hash = component;
    } else {
      TF.Globals.FAKE_HASH = component;
    }
  }

  /**
   * Convert dictionary of strings into a URI Component.
   * All key value entries get added as key value pairs in the component,
   * with the exception of a key with the TAB value, which if present
   * gets prepended to the URI Component string for backwards comptability
   * reasons.
   */
  function _dictToComponent(items: StringDict): string {
    let component = '';

    // Add the tab name e.g. 'events', 'images', 'histograms' as a prefix
    // for backwards compatbility.
    if (items[TAB] !== undefined) {
      component += items[TAB];
    }

    // Join other strings with &key=value notation
    let nonTab = _.pairs(items)
                     .filter(function(pair) { return pair[0] !== TAB; })
                     .map(function(pair) {
                       return encodeURIComponent(pair[0]) + '=' +
                           encodeURIComponent(pair[1]);
                     })
                     .join('&');

    return nonTab.length > 0 ? (component + '&' + nonTab) : component;
  }

  /**
   * Convert a URI Component into a dictionary of strings.
   * Component should consist of key-value pairs joined by a delimiter
   * with the exception of the tabName.
   * Returns dict consisting of all key-value pairs and
   * dict[TAB] = tabName
   */
  function _componentToDict(component: string): StringDict {
    let items = {} as StringDict;

    let tokens = component.split('&');
    tokens.forEach(function(token) {
      let kv = token.split('=');
      // Special backwards compatibility for URI components like #events
      if (kv.length === 1 && _.contains(TF.Globals.TABS, kv[0])) {
        items[TAB] = kv[0];
      } else if (kv.length === 2) {
        items[decodeURIComponent(kv[0])] = decodeURIComponent(kv[1]);
      }
    });
    return items;
  }

  /**
   * Return a function that:
   * (1) Initializes a Polymer property with a default value, if its
   *     value is not already set
   * (2) Sets up listener that updates Polymer property on hash change.
   */
  function _getInitializer<T>(
      get: (name: string) => T, propertyName: string, defaultVal: T): Function {
    return function() {
      let URIStorageName = getURIStorageName(this, propertyName);
      let setComponentValue = () => {
        let uriValue = get(URIStorageName);
        this[propertyName] = uriValue !== undefined ? uriValue : defaultVal;
      };
      // Set the value on the property.
      setComponentValue();
      // Update it when the hashchanges.
      window.addEventListener('hashchange', setComponentValue);
    };
  }

  /**
   * Return a function that updates URIStorage when a property changes.
   */
  function _getObserver<T>(
      get: (name: string) => T, set: (name: string, newVal: T) => void,
      propertyName: string, defaultVal: T): Function {
    return function() {
      let URIStorageName = getURIStorageName(this, propertyName);
      let newVal = this[propertyName];
      if (!_.isEqual(newVal, get(URIStorageName))) {
        if (_.isEqual(newVal, defaultVal)) {
          _unset(URIStorageName);
        } else {
          set(URIStorageName, newVal);
        }
      }
    };
  }

  /**
   * Delete a key from the URI.
   */
  function _unset(key) {
    let items = _componentToDict(_readComponent());
    delete items[key];
    _writeComponent(_dictToComponent(items));
  }
}
