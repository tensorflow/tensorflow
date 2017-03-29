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
 *
 * It also allows saving the values to localStorage for long-term persistance.
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
   * Return a string stored in URI or localStorage.
   * Undefined if not found.
   */
  export function getString(key: string, useLocalStorage: boolean): string {
    if (useLocalStorage) {
      return window.localStorage.getItem(key);
    } else {
      return _componentToDict(_readComponent())[key];
    }
  }

  /**
   * Set a string in URI or localStorage.
   */
  export function setString(
      key: string, value: string, useLocalStorage: boolean) {
    if (useLocalStorage) {
      window.localStorage.setItem(key, value);
    } else {
      let items = _componentToDict(_readComponent());
      items[key] = value;
      _writeComponent(_dictToComponent(items));
    }
  }

  /**
   * Return a boolean stored in stored in URI or localStorage.
   * Undefined if not found.
   */
  export function getBoolean(key: string, useLocalStorage: boolean): boolean {
    let item = getString(key, useLocalStorage);
    return item === 'true' ? true : item === 'false' ? false : undefined;
  }

  /**
   * Store a boolean in URI or localStorage.
   */
  export function setBoolean(
      key: string, value: boolean, useLocalStorage = false) {
    setString(key, value.toString(), useLocalStorage);
  }

  /**
   * Return a number stored in stored in URI or localStorage.
   * Undefined if not found.
   */
  export function getNumber(key: string, useLocalStorage: boolean): number {
    let item = getString(key, useLocalStorage);
    return item === undefined ? undefined : +item;
  }

  /**
   * Store a number in URI or localStorage.
   */
  export function setNumber(
      key: string, value: number, useLocalStorage: boolean) {
    setString(key, '' + value, useLocalStorage);
  }

  /**
   * Return an object stored in stored in URI or localStorage.
   * Undefined if not found.
   */
  export function getObject(key: string, useLocalStorage: boolean): {} {
    let item = getString(key, useLocalStorage);
    return item === undefined ? undefined : JSON.parse(atob(item));
  }

  /**
   * Store an object in URI or localStorage.
   */
  export function setObject(key: string, value: {}, useLocalStorage: boolean) {
    setString(key, btoa(JSON.stringify(value)), useLocalStorage);
  }

  /**
   * Get a unique storage name for a (Polymer component, propertyName) tuple.
   *
   * DISAMBIGUATOR must be set on the component, if other components use the
   * same propertyName.
   */
  export function getURIStorageName(
      component: {}, propertyName: string): string {
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
      propertyName: string, defaultVal: boolean,
      useLocalStorage = false): Function {
    return _getInitializer(
        getBoolean, propertyName, defaultVal, useLocalStorage);
  }

  /**
   * Return a function that:
   * (1) Initializes a Polymer string property with a default value, if its
   *     value is not already set
   * (2) Sets up listener that updates Polymer property on hash change.
   */
  export function getStringInitializer(
      propertyName: string, defaultVal: string,
      useLocalStorage = false): Function {
    return _getInitializer(
        getString, propertyName, defaultVal, useLocalStorage);
  }

  /**
   * Return a function that:
   * (1) Initializes a Polymer number property with a default value, if its
   *     value is not already set
   * (2) Sets up listener that updates Polymer property on hash change.
   */
  export function getNumberInitializer(
      propertyName: string, defaultVal: number,
      useLocalStorage = false): Function {
    return _getInitializer(
        getNumber, propertyName, defaultVal, useLocalStorage);
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
      propertyName: string, defaultVal: {}, useLocalStorage = false): Function {
    return _getInitializer(
        getObject, propertyName, defaultVal, useLocalStorage);
  }

  /**
   * Return a function that updates URIStorage when a string property changes.
   */
  export function getBooleanObserver(
      propertyName: string, defaultVal: boolean,
      useLocalStorage = false): Function {
    return _getObserver(
        getBoolean, setBoolean, propertyName, defaultVal, useLocalStorage);
  }

  /**
   * Return a function that updates URIStorage when a string property changes.
   */
  export function getStringObserver(
      propertyName: string, defaultVal: string,
      useLocalStorage = false): Function {
    return _getObserver(
        getString, setString, propertyName, defaultVal, useLocalStorage);
  }

  /**
   * Return a function that updates URIStorage when a number property changes.
   */
  export function getNumberObserver(
      propertyName: string, defaultVal: number,
      useLocalStorage = false): Function {
    return _getObserver(
        getNumber, setNumber, propertyName, defaultVal, useLocalStorage);
  }

  /**
   * Return a function that updates URIStorage when an object property changes.
   * Generates a deep clone of the defaultVal to avoid mutation issues.
   */
  export function getObjectObserver(
      propertyName: string, defaultVal: {}, useLocalStorage = false): Function {
    let clone = _.cloneDeep(defaultVal);
    return _getObserver(
        getObject, setObject, propertyName, clone, useLocalStorage);
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
      get: (name: string, useLocalStorage: boolean) => T, propertyName: string,
      defaultVal: T, useLocalStorage): Function {
    return function() {
      let URIStorageName = getURIStorageName(this, propertyName);
      // setComponentValue will be called every time the hash changes, and is
      // responsible for ensuring that new state in the hash will be propagated
      // to the component with that property.
      // It is important that this function does not re-assign needlessly,
      // to avoid Polymer observer churn.
      let setComponentValue = () => {
        let uriValue = get(URIStorageName, false);
        let currentValue = this[propertyName];
        // if uriValue is undefined, we will ensure that the property has the
        // default value
        if (uriValue === undefined) {
          let valueToSet: T;
          // if we are using localStorage, we will set the value to the value
          // from localStorage. Then, the corresponding observer will proxy
          // the localStorage value into URI storage.
          // in this way, localStorage takes precedence over the default val
          // but not over the URI value.
          if (useLocalStorage) {
            let useLocalStorageValue = get(URIStorageName, true);
            valueToSet = useLocalStorageValue === undefined ?
                defaultVal :
                useLocalStorageValue;
          } else {
            valueToSet = defaultVal;
          }
          if (!_.isEqual(currentValue, valueToSet)) {
            // If we don't have an explicit URI value, then we need to ensure
            // the property value is equal to the default value.
            // We will assign a clone rather than the canonical default, because
            // the component receiving this property may mutate it, and we need
            // to keep a pristine copy of the default.
            this[propertyName] = _.clone(valueToSet);
          }
          // In this case, we have an explicit URI value, so we will ensure that
          // the component has an equivalent value.
        } else {
          if (!_.isEqual(uriValue, currentValue)) {
            this[propertyName] = uriValue;
          }
        }
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
      get: (name: string, useLocalStorage: boolean) => T,
      set: (name: string, newVal: T, useLocalStorage: boolean) => void,
      propertyName: string, defaultVal: T, useLocalStorage: boolean): Function {
    return function() {
      let URIStorageName = getURIStorageName(this, propertyName);
      let newVal = this[propertyName];
      // if this is a localStorage property, we always synchronize the value
      // in localStorage to match the one currently in the URI.
      if (useLocalStorage) {
        set(URIStorageName, newVal, true);
      }
      if (!_.isEqual(newVal, get(URIStorageName, false))) {
        if (_.isEqual(newVal, defaultVal)) {
          _unsetFromURI(URIStorageName);
        } else {
          set(URIStorageName, newVal, false);
        }
      }
    };
  }

  /**
   * Delete a key from the URI.
   */
  function _unsetFromURI(key) {
    let items = _componentToDict(_readComponent());
    delete items[key];
    _writeComponent(_dictToComponent(items));
  }
}
