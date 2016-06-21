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

/* tslint:disable:no-namespace */
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
   * Return a string stored in the URI, given a corresonding key.
   * Null if not found.
   */
  export function getString(key: string): string {
    let items = _componentToDict(_readComponent());
    return _.isUndefined(items[key]) ? null : items[key];
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
   */
  export function getNumber(key: string): number {
    let items = _componentToDict(_readComponent());
    return _.isUndefined(items[key]) ? null : +items[key];
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
   */
  export function getObject(key: string): Object {
    let items = _componentToDict(_readComponent());
    return _.isUndefined(items[key]) ? null : JSON.parse(atob(items[key]));
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
   * Read component from URI (e.g. returns "events&runPrefix=train*").
   */
  function _readComponent(): string { return window.location.hash.slice(1); }

  /**
   * Write component to URI.
   */
  function _writeComponent(component: string) {
    window.location.hash = component;
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
}
