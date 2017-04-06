# -*- coding: utf-8 -*-
"""
    werkzeug.contrib.jsrouting
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    Addon module that allows to create a JavaScript function from a map
    that generates rules.

    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
try:
    from simplejson import dumps
except ImportError:
    try:
        from json import dumps
    except ImportError:
        def dumps(*args):
            raise RuntimeError('simplejson required for jsrouting')

from inspect import getmro
from werkzeug.routing import NumberConverter
from werkzeug._compat import iteritems


def render_template(name_parts, rules, converters):
    result = u''
    if name_parts:
        for idx in range(0, len(name_parts) - 1):
            name = u'.'.join(name_parts[:idx + 1])
            result += u"if (typeof %s === 'undefined') %s = {}\n" % (name, name)
        result += '%s = ' % '.'.join(name_parts)
    result += """(function (server_name, script_name, subdomain, url_scheme) {
    var converters = [%(converters)s];
    var rules = %(rules)s;
    function in_array(array, value) {
        if (array.indexOf != undefined) {
            return array.indexOf(value) != -1;
        }
        for (var i = 0; i < array.length; i++) {
            if (array[i] == value) {
                return true;
            }
        }
        return false;
    }
    function array_diff(array1, array2) {
        array1 = array1.slice();
        for (var i = array1.length-1; i >= 0; i--) {
            if (in_array(array2, array1[i])) {
                array1.splice(i, 1);
            }
        }
        return array1;
    }
    function split_obj(obj) {
        var names = [];
        var values = [];
        for (var name in obj) {
            if (typeof(obj[name]) != 'function') {
                names.push(name);
                values.push(obj[name]);
            }
        }
        return {names: names, values: values, original: obj};
    }
    function suitable(rule, args) {
        var default_args = split_obj(rule.defaults || {});
        var diff_arg_names = array_diff(rule.arguments, default_args.names);

        for (var i = 0; i < diff_arg_names.length; i++) {
            if (!in_array(args.names, diff_arg_names[i])) {
                return false;
            }
        }

        if (array_diff(rule.arguments, args.names).length == 0) {
            if (rule.defaults == null) {
                return true;
            }
            for (var i = 0; i < default_args.names.length; i++) {
                var key = default_args.names[i];
                var value = default_args.values[i];
                if (value != args.original[key]) {
                    return false;
                }
            }
        }

        return true;
    }
    function build(rule, args) {
        var tmp = [];
        var processed = rule.arguments.slice();
        for (var i = 0; i < rule.trace.length; i++) {
            var part = rule.trace[i];
            if (part.is_dynamic) {
                var converter = converters[rule.converters[part.data]];
                var data = converter(args.original[part.data]);
                if (data == null) {
                    return null;
                }
                tmp.push(data);
                processed.push(part.name);
            } else {
                tmp.push(part.data);
            }
        }
        tmp = tmp.join('');
        var pipe = tmp.indexOf('|');
        var subdomain = tmp.substring(0, pipe);
        var url = tmp.substring(pipe+1);

        var unprocessed = array_diff(args.names, processed);
        var first_query_var = true;
        for (var i = 0; i < unprocessed.length; i++) {
            if (first_query_var) {
                url += '?';
            } else {
                url += '&';
            }
            first_query_var = false;
            url += encodeURIComponent(unprocessed[i]);
            url += '=';
            url += encodeURIComponent(args.original[unprocessed[i]]);
        }
        return {subdomain: subdomain, path: url};
    }
    function lstrip(s, c) {
        while (s && s.substring(0, 1) == c) {
            s = s.substring(1);
        }
        return s;
    }
    function rstrip(s, c) {
        while (s && s.substring(s.length-1, s.length) == c) {
            s = s.substring(0, s.length-1);
        }
        return s;
    }
    return function(endpoint, args, force_external) {
        args = split_obj(args);
        var rv = null;
        for (var i = 0; i < rules.length; i++) {
            var rule = rules[i];
            if (rule.endpoint != endpoint) continue;
            if (suitable(rule, args)) {
                rv = build(rule, args);
                if (rv != null) {
                    break;
                }
            }
        }
        if (rv == null) {
            return null;
        }
        if (!force_external && rv.subdomain == subdomain) {
            return rstrip(script_name, '/') + '/' + lstrip(rv.path, '/');
        } else {
            return url_scheme + '://'
                   + (rv.subdomain ? rv.subdomain + '.' : '')
                   + server_name + rstrip(script_name, '/')
                   + '/' + lstrip(rv.path, '/');
        }
    };
})""" % {'converters': u', '.join(converters),
         'rules': rules}

    return result


def generate_map(map, name='url_map'):
    """
    Generates a JavaScript function containing the rules defined in
    this map, to be used with a MapAdapter's generate_javascript
    method.  If you don't pass a name the returned JavaScript code is
    an expression that returns a function.  Otherwise it's a standalone
    script that assigns the function with that name.  Dotted names are
    resolved (so you an use a name like 'obj.url_for')

    In order to use JavaScript generation, simplejson must be installed.

    Note that using this feature will expose the rules
    defined in your map to users. If your rules contain sensitive
    information, don't use JavaScript generation!
    """
    from warnings import warn
    warn(DeprecationWarning('This module is deprecated'))
    map.update()
    rules = []
    converters = []
    for rule in map.iter_rules():
        trace = [{
            'is_dynamic':   is_dynamic,
            'data':         data
        } for is_dynamic, data in rule._trace]
        rule_converters = {}
        for key, converter in iteritems(rule._converters):
            js_func = js_to_url_function(converter)
            try:
                index = converters.index(js_func)
            except ValueError:
                converters.append(js_func)
                index = len(converters) - 1
            rule_converters[key] = index
        rules.append({
            u'endpoint':    rule.endpoint,
            u'arguments':   list(rule.arguments),
            u'converters':  rule_converters,
            u'trace':       trace,
            u'defaults':    rule.defaults
        })

    return render_template(name_parts=name and name.split('.') or [],
                           rules=dumps(rules),
                           converters=converters)


def generate_adapter(adapter, name='url_for', map_name='url_map'):
    """Generates the url building function for a map."""
    values = {
        u'server_name':     dumps(adapter.server_name),
        u'script_name':     dumps(adapter.script_name),
        u'subdomain':       dumps(adapter.subdomain),
        u'url_scheme':      dumps(adapter.url_scheme),
        u'name':            name,
        u'map_name':        map_name
    }
    return u'''\
var %(name)s = %(map_name)s(
    %(server_name)s,
    %(script_name)s,
    %(subdomain)s,
    %(url_scheme)s
);''' % values


def js_to_url_function(converter):
    """Get the JavaScript converter function from a rule."""
    if hasattr(converter, 'js_to_url_function'):
        data = converter.js_to_url_function()
    else:
        for cls in getmro(type(converter)):
            if cls in js_to_url_functions:
                data = js_to_url_functions[cls](converter)
                break
        else:
            return 'encodeURIComponent'
    return '(function(value) { %s })' % data


def NumberConverter_js_to_url(conv):
    if conv.fixed_digits:
        return u'''\
var result = value.toString();
while (result.length < %s)
    result = '0' + result;
return result;''' % conv.fixed_digits
    return u'return value.toString();'


js_to_url_functions = {
    NumberConverter:    NumberConverter_js_to_url
}
