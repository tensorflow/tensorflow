# -*- coding: utf-8 -*-
"""
    werkzeug.testapp
    ~~~~~~~~~~~~~~~~

    Provide a small test application that can be used to test a WSGI server
    and check it for WSGI compliance.

    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
import os
import sys
import werkzeug
from textwrap import wrap
from werkzeug.wrappers import BaseRequest as Request, BaseResponse as Response
from werkzeug.utils import escape
import base64

logo = Response(base64.b64decode('''
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
='''), mimetype='image/png')


TEMPLATE = u'''\
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"
  "http://www.w3.org/TR/html4/loose.dtd">
<title>WSGI Information</title>
<style type="text/css">
  @import url(http://fonts.googleapis.com/css?family=Ubuntu);

  body       { font-family: 'Lucida Grande', 'Lucida Sans Unicode', 'Geneva',
               'Verdana', sans-serif; background-color: white; color: #000;
               font-size: 15px; text-align: center; }
  #logo      { float: right; padding: 0 0 10px 10px; }
  div.box    { text-align: left; width: 45em; margin: auto; padding: 50px 0;
               background-color: white; }
  h1, h2     { font-family: 'Ubuntu', 'Lucida Grande', 'Lucida Sans Unicode',
               'Geneva', 'Verdana', sans-serif; font-weight: normal; }
  h1         { margin: 0 0 30px 0; }
  h2         { font-size: 1.4em; margin: 1em 0 0.5em 0; }
  table      { width: 100%%; border-collapse: collapse; border: 1px solid #AFC5C9 }
  table th   { background-color: #AFC1C4; color: white; font-size: 0.72em;
               font-weight: normal; width: 18em; vertical-align: top;
               padding: 0.5em 0 0.1em 0.5em; }
  table td   { border: 1px solid #AFC5C9; padding: 0.1em 0 0.1em 0.5em; }
  code       { font-family: 'Consolas', 'Monaco', 'Bitstream Vera Sans Mono',
               monospace; font-size: 0.7em; }
  ul li      { line-height: 1.5em; }
  ul.path    { font-size: 0.7em; margin: 0 -30px; padding: 8px 30px;
               list-style: none; background: #E8EFF0; }
  ul.path li { line-height: 1.6em; }
  li.virtual { color: #999; text-decoration: underline; }
  li.exp     { background: white; }
</style>
<div class="box">
  <img src="?resource=logo" id="logo" alt="[The Werkzeug Logo]" />
  <h1>WSGI Information</h1>
  <p>
    This page displays all available information about the WSGI server and
    the underlying Python interpreter.
  <h2 id="python-interpreter">Python Interpreter</h2>
  <table>
    <tr>
      <th>Python Version
      <td>%(python_version)s
    <tr>
      <th>Platform
      <td>%(platform)s [%(os)s]
    <tr>
      <th>API Version
      <td>%(api_version)s
    <tr>
      <th>Byteorder
      <td>%(byteorder)s
    <tr>
      <th>Werkzeug Version
      <td>%(werkzeug_version)s
  </table>
  <h2 id="wsgi-environment">WSGI Environment</h2>
  <table>%(wsgi_env)s</table>
  <h2 id="installed-eggs">Installed Eggs</h2>
  <p>
    The following python packages were installed on the system as
    Python eggs:
  <ul>%(python_eggs)s</ul>
  <h2 id="sys-path">System Path</h2>
  <p>
    The following paths are the current contents of the load path.  The
    following entries are looked up for Python packages.  Note that not
    all items in this path are folders.  Gray and underlined items are
    entries pointing to invalid resources or used by custom import hooks
    such as the zip importer.
  <p>
    Items with a bright background were expanded for display from a relative
    path.  If you encounter such paths in the output you might want to check
    your setup as relative paths are usually problematic in multithreaded
    environments.
  <ul class="path">%(sys_path)s</ul>
</div>
'''


def iter_sys_path():
    if os.name == 'posix':
        def strip(x):
            prefix = os.path.expanduser('~')
            if x.startswith(prefix):
                x = '~' + x[len(prefix):]
            return x
    else:
        strip = lambda x: x

    cwd = os.path.abspath(os.getcwd())
    for item in sys.path:
        path = os.path.join(cwd, item or os.path.curdir)
        yield strip(os.path.normpath(path)), \
            not os.path.isdir(path), path != item


def render_testapp(req):
    try:
        import pkg_resources
    except ImportError:
        eggs = ()
    else:
        eggs = sorted(pkg_resources.working_set,
                      key=lambda x: x.project_name.lower())
    python_eggs = []
    for egg in eggs:
        try:
            version = egg.version
        except (ValueError, AttributeError):
            version = 'unknown'
        python_eggs.append('<li>%s <small>[%s]</small>' % (
            escape(egg.project_name),
            escape(version)
        ))

    wsgi_env = []
    sorted_environ = sorted(req.environ.items(),
                            key=lambda x: repr(x[0]).lower())
    for key, value in sorted_environ:
        wsgi_env.append('<tr><th>%s<td><code>%s</code>' % (
            escape(str(key)),
            ' '.join(wrap(escape(repr(value))))
        ))

    sys_path = []
    for item, virtual, expanded in iter_sys_path():
        class_ = []
        if virtual:
            class_.append('virtual')
        if expanded:
            class_.append('exp')
        sys_path.append('<li%s>%s' % (
            class_ and ' class="%s"' % ' '.join(class_) or '',
            escape(item)
        ))

    return (TEMPLATE % {
        'python_version':   '<br>'.join(escape(sys.version).splitlines()),
        'platform':         escape(sys.platform),
        'os':               escape(os.name),
        'api_version':      sys.api_version,
        'byteorder':        sys.byteorder,
        'werkzeug_version': werkzeug.__version__,
        'python_eggs':      '\n'.join(python_eggs),
        'wsgi_env':         '\n'.join(wsgi_env),
        'sys_path':         '\n'.join(sys_path)
    }).encode('utf-8')


def test_app(environ, start_response):
    """Simple test application that dumps the environment.  You can use
    it to check if Werkzeug is working properly:

    .. sourcecode:: pycon

        >>> from werkzeug.serving import run_simple
        >>> from werkzeug.testapp import test_app
        >>> run_simple('localhost', 3000, test_app)
         * Running on http://localhost:3000/

    The application displays important information from the WSGI environment,
    the Python interpreter and the installed libraries.
    """
    req = Request(environ, populate_request=False)
    if req.args.get('resource') == 'logo':
        response = logo
    else:
        response = Response(render_testapp(req), mimetype='text/html')
    return response(environ, start_response)


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, test_app, use_reloader=True)
