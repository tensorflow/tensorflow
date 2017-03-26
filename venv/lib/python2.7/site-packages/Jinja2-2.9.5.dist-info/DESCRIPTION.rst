Jinja2
~~~~~~

Jinja2 is a template engine written in pure Python.  It provides a
`Django`_ inspired non-XML syntax but supports inline expressions and
an optional `sandboxed`_ environment.

Nutshell
--------

Here a small example of a Jinja template::

    {% extends 'base.html' %}
    {% block title %}Memberlist{% endblock %}
    {% block content %}
      <ul>
      {% for user in users %}
        <li><a href="{{ user.url }}">{{ user.username }}</a></li>
      {% endfor %}
      </ul>
    {% endblock %}

Philosophy
----------

Application logic is for the controller but don't try to make the life
for the template designer too hard by giving him too few functionality.

For more informations visit the new `Jinja2 webpage`_ and `documentation`_.

.. _sandboxed: http://en.wikipedia.org/wiki/Sandbox_(computer_security)
.. _Django: http://www.djangoproject.com/
.. _Jinja2 webpage: http://jinja.pocoo.org/
.. _documentation: http://jinja.pocoo.org/2/documentation/


