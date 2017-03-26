Flask
-----

Flask is a microframework for Python based on Werkzeug, Jinja 2 and good
intentions. And before you ask: It's BSD licensed!

Flask is Fun
````````````

Save in a hello.py:

.. code:: python

    from flask import Flask
    app = Flask(__name__)

    @app.route("/")
    def hello():
        return "Hello World!"

    if __name__ == "__main__":
        app.run()

And Easy to Setup
`````````````````

And run it:

.. code:: bash

    $ pip install Flask
    $ python hello.py
     * Running on http://localhost:5000/

 Ready for production? `Read this first <http://flask.pocoo.org/docs/deploying/>`.

Links
`````

* `website <http://flask.pocoo.org/>`_
* `documentation <http://flask.pocoo.org/docs/>`_
* `development version
  <http://github.com/pallets/flask/zipball/master#egg=Flask-dev>`_



