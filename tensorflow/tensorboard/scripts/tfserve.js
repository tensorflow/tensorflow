/**
 * Simple server for running TensorBoard during development.
 */

var express = require('express');
var http = require('http');
var fs = require('fs');
var parseUrl = require('url').parse;
var send = require('send');

function pathExists(path) {
  try {
    fs.statSync(path);
    return true;
  } catch (e) {
    return false;
  }
}

function serveTensorBoard(options) {
  var logger = options.verbose ? console.log : new Function();

  function serveFile(req, res) {
    var url = parseUrl(req.url, true);
    var splitPath = url.pathname.split('/').slice(1);
    var filePath = splitPath.join('/');
    logger('serve file path:', filePath);
    send(req, filePath).pipe(res);
  }

  options.port = options.port || 8000;
  options.host = options.host || 'localhost';
  console.log('Serving TensorBoard at', options.host, ':', options.port);
  logger('Serving in verbose mode.');
  var app = express();
  app.get('/', function(req, res) {
    // redirect to the demo page
    logger('Redirecting from / to the demo page');
    res.redirect(301, '/components/tf-tensorboard/demo/index.html');
  });
  app.get('/components/imports/local-imports/*', serveFile);
  app.get('/components/imports/*', function(req, res) {
    var url = parseUrl(req.url, true).pathname;
    var newUrl = url.replace('imports', 'imports/local-imports');
    logger('redirect:', url, '->', newUrl);
    res.redirect(301, newUrl);
  });
  app.get('/components/*', function(req, res) {
    // serve from bower_components if possible, components otherwise
    var url = parseUrl(req.url, true);
    var splitPath = url.pathname.split('/').slice(2);
    var bowerPath = ['bower_components'].concat(splitPath).join('/');
    var componentsPath = ['components'].concat(splitPath).join('/');
    var path;
    if (pathExists(bowerPath)) {
      path = bowerPath;
    } else if (pathExists(componentsPath)) {
      path = componentsPath;
    } else {
      console.error('Unable to find path:', componentsPath);
      res.status(404).send('404 - couldnt find', componentsPath);
      return;
    }
    logger('sending file:', path);
    send(req, path).pipe(res);
  });
  app.get('*', serveFile);

  var server = http.createServer(app);
  server.listen(options.port, options.host);

  server.on('error', function(err) {
    if (err.code === 'EADDRINUSE') {
      console.error('tfserve.js: Error - Port in use:', options.port);
    }
  });
}

module.exports = serveTensorBoard;
