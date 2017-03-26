$(function() {
  if (!EVALEX_TRUSTED) {
    initPinBox();
  }

  /**
   * if we are in console mode, show the console.
   */
  if (CONSOLE_MODE && EVALEX) {
    openShell(null, $('div.console div.inner').empty(), 0);
  }

  $('div.traceback div.frame').each(function() {
    var
      target = $('pre', this),
      consoleNode = null,
      frameID = this.id.substring(6);

    target.click(function() {
      $(this).parent().toggleClass('expanded');
    });

    /**
     * Add an interactive console to the frames
     */
    if (EVALEX && target.is('.current')) {
      $('<img src="?__debugger__=yes&cmd=resource&f=console.png">')
        .attr('title', 'Open an interactive python shell in this frame')
        .click(function() {
          consoleNode = openShell(consoleNode, target, frameID);
          return false;
        })
        .prependTo(target);
    }
  });

  /**
   * toggle traceback types on click.
   */
  $('h2.traceback').click(function() {
    $(this).next().slideToggle('fast');
    $('div.plain').slideToggle('fast');
  }).css('cursor', 'pointer');
  $('div.plain').hide();

  /**
   * Add extra info (this is here so that only users with JavaScript
   * enabled see it.)
   */
  $('span.nojavascript')
    .removeClass('nojavascript')
    .html('<p>To switch between the interactive traceback and the plaintext ' +
          'one, you can click on the "Traceback" headline.  From the text ' +
          'traceback you can also create a paste of it. ' + (!EVALEX ? '' :
          'For code execution mouse-over the frame you want to debug and ' +
          'click on the console icon on the right side.' +
          '<p>You can execute arbitrary Python code in the stack frames and ' +
          'there are some extra helpers available for introspection:' +
          '<ul><li><code>dump()</code> shows all variables in the frame' +
          '<li><code>dump(obj)</code> dumps all that\'s known about the object</ul>'));

  /**
   * Add the pastebin feature
   */
  $('div.plain form')
    .submit(function() {
      var label = $('input[type="submit"]', this);
      var old_val = label.val();
      label.val('submitting...');
      $.ajax({
        dataType:     'json',
        url:          document.location.pathname,
        data:         {__debugger__: 'yes', tb: TRACEBACK, cmd: 'paste',
                       s: SECRET},
        success:      function(data) {
          $('div.plain span.pastemessage')
            .removeClass('pastemessage')
            .text('Paste created: ')
            .append($('<a>#' + data.id + '</a>').attr('href', data.url));
        },
        error:        function() {
          alert('Error: Could not submit paste.  No network connection?');
          label.val(old_val);
        }
      });
      return false;
    });

  // if we have javascript we submit by ajax anyways, so no need for the
  // not scaling textarea.
  var plainTraceback = $('div.plain textarea');
  plainTraceback.replaceWith($('<pre>').text(plainTraceback.text()));
});

function initPinBox() {
  $('.pin-prompt form').submit(function(evt) {
    evt.preventDefault();
    var pin = this.pin.value;
    var btn = this.btn;
    btn.disabled = true;
    $.ajax({
      dataType: 'json',
      url: document.location.pathname,
      data: {__debugger__: 'yes', cmd: 'pinauth', pin: pin,
             s: SECRET},
      success: function(data) {
        btn.disabled = false;
        if (data.auth) {
          EVALEX_TRUSTED = true;
          $('.pin-prompt').fadeOut();
        } else {
          if (data.exhausted) {
            alert('Error: too many attempts.  Restart server to retry.');
          } else {
            alert('Error: incorrect pin');
          }
        }
        console.log(data);
      },
      error: function() {
        btn.disabled = false;
        alert('Error: Could not verify PIN.  Network error?');
      }
    });
  });
}

function promptForPin() {
  if (!EVALEX_TRUSTED) {
    $.ajax({
      url: document.location.pathname,
      data: {__debugger__: 'yes', cmd: 'printpin', s: SECRET}
    });
    $('.pin-prompt').fadeIn(function() {
      $('.pin-prompt input[name="pin"]').focus();
    });
  }
}


/**
 * Helper function for shell initialization
 */
function openShell(consoleNode, target, frameID) {
  promptForPin();
  if (consoleNode)
    return consoleNode.slideToggle('fast');
  consoleNode = $('<pre class="console">')
    .appendTo(target.parent())
    .hide()
  var historyPos = 0, history = [''];
  var output = $('<div class="output">[console ready]</div>')
    .appendTo(consoleNode);
  var form = $('<form>&gt;&gt;&gt; </form>')
    .submit(function() {
      var cmd = command.val();
      $.get('', {
          __debugger__: 'yes', cmd: cmd, frm: frameID, s: SECRET}, function(data) {
        var tmp = $('<div>').html(data);
        $('span.extended', tmp).each(function() {
          var hidden = $(this).wrap('<span>').hide();
          hidden
            .parent()
            .append($('<a href="#" class="toggle">&nbsp;&nbsp;</a>')
              .click(function() {
                hidden.toggle();
                $(this).toggleClass('open')
                return false;
              }));
        });
        output.append(tmp);
        command.focus();
        consoleNode.scrollTop(consoleNode.get(0).scrollHeight);
        var old = history.pop();
        history.push(cmd);
        if (typeof old != 'undefined')
          history.push(old);
        historyPos = history.length - 1;
      });
      command.val('');
      return false;
    }).
    appendTo(consoleNode);

  var command = $('<input type="text">')
    .appendTo(form)
    .keydown(function(e) {
      if (e.charCode == 100 && e.ctrlKey) {
        output.text('--- screen cleared ---');
        return false;
      }
      else if (e.charCode == 0 && (e.keyCode == 38 || e.keyCode == 40)) {
        if (e.keyCode == 38 && historyPos > 0)
          historyPos--;
        else if (e.keyCode == 40 && historyPos < history.length)
          historyPos++;
        command.val(history[historyPos]);
        return false;
      }
    });

  return consoleNode.slideDown('fast', function() {
    command.focus();
  });
}
