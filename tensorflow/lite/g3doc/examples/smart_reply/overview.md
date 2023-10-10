# Smart reply

<img src="../images/smart_reply.png" class="attempt-right" />

## Get started

Our smart reply model generates reply suggestions based on chat messages. The
suggestions are intended to be contextually relevant, one-touch responses that
help the user to easily reply to an incoming message.

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/smartreply/1/default/1?lite-format=tflite">Download
starter model</a>

### Sample application

There is a TensorFlow Lite sample application that demonstrates the smart reply
model on Android.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/smart_reply/android">View
Android example</a>

Read the
[GitHub page](https://github.com/tensorflow/examples/tree/master/lite/examples/smart_reply/android/)
to learn how the app works. Inside this project, you will also learn how to
build an app with custom C++ ops.

## How it works

The model generates reply suggestions to conversational chat messages.

The on-device model comes with several benefits. It is:
<ul>
  <li>Fast: The model resides on the device and does not require internet connectivity. Thus, inference is very fast and has an average latency of only a few milliseconds.</li>
  <li>Resource efficient: The model has a small memory footprint on the device.</li>
  <li>Privacy-friendly: User data never leaves the device.</li>
</ul>

## Example output

<img alt="Animation showing smart reply" src="images/smart_reply.gif" style="max-width: 300px"/>

## Read more about this

<ul>
  <li><a href="https://arxiv.org/pdf/1708.00630.pdf">Research paper</a></li>
  <li><a href="https://github.com/tensorflow/examples/tree/master/lite/examples/smart_reply/android">Source code</a></li>
</ul>

## Users

<ul>
  <li><a href="https://www.blog.google/products/gmail/save-time-with-smart-reply-in-gmail/">Gmail</a></li>
  <li><a href="https://www.blog.google/products/gmail/computer-respond-to-this-email/">Inbox</a></li>
  <li><a href="https://blog.google/products/allo/google-allo-smarter-messaging-app/">Allo</a></li>
  <li><a href="https://research.googleblog.com/2017/02/on-device-machine-intelligence.html">Smart Replies on Android Wear</a></li>
</ul>
