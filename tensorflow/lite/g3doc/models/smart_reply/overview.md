# Smart reply
<img src="../images/smart_reply.png" class="attempt-right" />

Smart replies are contextually relevant, one-touch responses that help the user to reply to an incoming text message (or email) efficiently and effortlessly.

<a class="button button-primary" href="http://download.tensorflow.org/models/tflite/smartreply_1.0_2017_11_01.zip">Download starter model and labels</a>

## Tutorials (coming soon)
<a class="button button-primary" href="">iOS</a>
<a class="button button-primary" href="">Android</a>

## How it works
The model generates reply suggestions to input conversational chat messages with an efficient inference that can be easily be plugged in to your chat application to power on-device conversational intelligence.

The on-device model comes with several benefits. It is:
<ul>
  <li>Faster: The model resides on the device and does not require internet connectivity. Thus, the inference is very fast and has an average latency of only a few milliseconds.</li>
  <li>Resource efficient: The model has a small memory footprint on the device.</li>
  <li>Privacy-friendly: The user data never leaves the device and this eliminates any privacy restrictions.</li>
</ul>

## Example output
<img src="images/smart_reply.gif" />

## How to use this model?
We have provided a pre-built demo APK that you can download, install, and test on your phone. Go to the <a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/models/smartreply/g3doc">GitHub page</a> for instructions and list of support ops and functionalities.

## Read more about this
<ul>
  <li><a href="https://arxiv.org/pdf/1708.00630.pdf">Research paper</a></li>
  <li><a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/models/smartreply/">Source code</a></li>
</ul>

## Users
<ul>
  <li><a href="https://www.blog.google/products/gmail/save-time-with-smart-reply-in-gmail/">Gmail</a></li>
  <li><a href="https://www.blog.google/products/gmail/computer-respond-to-this-email/">Inbox</a></li>
  <li><a href="https://blog.google/products/allo/google-allo-smarter-messaging-app/">Allo</a></li>
  <li><a href="https://research.googleblog.com/2017/02/on-device-machine-intelligence.html">Smart Replies on Android Wear</a></li>
</ul>
