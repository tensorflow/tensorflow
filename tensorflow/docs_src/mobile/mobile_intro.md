# Introduction to TensorFlow Mobile

TensorFlow was designed from the ground up to be a good deep learning solution
for mobile platforms like Android and iOS. This mobile guide should help you
understand how machine learning can work on mobile platforms and how to
integrate TensorFlow into your mobile apps effectively and efficiently.

## About this Guide

This guide is aimed at developers who have a TensorFlow model that’s
successfully working in a desktop environment, who want to integrate it into
a mobile application, and cannot use TensorFlow Lite. Here are the
main challenges you’ll face during that process:

- Understanding how to use Tensorflow for mobile.
- Building TensorFlow for your platform.
- Integrating the TensorFlow library into your application.
- Preparing your model file for mobile deployment.
- Optimizing for latency, RAM usage, model file size, and binary size.

## Common use cases for mobile machine learning

**Why run TensorFlow on mobile?**

Traditionally, deep learning has been associated with data centers and giant
clusters of high-powered GPU machines. However, it can be very expensive and
time-consuming to send all of the data a device has access to across a network
connection. Running on mobile makes it possible to deliver very interactive
applications in a way that’s not possible when you have to wait for a network
round trip.

Here are some common use cases for on-device deep learning:

### Speech Recognition

There are a lot of interesting applications that can be built with a
speech-driven interface, and many of these require on-device processing. Most of
the time a user isn’t giving commands, and so streaming audio continuously to a
remote server would be a waste of bandwidth, since it would mostly be silence or
background noises. To solve this problem it’s common to have a small neural
network running on-device @{$tutorials/audio_recognition$listening out for a particular keyword}.
Once that keyword has been spotted, the rest of the
conversation can be transmitted over to the server for further processing if
more computing power is needed.

### Image Recognition

It can be very useful for a mobile app to be able to make sense of a camera
image. If your users are taking photos, recognizing what’s in them can help your
camera apps apply appropriate filters, or label the photos so they’re easily
findable. It’s important for embedded applications too, since you can use image
sensors to detect all sorts of interesting conditions, whether it’s spotting
endangered animals in the wild
or
[reporting how late your train is running](https://svds.com/tensorflow-image-recognition-raspberry-pi/).

TensorFlow comes with several examples of recognizing the types of objects
inside images along with a variety of different pre-trained models, and they can
all be run on mobile devices. You can try out
our
[Tensorflow for Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html#0) and
[Tensorflow for Poets 2: Optimize for Mobile](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/index.html#0) codelabs to
see how to take a pretrained model and run some very fast and lightweight
training to teach it to recognize specific objects, and then optimize it to
run on mobile.

### Object Localization

Sometimes it’s important to know where objects are in an image as well as what
they are. There are lots of augmented reality use cases that could benefit a
mobile app, such as guiding users to the right component when offering them
help fixing their wireless network or providing informative overlays on top of
landscape features. Embedded applications often need to count objects that are
passing by them, whether it’s pests in a field of crops, or people, cars and
bikes going past a street lamp.

TensorFlow offers a pretrained model for drawing bounding boxes around people
detected in images, together with tracking code to follow them over time. The
tracking is especially important for applications where you’re trying to count
how many objects are present over time, since it gives you a good idea when a
new object enters or leaves the scene. We have some sample code for this
available for Android [on
Github](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android),
and also a [more general object detection
model](https://github.com/tensorflow/models/tree/master/object_detection/README.md)
available as well.

### Gesture Recognition

It can be useful to be able to control applications with hand or other
gestures, either recognized from images or through analyzing accelerometer
sensor data. Creating those models is beyond the scope of this guide, but
TensorFlow is an effective way of deploying them.

### Optical Character Recognition

Google Translate’s live camera view is a great example of how effective
interactive on-device detection of text can be.

<div class="video-wrapper">
  <iframe class="devsite-embedded-youtube-video" data-video-id="06olHmcJjS0"
            data-autohide="1" data-showinfo="0" frameborder="0" allowfullscreen>
  </iframe>
</div>

There are multiple steps involved in recognizing text in images. You first have
to identify the areas where the text is present, which is a variation on the
object localization problem, and can be solved with similar techniques. Once you
have an area of text, you then need to interpret it as letters, and then use a
language model to help guess what words they represent. The simplest way to
estimate what letters are present is to segment the line of text into individual
letters, and then apply a simple neural network to the bounding box of each. You
can get good results with the kind of models used for MNIST, which you can find
in TensorFlow’s tutorials, though you may want a higher-resolution input.  A
more advanced alternative is to use an LSTM model to process a whole line of
text at once, with the model itself handling the segmentation into different
characters.

### Translation

Translating from one language to another quickly and accurately, even if you
don’t have a network connection, is an important use case. Deep networks are
very effective at this sort of task, and you can find descriptions of a lot of
different models in the literature. Often these are sequence-to-sequence
recurrent models where you’re able to run a single graph to do the whole
translation, without needing to run separate parsing stages.

### Text Classification

If you want to suggest relevant prompts to users based on what they’re typing or
reading, it can be very useful to understand the meaning of the text. This is
where text classification comes in. Text classification is an umbrella term
that covers everything from sentiment analysis to topic discovery. You’re likely
to have your own categories or labels that you want to apply, so the best place
to start is with an example
like
[Skip-Thoughts](https://github.com/tensorflow/models/tree/master/skip_thoughts/),
and then train on your own examples.

### Voice Synthesis

A synthesized voice can be a great way of giving users feedback or aiding
accessibility, and recent advances such as
[WaveNet](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) show
that deep learning can offer very natural-sounding speech.

## Mobile machine learning and the cloud

These examples of use cases give an idea of how on-device networks can
complement cloud services. Cloud has a great deal of computing power in a
controlled environment, but running on devices can offer higher interactivity.
In situations where the cloud is unavailable, or your cloud capacity is limited,
you can provide an offline experience, or reduce cloud workload by processing
easy cases on device.

Doing on-device computation can also signal when it's time to switch to working
on the cloud. A good example of this is hotword detection in speech. Since
devices are able to constantly listen out for the keywords, this then triggers a
lot of traffic to cloud-based speech recognition once one is recognized. Without
the on-device component, the whole application wouldn’t be feasible, and this
pattern exists across several other applications as well. Recognizing that some
sensor input is interesting enough for further processing makes a lot of
interesting products possible.

## What hardware and software should you have?

TensorFlow runs on Ubuntu Linux, Windows 10, and OS X. For a list of all
supported operating systems and instructions to install TensorFlow, see
@{$install$Installing Tensorflow}.

Note that some of the sample code we provide for mobile TensorFlow requires you
to compile TensorFlow from source, so you’ll need more than just `pip install`
to work through all the sample code.

To try out the mobile examples, you’ll need a device set up for development,
using
either [Android Studio](https://developer.android.com/studio/install.html),
or [XCode](https://developer.apple.com/xcode/) if you're developing for iOS.

## What should you do before you get started?

Before thinking about how to get your solution on mobile:

1. Determine whether your problem is solvable by mobile machine learning
2. Create a labelled dataset to define your problem
3. Pick an effective model for the problem

We'll discuss these in more detail below.

### Is your problem solvable by mobile machine learning?

Once you have an idea of the problem you want to solve, you need to make a plan
of how to build your solution. The most important first step is making sure that
your problem is actually solvable, and the best way to do that is to mock it up
using humans in the loop.

For example, if you want to drive a robot toy car using voice commands, try
recording some audio from the device and listen back to it to see if you can
make sense of what’s being said. Often you’ll find there are problems in the
capture process, such as the motor drowning out speech or not being able to hear
at a distance, and you should tackle these problems before investing in the
modeling process.

Another example would be giving photos taken from your app to people see if they
can classify what’s in them, in the way you’re looking for. If they can’t do
that (for example, trying to estimate calories in food from photos may be
impossible because all white soups look the same), then you’ll need to redesign
your experience to cope with that. A good rule of thumb is that if a human can’t
handle the task then it will be difficult to train a computer to do better.

### Create a labelled dataset

After you’ve solved any fundamental issues with your use case, you need to
create a labeled dataset to define what problem you’re trying to solve. This
step is extremely important, moreso than picking which model to use. You want it
to be as representative as possible of your actual use case, since the model
will only be effective at the task you teach it. It’s also worth investing in
tools to make labeling the data as efficient and accurate as possible. For
example, if you’re able to switch from having to click a button on a web
interface to simple keyboard shortcuts, you may be able to speed up the
generation process a lot. You should also start by doing the initial labeling
yourself, so you can learn about the difficulties and likely errors, and
possibly change your labeling or data capture process to avoid them. Once you
and your team are able to consistently label examples (that is once you
generally agree on the same labels for most examples), you can then try and
capture your knowledge in a manual and teach external raters how to run the same
process.

### Pick an effective model

The next step is to pick an effective model to use. You might be able to avoid
training a model from scratch if someone else has already implemented a model
similar to what you need; we have a repository of models implemented in
TensorFlow [on Github](https://github.com/tensorflow/models) that you can look
through. Lean towards the simplest model you can find, and try to get started as
soon as you have even a small amount of labelled data, since you’ll get the best
results when you’re able to iterate quickly. The shorter the time it takes to
try training a model and running it in s real application, the better overall
results you’ll see. It’s common for an algorithm to get great training accuracy
numbers but then fail to be useful within a real application because there’s a
mismatch between the dataset and real usage. Prototype end-to-end usage as soon
as possible to create a consistent user experience.

## Next Steps

We suggest you get started by building one of our demos for
@{$mobile/android_build$Android} or @{$mobile/ios_build$iOS}.
