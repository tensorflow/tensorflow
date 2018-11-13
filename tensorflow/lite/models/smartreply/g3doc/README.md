# Smart Reply Model

## What is On-Device Smart Reply Model?

Smart Replies are contextually relevant, one-touch responses that help the user
to reply to an incoming text message (or email) efficiently and effortlessly.
Smart Replies have been highly successful across several Google products
including
[Gmail](https://www.blog.google/products/gmail/save-time-with-smart-reply-in-gmail/),
[Inbox](https://www.blog.google/products/gmail/computer-respond-to-this-email/)
and
[Allo](https://blog.google/products/allo/google-allo-smarter-messaging-app/).

The On-device Smart Reply model is targeted towards text chat use cases. It has
a completely different architecture from its cloud-based counterparts, and is
built specifically for memory constraints devices such as phones & watches. It
has been successfully used to provide [Smart Replies on Android
Wear](https://research.googleblog.com/2017/02/on-device-machine-intelligence.html)
to all first- & third-party apps.

The on-device model comes with several benefits. It is:

*   **Faster**: The model resides on the device and does not require internet
    connectivity. Thus, the inference is very fast and has an average latency of
    only a few milliseconds.
*   **Resource efficient**: The model has a small memory footprint on
    the device.
*   **Privacy-friendly**: The user data never leaves the device and this
    eliminates any privacy restrictions.

A caveat, though, is that the on-device model has lower triggering rate than its
cloud counterparts (triggering rate is the percentage of times the model
suggests a response for an incoming message).

## When to use this Model?

The On-Device Smart Reply model is aimed towards improving the messaging
experience for day-to-day conversational chat messages. We recommend using this
model for similar use cases. Some sample messages on which the model does well
are provided in this [tsv
file](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/models/testdata/smartreply_samples.tsv)
for reference. The file format is:

```
   {incoming_message  smart_reply1   [smart_reply2]   [smart_reply3]}
```

For the current model, we see a triggering rate of about 30-40% for messages
which are similar to those provided in the tsv file above.

In case the model does not trigger any response, the system falls back to
suggesting replies from a fixed back-off set that was compiled from popular
response intents observed in chat conversations. Some of the fallback responses
are `Ok, Yes, No, 👍, ☺`.

The model can only be used for inference at this time (i.e. it cannot be custom
trained). If you are interested to know how the model was trained, please refer
to this [blog
post](https://research.googleblog.com/2017/02/on-device-machine-intelligence.html)
and [research paper](https://arxiv.org/pdf/1708.00630).

## How to use this Model?

We have provided a pre-built demo APK that you can download, install and test on
your phone ([demo APK
here](http://download.tensorflow.org/deps/tflite/SmartReplyDemo.apk)).

The On-Device Smart Reply demo App works in the following way:

1.  Android app links to the JNI binary with a predictor library.

2.  In the predictor library, `GetSegmentPredictions` is called with a list of input
    strings.

    2.1 The input string can be 1-3 most recent messages of the conversations in
    form of string vector. The model will run on these input sentences and
    provide Smart Replies corresponding to them.

    2.2 The function performs some preprocessing on input data which includes:

    *   Sentence splitting: The input message will be split into sentences if
        message has more than one sentence. Eg: a message like “How are you?
        Want to grab lunch?” will be broken down into 2 different sentences.
    *   Normalization: The individual sentences will be normalized by converting
        them into lower cases, removing unnecessary punctuations, etc. Eg: “how
        are you????” will be converted to “how are you?” (refer for NORMALIZE op
        for more details).

        The input string content will be converted to tensors.

    2.3 The function then runs the prediction model on the input tensors.

    2.4 The function also performs some post-processing which includes
    aggregating the model predictions for the input sentences from 2.2 and
    returning the appropriate responses.

3.  Finally, it gets response(s) from `std::vector<PredictorResponse>`, and
    returns back to Android app. Responses are sorted in descending order of
    confidence score.

## Ops and Functionality Supported

Following are the ops supported for using On-Device Smart Reply model:

*   **NORMALIZE**

    This is a custom op which normalizes the sentences by:

    *   Converting all sentences into lower case.
    *   Removing unnecessary punctuations (eg: “how are you????” → “how are
        you?”).
    *   Expanding sentences wherever necessary (eg: “ I’m home” → “I am home”).

*   **SKIP_GRAM**

    This is an op inside TensorFlow Lite that converts sentences into a list of
    skip grams. The configurable parameters are `ngram_size` and
    `max_skip_size`. For the model provided, the values for these parameters are
    set to 3 & 2 respectively.

*   **EXTRACT_FEATURES**

    This is a custom op that hashes skip grams to features represented as
    integers. Longer skip-grams are allocated higher weights.

*   **LSH_PROJECTION**

    This is an op inside TensorFlow Lite that projects input features to a
    corresponding bit vector space using Locality Sensitive Hashing (LSH).

*   **PREDICT**

    This is a custom op that runs the input features through the projection
    model (details [here](https://arxiv.org/pdf/1708.00630.pdf)), computes the
    appropriate response labels along with weights for the projected features,
    and aggregates the response labels and weights together.

*   **HASHTABLE_LOOKUP**

    This is an op inside TensorFlow Lite that uses label id from predict op and
    looks up the response text from the given label id.

## Further Information

*   Open source code
    [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/models/smartreply/).
