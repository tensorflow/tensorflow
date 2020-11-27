# TensorFlow Lite Task Library

TensorFlow Lite Task Library contains a set of powerful and easy-to-use
task-specific libraries for app developers to create ML experiences with TFLite.
It provides optimized out-of-box model interfaces for popular machine learning
tasks, such as image classification, question and answer, etc. The model
interfaces are specifically designed for each task to achieve the best
performance and usability. Task Library works cross-platform and is supported on
Java, C++, and Swift.

## What to expect from the Task Library

*   **Clean and well-defined APIs usable by non-ML-experts** \
    Inference can be done within just 5 lines of code. Use the powerful and
    easy-to-use APIs in the Task library as building blocks to help you easily
    develop ML with TFLite on mobile devices.

*   **Complex but common data processing** \
    Supports common vision and natural language processing logic to convert
    between your data and the data format required by the model. Provides the
    same, shareable processing logic for training and inference.

*   **High performance gain** \
    Data processing would take no more than a few milliseconds, ensuring the
    fast inference experience using TensorFlow Lite.

*   **Extensibility and customization** \
    You can leverage all benefits the Task Library infrastructure provides and
    easily build your own Android/iOS inference APIs.

## Supported tasks

Below is the list of the supported task types. The list is expected to grow as
we continue enabling more and more use cases.

*   **Vision APIs**

    *   [ImageClassifier](image_classifier.md)
    *   [ObjectDetector](object_detector.md)
    *   [ImageSegmenter](image_segmenter.md)

*   **Natural Language (NL) APIs**

    *   [NLClassifier](nl_classifier.md)
    *   [BertNLCLassifier](bert_nl_classifier.md)
    *   [BertQuestionAnswerer](bert_question_answerer.md)

*   **Custom APIs**

    *   Extend Task API infrastructure and build
        [customized API](customized_task_api.md).
