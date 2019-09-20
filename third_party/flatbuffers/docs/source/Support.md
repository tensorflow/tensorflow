Platform / Language / Feature support    {#flatbuffers_support}
=====================================

FlatBuffers is actively being worked on, which means that certain platform /
language / feature combinations may not be available yet.

This page tries to track those issues, to make informed decisions easier.
In general:

  * Languages: language support beyond the ones created by the original
    FlatBuffer authors typically depends on community contributions.
  * Features: C++ was the first language supported, since our original
    target was high performance game development. It thus has the richest
    feature set, and is likely most robust. Other languages are catching up
    however.
  * Platforms: All language implementations are typically portable to most
    platforms, unless where noted otherwise.

NOTE: this table is a start, it needs to be extended.

Feature                        | C++    | Java   | C#     | Go     | Python | JS        | TS        | C       | PHP | Dart    | Lobster | Rust
------------------------------ | ------ | ------ | ------ | ------ | ------ | --------- | --------- | ------  | --- | ------- | ------- | ----
Codegen for all basic features | Yes    | Yes    | Yes    | Yes    | Yes    | Yes       | Yes       | Yes     | WiP | Yes     | Yes     | Yes
JSON parsing                   | Yes    | No     | No     | No     | No     | No        | No        | Yes     | No  | No      | Yes     | No
Simple mutation                | Yes    | Yes    | Yes    | Yes    | No     | No        | No        | No      | No  | No      | No      | No
Reflection                     | Yes    | No     | No     | No     | No     | No        | No        | Basic   | No  | No      | No      | No
Buffer verifier                | Yes    | No     | No     | No     | No     | No        | No        | Yes     | No  | No      | No      | No
Testing: basic                 | Yes    | Yes    | Yes    | Yes    | Yes    | Yes       | Yes       | Yes     | ?   | Yes     | Yes     | Yes
Testing: fuzz                  | Yes    | No     | No     | Yes    | Yes    | No        | No        | No      | ?   | No      | No      | Yes
Performance:                   | Superb | Great  | Great  | Great  | Ok     | ?         | ?         | Superb  | ?   | ?       | Great   | Superb
Platform: Windows              | VS2010 | Yes    | Yes    | ?      | ?      | ?         | Yes       | VS2010  | ?   | Yes     | Yes     | Yes
Platform: Linux                | GCC282 | Yes    | ?      | Yes    | Yes    | ?         | Yes       | Yes     | ?   | Yes     | Yes     | Yes
Platform: OS X                 | Xcode4 | ?      | ?      | ?      | Yes    | ?         | Yes       | Yes     | ?   | Yes     | Yes     | Yes
Platform: Android              | NDK10d | Yes    | ?      | ?      | ?      | ?         | ?         | ?       | ?   | Flutter | Yes     | ?
Platform: iOS                  | ?      | ?      | ?      | ?      | ?      | ?         | ?         | ?       | ?   | Flutter | Yes     | ?
Engine: Unity                  | ?      | ?      | Yes    | ?      | ?      | ?         | ?         | ?       | ?   | ?       | No      | ?
Primary authors (github)       | aard*  | aard*  | ev*/js*| rw     | rw     | evanw/ev* | kr*       | mik*    | ch* | dnfield | aard*   | rw

  * aard = aardappel (previously: gwvo)
  * ev = evolutional
  * js = jonsimantov
  * mik = mikkelfj
  * ch = chobie
  * kr = krojew

<br>
