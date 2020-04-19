<!--
• This is a README.md template we encourage you to use when you release your model.
• There are general sections we added to this template for various ML models.
• You may need to add or remove sections depending on your needs.
-->

# Project Name

## Authors
The **1st place winner** of the **4th On-device Visual Intelligence Competition** ([OVIC](https://docs.google.com/document/d/1Rxm_N7dGRyPXjyPIdRwdhZNRye52L56FozDnfYuCi0k/edit#)) of Low-Power Computer Vision Challenge ([LPCVC](https://lpcv.ai/))

* Last name, First name ([@GitHubUsername](https://github.com/username))
* Last name, First name ([@GitHubUsername](https://github.com/username))
* Last name, First name ([@GitHubUsername](https://github.com/username))

## Description
<!-- Provide description of the model -->
The model submitted for the OVIC and full implementation code for training, evaluation, and inference

* OVIC track: Image Classification, Object Detection

## Algorithm
<!-- Provide details of the algorithms used -->

## Requirements
<!--
• Provide description of the model 
• Provide brief information of the algorithms used
-->

To install requirements:

```setup
pip install -r requirements.txt
```

## Pre-trained Models

| Model | Download | MD5 checksum |
|-------|----------|--------------|
| Model Name | Download Link (Size: KB) | MD5 checksum |

The model tar file contains the followings:
* Trained model checkpoint
* Frozen trained model
* TensorFlow Lite model

## Results

### [4th OVIC Public Ranked Leaderboard](https://lpcvc.ecn.purdue.edu/score_board_r4/?contest=round4)

#### Image Classification (from the Leaderboard)

| Rank | Username | Latency | Accuracy on Classified | # Classified | Accuracy/Time | Metric | Reference Accuracy |
|------|----------|---------|------------------------|--------------|---------------|--------|--------------------|
| 1 | Username | xx.x | 0.xxxx | 20000.0 | xxx | 0.xxxxx | 0.xxxxx |

 * **Metric**: Accuracy improvement over the reference accuracy from the Pareto optimal curve
 * **Accuracy on Classified**: The accuracy in [0, 1] computed based only on the images classified within the wall-time
 * **\# Classified**: The number of images classified within the wall-time
 * **Accuracy/Time**: The accuracy divided by either the total inference time or the wall-time, whichever is longer
 * **Reference accuracy**: The reference accuracy of models from the Pareto optimal curve that have the same latency as the submission

#### Object Detection

| Rank | Username | Metric | Runtime | mAP over time | mAP of processed |
|------|----------|--------|---------|---------------|------------------|
| 1 | Username | 0.xxxxx | xxx.x | xxx | xxx |

* **Metric**: COCO mAP computed on the entire minival dataset
* **mAP over time**: COCO mAP on the minival dataset divided by latency per image
* **mAP of processed**: COCO mAP computed only on the processed images

## Dataset
<!--
• Provide detailed information of the dataset used
-->

## Training
<!--
• Provide detailed training information (preprocessing, hyperparameters, random seeds, and environment) 
• Provide a command line example for training.
-->

Please run this command line for training.

```shell
python3 ...
```

## Evaluation
<!--
• Provide evaluation script with details of how to reproduce results.
• Describe data preprocessing / postprocessing steps
• Provide a command line example for evaluation.
-->

Please run this command line for evaluation.

```shell
python3 ...
```

## References
<!-- Link to references -->

## License
<!--
• Place your license text in a file named LICENSE.txt (or LICENSE.md) in the root of the repository.
• Please also include information about your license in this README.md file.
e.g., [Adding a license to a repository](https://help.github.com/en/github/building-a-strong-community/adding-a-license-to-a-repository)
-->

This project is licensed under the terms of the **Apache License 2.0**.

## Citation
<!--
If you want to make your repository citable, please follow the instructions at [Making Your Code Citable](https://guides.github.com/activities/citable-code/)
-->

If you want to cite this repository in your research paper, please use the following information.
