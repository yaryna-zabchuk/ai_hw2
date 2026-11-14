---
title: Multiple Object Video Detection
author: Yaryna Zabchuk and Yurii Fito
course: Artificial Intelligence · UCU
date: 2026-11-16
---

## Introduction

We study object detection in drone video frames from the UCU AI 2026 competition. The dataset contains eight target classes and many small, partially occluded objects. Our goal is to compare detector families under a common validation protocol and develop an inference pipeline that improves detection of small objects without changing the evaluation data.

## Data Understanding and Preprocessing

We retained the supplied split by video sequence to avoid leakage from near-duplicate frames. It contains 56 train clips and 7 validation clips with no shared clip names; one high-resolution clip contributes 34% of validation frames.

| Split | Clips | Frames | Objects | Ignored regions |
|---|---:|---:|---:|---:|
| Train | 56 | 24,201 | 1,105,516 | 63,843 |
| Validation | 7 | 2,846 | 114,132 | 3,995 |

COCO annotations with `iscrowd = 0` were used as targets; `iscrowd = 1` boxes were ignored. A check on 400 sampled frames matched YOLO and COCO boxes by class and geometry. Object size was measured as $\sqrt{wh}$: the median was 46.4 pixels and 24% of objects were COCO-small. Person and car dominated the data, while bus represented less than 1%.

![Class distribution](images/eda_class_distribution.png)

Scene density varied from 5 to 143 objects per frame, with a maximum close to 180, so evaluation used `maxDets = 300`. Ignored regions occurred in 72% of frames but covered only 2.3% of image area on average. They were mean-colour masked during training and preserved for COCO evaluation. Since 47% of objects were partly occluded, Soft-NMS was also tested.

![Object size distribution](images/eda_object_sizes.png)

![Objects per frame](images/eda_objects_per_frame.png)

At a 640-pixel long side, more than half of objects were below 16 pixels. We therefore used 1280, mild scale/colour/blur augmentation, and no validation augmentation. K-means box sizes at input 1024 spanned roughly 7-100 pixels, motivating smaller anchors, a stride-4 head and tiled inference. Main experiments used every fifth frame; temporal experiments used all validation frames.

## Faster R-CNN

**Protocol.** We used COCO-pretrained Faster R-CNN with a ResNet-50 FPN for six epochs (batch size 4, SGD with OneCycle scheduling, 1280-pixel inputs). Smaller dataset-derived anchors and a 300-box limit addressed small and crowded scenes.

**Result.**

| mAP | AP50 | AP_small | FPS |
|---:|---:|---:|---:|
| 0.2023 | 0.4271 | 0.1019 | 5.74 |

Increasing RPN proposals from 500 to 2,000 changed mAP only from 0.2000 to 0.2026 while reducing speed. GIoU box loss also failed to improve mAP (0.2009 versus 0.2023), so we retained the baseline.

The RPN first filters the image into candidate regions, while the RoI head classifies and refines them. This helps suppress background in crowded frames. However, the limited gain from extra proposals shows that proposal recall was not the only bottleneck; small-object features and precise localization remained difficult.

**Errors.** Tiny objects were often missed, dense scenes produced localization errors, and visually similar classes caused misclassification. These findings motivated later high-resolution and tiled inference.

## RetinaNet

**Protocol.** RetinaNet used the same pretrained backbone, data split, six-epoch budget and COCO evaluation as Faster R-CNN. This made the comparison between two-stage and one-stage anchor-based detectors more controlled.

**Result.**

| mAP | AP50 | AP_small | FPS |
|---:|---:|---:|---:|
| 0.1360 | 0.2681 | 0.0508 | 5.36 |

Dataset-derived smaller anchors reduced mAP to 0.0975, while a focal-loss variant reached 0.1037.

Unlike Faster R-CNN, RetinaNet predicts classes and box offsets densely at every feature-map location without a proposal filter. This creates many easy background examples in dense scenes. Focal loss was intended to reduce this imbalance, but remained below the default configuration.

**Conclusion.** Neither anchor matching nor the tested loss change solved the dense foreground/background assignment problem. The low AP_small and crowded-scene errors suggest that removing the proposal stage is costly for this dataset.

## YOLOv8: One-stage Anchor-free Detector

**Protocol.** We trained YOLOv8m for six epochs with the same 1280-pixel inputs, sampled frames, batch size and COCO evaluation. The model is anchor-free: feature-map locations predict class scores and distances to box sides, followed by NMS. We tested an extra stride-4 P2 head for small objects and a blended Normalized Wasserstein Distance (NWD) box loss.

**Results.**

| Variant | mAP | AP50 | AP_small | FPS |
|---|---:|---:|---:|---:|
| YOLOv8 baseline | 0.1996 | 0.3795 | 0.1148 | 19.69 |
| YOLOv8 + P2 head | 0.1966 | 0.3872 | 0.1413 | 14.73 |
| YOLOv8 + NWD | 0.1813 | 0.3590 | 0.1333 | 10.50 |

The P2 head slightly reduced overall mAP but improved AP_small from 0.1148 to 0.1413, confirming its value for tiny objects at a speed cost. NWD also improved small-object AP relative to the baseline, but hurt overall detection. Its weaker mAP suggests that tolerance to small localization shifts can reduce precision on medium and large objects. YOLOv8 was the fastest detector, but RT-DETR was more accurate under the same short training budget.

## RT-DETR: Query-based Detector

**Protocol.** RT-DETR-L was trained for six epochs with AdamW, the same sampled frames and input resolution, and COCO evaluation. Unlike dense CNN detectors, it predicts a fixed set of object queries. Hungarian bipartite matching assigns predictions to ground truth during training, so inference does not require the same class-wise NMS stage.

**Results.**

| Variant | mAP | AP50 | AP_small | FPS |
|---|---:|---:|---:|---:|
| RT-DETR baseline | 0.2422 | 0.4579 | 0.1214 | 7.31 |
| RT-DETR, 500 queries | **0.2508** | **0.4801** | **0.1393** | 4.97 |
| RT-DETR, higher box costs | 0.2136 | 0.4337 | 0.1266 | 4.97 |

Increasing the query budget from 300 to 500 improved all accuracy metrics, especially AP50, but reduced throughput. Increasing box and GIoU weights in the matching cost instead hurt convergence within the six-epoch budget. RT-DETR gave the best single-model mAP and handled crowded scenes without duplicate suppression, but its lower FPS motivated using it selectively in the final inference pipeline.

