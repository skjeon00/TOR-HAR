# Temporal Over-Resolution in Wearable Human Activity Recognition: A Controlled Study of Reduced Temporal Density <br>
<img width="1376" height="462" alt="budget_har" src="https://github.com/user-attachments/assets/9a9ae7c5-a649-4f09-a155-8f503901d061" />

## Paper Overview
**Abstract:** Wearable human activity recognition (HAR) pipelines commonly process inertial signals at the native sampling rate, although this temporal resolution may be finer than required for task discrimination. Prior studies have shown that lower sampling rates can preserve HAR accuracy, but they provide limited evidence that this behavior reflects temporal redundancy rather than reduction- or model-specific effects. We investigate this issue across multiple HAR datasets using operator-diverse, budget-matched, paired, and receptive-field-matched analyses. Quarter-resolution processing preserves or improves mean recognition performance on the three primary benchmarks while reducing model FLOPs by approximately 75%, whereas PPG-DaLiA shows a narrower operating region: half-resolution average pooling nearly preserves the full-resolution mean, while quarter-resolution processing produces clearer degradation. Across the operator-diverse sweeps, performance retention is observed under different reduction operators; under matched downstream model-computation budgets, full-span reduced-resolution processing outperforms selected native rate evidence; and receptive-field-matched controls show no consistent mean advantage for fourfold denser native-rate inputs. These findings provide converging mean-level evidence consistent with temporal over-resolution, supporting temporal resolution as a task-dependent model-input design variable rather than fixed acquisition metadata.

## Dataset
This repository does not include datasets. Please download them from the official sources below and configure the dataset path accordingly.
- **UCI-HAR** dataset is available at https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **PAMAP2** dataset is available at
- **MotionSense** dataset is available at https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset
- **PPG-DaLiA** dataset is available at https://archive.ics.uci.edu/dataset/495/ppg+dalia
- **USC-HAD** dataset is available at

## Contact
For questions or issues, please contact:
- Seokyeong Jeon : seo1270@gachon.ac.kr
