# Automated Detection of Maxillary Sinus Opacifications from CT Images

Research code accompanying the paper **“Automated detection of maxillary sinus
opacifications compatible with sinusitis from CT images”**, published in
*Dentomaxillofacial Radiology* (2024).

> Kwon KW, Kim J, Kang D. Dentomaxillofac Radiol. 2024;53(8):549–557.  
> https://doi.org/10.1093/dmfr/twae042

## Overview

This project uses YOLOv8 to localize the left and right maxillary sinuses in
coronal CT images and classify each region as:

- `normal`
- `cyst`
- `sinusitis`

The published study used 1,080 coronal CT images containing 2,158 maxillary
sinuses. Images were split into 648 training, 216 validation, and 216 test
images. The primary YOLOv8n transfer-learning model achieved 97.1% overall
precision, 93.8% recall, 96.6% mAP50, 85.3% mAP50–95, and a 95.4% F1-score on
the test set. Please refer to the paper for the complete study design and
results.

## Data availability

**The CT images, annotations, trained weights, and patient-level data are not
included in this repository.** The source data are clinical data governed by
institutional approval and patient privacy requirements, and cannot be made
public through this repository.

Consequently, this repository documents the training and evaluation pipeline,
but the published results cannot be reproduced without an independently
authorized dataset prepared in the same format. No synthetic or example
medical images are presented as study data.

## Repository layout

```text
train.py                                   Original training/evaluation script retained from the study workspace
test.py                                    Original test-set evaluation script
cuda.py                                    GPU availability check used during development
weight                                     Original weight-inspection snippet, kept under its original filename
ultralytics/                               Ultralytics 8.0.170 source with study-specific modifications
ultralytics/cfg/datasets/sinus_data.yaml   Original dataset configuration retained from the workspace
configs/sinus_ct.yaml                      Clean public dataset configuration template
src/train.py                               Reconstructed command-line training entry point
src/evaluate.py                            Reconstructed command-line test evaluator
homework2.ipynb                            Unrelated MNIST/PCA/SVM coursework retained for archival completeness
requirements.txt                           Original workspace dependencies
CITATION.cff                               Article citation metadata
```

The root-level scripts and modified `ultralytics/` package are retained from
the original research workspace. The scripts under `src/` were reconstructed
later from the retained workspace and the published methods to provide a
clearer command-line interface. They are not represented as the exact scripts
executed for every result reported in the article.

### Study-specific Ultralytics modifications

The retained workspace includes changes to Ultralytics configuration, metrics,
and visualization code, including:

- training image size and epoch defaults;
- a maximum of two detections per image;
- augmentation settings used in intermediate experiments;
- metric labels and confusion-matrix text size;
- suppression of filenames and bounding-box labels in selected result plots.

Some retained files represent intermediate experiments and may differ from the
final protocol reported in the paper. Where the code and article differ, the
peer-reviewed article is the authoritative description of the published
experiment.

## Environment

The original experiments used Python/PyTorch with Ultralytics YOLOv8 on one
NVIDIA RTX 3090 GPU (24 GB). This repository retains the Ultralytics 8.0.170
source tree on which the research workspace was based.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset format

Prepare an independently authorized dataset in Ultralytics YOLO detection
format:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Each label row must follow `class x_center y_center width height`, with
normalized coordinates. Edit `path` in `configs/sinus_ct.yaml` to point to the
dataset root.

## Training

The published primary experiment fine-tuned COCO-pretrained YOLOv8n weights for
up to 300 epochs, using 512 × 512 images, batch size 32, and early-stopping
patience 50.

```bash
python src/train.py --data configs/sinus_ct.yaml --device 0
```

To run the comparison trained from scratch:

```bash
python src/train.py --data configs/sinus_ct.yaml --from-scratch --device 0
```

Ultralytics stores checkpoints and logs under `runs/`, which is ignored by Git.

## Evaluation

```bash
python src/evaluate.py \
  --weights runs/sinus_ct/yolov8n_transfer/weights/best.pt \
  --data configs/sinus_ct.yaml \
  --device 0
```

The evaluator uses the held-out `test` split and limits detections to two per
image, reflecting the two maxillary sinuses expected in a typical CT slice.

## Scope of this archive

This repository combines files retained from the original development
workspace with clearly identified convenience scripts reconstructed from the
published methods. It is not a clinical product, has not been packaged as a
medical device, and must not be used for diagnosis or treatment decisions.

## Citation

If this work is useful in your research, please cite:

```bibtex
@article{kwon2024maxillary,
  author  = {Kwon, Kyung Won and Kim, Jihun and Kang, Dongwoo},
  title   = {Automated detection of maxillary sinus opacifications compatible with sinusitis from CT images},
  journal = {Dentomaxillofacial Radiology},
  volume  = {53},
  number  = {8},
  pages   = {549--557},
  year    = {2024},
  doi     = {10.1093/dmfr/twae042}
}
```

## Acknowledgements

The detection pipeline is built with
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics). See the paper
for institutional, ethical, and author-contribution details.
