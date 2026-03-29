# Experiment 7: Instance Segmentation using Mask R-CNN

## Objective
Perform instance segmentation on an input image using a pretrained **Mask R-CNN** model with a ResNet-50 FPN backbone from PyTorch's `torchvision` library.

## Theory
Instance segmentation combines **object detection** and **semantic segmentation** — it detects individual objects in an image and generates a pixel-level mask for each one. Unlike semantic segmentation (which labels every pixel by class), instance segmentation distinguishes between separate instances of the same class.

**Mask R-CNN** extends Faster R-CNN by adding a branch for predicting segmentation masks on each Region of Interest (RoI), in parallel with the existing branch for bounding box classification and regression.

### Key Concepts
| Concept | Description |
|---|---|
| **Instance Segmentation** | Detecting and delineating each distinct object in an image at the pixel level |
| **Mask R-CNN** | A two-stage framework: first generates region proposals, then classifies and generates masks for each |
| **ResNet-50 FPN** | Feature Pyramid Network backbone built on ResNet-50 for multi-scale feature extraction |
| **COCO Dataset** | The model is pretrained on the 80-class COCO object detection dataset |
| **Confidence Threshold** | A score cutoff (default `0.5`) to filter low-confidence predictions |

## Requirements

```
torch
torchvision
opencv-python
matplotlib
certifi
```

Install all dependencies:
```bash
pip install torch torchvision opencv-python matplotlib certifi
```

## Project Structure

```
Exp_7/
├── main.ipynb      # Main notebook with full pipeline
├── 1.jpeg          # Input image
├── output.jpg      # Generated output (after running)
└── README.md       # This file
```

## How to Run

1. Place your input image in the project directory (default: `1.jpeg`).
2. Open `main.ipynb` in Jupyter Notebook or VS Code.
3. Run all cells sequentially.
4. The output image with bounding boxes, labels, and segmentation masks will be displayed and saved as `output.jpg`.

## Pipeline

```
Input Image → Preprocessing → Mask R-CNN Model → Predictions
    → Filter by Confidence → Draw Masks + Bounding Boxes + Labels → Output
```

### Steps
1. **SSL Fix** — Handles macOS SSL certificate issues for model download.
2. **Import Libraries** — PyTorch, torchvision, OpenCV, matplotlib.
3. **Load Pretrained Model** — Mask R-CNN ResNet-50 FPN with COCO weights.
4. **Define COCO Labels** — 80 object classes from the COCO dataset.
5. **Read Input Image** — Load and convert from BGR to RGB.
6. **Preprocess** — Convert image to tensor.
7. **Model Prediction** — Run inference to get boxes, labels, scores, and masks.
8. **Apply Threshold** — Filter detections with confidence > 0.5.
9. **Visualize** — Overlay green masks, red bounding boxes, and class labels.
10. **Save Output** — Write the annotated image to `output.jpg`.

## Sample Output

The model produces:
- **Green overlay** — Pixel-level segmentation mask for each detected object
- **Red bounding box** — Object location
- **Label + score** — Class name with confidence percentage (e.g., `person (0.98)`)

## References
- [Mask R-CNN Paper (He et al., 2017)](https://arxiv.org/abs/1703.06870)
- [PyTorch torchvision Models](https://pytorch.org/vision/stable/models.html)
- [COCO Dataset](https://cocodataset.org/)
