# Transfer Learning on Image Dataset (RVL-CDIP)

A deep learning assignment exploring three distinct **transfer learning strategies** using VGG-16 pretrained on ImageNet for 16-class document image classification on the RVL-CDIP dataset.

---

## Dataset

**RVL-CDIP** (Ryerson Vision Lab Complex Document Image Dataset)

| Property | Details |
|----------|---------|
| Total Images | 48,000 (`.tif` format) |
| Classes | 16 document categories |
| Train Split | 36,000 images (75%) |
| Validation Split | 12,000 images (25%) |
| Input Size | 224 × 224 × 3 (RGB) |
| Labels | `labels_final.csv` (path, class) |

**Document Classes:**

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | Letter | 8 | File Folder |
| 1 | Form | 9 | News Article |
| 2 | Email | 10 | Budget |
| 3 | Handwritten | 11 | Invoice |
| 4 | Advertisement | 12 | Presentation |
| 5 | Scientific Report | 13 | Questionnaire |
| 6 | Scientific Publication | 14 | Resume |
| 7 | Specification | 15 | Memo |

---

## Approach

All three models share the same **VGG-16 backbone** (pretrained on ImageNet, top FC layers removed, input shape 224×224×3) but differ in what is frozen vs. trained and in the custom head architecture.

### Model 1 — VGG-16 + Custom Conv Block + FC Layers

**Architecture:**
```
VGG-16 (frozen) → Conv2D(512, 3×3, same) → MaxPool2D(2×2)
→ Flatten → Dense(256, ReLU) → Dense(128, ReLU) + Dropout(0.2)
→ Dense(16, Softmax)
```

| Parameters | Count |
|-----------|-------|
| Total | 18,289,360 |
| Trainable | 3,574,672 (19.5%) |
| Frozen (VGG-16) | 14,714,688 |

**Results (6 epochs):**

| Metric | Training | Validation |
|--------|----------|-----------|
| Accuracy | 81.46% | 74.12% |
| Loss | 0.5982 | 0.9063 |

**Verdict:** Best performing model. Minimal overfitting with strong generalization.

---

### Model 2 — VGG-16 + Convolutional Head (No FC Layers)

**Architecture:**
```
VGG-16 (frozen) → Conv2D(192, 7×7, ReLU) → Conv2D(96, 1×1, ReLU)
→ Flatten → Dropout(0.2) → Dense(16, Softmax)
```

The 7×7 convolution replaces the first dense layer — equivalent to a fully-connected operation on the 7×7 spatial output of VGG-16, preserving spatial information while reducing parameters.

| Parameters | Count |
|-----------|-------|
| Total | 19,551,856 |
| Trainable | 4,837,168 (24.7%) |
| Frozen (VGG-16) | 14,714,688 |

**Results (6 epochs):**

| Metric | Training | Validation |
|--------|----------|-----------|
| Accuracy | 78.45% | 71.17% |
| Loss | 0.7058 | 1.0243 |

**Verdict:** Slightly lower accuracy than Model 1 but more interpretable; the convolutional head retains spatial structure and is more parameter-efficient relative to dense layers.

---

### Model 3 — Fine-tuned VGG-16 Block 5 + Convolutional Head

**Architecture:**
```
VGG-16 Block 1–4 (frozen) → VGG-16 Block 5 (trainable, last 6 layers)
→ Conv2D(256, 7×7, ReLU) → Conv2D(128, 1×1, ReLU)
→ Flatten → Dropout(0.2) → Dense(16, Softmax)
```

| Parameters | Count |
|-----------|-------|
| Total | 21,172,432 |
| Trainable | 21,172,432 (100%) |
| Frozen | 0 |

**Results (6 epochs):**

| Metric | Training | Validation |
|--------|----------|-----------|
| Accuracy | 6.13% | 6.13% |
| Loss | 2.7729 | 2.7730 |

**Verdict:** Complete failure — random guessing level (~1/16 = 6.25%). Unfreezing pretrained layers with only 6 epochs causes **catastrophic forgetting**. Fine-tuning requires significantly more epochs (50+) and a lower learning rate.

---

## Training Configuration

| Setting | Value |
|---------|-------|
| Loss Function | Categorical Crossentropy |
| Optimizer | Adam |
| Batch Size | 16 |
| Epochs | 6 |
| Image Preprocessing | Rescaling (1/255) |
| Callbacks | TensorBoard |
| Environment | Google Colab (GPU) |

---

## Key Takeaways

1. **Freeze and train a custom head** (Model 1) is the most effective strategy with limited training epochs — achieves 74% validation accuracy on a challenging 16-class document classification task.

2. **Replacing FC layers with Conv layers** (Model 2) is a valid parameter-efficient alternative, trading ~3% accuracy for a more spatially aware head.

3. **Fine-tuning without sufficient epochs** (Model 3) leads to catastrophic forgetting. Unfreezing pretrained weights requires a much lower learning rate and far more training steps before the model converges.

4. **ImageNet features transfer well** to document classification despite the domain difference, demonstrating the generalizability of deep CNN features.

---

## Project Structure

```
TRANSFER-LEARNING-ON-IMAGE-DATASET/
├── Transfer_Learning_Assignment.ipynb   # Full implementation with results
├── requirements.txt                     # Python dependencies
└── README.md
```

---

## Setup & Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset

Download the dataset from Kaggle (`brahma0545/aaic-assignment-tl`) and place `labels_final.csv` and the image folder in your working directory. If using Google Colab, the notebook includes the Kaggle API download steps.

### 3. Run the Notebook

```bash
jupyter notebook Transfer_Learning_Assignment.ipynb
```

Or open directly in Google Colab.

---

## Results Summary

| Model | Val Accuracy | Val Loss | Strategy |
|-------|-------------|---------|---------|
| Model 1 | **74.12%** | 0.9063 | Frozen VGG-16 + Conv+FC head |
| Model 2 | 71.17% | 1.0243 | Frozen VGG-16 + Conv-only head |
| Model 3 | 6.13% | 2.7730 | Fine-tuned VGG-16 Block 5 + Conv head |

---

## Tech Stack

- Python 3.7+
- TensorFlow 2.9.2 / Keras
- NumPy, Pandas
- Pillow
- TensorBoard
- Google Colab (GPU runtime)
