# CoANet: Connectivity Attention Network for Road Extraction from Satellite Imagery

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![IEEE TIP](https://img.shields.io/badge/Paper-IEEE%20TIP%202021-00629B.svg?style=flat-square)](https://ieeexplore.ieee.org/document/9525352)
[![Dataset](https://img.shields.io/badge/Dataset-DeepGlobe-4B8BBE.svg?style=flat-square)](http://deepglobe.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

An end-to-end, modular PyTorch implementation of **CoANet (Connectivity Attention Network)** based on the IEEE Transactions on Image Processing (TIP) 2021 paper: *"CoANet: Connectivity Attention Network for Road Extraction from Satellite Imagery"* by Mei et al.

This repository provides a clean, memory-optimized pipeline for extracting topologically connected road networks from high-resolution satellite imagery, evaluated on the **DeepGlobe Road Extraction Dataset**.

---

## 1. Problem Statement & Theoretical Motivation

Extracting continuous road networks from high-resolution satellite imagery is fundamentally challenged by complex urban disturbances, such as shadows cast by buildings, tree canopy occlusions, and visually similar non-road structures. 

Conventional semantic segmentation architectures (e.g., U-Net, FCN, DeepLab) formulate road extraction strictly as a dense pixel-wise binary classification task. Consequently, they suffer from two structural limitations:
1. **Isotropic Receptive Fields:** Standard square convolutional kernels ($3 \times 3$) aggregate context equally from all directions, inevitably incorporating irrelevant background noise when modeling narrow, elongated, and continuous road spans.
2. **Topological Fragmentation:** Pixel-wise cross-entropy loss ignores graphical and structural dependencies between neighboring pixels, leading to broken road segments in occluded regions.

**CoANet** addresses both challenges simultaneously by introducing:
* **Strip Convolution Module (SCM):** Leverages directional linear convolutions along four orientations (Horizontal, Vertical, Left-Diagonal, Right-Diagonal) to capture long-range geometric context while suppressing background interference.
* **Connectivity Attention Module (CoA):** Explicitly models graphical topology by predicting pair-wise pixel dependencies across 8 neighboring directions at multiple spatial scales ($d=1$ and $d=3$), preserving road continuity across severe occlusions.

---

## 2. Model Architecture & Mathematical Formulation

The network follows a dual-branch Encoder-Decoder paradigm with a total of **62,717,713 trainable parameters**.

```text
[Input RGB Image: B x 3 x 512 x 512]
                 │
                 ▼
[Encoder: Dilated ResNet-101 Backbone (Output Stride = 8)]
                 │
                 ▼
[Multi-Scale Context: 5-Branch ASPP Module (2048 -> 256)]
                 │
                 ▼
[Decoder: 4 x Strip Convolution Blocks (SCM) + Skip Connections]
                 │
        ┌────────┴───────────────────────────────┐
        ▼                                        ▼
[Segmentation Head]               [Connectivity Branch (Parallel CoA)]
        │                                        │
        ▼                                        ▼
[Seg Logits: B x 1 x 512 x 512]   [Conn Cubes (d=1, d=3): B x 8 x 512 x 512]
```

### 2.1. Dilated ResNet-101 Encoder & ASPP
To prevent severe spatial resolution loss on thin road structures, the standard stride-2 downsampling operations in `Layer3` (`Conv4_x`) and `Layer4` (`Conv5_x`) of the ImageNet-pretrained **ResNet-101** backbone are replaced with **Atrous (Dilated) Convolutions** with dilation rates $r=2$ and $r=4$, respectively. This maintains a feature map resolution of $\frac{H}{8} \times \frac{W}{8}$ ($64 \times 64$). An **Atrous Spatial Pyramid Pooling (ASPP)** module with dilation rates $r \in \{6, 12, 18\}$, a $1 \times 1$ branch, and global average pooling aggregates multi-scale context into 256 channels.

### 2.2. Strip Convolution Module (SCM)
Each of the 4 decoder stages progressively upsamples features and fuses multi-level encoder representations (`Skip_2` and `Skip_4`) using Strip Convolution Blocks. Given a kernel length of $2k+1 = 9$ ($k=4$), SCM applies four parallel 1D strip convolutions along directional vectors $D \in \{(0,1), (1,0), (1,1), (-1,1)\}$, concatenating their outputs to capture roads of arbitrary orientation.

### 2.3. Connectivity Attention Module (CoA) & Dynamic Ground-Truth Cubes
Instead of relying solely on binary segmentation masks $Y \in \{0, 1\}^{H \times W}$, CoANet constructs an 8-channel **Connectivity Cube** $O^{(d)} \in \{0, 1\}^{8 \times H \times W}$ for distance $d \in \{1, 3\}$. For a pixel at $(i, j)$ and neighbor offset $(\Delta y_c, \Delta x_c)$ in direction $c \in \{1, \dots, 8\}$:

$$O_{c, i, j}^{(d)} = Y_{i, j} \times Y_{i + d \cdot \Delta y_c, \, j + d \cdot \Delta x_c}$$

A channel $c$ is active ($1$) if and only if both the center pixel and its $c$-th neighbor at distance $d$ belong to the road class. The CoA head processes shared decoder features via a $3 \times 3$ convolution, a dilated $3 \times 3$ convolution ($r=d$), and a **Squeeze-and-Excitation (SE)** channel calibration block before projecting to 8 directional logits.

### 2.4. Multi-Task Loss Function
The network is optimized end-to-end via a joint objective combining segmentation loss ($\mathcal{L}_{seg}$) and multi-scale connectivity loss ($\mathcal{L}_{con}$):

$$\mathcal{L}_{CoANet} = \mathcal{L}_{seg} + \lambda \mathcal{L}_{con}$$

$$\mathcal{L}_{seg} = \mathcal{L}_{BCE}(\hat{Y}_{seg}, Y) + \alpha \left(1 - \frac{2 \sum_{i=1}^{N} y_i \hat{y}_i}{\sum_{i=1}^{N} y_i^2 + \sum_{i=1}^{N} \hat{y}_i^2 + \epsilon}\right)$$

$$\mathcal{L}_{con} = \mathcal{L}_{BCE}(\hat{O}^{(1)}, O^{(1)}) + \beta \mathcal{L}_{BCE}(\hat{O}^{(3)}, O^{(3)})$$

where $\alpha = 1.0, \beta = 1.0, \lambda = 1.0$.

---

## 3. Engineering Highlights & Implementation Details

This implementation was engineered from the ground up in PyTorch with a strict focus on tensor shape contracts, computational efficiency, and hardware constraints:

* **Vectorized On-the-Fly Connectivity Cube Generation:** Storing pre-computed 8-channel float32 cubes for thousands of high-resolution images requires substantial disk I/O. We implemented a pure PyTorch tensor-slicing generator (`generate_connectivity_cube`) that computes $O^{(1)}$ and $O^{(3)}$ dynamically inside `__getitem__` with zero Python loops.
* **Masked Diagonal Strip Convolutions:** Since native PyTorch `nn.Conv2d` only supports rectangular kernels, diagonal strip convolutions ($D=(1,1)$ and $D=(-1,1)$) are engineered using $9 \times 9$ kernels multiplied by non-trainable binary diagonal masks registered via `register_buffer`. This guarantees exact diagonal weight support during both forward propagation and gradient backpropagation.
* **VRAM Optimization via AMP & Gradient Accumulation:** To replicate the paper's effective batch size of $16$ at $512 \times 512$ resolution on a single GPU (15GB VRAM), the training engine utilizes **Automatic Mixed Precision (`torch.amp`)** alongside **Gradient Accumulation**, paired with polynomial learning rate decay ($lr = lr_{base} \cdot (1 - \frac{iter}{max\_iter})^{3.0}$) and gradient norm clipping (`max_norm=10.0`).
* **Inference Fusion Strategy:** During evaluation, predictions from the connectivity branches are thresholded at $t=0.5$, summed across the 8 directional channels to form topological enhancement masks, and fused with the base segmentation probability map.

---

## 4. Experimental Results & Performance Analysis

### 4.1. Quantitative Evaluation (Early-Stage Convergence @ Epoch 4)
The model was evaluated on a held-out validation split of **622 images** ($512 \times 512$) from the **DeepGlobe Road Extraction Dataset**. Even at an early training stage (**Epoch 4**), the network demonstrates strong topological feature learning:

```text
==================================================
 CoANet Professional Evaluation Report 
==================================================
 Total Validation Images : 622
 Checkpoint Epoch        : 4
--------------------------------------------------
 Overall Pixel Accuracy  : 95.88 %
 Precision               : 50.99 %
 Recall                  : 59.72 %
 F1-Score (Dice)         : 55.01 %
 Intersection over Union : 37.94 %
==================================================
```

| Evaluation Metric | Validation Score (Epoch 4) | Paper Full-Training Benchmark (DeepGlobe) |
| :--- | :---: | :---: |
| **Overall Pixel Accuracy** | **95.88 %** | - |
| **Precision** | **50.99 %** | - |
| **Recall** | **59.72 %** | - |
| **F1-Score (Dice)** | **55.01 %** | ~ 79.80 % |
| **Intersection over Union (IoU)** | **37.94 %** | ~ 66.40 % |

### 4.2. Analytical Plots & Engineering Interpretation

![Quantitative Analytics](assets/quantitative_analytics.png)

**Technical Analysis of the Model's Behavior:**
1. **Recall vs. Precision Trade-off (59.72% vs. 50.99%):** In satellite road extraction, road pixels typically constitute less than 5% of the total image area (extreme class imbalance). Achieving a **Recall of nearly 60%** by Epoch 4 confirms that the squared-denominator Soft Dice Loss and the Connectivity Attention (CoA) branches effectively prevent the model from collapsing into an all-background trivial solution—a common failure mode when training deep segmentation networks with standard Cross-Entropy.
2. **Inference Fusion Impact:** Because the fused prediction incorporates active neighboring links from $\hat{O}^{(1)}$ and $\hat{O}^{(3)}$, the model actively bridges small gaps and favors road continuity. At early epochs, this slightly widens predicted road boundaries, increasing False Positives (lowering Precision relative to Recall) while significantly boosting topological connectivity.
3. **IoU Distribution & PR Curve Stability:** As shown in the Validation IoU Histogram and Precision-Recall Curve, the model exhibits a unimodal, consistent IoU distribution across diverse rural and urban scenes rather than overfitting to a narrow subset of textures. Extending training along the polynomial learning rate schedule sharpens boundary localization and drives Precision and IoU toward the paper's asymptotic benchmark.

---

### 4.3. Qualitative Visual Evaluation

The figure below illustrates dynamic validation samples comparing the input satellite imagery, Ground Truth mask, raw segmentation branch output, and the final **CoANet Fused Prediction**:

![Qualitative Visual Evaluation](assets/qualitative_results.png)

* **Column 1 (Input Image):** High-resolution RGB satellite imagery exhibiting shadows, vegetation occlusions, and unpaved rural tracks.
* **Column 2 (Ground Truth):** Binary road mask target.
* **Column 3 (Base Segmentation Prediction):** Output from the segmentation head alone, which occasionally suffers from disconnections in low-contrast or occluded segments.
* **Column 4 (CoANet Fused Prediction):** Final output after integrating the $d=1$ and $d=3$ connectivity cubes, visibly reinforcing linear continuity and recovering faint road links.

---

## 5. Repository Structure & Usage

```text
├── assets/
│   ├── quantitative_analytics.png   # Confusion Matrix, PR Curve, and IoU Histogram
│   └── qualitative_results.png      # Visual comparison of road extraction results
├── CoANet_Implementation.ipynb      # Complete modular notebook (Data -> Model -> Train -> Eval)
└── README.md                        # Project documentation and technical analysis
```

### Quick Start (Google Colab / Local Jupyter)
1. Clone the repository and upload `CoANet_Implementation.ipynb` to Google Colab.
2. Provide your `kaggle.json` API token in **Cell 1** to automatically download and extract the **DeepGlobe Road Extraction Dataset**.
3. Run **Cells 3–9** to verify the vectorized connectivity generator, `DataLoader` shape contracts (`[B, 3, 512, 512]`, `[B, 1, 512, 512]`, `[B, 8, 512, 512]`), and the `CoANet` forward pass.
4. Execute **Cell 14** to start or resume Mixed-Precision training with automatic Google Drive checkpointing (`best_coanet.pth` and `last_checkpoint.pth`).
5. Run **Cells 15–17** to generate the quantitative evaluation report, qualitative visual comparisons, and analytical charts.

---

## 6. Reference & Citation

If you find this implementation or analysis useful, please cite the original IEEE TIP 2021 paper:

```bibtex
@ARTICLE{9525352,
  author={Mei, Jie and Li, Rou-Jing and Gao, Wang and Cheng, Ming-Ming},
  journal={IEEE Transactions on Image Processing}, 
  title={CoANet: Connectivity Attention Network for Road Extraction From Satellite Imagery}, 
  year={2021},
  volume={30},
  pages={8540-8552},
  doi={10.1109/TIP.2021.3117076}
}
```