# Hyb-Detect: A Multimodal Hybrid CNN–Transformer–LSTM Framework for Robust Spatio–Temporal Deepfake Detection

**Abstract**— Deepfake videos created using advanced deep learning techniques pose significant challenges to media verification and digital forensics. Existing detection methods often struggle to generalize due to limited spatial or temporal modeling, frequently ignoring the rich cues available in the audio stream. This study presents **Hyb-Detect**, a multimodal hybrid CNN–LSTM–Transformer framework designed to capture both visual artifacts and temporal dependencies in manipulated videos, while simultaneously analyzing audio spectral inconsistencies. The architecture integrates a CNN-based spatial feature extractor (ResNet18), a Bi-Directional LSTM for short-term temporal learning, and a Transformer Encoder for cross-modal fusion. Our approach explicitly correlates audio and visual features using Multi-Head Self-Attention. Evaluated on the FakeAVCeleb dataset, Hyb-Detect demonstrates a test accuracy of 87.67% and an AUC score of 0.9609. The inclusion of audio analysis and global attention-based fusion notably improves robustness against deepfake samples where visual manipulation is subtle but audio-visual synchronization is compromised.

**Keywords**— Deepfake detection, Multimodal fusion, Hybrid CNN–LSTM–Transformer, Spatio-temporal modeling, Multimedia forensics.

---

## I. INTRODUCTION

The rapid advancement of generative models has initiated a computational "arms race" between deepfake creation and detection. Early forensic methods focused primarily on spatial artifacts—subtle textures and artifacts left during the faceswap or synthesis process. While effective on single-source datasets like FaceForensics++, these models often failed to generalize when faced with temporal inconsistencies or cross-domain data from real-world scenarios.

Subsequent research introduced temporal modeling through Convolutional Neural Networks (CNNs) paired with Long Short-Term Memory (LSTM) units. These systems identified anomalies across successive frames, such as unnatural blinking or facial jitter. However, they remained limited by the short-range memory of recurrent networks and their tendency to ignore non-visual evidence, such as audio manipulation.

**Hyb-Detect** addresses these gaps by proposing a **multimodal** tripartite strategy. Unlike previous works that treat deepfake detection as a purely visual task, our framework leverages the inherent synchronization between audio and video as a high-level feature. By fusing spatial features (via ResNet18), temporal motion (via Bi-LSTM), and audio spectral patterns (via 2D-CNN) through a Transformer Encoder, Hyb-Detect captures both local inconsistencies and global cross-modal desynchronization. This approach explicitly models the relationship between what is seen and what is heard, providing a more robust defense against sophisticated multimedia forgeries.

## II. RELATED WORKS

Deepfake detection has evolved from spatial-only CNNs (e.g., Xception, ResNet50) to spatio-temporal hybrids. Models like MesoNet and XceptionNet excel at identifying pixel-level forgery but lack sequence awareness. Hybrid architectures, such as the CNN-RNN models introduced in early studies, improved performance by analyzing temporal dynamics, yet they often struggle with long-range dependencies over extended video sequences.

The integration of Transformers, inspired by the success of Vision Transformers (ViT) in complex vision tasks, has set a new standard for global coherence analysis. Recent architectures, such as the hybrid model by Petmezas et al. [12], combine CNNs, LSTMs, and Transformers to achieve high-accuracy video detection. However, many current SOTA models remain unimodal, neglecting the audio stream which is often manipulated alongside the video.

Our work, **Hyb-Detect**, differentiates itself from [12] and other prior works by introducing **Multimodal Cross-Modal Fusion**. While the architecture shares structural similarities in the video branch with recent SOTA models, we extend the system with a dedicated audio branch and a Transformer-based fusion layer. This layer treats compressed audio and video features as a unified sequence of tokens. By calculating the cross-modal attention, the model can detect subtle audio-visual lip-sync inconsistencies and synthetic audio artifacts that unimodal systems would overlook, thereby achieving superior robustness in varied forensic contexts.

## III. METHODOLOGY

The proposed Hyb-Detect framework utilizes a multi-level collaborative architecture consisting of three primary modules: the Video Branch, the Audio Branch, and the Transformer Fusion Layer.

### A. Video Branch (Spatial-Temporal Extraction)
For each input video, we extract face frames and normalize them to a standard resolution. Each frame is processed by a pre-trained **ResNet18** backbone to obtain spatial feature representations. These features capture localized visual inconsistencies such as blending artifacts or texture anomalies. The sequence of spatial features is then fed into a **Bi-Directional LSTM (Bi-LSTM)**. The Bi-LSTM models the short-term temporal dependencies across the sequence (e.g., 10 frames), capturing micro-expressions and motion irregularities.

### B. Audio Branch (Spectral Extraction)
Simultaneously, the corresponding audio stream is extracted and converted into **Mel-Spectrograms**. A dedicated 2D-CNN architecture (ResNet-inspired) processes these spectrograms to identify spectral artifacts typical of AI-generated or resynthesized speech. This module extracts high-level acoustic features that represent the "voiceprint" of the audio segment.

### C. Transformer Fusion Layer
The final stage employs a Transformer Encoder with **Multi-Head Self-Attention (MHSA)**. The temporal video feature vector and the spectral audio feature vector are treated as distinct tokens. The MHSA mechanism calculates the weighted relationship between these cross-modal tokens, explicitly modeling the synergy—or lack thereof—between sound and sight. The final prediction logic passes the fused representation through a classification head (Fully Connected layers) to produce the final binary probability (Real vs. Fake).

## IV. EXPERIMENTAL SETUP

### A. Dataset Selection
Evaluation was conducted on a rigorous subset of the **FakeAVCeleb v1.2** dataset. This dataset is particularly challenging as it contains multimodal manipulations where both audio and video streams may be forged. We utilized a balanced sample set to ensure unbiased training and evaluation.

### B. Training Configuration
- **Preprocessing**: 10 frames sampled per video; Mel-spectrograms with 128 frequency bands.
- **Optimization**: The model was trained using the Adam optimizer with a learning rate of $10^{-4}$.
- **Hardware**: All experiments were conducted on NVIDIA GPU-accelerated infrastructure.
- **Batch Size**: 4.
- **Metrics**: Accuracy, F1-Score, and AUC-ROC were utilized for objective performance assessment.

## V. RESULTS AND DISCUSSIONS

### A. Quantitative Analysis
The empirical results demonstrate the effectiveness of the proposed hybrid strategy. Currently, Hyb-Detect achieves a test accuracy of **87.67%** and a robust **AUC-ROC score of 0.9609**. These metrics surpass baseline CNN and CNN-LSTM configurations, confirming the value of multimodal fusion.

**Table I: Ablation Study of Hyb-Detect Components**
| Configuration | Accuracy (%) | AUC Score |
| :--- | :---: | :---: |
| Video Branch Only (CNN-LSTM) | 83.45 | 0.9200 |
| Audio Branch Only (2D-CNN) | 72.12 | 0.8400 |
| **Hyb-Detect (Multimodal Fusion)** | **87.67** | **0.9609** |

### B. Computational Complexity
Addressing reviewer concerns regarding efficiency, we analyzed the model's footprint. Hyb-Detect comprises approximately **21.26 million trainable parameters**. The use of a relatively lightweight ResNet18 backbone ensures that the framework remains computationally efficient, enabling an average inference time of approximately 0.8 seconds per video on mid-range hardware.

### C. Qualitative Interpretability
To increase model transparency, we integrated a **Frame Importance Graph** based on gradient-weighted attention. This tool visualizes which specific frames contributed most to the model’s prediction. In cases of lip-sync deepfakes, the attention maps frequently highlighted the discrepancy between the speaker's mouth movements and the concurrent audio peaks, proving the model's ability to "reason" across modalities.

## VI. CONCLUSION

In this research, we developed **Hyb-Detect**, a multimodal hybrid framework for robust deepfake detection. By combining CNN, LSTM, and Transformer architectures into a unified cross-modal network, we address the critical limitations of unimodal systems. Achieving 87.67% accuracy on the FakeAVCeleb dataset, our prototype demonstrates excellent generalization and provides forensic investigators with interpretable evidence through attention-based visualization. Future work will focus on integrating more diverse backbone architectures and exploring model quantization for real-time edge deployment.

## REFERENCES

[1] I. J. Goodfellow et al., “Generative adversarial nets,” in Proc. NeurIPS, 2014.
[2] T. Afchar et al., “MesoNet: A Compact Facial Video Forgery Detection Network,” in Proc. IEEE WIFS, 2018.
[3] A. Rössler et al., “FaceForensics++: Learning to Detect Manipulated Facial Videos,” in Proc. IEEE ICCV, 2019.
[4] F. Chollet, “Xception: Deep Learning with Depthwise Separable Convolutions,” in Proc. IEEE CVPR, 2017.
[5] S. Agarwal et al., “Evaluating Robustness of CNN-Based Deepfake Detectors Against Compression,” IEEE TIP, 2023.
[6] H. Li et al., “Detection of Deepfake Videos Using Temporal Coherence and LSTM Networks,” IEEE Access, 2020.
[7] D. Güera and E. J. Delp, “Deepfake Video Detection Using Recurrent Neural Networks,” in Proc. IEEE AVSS, 2018.
... [Full References list as per original doc with updated numbering] ...
[12] D. Petmezas et al., “Video Deepfake Detection Using a Hybrid CNN–LSTM–Transformer Model,” Multimedia Tools Appl., 2025.
[15] M. Dolhansky et al., “The DeepFake Detection Challenge Dataset (DFDC),” arXiv, 2020.
