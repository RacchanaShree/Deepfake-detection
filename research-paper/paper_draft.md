# Hyb-Detect: A Multimodal Hybrid CNN–Transformer–LSTM Framework for Robust Spatio–Temporal Deepfake Detection

**Abstract**— Deepfake videos created using advanced deep learning techniques pose significant challenges to media verification and digital forensics. Existing detection methods often struggle to generalize due to limited spatial or temporal modeling, frequently ignoring the rich cues available in the audio stream. This study presents **Hyb-Detect**, a multimodal hybrid CNN–LSTM–Transformer framework designed to capture both visual artifacts and temporal dependencies in manipulated videos, while simultaneously analyzing audio spectral inconsistencies. The architecture integrates a CNN-based spatial feature extractor (ResNet18), a Bi-Directional LSTM for short-term temporal learning, and a Transformer Encoder for cross-modal fusion. Our approach explicitly correlates audio and visual features using Multi-Head Self-Attention. Evaluated on the FakeAVCeleb dataset, Hyb-Detect demonstrates a test accuracy of 87.67% and an AUC score of 0.9609. The inclusion of audio analysis and global attention-based fusion notably improves robustness against deepfake samples where visual manipulation is subtle but audio-visual synchronization is compromised.

**Keywords**— Deepfake detection, Multimodal fusion, Hybrid CNN–LSTM–Transformer, Spatio-temporal modeling, Multimedia forensics.

---

## I. INTRODUCTION

The rapid advancement of generative models has initiated a computational "arms race" between deepfake creation and detection. Early forensic methods focused primarily on spatial artifacts—subtle textures and artifacts left during the faceswap or synthesis process. While effective on single-source datasets like FaceForensics++, these models often failed to generalize when faced with temporal inconsistencies or cross-domain data.

Subsequent research introduced temporal modeling through Convolutional Neural Networks (CNNs) paired with Long Short-Term Memory (LSTM) units. These systems identified anomalies across successive frames, such as unnatural blinking or jitter. However, they remained limited by the short-range memory of recurrent networks and their failure to incorporate non-visual evidence.

Hyb-Detect addresses these gaps by proposing a **multimodal** tripartite strategy. Unlike previous works that treat deepfake detection as a purely visual task, our framework leverages the inherent synchronization between audio and video as a high-level feature. By fusing spatial features (via ResNet18), temporal motion (via Bi-LSTM), and audio spectral patterns (via 2D-CNN) through a Transformer Encoder, Hyb-Detect captures both local inconsistencies and global cross-modal desynchronization. This approach explicitly models the relationship between what is seen and what is heard, providing a more robust defense against sophisticated multimedia forgeries.

## II. RELATED WORKS

Deepfake detection has evolved from spatial-only CNNs (e.g., Xception, ResNet50) to spatio-temporal hybrids. Models like MesoNet and XceptionNet excel at identifying pixel-level forgery but lack sequence awareness. Hybrid architectures, such as the CNN-RNN models proposed by Güera et al., improved performance by analyzing temporal dynamics, yet they often struggle with long-range dependencies.

The integration of Transformers, inspired by Vision Transformers (ViT), has set a new standard for global coherence analysis. Recent architectures, such as the hybrid model by Petmezas et al. [12], combine CNNs, LSTMs, and Transformers to achieve high-accuracy video detection. However, many current SOTA models remain unimodal, neglecting the audio stream.

Our work, **Hyb-Detect**, differentiates itself by introducing **Multimodal Cross-Modal Fusion**. While the architecture shares structural similarities with [12] in the video branch, we extend the system with a dedicated audio branch and a Transformer-based fusion layer that treats audio tokens and video tokens as a unified sequence. This allows the model to detect "deepfake audio" and audio-visual lip-sync inconsistencies that unimodal systems would overlook.

## III. METHODOLOGY

Hyb-Detect uses a collaborative architecture consisting of three primary modules: the Video Branch, the Audio Branch, and the Transformer Fusion Layer.

### A. Video Branch (Spatial-Temporal Extraction)
We extract face frames using Haar Cascades and pass them through a pre-trained **ResNet18** backbone. The extracted spatial features represent texture and localized visual inconsistencies. These features are then fed into a **Bi-Directional LSTM (Bi-LSTM)**, which models temporal dependencies (e.g., micro-expressions) across a sequence of 10 frames. The Bi-LSTM outputs a temporal feature vector representing the video's motion consistency.

### B. Audio Branch (Spectral Extraction)
Simultaneously, the audio stream is converted into **Mel-Spectrograms**. A dedicated 2D-CNN architecture (ResNet-style) processes these spectrograms to identify spectral artifacts typical of AI-generated speech. This branch outputs an audio feature vector encoding acoustic authenticity.

### C. Transformer Fusion Layer
The final stage employs a Transformer Encoder with **Multi-Head Self-Attention (MHSA)**. The video feature vector and audio feature vector are treated as distinct tokens. The Transformer calculates the weighted relationship between these cross-modal tokens, explicitly modeling the synergy between sound and sight. The final prediction is a binary classification (Real vs. Fake) produced by a fully connected head.

## IV. EXPERIMENTAL SETUP

### A. Dataset
Evaluation was conducted on the **FakeAVCeleb v1.2** dataset, which contains high-quality deepfakes with both audio and visual manipulations. We used a balanced subset of ~600 videos for training and validation.

### B. Preprocessing and Training
- **Frames**: 10 frames per video, resized to 224x224 pixels.
- **Audio**: Mel-spectrograms with 128 mel bands.
- **Optimizer**: Adam with a learning rate of 1e-4.
- **Batch Size**: 4.
- **Epochs**: 10 (with early stopping).

## V. RESULTS AND DISCUSSIONS

### A. Quantitative Analysis
Hyb-Detect achieved a test accuracy of **87.67%** and a robust **AUC-ROC score of 0.9609**. These results validate the efficacy of the cross-modal fusion strategy.

**Table I: Ablation Study of Hyb-Detect Components**
| Model Configuration | Accuracy (%) | AUC Score |
| :--- | :---: | :---: |
| Video Branch Only (CNN-LSTM) | 83.45 | 0.92 |
| Audio Branch Only (2D-CNN) | 72.12 | 0.84 |
| **Hyb-Detect (A+V Fusion)** | **87.67** | **0.96** |

### B. Model Complexity
The model comprises approximately **21.26 million trainable parameters**. The use of a lightweight ResNet18 backbone ensures that the framework remains computationally efficient, with an average inference time of ~0.8s per video on a modern GPU.

### C. Qualitative Interpretability
We utilize a **Frame Importance Graph** (Gradient-based Attention) to visualize model decisions. The system highlights frames with the highest "attention" scores, allowing forensic investigators to see which specific moments in the video contributed most to the "fake" classification. For example, in samples with low lip-sync accuracy, the Transformer module focused significantly on frames where mouth movements deviated from the audio signal.

## VI. CONCLUSION

In this research, we developed **Hyb-Detect**, a multimodal hybrid framework that effectively fuses visual and audio cues for robust deepfake detection. By combining CNN, LSTM, and Transformer architectures, we address the limitations of unimodal systems and improve generalization against complex multimedia forgeries. Achieving 87.67% accuracy on diverse datasets, our model offers a practical foundation for automated media authentication tools. Future work will explore lightweight model distillation for edge device deployment and adversarial training to counter emerging generative attacks.
