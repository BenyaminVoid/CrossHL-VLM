# # CrossHL-VLM: Vision–Language Enhanced Cross-HL Transformer for HSI–LiDAR Classification


The repository contains the implementations for [Cross Hyperspectral and LiDAR Attention Transformer: An Extended Self-Attention for Land Use and Land Cover Classification](https://ieeexplore.ieee.org/document/10462184)
And
Cross Hyperspectral and LiDAR Attention Transformer: An Extended Self-Attention for Land Use and Land Cover Classification [Swalpa Kumar Roy](https://swalpa.github.io), Atri Sukul, [Ali Jamali](https://www.researchgate.net/profile/Ali-Jamali), [Juan Mario Haut](https://mhaut.github.io), and [Pedram Ghamisi](http://www.ai4rs.com)


## Sample Dataset
Get the disjoint dataset (<a href="https://drive.google.com/folderview?id=1Wy939ZoRWqIRkPE7NBcndn1LhHTW9HQi" target="_blank">TrentoDataset</a> folder) from Google Drive.

Get the disjoint dataset (<a href="https://drive.google.com/folderview?id=1zn32OnII2DVVeJ2ypMaF71BfLJnlxMu3" target="_blank">HoustonDataset</a> folder) from Google Drive.

Get the disjoint dataset (<a href="https://drive.google.com/folderview?id=1xZx3kMGOc3MmA1GlgaHSTYE0fb3rWc4X" target="_blank">MUUFL_Dataset</a> folder) from Google Drive.

This repository presents **CrossHL-VLM**, an extension of the Cross-HL Transformer that integrates **Vision–Language Model (VLM) semantic supervision** using **CLIP** for hyperspectral–LiDAR land-cover classification.

The proposed method introduces a **semantic embedding head** aligned with text prompts, enabling **semantic-guided multimodal learning** while preserving the original Cross-HL classification pipeline.

---

Overview:

Hyperspectral (HSI) and LiDAR data provide complementary spectral and structural information for land-cover classification.  
Cross-HL effectively fuses these modalities via cross-attention and it relies purely on numeric supervision.

CrossHL-VLM is a replication and modification study in order to test Cross-HL by injecting semantic knowledge through a vision–language model (CLIP), aligning learned image representations with textual class descriptions during training.

---

Key Contributions:

- Extension of Cross-HL Transformer with Vision–Language semantic supervision
- Dual-head training:
  - **Classification head** (standard Cross-HL)
  - **Semantic embedding head** aligned with CLIP text embeddings
- Semantic regularization via image–text contrastive alignment
- Evaluation of both classification accuracy and semantic head accuracy
- End-to-end training on HSI–LiDAR data (Trento dataset)

---


Architecture:

1. **Input**:  
   - HSI patch (11×11×63)  
   - LiDAR patch (11×11×1)

2. **Feature Extraction**:
   - 3D convolution over HSI spectral–spatial dimensions
   - Heterogeneous convolution (HetConv)

3. **Cross-HL Attention**:
   - LiDAR features act as **queries**
   - HSI tokens act as **keys/values**
   - Enables LiDAR-guided attention over HSI spatial tokens

4. **Transformer Encoder**:
   - CLS token summarizes fused representation

5. **Two Output Heads**:
   - **Classification Head** → class logits
   - **VLM Projection Head** → 512-D embedding aligned with CLIP text space

---

Vision–Language Integration (CLIP model)

CLIP is used only as a semantic prior, not for zero-shot classification.

### Text Prompts used here:
Each class is represented by a natural-language prompt:
"a remote sensing patch of Buildings"
"a remote sensing patch of Woods"