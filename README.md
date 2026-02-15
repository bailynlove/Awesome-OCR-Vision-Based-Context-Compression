# 📂 Awesome-OCR: Vision-Based Context Compression

<p align="center">
  <a href="https://awesome.re"><img src="https://awesome.re/badge.svg"></a>
  <img src="https://img.shields.io/github/stars/bailynlove/Awesome-OCR-Vision-Based-Context-Compression">
  <img src="https://img.shields.io/badge/PRs-Welcome-red">
  <img src="https://img.shields.io/github/last-commit/bailynlove/Awesome-OCR-Vision-Based-Context-Compression">
</p>

> **This repository curates recent research (2024–2026) on text-to-vision context compression – techniques that render textual context into images to overcome language models’ context length limits. By leveraging the higher information density of visual tokens, these approaches compress long text (documents, code, or dialog histories) into compact images that multimodal models or OCR decoders can interpret.**

---

## 📚 Table of Contents
- [01. Visual Token Efficiency & Scaling](#01-visual-token-efficiency--scaling) (The Foundation)
- [02. Visualized Agent Memory](#02-visualized-agent-memory) (The Application)
- [03. Visualized Structure: Code & Data](#03-visualized-structure-code--data) (The New Frontier)
- [04. Pixel-based Language Modeling](#04-pixel-based-language-modeling) (End-to-End Approaches)
- [05. Benchmarks & Datasets](#05-benchmarks--datasets)

---

### 01. Visual Token Efficiency & Scaling
*Focus: How to pack more information into fewer visual tokens. The fundamental mechanism of "Visual > Text".*

- **Glyph: Scaling Context Windows via Visual-Text Compression**
  - *Authors:* Jiale Cheng, et al. (Tsinghua & Zhipu AI)
  - *Venue:* **arXiv 2025**
  - *Tags:* `[Core Technique]`, `[3-4x Compression]`
  - [Paper](https://arxiv.org/abs/2510.17800) | [Repo](https://github.com/thu-coai/Glyph)

- **DeepSeek-OCR: Contexts Optical Compression**
  - *Authors:* DeepSeek-AI Team
  - *Venue:* **arXiv 2025**
  - *Tags:* `[Optical Compression]`, `[DeepSeek]`
  - [Paper](https://arxiv.org/abs/2510.18234) | [Repo](https://github.com/deepseek-ai/DeepSeek-OCR) 

- **Vision-centric Token Compression in Large Language Model (VIST)**
  - *Authors:* Ling Xing, et al.
  - *Venue:* **arXiv 2025**
  - *Tags:* `[Token Pruning]`, `[Efficiency]`
  - [Paper](https://arxiv.org/abs/2502.00791) | [Repo](https://github.com/CSU-JPG/VIST)

- **Text or Pixels? It Takes Half: On the Token Efficiency of Visual Text Inputs**
  - *Authors:* Yanhong Li, et al.
  - *Venue:* **arXiv 2025**
  - *Tags:* `[Empirical Study]`, `[50% Reduction]`
  - [Paper](https://arxiv.org/abs/2510.18279) | [Repo](https://github.com/yanhong-lbh/text_or_pixels)

- **Recoverable Compression: A Multimodal Vision Token Recovery Mechanism Guided by Text Information**
  - *Authors:* Yi Chen, et al.
  - *Venue:* **AAAI 2025**
  - *Tags:* `[Compression]`, `[Recovery]`
  - [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/32229)

- **Large Language Model for Lossless Image Compression with Visual Prompts**
  - *Authors:* Junhao Du, et al.
  - *Venue:* **arXiv 2025**
  - *Tags:* `[Compression]`, `[Visual Prompts]`
  - [Paper](https://arxiv.org/abs/2502.16163)

- **VoCo-LLaMA: Towards Vision Compression with Large Language Models**
  - *Authors:* Xubing Ye, et al.
  - *Venue:* **arXiv 2025**
  - *Tags:* `[Vision Compression]`
  - [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Ye_VoCo-LLaMA_Towards_Vision_Compression_with_Large_Language_Models_CVPR_2025_paper.html) | [Repo](https://github.com/Yxxxb/VoCo-LLaMA)

- **Global Compression Commander: Plug-and-Play Inference Acceleration for High-Resolution Large Vision-Language Models**
  - *Authors:* Xuyang Liu, et al.
  - *Venue:* **arXiv 2025**
  - *Tags:* `[Inference Acceleration]`, `[High-Resolution]`
  - [Paper](https://arxiv.org/abs/2501.05179) | [Repo](https://github.com/xuyang-liu16/GlobalCom2)

- **Leveraging Visual Tokens for Extended Text Contexts in Multi-Modal Learning**
  - *Authors:* Alex Jinpeng Wang, et al.
  - *Venue:* **arXiv 2024**
  - *Tags:* `[Multi-Modal]`, `[Extended Context]`
  - [Paper](https://arxiv.org/abs/2406.02547) | [Repo](https://fingerrec.github.io/visincontext/)

- **VTC-R1: Vision-Text Compression for Efficient Long-Context Reasoning**
  - *Authors:* Yibo Wang, et al.
  - *Venue:* **arXiv 2026**
  - *Tags:* `[Compression]`, `[Long-Context]`
  - [Paper](https://arxiv.org/abs/2601.22069) | [Repo](https://github.com/w-yibo/VTC-R1)

- **Global Context Compression with Interleaved Vision-Text Transformation**
  - *Authors:* Dian Jiao, et al.
  - *Venue:* **arXiv 2026**
  - *Tags:* `[Compression]`, `[Vision-Text]`
  - [Paper](https://arxiv.org/abs/2601.10378)

- **DeepSeek-OCR 2: Visual Causal Flow**
  - *Authors:* Haoran Wei, et al.
  - *Venue:* **arXiv 2026**
  - *Tags:* `[OCR]`
  - [Paper](https://arxiv.org/abs/2601.20552)
---

### 02. Visualized Agent Memory
*Focus: Using images to store agent interaction history, enabling "Infinite Memory" agents.*

- **MemOCR: Layout-Aware Visual Memory for Efficient Long-Horizon Reasoning**
  - *Authors:* Yaorui Shi, et al.
  - *Venue:* **arXiv 2026**
  - *Tags:* `[Agent Memory]`, `[Layout-Aware]`
  - [Paper](https://arxiv.org/abs/2601.21468) | [Repo](https://github.com/syr-cn/MemOCR)

- **AgentOCR: Reimagining Agent History via Optical Self-Compression**
  - *Authors:* Lang Feng, et al.
  - *Venue:* **arXiv 2026**
  - *Tags:* `[Agent Memory]`, `[Optical Compression]`
  - [Paper](https://arxiv.org/abs/2601.04786)
---

### 03. Visualized Structure: Code & Data
*Focus: Compressing highly structured data (Code, JSON, Logs) where visual layout conveys logic.*

- **CodeOCR: On the Effectiveness of Vision Language Models in Code Understanding**
  - *Authors:* Yuling Shi, et al.
  - *Venue:* **arXiv 2026**
  - *Tags:* `[Code]`, `[Syntax Highlighting]`, `[8x Compression]`
  - [Paper](https://arxiv.org/abs/2602.01785) | [Repo](https://github.com/YerbaPage/CodeOCR)

- **Can Vision-Language Models Handle Long-Context Code? An Empirical Study on Visual Compression**
  - *Authors:* Jianping Zhong, et al.
  - *Venue:* **arXiv 2026**
  - *Tags:* `[Empirical Study]`, `[Code Compression]`
  - [Paper](https://www.arxiv.org/abs/2602.00746)

- **When Text-as-Vision Meets Semantic IDs in Generative Recommendation: An Empirical Study**
  - *Authors:* Shutong Qiao et al.
  - *Venue:* **arXiv 2026**
  - *Tags:* `[Text-as-Vision]`, `[Semantic IDs]`
  - [Paper](https://arxiv.org/abs/2601.14697)

---

### 04. Pixel-based Language Modeling
*Focus: Radical approaches that abandon text tokenizers entirely.*

- **Pixology: Probing the Linguistic and Visual Capabilities of Pixel-based Language Models**
  - *Authors:* Kushal Tatariya, et al.
  - *Venue:* **EMNLP 2024**
  - *Tags:* `[Analysis]`, `[Pixel-only]`
  - [Paper](https://aclanthology.org/2024.emnlp-main.194/) | [Repo](https://github.com/kushaltatariya/Pixology)

- **Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models **
  - *Authors:* Haoran Wei, et al.
  - *Venue:* **arXiv 2023**
  - *Tags:* `[Scaling]`, `[Vision Vocabulary]`
  - [Paper](https://arxiv.org/abs/2312.06109) | [Repo](https://github.com/Ucas-HaoranWei/Vary)

---

### 05. Benchmarks & Datasets
*Focus: Where to test these Visual Context models.*

- **MMDocBench: Benchmarking Large Vision-Language Models for Fine-Grained Visual Document Understanding**
  - [Paper](https://arxiv.org/abs/2410.21311) | [Repo](https://github.com/MMDocBench)

- **MemoryAgentBench (2025)**
  - [Paper](https://arxiv.org/abs/2507.05257) | [Repo](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench)

- **CC-OCR: A Comprehensive and Challenging OCR Benchmark for Evaluating Large Multimodal Models in Literacy**
  - [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_CC-OCR_A_Comprehensive_and_Challenging_OCR_Benchmark_for_Evaluating_Large_ICCV_2025_paper.html) | [Repo](https://huggingface.co/datasets/wulipc/CC-OCR)

- **OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation**
  - [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_OCR_Hinders_RAG_Evaluating_the_Cascading_Impact_of_OCR_on_ICCV_2025_paper.html) | [Repo](https://huggingface.co/datasets/opendatalab/OHR-Bench)

- **LVLM-Compress-Bench: Benchmarking the Broader Impact of Large Vision-Language Model Compression**
  - [Paper](https://arxiv.org/abs/2503.04982)

- **LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding**
  - [Paper](https://aclanthology.org/2024.acl-long.172/) | [Repo](https://huggingface.co/datasets/zai-org/LongBench)

- **MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly**
  - [Paper](https://arxiv.org/abs/2505.10610) | [Repo](https://huggingface.co/datasets/ZhaoweiWang/MMLongBench)

- **LONGCODEU: Benchmarking Long-Context Language Models on
Long Code Understanding**
  - [Paper](https://aclanthology.org/2025.acl-long.1324.pdf) | [Repo](https://huggingface.co/datasets/longcodeu/longcodeu-dataset)

- **SynthVLM: Towards High-Quality and Efficient Synthesis of Image-Caption Datasets for Vision-Language Models**
  - [Paper](https://arxiv.org/abs/2407.20756) | [Repo](https://github.com/starriver030515/synthvlm)

---

### 💡 Contribution Guide
Please submit a PR if you find new papers that fit the **"Visual Context Compression"** theme. We are specifically looking for papers that quantify **Compression Ratio (e.g., Visual Tokens vs. Text Tokens)**.