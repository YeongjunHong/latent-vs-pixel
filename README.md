# latent-vs-pixel

> **"지능은 픽셀을 복원하는가, 아니면 잠재 공간의 의미를 예측하는가?"**

이 프로젝트는 Yann LeCun이 제안한 **JEPA(Joint-Embedding Predictive Architecture)**와 기존의 생성형 모델(**GPT/BERT**)을 밑바닥(From Scratch)부터 구현하며, 기계 지능의 표현 학습(Representation Learning) 메커니즘을 대조·분석하는 연구용 'Dojo'입니다.

##  Project Goals
* **Back to the Dojo:** 라이브러리 호출이 아닌, PyTorch 기본 연산을 이용한 Transformer 및 JEPA 핵심 로직 구현.
* **Paradigm Contrast:** 픽셀/토큰 단위 복원(Reconstruction)과 잠재 공간 예측(Latent Prediction)의 효율성 비교.
* **Modular Architecture:** Interface-Stage-Plugin의 3층 구조를 통해 V-JEPA 및 사유 루프(Reasoning Loop)로의 확장성 확보.

##  3-Tier Architecture
본 프로젝트는 확장성과 유지보수를 위해 3계층 위계 구조로 설계되었습니다.
1. **Components (Plugins):** Attention, ViT Block, Masker 등 독립적인 연산 모듈.
2. **Models (Adapters):** I-JEPA, GPT, BERT 등 컴포넌트들의 논리적 결합체.
3. **Workflows (Stages):** 각 모델 특유의 학습 루프(EMA 업데이트, Causal Masking 등) 및 추론 로직.

##  References
* [I-JEPA] Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (2023)
* [Path to AI] A Path Towards Autonomous Machine Intelligence (Yann LeCun, 2022)
* [GPT] Improving Language Understanding by Generative Pre-Training (2018)
* [BERT] Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)

---
*Created by 영준 (Unbubble the Babble)*