# Intro
My name is Matthew Busenlener, and I am a fourth-year student at USC studying Computer Science and Cognitive Science with an emphasis in AI. This repository hosts my AI and SWE related personal projects.

# Projects
### Latent Memory-Augmented LLM
* A memory-augmented LLM that encodes previous hidden states into latent memory tokens that are “recalled” during decoding.
* This model achieves a 22% improvement in F1 on HotpotQA compared to the In-Context baseline while compressing long contexts by 8x.
* https://github.com/Mbusenlener/Portfolio/tree/main/MemoryLLM
### Multi-Class Adaptive Rag
* RAG pipeline that predicts two unique kinds of hallucinations from the LLM’s hidden states: 1. Hallucination due to lack of parametric knowledge and 2. Hallucination despite parametric knowledge.
* The classifier achieves 62% balanced 3-class accuracy and dynamically routes the prompt to RAG retrieval, soft-prompt denoising, or direct generation, mitigating most hallucinations while maintaining token-efficiency.
* https://github.com/Hallucination-Detection-CSCI544/HallucinationDetection
### AI Music Generator
* A transformer-based Pytorch model trained from scratch that generates original, polyphonic music from text descriptions.
* This model uses reversible layers and FAVOR+ attention to handle sequences up to 4500 tokens.
* https://github.com/Mbusenlener/Portfolio/tree/main/MusicGenerator
### EEG Fatigue Classification BCI
* BCI application and EEG toolkit that quantifies mental fatigue on a continuous scale using live EEG recordings.
* Developed for the 2025 California Neurotech Conference with USC Neurotech.
* https://github.com/Neurotech-BCI/ClassificationPipeline
### EEG MIIR-Net (Music Imagery Information Retrieval) 
* A CNN model implemented in Pytorch for classifying raw EEG waveforms from the OpenMiir dataset of 9 participants listening to 8 different songs.
* This model is trained to predict which of the 8 songs an unseen participant is listening to based on high-resolution EEG recordings of the participant's brain activity.
* https://github.com/Mbusenlener/Portfolio/blob/main/EEGMiirNet
### Trojan Housing Finder
* A full-stack web application built with React, SpringBoot, and MySQL allowing USC students to search, filter, and review available rental properties near campus.
* https://github.com/TrojanHousing
