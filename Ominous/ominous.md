# Ominous

## Reading List

Papers

#Multimodal #End2end Understanding+Generation:

| Paper                                                                                                                            | Base Model          | Framework                                        | Data                                                      | Code                                                         | Publication | Preprint                                        | Affiliation     |
| -------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------------------ | --------------------------------------------------------- | ------------------------------------------------------------ | ----------- | ----------------------------------------------- | --------------- |
| One Single Transformer to Unify Multimodal Understanding and Generation                                                          | Phi + MagViT2       | PT + FT (LM-loss + MAE-loss)                     | mixture (image)                                           | [Show-o](https://github.com/showlab/Show-o)                     |             | [2408.12528](https://arxiv.org/abs/2408.12528)     | NUS             |
| Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model                                                | - (transfusion)     | PT (LM-loss + DDPM-loss)                         | self-collect (image)                                      |                                                              |             | [2408.11039](https://www.arxiv.org/abs/2408.11039) | Meta            |
| Anole: An Open, Autoregressive and Native Multimodal Models for Interleaved Image-Text Generation                                | chameleon           | INST (interleaved)                               | mixture (image)                                           | [anole](https://github.com/GAIR-NLP/anole)                      |             | [2407.06135](https://arxiv.org/abs/2407.06135)     | SJTU            |
| Explore the Limits of Omni-modal Pretraining at Scale                                                                            | vicuna              | PT + INST                                        | mixture (image, video, audio, depth -> text)              | [MiCo](https://github.com/invictus717/MiCo)                     |             | [2406.09412](https://arxiv.org/pdf/2406.09412)     | Shanghai AI Lab |
| X-VILA: Cross-Modality Alignment for Large Language Model                                                                        | vicuna + SD         | INST + Diffusion Decoder                         | mixture (image, video, audio)                             |                                                              |             | [2405.19335](https://arxiv.org/abs/2405.19335)     | NVIDIA          |
| Chameleon: Mixed-Modal Early-Fusion Foundation Models                                                                            | - (chameleon)       | PT + FT (AR + image detokenizer)                 | mixture (image)                                           | [chameleon](https://github.com/facebookresearch/chameleon)      |             | [2405.09818](https://arxiv.org/abs/2405.09818)     | Meta            |
| SEED-X: Multimodal Models with Unified Multi-granularity Comprehension and Generation                                            | LLaMA + SD          | PT + INST (interleaved)                          | mixture (image)                                           | [SEED-X](https://github.com/AILab-CVC/SEED-X)                   |             | [2404.14396](https://arxiv.org/abs/2404.14396)     | Tencent         |
| AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling                                                                   | LLaMA2 + SD         | INST + NAR-decoder                               | mixture (image, speech, music)                            | [AnyGPT](https://github.com/OpenMOSS/AnyGPT)                    |             | [2402.12226](https://arxiv.org/abs/2402.12226)     | FDU             |
| World Model on Million-Length Video And Language With Blockwise RingAttention                                                    | LLaMA + VQGAN (LWM) | PT (long-context)                                | mixture (image, video)                                    | [LWM](https://github.com/LargeWorldModel/LWM)                   |             | [2402.08268](https://arxiv.org/abs/2402.08268)     | UCB             |
| MM-Interleaved: Interleaved Image-Text Generative Modeling via Multi-modal Feature Synchronizer                                  | Vicuna + SD         | PT + INST                                        | mixture (image)                                           | [MM-Interleaved](https://github.com/OpenGVLab/MM-Interleaved)   |             | [2401.10208](https://arxiv.org/abs/2401.10208)     | Shanghai AI Lab |
| Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action                                  | T5X + VQGAN         | PT + INST                                        | mixture (image, audio, video, 3d)                         | [unified-io-2](https://github.com/allenai/unified-io-2)         |             | [2312.17172](https://arxiv.org/abs/2312.17172)     | AI2             |
| VL-GPT: A Generative Pre-trained Transformer for Vision and Language Understanding and Generation                                | LLaMA + SD          | PT + INST (interleaved)                          | mixture (image)                                           | [VL-GPT](https://github.com/AILab-CVC/VL-GPT)                   |             | [2312.09251](https://arxiv.org/abs/2312.09251)     | Tencent         |
| OneLLM: One Framework to Align All Modalities with Language                                                                      | LLaMA2              | PT + INST (universal encoder + moe projector)    | mixture (image, audio, point, depth, IMU, fMRI -> text)   | [OneLLM](https://github.com/csuhan/OneLLM)                      | CVPR2024    | [2312.03700](https://arxiv.org/abs/2312.03700)     | Shanghai AI lab |
| LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment                            | -                   | INST                                             | mixture (video, infrared, depth, audio -> text)           | [LanguageBind](https://github.com/PKU-YuanGroup/LanguageBind)   | ICLR2024    | [2310.01852](https://arxiv.org/abs/2310.01852)     | PKU             |
| DreamLLM: Synergistic Multimodal Comprehension and Creation                                                                      | Vicuna + SD         | PT + INST with projector (interleaved)           | mixture (image)                                           | [DreamLLM](https://github.com/RunpeiDong/DreamLLM)              | ICLR2024    | [2309.11499](https://arxiv.org/abs/2309.11499)     | MEGVII          |
| NExT-GPT: Any-to-Any Multimodal LLM                                                                                              | Vicuna + SD         | INST with projector                              | mixture (text -> audio/image/video)                       | [NExT-GPT](https://github.com/NExT-GPT/NExT-GPT)                | ICML2024    | [2309.05519](https://arxiv.org/abs/2309.05519)     | NUS             |
| LaVIT: Empower the Large Language Model to Understand and Generate Visual Content,[video version](https://arxiv.org/abs/2402.03161) | LLaMA  + SD        | PT + INST (vector quantization: CE + regression) | mixture (image)                                           | [LaVIT](https://github.com/jy0205/LaVIT)                        | ICLR2024    | [2309.04669](https://arxiv.org/abs/2309.04669)     | Kuaishou        |
| Emu: Generative Pretraining in Multimodality,[v2](https://arxiv.org/abs/2312.13286)                                                 | LLaMA + SD          | PT (AR: CE + regression )                       | mixture (image)                                           | [Emu](https://github.com/baaivision/Emu)                        | ICLR2024    | [2307.05222](https://arxiv.org/abs/2307.05222)     | BAAI            |
| Any-to-Any Generation via Composable Diffusion                                                                                   | SD-1.5              | individual diffusion -> latent attention         | mixture (text -> audio/image/video; image -> audio/video) | [CoDi](https://github.com/microsoft/i-Code/tree/main/i-Code-V3) | NeurIPS2023 | [2305.11846](https://arxiv.org/abs/2305.11846)     | Microsoft       |
| ImageBind: One Embedding Space To Bind Them All                                                                                  | CLIP                | Contrastive + Diffusion Decoder                  | mixture(image, video, audio, depth)                       | [ImageBind](https://github.com/facebookresearch/ImageBind)      |             | [2305.05665](https://arxiv.org/abs/2305.05665)     | Meta            |

#Streaming #Real-Time #Online

| Paper                                                                    | Base Model      | Framework                                                    | Data                                 | Code                                                                                            | Publication | Preprint                                    | Affiliation |
| ------------------------------------------------------------------------ | --------------- | ------------------------------------------------------------ | ------------------------------------ | ----------------------------------------------------------------------------------------------- | ----------- | ------------------------------------------- | ----------- |
| Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming    | Qwen2           | audio generation with text instruction + parallel generation | self-construct (VoiceAssistant-400K) | [mini-omni](https://github.com/gpt-omni/mini-omni)                                                 |             | [2408.16725](https://arxiv.org/abs/2408.16725) | THU         |
| VITA: Towards Open-Source Interactive Omni Multimodal LLM                | Mixtral-8x7B    | special tokens (<1>: audio; <2>: EOS; <3> text)              | mixture                              | [VITA](https://github.com/VITA-MLLM/VITA)                                                          |             | [2408.05211](https://arxiv.org/abs/2408.05211) | Tencent     |
| VideoLLM-online: Online Large Language Model for Streaming Video         | Llama2/3        | Multi-turn dialogue + streaming loss                         | Ego4D                                | [videollm-online](https://github.com/showlab/videollm-online)                                      |             | [2406.11816](https://arxiv.org/abs/2406.11816) | NUS         |
| RT-DETR: DETRs Beat YOLOs on Real-time Object Detection                  | Dino + DETR     | anchor-free                                                  | COCO                                 | [RT-DETR](https://github.com/lyuwenyu/RT-DETR)                                                     |             | [2304.08069](https://arxiv.org/abs/2304.08069) | Baidu       |
| Streaming Dense Video Captioning                                         | GIT/VidSeq + T5 | cluster visual token (memory)                                |                                      | [streaming_dvc](https://github.com/google-research/scenic/tree/main/scenic/projects/streaming_dvc) | CVPR2024    | [2304.08069](https://arxiv.org/abs/2404.01297) | Google      |
| Deformable DETR: Deformable Transformers for End-to-End Object Detection | ResNet+DETR     | deformable-attention                                         | COCO                                 | [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)                            | ICLR2021    | [2010.04159](https://arxiv.org/abs/2010.04159) | SenseTime   |

Projects:

- [2024.07] [SAM2](https://github.com/facebookresearch/segment-anything-2), Introducing Meta Segment Anything Model 2 (SAM 2)
  - [2024.08] [segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time), Run Segment Anything Model 2 on a live video stream
- [2024.06] [LLaVA-Magvit2](https://github.com/lucasjinreal/LLaVA-Magvit2), LLaVA MagVit2: Combines MLLM Understanding and Generation with MagVit2
- [2024.05] [GPT-4o system card](https://openai.com/index/hello-gpt-4o/), We’re announcing GPT-4o, our new flagship model that can reason across audio, vision, and text in real time.

## Dataset

#omininou-modality

- [2024.06] [ShareGPT4Omni Dataset](https://sharegpt4omni.github.io/), ShareGPT4Omni: Towards Building Omni Large Multi-modal Models with Comprehensive Multi-modal Annotations.

#streaming-data

- [2024.06] VideoLLM-online: Online Large Language Model for Streaming Video
- [2024.05] Streaming Long Video Understanding with Large Language Models

## Benchmark

#timestampQA

- [2024.06] [VStream-QA](https://invinciblewyq.github.io/vstream-page/), Flash-VStream: Memory-Based Real-Time Understanding for Long Video Streams

#state#episodic

- [2024.04] [OpenEQA](https://open-eqa.github.io/), OpenEQA: Embodied Question Answering in the Era of Foundation Models
- [2021.10] [Env-QA](https://envqa.github.io/), Env-QA: A Video Question Answering Benchmark for Comprehensive Understanding of Dynamic Environments
