# Image

Table of Contents

- [Reading List](#reading-list)
- [Dataset](#dataset)
- [Benchmarks](#benchmarks)

## Reading List

| Paper                                                                                                  | Base Language Model       | Code                                                                                          | Publication | Preprint                                    | Affiliation |
| ------------------------------------------------------------------------------------------------------ | ------------------------- | --------------------------------------------------------------------------------------------- | ----------- | ------------------------------------------- | ----------- |
| Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models | LLaMA | [LaVIN](https://github.com/luogen1996/LaVIN) | | [2305.15023](https://arxiv.org/abs/2305.15023) | Xiamen Univ.|
| VisionLLM: Large Language Model is also an Open-Ended Decoder for Vision-Centric Tasks | Alpaca | [VisionLLM](https://github.com/OpenGVLab/VisionLLM) | | [2305.11175](https://arxiv.org/abs/2305.11175) | Shanghai AI Lab. | 
| Otter: A Multi-Modal Model with In-Context Instruction Tuning | Flamingo | [Otter](https://github.com/Luodian/Otter) | | [2305.03726](https://arxiv.org/abs/2305.03726) | NTU |
| X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages | ChatGPT | [X-LLM](https://github.com/phellonchen/X-LLM) | | [2305.04160](https://arxiv.org/abs/2305.04160) | CAS |
| Multimodal Procedural Planning via Dual Text-Image Prompting | OFA, BLIP, GPT3  | [TIP](https://github.com/YujieLu10/TIP) | | [2305.01795](https://arxiv.org/abs/2305.01795) | UCSB |
| LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model | LLaMA | [LLaMA-Adapter](https://github.com/ZrrSkywalker/LLaMA-Adapter) | | [2304.15010](https://arxiv.org/abs/2304.15010) | Shanghai AI Lab. |
| mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality, [mPLUG](https://aclanthology.org/2022.emnlp-main.488/), [mPLUG-2](https://arxiv.org/abs/2302.00402) | LLaMA | [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl) |  | [2304.14178](https://arxiv.org/abs/2304.14178) | DAMO Academy |
| MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models | Vicunna | [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4) | | [2304.10592](https://arxiv.org/abs/2304.10592) | KAUST |
| Visual Instruction Tuning                                        | LLaMA                   | [LLaVA](https://github.com/haotian-liu/LLaVA)                                                |             | [2304.08485](https://arxiv.org/abs/2304.08485) | Microsoft   |
| MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action                                        | ChatGPT                   | [MM-REACT](https://github.com/microsoft/MM-REACT)                                                |             | [2303.11381](https://arxiv.org/abs/2303.11381) | Microsoft   |
| ViperGPT: Visual Inference via Python Execution for Reasoning                                          | Codex                     | [ViperGPT](https://github.com/cvlab-columbia/viper)                                              |             | [2303.08128](https://arxiv.org/abs/2303.08128) | Columbia    |
| Scaling Vision-Language Models with Sparse Mixture of Experts                                          | (MOE + Scaling)           |                                                                                               |             | [2303.07226](https://arxiv.org/abs/2303.07226) | Microsoft   |
| ChatGPT Asks, BLIP-2 Answers: Automatic Questioning Towards Enriched Visual Descriptions               | ChatGPT, Flan-T5 (BLIP2)  | [ChatCaptioner](https://github.com/Vision-CAIR/ChatCaptioner)                                    |             | [2303.06594](https://arxiv.org/abs/2303.06594) | KAUST       |
| Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models                             | ChatGPT                   | [Visual ChatGPT](https://github.com/microsoft/visual-chatgpt)                                    |             | [2303.04671](https://arxiv.org/abs/2303.04671) | Microsoft   |
| PaLM-E: An Embodied Multimodal Language Model                                                          | PaLM                      |                                                                                               |             | [2303.03378](https://arxiv.org/abs/2303.03378) | Google      |
| Prismer: A Vision-Language Model with An Ensemble of Experts                                           | RoBERTa, OPT, BLOOM       | [Prismer](https://github.com/NVlabs/prismer)                                                     |             | [2303.02506](https://arxiv.org/abs/2303.02506) | Nvidia      |
| Prompting Large Language Models with Answer Heuristics for Knowledge-based Visual Question Answering | GPT3 | [Prophet](https://github.com/MILVLG/prophet) | CVPR2023 | [2303.01903](https://arxiv.org/abs/2303.01903) | HDU |
| Language Is Not All You Need: Aligning Perception with Language Models                                 | Magneto                   | [KOSMOS-1](https://github.com/microsoft/unilm)                                                   |             | [2302.14045](https://arxiv.org/abs/2302.14045) | Microsoft   |
| Scaling Vision Transformers to 22 Billion Parameters                                                   | (CLIP + Scaling)          |                                                                                               |             | [2302.05442](https://arxiv.org/abs/2302.05442) | Google      |
| Multimodal Chain-of-Thought Reasoning in Language Models                                               | T5                        | [MM-COT](https://github.com/amazon-science/mm-cot)                                               |             | [2302.00923](https://arxiv.org/abs/2302.00923) | Amazon      |
| Re-ViLM: Retrieval-Augmented Visual Language Model for Zero and Few-Shot Image Caption                 | RETRO                     |                                                                                               |             | [2302.04858](https://arxiv.org/abs/2302.04858) | Nvidia      |
| BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models | Flan-T5                   | [BLIP2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)                            |     ICML2023        | [2301.12597](https://arxiv.org/abs/2301.12597) | Salesforce  |
| See, Think, Confirm: Interactive Prompting Between Vision and Language Models for Knowledge-based Visual Reasoning | OPT | | | [2301.05226](https://arxiv.org/abs/2301.05226) | MIT-IBM |
| Generalized Decoding for Pixel, Image, and Language                                                    | GPT3                      | [X-GPT](https://github.com/microsoft/X-Decoder/tree/xgpt)                                        |             | [2212.11270](https://arxiv.org/abs/2212.11270) | Microsoft   |
| From Images to Textual Prompts: Zero-shot Visual Question Answering with Frozen Large Language Models | OPT | [Img2LLM](https://github.com/salesforce/LAVIS/tree/main/projects/img2prompt-vqa) | CVPR2023 | [2212.10846](https://arxiv.org/abs/2212.10846) | Salesforce |
| Language Models are General-Purpose Interfaces                                                         | DeepNorm                  | [METALM](https://github.com/microsoft/unilm)                                                     |             | [2206.06336](https://arxiv.org/abs/2206.06336) | Microsoft   |
| Language Models Can See: Plugging Visual Controls in Text Generation                                   | GPT2                      | [MAGIC](https://github.com/yxuansu/MAGIC)                                                        |             | [2205.02655](https://arxiv.org/abs/2205.02655) | Tencent     |
| Flamingo: a Visual Language Model for Few-Shot Learning                                                | Chinchilla                | [Flamingo](https://github.com/lucidrains/flamingo-pytorch)                                       | NIPS 2022   | [2204.14198](https://arxiv.org/abs/2204.14198) | DeepMind    |
| An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA | GPT3 | [PICa](https://github.com/microsoft/PICa) | AAAI2022 | [2109.05014](https://arxiv.org/abs/2109.05014) | Microsoft |
| Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language                                | GPT3, RoBERTa             | [Socratic Models](https://github.com/google-research/google-research/tree/master/socraticmodels) | ICLR 2023   | [2204.00598](https://arxiv.org/abs/2204.00598) | Google      |
| Learning Transferable Visual Models From Natural Language Supervision                                  | Bert                      | [CLIP](https://github.com/openai/CLIP)                                                           | ICML 2021   | [2103.00020](https://arxiv.org/abs/2103.00020) | OpenAI      |

## Datasets
- [DataComp](https://github.com/mlfoundations/datacomp)



## Benchmarks

| Benchmark                                                 | Task | Data                                                                                 | Paper                                                                                   | Preprint                                    | Publication | Affiliation |
| --------------------------------------------------------- | ---- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- | ------------------------------------------- | ----------- | ----------- |
| [INFOSEEK](https://open-vision-language.github.io/infoseek/) | VQA  | [OVEN (open domain image)](https://open-vision-language.github.io/oven/) + Human Anno. | Can Pre-trained Vision and Language Models Answer Visual Information-Seeking Questions? | [2302.11713](https://arxiv.org/abs/2302.11713) |             | Google      |