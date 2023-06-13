# Awesome-Colorful Large Language Model [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of Large Language Model ➕ Vision/Audio/Robotic and Augmented Language Model (action, reasoning).

**CONTENTS**

- [Vision](#vision)
  - [Benchmarks](#benchmarks)
  - [Image Language Model](#image-language-model)
    - [Reading List](#reading-list)
  - [Video Language Model](#video-language-model)
    - [Reading List](#reading-list-1)
    - [Pretraining Tasks](#pretraining-tasks)
    - [Datasets](#datasets)
  - [Tutorials](#tutorials)
  - [Other Curated Lists](#other-curated-lists)
    - [Model](#model)
    - [Dataset](#dataset)
- [Audio](#audio)
  - [Other Curated Lists](#other-curated-lists-1)
- [Robotic](#robotic)
  - [Reading List](#reading-list-2)
  - [Other Curated Lists](#other-curated-lists-2)
- [Augmented Language Model](#augmented-language-model)
  - [Reading List](#reading-list-3)
  - [Projects](#projects)
  - [Other Curated Lists](#other-curated-lists-3)
- [Related](#related)

## VISION

### Benchmarks

| Benchmark                                                 | Task | Data                                                                                 | Paper                                                                                   | Preprint                                    | Publication | Affiliation |
| --------------------------------------------------------- | ---- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- | ------------------------------------------- | ----------- | ----------- |
| [INFOSEEK](https://open-vision-language.github.io/infoseek/) | VQA  | [OVEN (open domain image)](https://open-vision-language.github.io/oven/) + Human Anno. | Can Pre-trained Vision and Language Models Answer Visual Information-Seeking Questions? | [2302.11713](https://arxiv.org/abs/2302.11713) |             | Google      |

### Image Language Model

#### Reading List

| Paper                                                                                                  | Base Language Model       | Code                                                                                          | Publication | Preprint                                    | Affiliation |
| ------------------------------------------------------------------------------------------------------ | ------------------------- | --------------------------------------------------------------------------------------------- | ----------- | ------------------------------------------- | ----------- |
| Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models | LLaMA | [LaVIN](https://github.com/luogen1996/LaVIN) | | [2305.15023](https://arxiv.org/abs/2305.15023) | Xiamen Univ.|
| VisionLLM: Large Language Model is also an Open-Ended Decoder for Vision-Centric Tasks | Alpaca | [VisionLLM](https://github.com/OpenGVLab/VisionLLM) | | [2305.11175](https://arxiv.org/abs/2305.11175) | Shanghai AI Lab. | 
| Otter: A Multi-Modal Model with In-Context Instruction Tuning | Flamingo | [Otter](https://github.com/Luodian/Otter) | | [2305.03726](https://arxiv.org/abs/2305.03726) | NTU |
| X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages | ChatGPT | [X-LLM](https://github.com/phellonchen/X-LLM) | | [2305.04160](https://arxiv.org/abs/2305.04160) | CAS |
| Multimodal Procedural Planning via Dual Text-Image Prompting | OFA, BLIP, GPT3  | [TIP](https://github.com/YujieLu10/TIP) | | [2305.01795](https://arxiv.org/abs/2305.01795) | UCSB |
| LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model | LLaMA | [LLaMA-Adapter](https://github.com/ZrrSkywalker/LLaMA-Adapter) | | [2304.15010](https://arxiv.org/abs/2304.15010) | Shanghai AI Lab. |
| mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality, [mPLUG](https://aclanthology.org/2022.emnlp-main.488/), [mPLUG-2](https://arxiv.org/abs/2302.00402) | LLaMA | [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl) |  | [2304.14178](https://arxiv.org/abs/2304.14178) | DAMO Academy |
| MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models | Vicunna | [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4) | | [2304.github](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/MiniGPT_4.pdf) | KAUST |
| Visual Instruction Tuning                                        | LLaMA                   | [LLaVA](https://github.com/haotian-liu/LLaVA)                                                |             | [2304.02643](https://arxiv.org/abs/2304.02643) | Microsoft   |
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

#### Dataset
- [DataComp](https://github.com/mlfoundations/datacomp)

### Video Language Model

#### Reading List

| Paper                                                                                                                     | Base Language Model | Code                                                                                                     | Publication         | Preprint                                    | Affiliation |
| ------------------------------------------------------------------------------------------------------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------------- | ----------- |
| Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models | Vicuna | [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT) | | [2306.05424](https://arxiv.org/abs/2306.05424) | MBZUAI |
| Macaw-LLM: Multi-Modal Language Modeling with Image, Video, Audio, and Text Integration | LLaMA | [Macaw-LLM](https://github.com/lyuchenyang/Macaw-LLM) | | [2305.github](https://github.com/lyuchenyang/Macaw-LLM) | Tencent |
| Video-LLaMA: An Instruction-Finetuned Visual Language Model for Video Understanding | LLaMA | [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA) | | [2305.github](https://github.com/DAMO-NLP-SG/Video-LLaMA) | Alibaba | 
| Self-Chained Image-Language Model for Video Localization and Question Answering | BLIP2 | [SeViLA](https://github.com/Yui010206/SeViLA) | | [2305.06988](https://arxiv.org/abs/2305.06988) | UNC | 
| VideoChat: Chat-Centric Video Understanding | Blip2 | [VideoChat](https://github.com/OpenGVLab/Ask-Anything) | | [2305.06355](https://arxiv.org/abs/2305.06355) | Shanghai AI Lab |
| X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages | ChatGPT | [X-LLM](https://github.com/phellonchen/X-LLM) | | [2305.04160](https://arxiv.org/abs/2305.04160) | CAS |
| VALOR: Vision-Audio-Language Omni-Perception Pretraining Model and Dataset | Bert | [VALOR](https://github.com/TXH-mercury/VALOR) | | [2304.08345](https://arxiv.org/abs/2304.08345) | UCAS |
| Verbs in Action: Improving verb understanding in video-language models | PaLM |  | | [2304.06708](https://arxiv.org/abs/2304.06708) | Google |
| Video ChatCaptioner: Towards the Enriched Spatiotemporal Descriptions                                  | ChatGPT, Flan-T5 (BLIP2) | [ChatCaptioner](https://github.com/Vision-CAIR/ChatCaptioner)                                    |             | [2304.04227](https://arxiv.org/abs/2304.04227) | KAUST       |
| Language Models are Causal Knowledge Extractors for Zero-shot Video Question Answering  | GPT2, GPT-Neo, GPT3 | | CVPR2023 workshop | [2304.03754](https://arxiv.org/abs/2304.03754) | Columbia Univ. |
| Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning                                    | T5                  | [Vid2Seq](https://github.com/google-research/scenic/tree/main/scenic/projects/vid2seq)                      |                     | [2302.14115](https://arxiv.org/abs/2302.14115) | Google      |
| HiTeA: Hierarchical Temporal-Aware Video-Language Pre-training                                                            | Bert                |                                                                                                          |                     | [2212.14546](https://arxiv.org/abs/2212.14546) | Alibaba     |
| VindLU: A Recipe for Effective Video-and-Language Pretraining                                                             | Bert                | [VindLU](https://github.com/klauscc/VindLU)                                                                 |                     | [2212.05051](https://arxiv.org/abs/2212.05051) | UNC         |
| SMAUG: Sparse Masked Autoencoder for Efficient Video-Language Pre-training                                                | Bert                |                                                                                                          |                     | [2211.11446](https://arxiv.org/abs/2211.11446) | UW          |
| CLOP: Video-and-Language Pre-Training with Knowledge Regularizations                                                      | Roberta             |                                                                                                          | MM 2022             | [2211.03314](https://arxiv.org/abs/2211.03314) | Baidu       |
| Long-Form Video-Language Pre-Training with Multimodal Temporal Contrastive Learning                                       | Bert                |                                                                                                          | NIPS 2022           | [2210.06031](https://arxiv.org/abs/2210.06031) | Microsoft   |
| OmniVL: One Foundation Model for Image-Language and Video-Language Tasks                                                  | Bert                |                                                                                                          | NIPS 2022           | [2209.07526](https://arxiv.org/abs/2209.07526) | Microsoft   |
| Clover: Towards A Unified Video-Language Alignment and Fusion Model                                                       | Bert                | [Clover](https://github.com/LeeYN-43/Clover)                                                                |                     | [2207.07885](https://arxiv.org/abs/2207.07885) | Bytedance   |
| LAVENDER: Unifying Video-Language Understanding as Masked Language Modeling                                               | Bert-like           | [LAVENDER](https://github.com/microsoft/LAVENDER)                                                           | CVPR 2023           | [2206.07160](https://arxiv.org/abs/2206.07160) | Microsoft   |
| Revealing Single Frame Bias for Video-and-Language Learning                                                               | Bert                | [Singularity](https://github.com/jayleicn/singularity)                                                      |                     | [2206.03428](https://arxiv.org/abs/2206.03428) | UNC         |
| Flamingo: a Visual Language Model for Few-Shot Learning                                                                   | Chinchilla          | [Flamingo](https://github.com/lucidrains/flamingo-pytorch)                                                  | NIPS 2022           | [2204.14198](https://arxiv.org/abs/2204.14198) | DeepMind    |
| All in One: Exploring Unified Video-Language Pre-training                                                                 | Bert-like           | [All-In-One](https://github.com/showlab/all-in-one)                                                         | CVPR 2023           | [2203.07303](https://arxiv.org/abs/2203.07303) | NUS         |
| End-to-end Generative Pretraining for Multimodal Video Captioning                                                         | Bert+GPT2           |                                                                                                          | CVPR 2022           | [2201.08264](https://arxiv.org/abs/2201.08264) | Google      |
| Align and Prompt: Video-and-Language Pre-training with Entity Prompts                                                     | Bert-like           | [ALPRO](https://github.com/salesforce/ALPRO)                                                                | CVPR 2022           | [2112.09583](https://arxiv.org/abs/2112.09583) | Salesforce  |
| VIOLET : End-to-End Video-Language Transformers with Masked Visual-token Modeling,[V2](https://arxiv.org/pdf/2209.01540.pdf) | Bert                | [VIOLET](https://github.com/tsujuifu/pytorch_violet)                                                        |                     | [2111.12681](https://arxiv.org/abs/2111.12681) | Microsoft   |
| VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding                                                | Bert                | [VideoCLIP](https://github.com/facebookresearch/fairseq/tree/main/examples/MMPT)                            | EMNLP 2021          | [2109.14084](https://arxiv.org/abs/2109.14084) | Facebook    |
| MERLOT: Multimodal Neural Script Knowledge Models,[V2](https://arxiv.org/abs/2201.02639)                                     | Roberta             | [MERLOT](https://github.com/rowanz/merlot)                                                                  | NIPS 2021           | [2106.02636](https://arxiv.org/abs/2106.02636) | AI2         |
| VLM: Task-agnostic Video-Language Model Pre-training for Video Understanding                                              | Bert                | [VLP](https://github.com/facebookresearch/fairseq/blob/main/examples/MMPT/README.md)                        | ACL Findings 2021   | [2105.09996](https://arxiv.org/abs/2105.09996) | Facebook    |
| VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text                                 | Bert-like           |                                                                                                          | NIPS 2021           | [2104.11178](https://arxiv.org/abs/2104.11178) | Google      |
| CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval                                                 | Bert-like           | [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip)                                                          | Neurocomputing 2022 | [2104.08860](https://arxiv.org/abs/2104.08860) | Microsoft   |
| Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval                                                  | Bert                | [Frozen-in-Time](https://github.com/m-bain/frozen-in-time)                                                  | ICCV 2021           | [2104.00650](https://arxiv.org/abs/2104.00650) | Oxford      |
| Less is More: CLIPBERT for Video-and-Language Learning via Sparse Sampling                                                | Bert                | [ClipBert](https://github.com/jayleicn/ClipBERT)                                                            | CVPR 2021           | [2102.06183](https://arxiv.org/abs/2102.06183) | Microsoft   |
| ActBERT: Learning Global-Local Video-Text Representations                                                                 | Bert                | [ActBert](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/en/model_zoo/multimodal/actbert.md) | CVPR 2020           | [2011.07231](https://arxiv.org/abs/2011.07231) | Baidu       |
| Video Understanding as Machine Translation                                                                                | T5                  |                                                                                                          |                     | [2006.07203](https://arxiv.org/abs/2006.07203) | Facebook    |
| HERO: Hierarchical Encoder for Video+Language Omni-representation Pre-training                                            | Bert                | [HERO](https://github.com/linjieli222/HERO)                                                                 | EMNLP 2020          | [2005.00200](https://arxiv.org/abs/2005.00200) | Microsoft   |
| UniVL: A Unified Video and Language Pre-Training Model for Multimodal Understanding and Generation                        | Bert                | [UniVL](https://github.com/microsoft/UniVL)                                                                 |                     | [2002.06353](https://arxiv.org/abs/2002.06353) | Microsoft   |
| Learning Video Representations using Contrastive Bidirectional Transformer                                                | Bert                |                                                                                                          |                     | [1906.05743](https://arxiv.org/abs/1906.05743) | Google      |
| VideoBERT: A Joint Model for Video and Language Representation Learning                                                   | Bert                | [VideoBert (non-official)](https://github.com/ammesatyajit/VideoBERT)                                       | ICCV 2019           | [1904.01766](https://arxiv.org/abs/1904.01766) | Google      |

#### Pretraining Tasks

<details><summary>Commmonly Used Pretraining Tasks</summary>

- Masked Language Modeling (MLM)
- Causal Language Modeling (LM)
- Masked Vision Modeling (MLM)
  - Vision = Frame
  - Vision = Patch
  - VIsion = Object
- Video Language Matching (VLM)
- Video Language Contrastive (VLC)

</details>

#### Datasets

<details><summary>Commmonly Used Video Corpus for Pretraining</summary>

| Paper                                                                                         | Video Clips | Duration | Sentences | Domain      | Download Link                                                              |
| --------------------------------------------------------------------------------------------- | ----------- | -------- | --------- | ----------- | -------------------------------------------------------------------------- |
| Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval                      | 2.5M        | 18s      | 2.5M      | open        | [WebVid-2M](https://github.com/m-bain/webvid)                                 |
| Howto100m: Learning a text-video embedding by watching hundred million narrated video clips   | 136M        | 4s       | 136M      | instruction | [HowTo100M](https://www.di.ens.fr/willow/research/howto100m/)                 |
| Merlot: Multimodal neural script knowledge models. Advances in Neural Information Processing  | 6M          | -20m     | ~720M     | open        | [YT-Temporal-180M](https://rowanzellers.com/merlot/)                          |
| Advancing High-Resolution Video-Language Representation with Large-Scale Video Transcriptions | 100M        | 13.4s    | 100M      | open        | [HD-VILA-100M](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m) |
| CHAMPAGNE: Learning Real-world Conversation from Large-Scale Web Videos                       | 18M         | 60s      |           | open        | [YTD-18M](https://seungjuhan.me/champagne/)                                   |

</details>

<details><summary>Commmonly Used Downsteam Tasks</summary>

| **Task**          | **Paper**                                                                                                         | **Download Link**                                                                                                     | **Publication** |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| **Retrieval**     | **Collecting Highly Parallel Data for Paraphrase Evaluation**                                                     | [MSVD](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/)                                                             | **ACL 2011**    |
| **Retrieval**     | **A Dataset for Movie Description**                                                                               | [LSMDC](https://sites.google.com/site/describingmovies/download)                                                               | **CVPR 2015**   |
| **Retrieval**     | **MSR-VTT: A Large Video Description Dataset for Bridging Video and Language**                                    | [MSR-VTT](https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip)                                                 | **CVPR 2016**   |
| **Retrieval**     | **Localizing Moments in Video with Natural Language**                                                             | [DiDeMo](https://github.com/LisaAnne/LocalizingMoments)                                                                        | **ICCV 2017**   |
| **Retrieval**     | **Dense-Captioning Events in Videos**                                                                             | [ActivityNet Caption](https://cs.stanford.edu/people/ranjaykrishna/densevid/)                                                  | **ICCV 2017**   |
| **Retrieval**     | **Towards Automatic Learning of Procedures from Web Instructional Videos**                                        | [YouCook2](http://youcook2.eecs.umich.edu/download)                                                                            | **AAAI 2018**   |
| **OE QA**         | **TGIF-QA: Toward Spatio-Temporal Reasoning in Visual Question Answering**                                        | [TGIF-Frame](https://github.com/YunseokJANG/tgif-qa/tree/cvpr2017/dataset)                                                     | **CVPR 2017**   |
| **OE QA**         | **A dataset and exploration of models for understanding video data through fill-in-the-blank question-answering** | [LSMDC-FiB](https://github.com/yj-yu/lsmdc)                                                                                    | **CVPR 2017**   |
| **OE QA**         | **Video Question Answering via Gradually Refined Attention over Appearance and Motion**                           | [MSRVTT-QA](https://github.com/xudejing/video-question-answering),[MSVD-QA](https://github.com/xudejing/video-question-answering) | **MM 2017**     |
| **OE QA**         | **ActivityNet-QA: A Dataset for Understanding Complex Web Videos via Question Answering**                         | [ActivityNet-QA](https://github.com/MILVLG/activitynet-qa)                                                                     | **AAAI 2019**   |
| **MC QA**         | **Learning Language-Visual Embedding for Movie Understanding with Natural-Language**                              | [LSMDC-MC](https://github.com/yj-yu/lsmdc)                                                                                     |                       |
| **MC  QA**        | **TGIF-QA: Toward Spatio-Temporal Reasoning in Visual Question Answering**                                        | [TGIF-Action, TGIF-Transition](https://github.com/YunseokJANG/tgif-qa/tree/cvpr2017/dataset)                                   | **CVPR 2017**   |
| **MC QA**         | **A Joint Sequence Fusion Model for Video Question Answering and Retrieval**                                      | [MSRVTT-MC](https://github.com/yj-yu/lsmdc)                                                                                    | **ECCV 2018**   |
| **Caption**       | **Collecting Highly Parallel Data for Paraphrase Evaluation**                                                     | [MSVD](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/)                                                             | **ACL 2011**    |
| **Caption**       | **MSR-VTT: A Large Video Description Dataset for Bridging Video and Language**                                    | [MSR-VTT](https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip)                                                 | **CVPR 2016**   |
| **Dense Caption** | **Dense-Captioning Events in Videos**                                                                             | [ActivityNet Caption](https://cs.stanford.edu/people/ranjaykrishna/densevid/)                                                  | **ICCV 2017**   |
| **Dense Caption** | **Towards Automatic Learning of Procedures from Web Instructional Videos**                                        | [YouCook2](http://youcook2.eecs.umich.edu/download)                                                                            | **AAAI 2018**   |
| **Dense Caption** | **Multimodal Pretraining for Dense Video Captioning**                                                             | [ViTT](https://github.com/google-research-datasets/Video-Timeline-Tags-ViTT)                                                   | **AACL 2020**   |

</details>

<details><summary>Advanced Video Language Tasks</summary>

| paper                                                                                                          | task                     | duration | domain       | link                                                                                         | publication |
| -------------------------------------------------------------------------------------------------------------- | ------------------------ | -------- | ------------ | -------------------------------------------------------------------------------------------- | ----------- |
| From Representation to Reasoning: Towards both Evidence and Commonsense Reasoning for Video Question-Answering | Video QA                 | 9s       | open         | [Causal-VidQA](https://github.com/bcmi/Causal-VidQA)                                            | CVPR 2022   |
| VIOLIN: A Large-Scale Dataset for Video-and-Language Inference                                                 | Video Language Inference | 35.2s    | movie        | [VIOLIN](https://github.com/jimmy646/violin)                                                    | CVPR 2020   |
| TVQA: Localized, Compositional Video Question Answering                                                        | Video QA                 | 60-90s   | movie        | [TVQA](https://tvqa.cs.unc.edu/)                                                                | EMNLP 2018  |
| AGQA: A Benchmark for Compositional Spatio-Temporal Reasoning                                                  | Video QA                 | 30s      | open         | [AGQA](https://cs.stanford.edu/people/ranjaykrishna/agqa/)                                      | CVPR 2021   |
| NExT-QA: Next Phase of Question-Answering to Explaining Temporal Actions                                       | Video QA                 | 44s      | open         | [NExT-QA-MC](https://github.com/doc-doc/NExT-QA), [NExT-QA-OE](https://github.com/doc-doc/NExT-OE) | CVPR 2021   |
| STAR: A Benchmark for Situated Reasoning in Real-World Videos                                                  | Video QA                 | 12s      | open         | [Star](https://github.com/csbobby/STAR_Benchmark)                                               | NIPS 2021   |
| Env-QA: A Video Question Answering Benchmark for Comprehensive Understanding of Dynamic Environments           | Video QA                 | 20s      | virtual env. | [Env-QA](https://envqa.github.io/)                                                              | ICCV 2021   |
| Social-IQ: A Question Answering Benchmark for Artificial Social Intelligence                                   | Video QA                 | 60s      | open         | [Social-IQ](https://www.thesocialiq.com/)                                                       | CVPR 2019   |

</details>

### Image Generation
#### Reading List


### Tutorials

- [[CVPR2022 Tutorial] Recent Advances in Vision-and-Language Pre-training](https://vlp-tutorial.github.io/)

### Other Curated Lists

#### Model:

- [LLM-in-Vision](https://github.com/DirtyHarryLYL/LLM-in-Vision)
- [Awesome-Transformer-Attention](https://github.com/cmhungsteve/Awesome-Transformer-Attention)
- [Awesome-Vision-and-Language](https://github.com/sangminwoo/awesome-vision-and-language#survey)
- [Awesome-Diffusion-Models](https://github.com/heejkoo/Awesome-Diffusion-Models)
- [Awesome-3D-Vision-and-Language](https://github.com/jianghaojun/Awesome-3D-Vision-and-Language)

#### Dataset:

- [Awesome-Video-Datasets](https://github.com/xiaobai1217/Awesome-Video-Datasets#Video-and-Language)

## Audio

### Other Curated Lists

- [Audio-AI-Timeline](https://github.com/archinetai/audio-ai-timeline)

## Robotic

### Reading List

| Paper                                                                                    | Base Language Model | Code | Publication | Preprint                                    | Affiliation         |
| ---------------------------------------------------------------------------------------- | ------------------- | ---- | ----------- | ------------------------------------------- | ------------------- |
| Chat with the Environment: Interactive Multimodal Perception using Large Language Models | GPT3                |      |             | [2303.08268](https://arxiv.org/abs/2303.08268) | Universitat Hamburg |

### Other Curated Lists

- [Awesome-LLM-Robotics](https://github.com/GT-RIPL/Awesome-LLM-Robotics)
- [PromptCraft-Robotics](https://github.com/microsoft/PromptCraft-Robotics)
- [Awesome-Robotics](https://github.com/ahundt/awesome-robotics)

## Augmented Language Model

### Reading List

Survey

- (2023-04) Tool Learning with Foundation Models [paper](https://arxiv.org/abs/2304.08354)
- (2023-02) Augmented Language Models: a Survey [paper](https://arxiv.org/abs/2302.07842)

Reading List

| Paper                                                                                                          | LLM                | Code                                                                           | Publication | Preprint                                    | Affiliation   |
| -------------------------------------------------------------------------------------------------------------- | ------------------ | ------------------------------------------------------------------------------ | ----------- | ------------------------------------------- | ------------- |
| LLM+P: Empowering Large Language Models with Optimal Planning Proficiency | GPT4 | [LLM-PDDL](https://github.com/Cranial-XIX/llm-pddl) | | [2304.11477](https://arxiv.org/abs/2304.11477) | UTEXAS |
| Can GPT-4 Perform Neural Architecture Search? | GPT4 | [GENIUS](https://github.com/mingkai-zheng/GENIUS) | | [2304.10970](https://arxiv.org/abs/2304.10970) | Cambridge | 
| Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models | GPT4 | [Chameleon](https://github.com/lupantech/chameleon-llm) | | [2304.09842](https://arxiv.org/abs/2304.09842) | Microsoft |
| OpenAGI: When LLM Meets Domain Experts                                                                         | ChatGPT            | [OpenAGI](https://github.com/agiresearch/OpenAGI)                                 |             | [2304.04370](https://arxiv.org/abs/2304.04370) | Rutgers Univ. |
| HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace                                       | ChatGPT            | [JARVIS](https://github.com/microsoft/JARVIS)                                     |             | [2303.17580](https://arxiv.org/abs/2303.17580) | Microsoft     |
| Language Models can Solve Computer Tasks                                                                       | ChatGPT, GPT3, etc | [RCI Agent](https://github.com/posgnu/rci-agent)                                  |             | [2303.17491](https://arxiv.org/abs/2303.17491) | CMU           |
| TaskMatrix.AI: Completing Tasks by Connecting Foundation Models with Millions of APIs                          | ChatGPT            | [TaskMatrix](https://github.com/microsoft/visual-chatgpt/tree/main/TaskMatrix.AI) |             | [2303.16434](https://arxiv.org/abs/2303.16434) | Microsoft     |
| MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action                                        | ChatGPT                   | [MM-REACT](https://github.com/microsoft/MM-REACT)                                                |             | [2303.11381](https://arxiv.org/abs/2303.11381) | Microsoft   |
| ART: Automatic multi-step reasoning and tool-use for large language models                                    | GPT3, Codex        | [Language-Programmes](https://github.com/bhargaviparanjape/language-programmes)   |             | [2303.09014](https://arxiv.org/abs/2303.09014) | Microsoft     |
| Foundation Models for Decision Making: Problems, Methods, and Opportunities                                    | -                  | -                                                                              |             | [2303.04129](https://arxiv.org/abs/2303.04129) | Google        |
| Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback | ChatGPT            | [LLM-Augmenter](https://github.com/pengbaolin/LLM-Augmenter)                      |             | [2302.12813](https://arxiv.org/abs/2302.12813) | Microsoft     |
| Toolformer: Language Models Can Teach Themselves to Use Tools                                                  | GPT-J, OPT, GPT3   | [Toolformer (Unofficial)](https://github.com/lucidrains/toolformer-pytorch)       |             | [2302.04761](https://arxiv.org/abs/2302.04761) | Meta          |
| Visual Programming: Compositional visual reasoning without training | GPT3 | [VisProg](https://github.com/allenai/visprog) | CVPR2023 | [2211.11559](https://arxiv.org/abs/2211.11559) | AI2 | 

### Projects
- (2023-04) AgentGPT [code](https://github.com/reworkd/AgentGPT)
- (2023-04) Auto-GPT [code](https://github.com/Torantulino/Auto-GPT)
- (2023-04) BabyAGI [code](https://github.com/yoheinakajima/babyagi), [blog](https://twitter.com/yoheinakajima/status/1640934493489070080?s=20)
- (2023-05)  闻达[code](https://github.com/l15y/wenda)


### Other Curated Lists

- [ToolLearningPapers](https://github.com/thunlp/ToolLearningPapers)

## Related

- [Awesome-Multimodal-Research](https://github.com/Eurus-Holmes/Awesome-Multimodal-Research)
- [Awesome-Multimodal-ML](https://github.com/pliang279/awesome-multimodal-ml)
- [Awesome-ALM](https://github.com/pbhu1024/awesome-augmented-language-model#action-and-plan)
- [LLMsPracticalGuide](https://github.com/Mooler0410/LLMsPracticalGuide)

## Contributing

Please freely create [pull request](https://github.com/patrick-tssn/Awesome-Colorful-LLM/pulls) or drop me an [email](flagwyx@gmail.com)
