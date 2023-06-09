# Augmented Language Model

Table of Contents

- [Reading List](#reading-list)
- [Projects](#projects)
- [Applications](#applications)
- [Reference](#reference)

## Reading List

Survey

- (2023-04) Tool Learning with Foundation Models [paper](https://arxiv.org/abs/2304.08354), [BMTools](https://github.com/OpenBMB/BMTools)
- (2023-02) Augmented Language Models: a Survey [paper](https://arxiv.org/abs/2302.07842)

Reading List

| Paper                                                                                                          | LLM                | Code                                                                           | Publication | Preprint                                    | Affiliation   |
| -------------------------------------------------------------------------------------------------------------- | ------------------ | ------------------------------------------------------------------------------ | ----------- | ------------------------------------------- | ------------- |
| AssistGPT: A General Multi-modal Assistant that can Plan, Execute, Inspect, and Learn                          | ChatGPT            | [AssistGPT](https://github.com/COOORN/AssistGPT)                                  |             | [2306.08640](https://arxiv.org/abs/2306.08640) | NUS           |
| GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction                                    | Vicuna-13B         | [GPT4Tools](https://github.com/StevenGrove/GPT4Tools)                             |             | [2305.18752](https://arxiv.org/abs/2305.18752) | Tencent       |
| AdaPlanner: Adaptive Planning from Feedback with Language Models                                              | GPT3/3.5           | [AdaPlanner](https://github.com/haotiansun14/AdaPlanner)                          |             | [2305.16653](https://arxiv.org/abs/2305.16653) | Gatech        |
| LLM+P: Empowering Large Language Models with Optimal Planning Proficiency                                      | GPT4               | [LLM-PDDL](https://github.com/Cranial-XIX/llm-pddl)                               |             | [2304.11477](https://arxiv.org/abs/2304.11477) | UTEXAS        |
| Can GPT-4 Perform Neural Architecture Search?                                                                  | GPT4               | [GENIUS](https://github.com/mingkai-zheng/GENIUS)                                 |             | [2304.10970](https://arxiv.org/abs/2304.10970) | Cambridge     |
| Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models                                    | GPT4               | [Chameleon](https://github.com/lupantech/chameleon-llm)                           |             | [2304.09842](https://arxiv.org/abs/2304.09842) | Microsoft     |
| OpenAGI: When LLM Meets Domain Experts                                                                         | ChatGPT            | [OpenAGI](https://github.com/agiresearch/OpenAGI)                                 |             | [2304.04370](https://arxiv.org/abs/2304.04370) | Rutgers Univ. |
| HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace                                       | ChatGPT            | [JARVIS](https://github.com/microsoft/JARVIS)                                     |             | [2303.17580](https://arxiv.org/abs/2303.17580) | Microsoft     |
| Language Models can Solve Computer Tasks                                                                       | ChatGPT, GPT3, etc | [RCI Agent](https://github.com/posgnu/rci-agent)                                  |             | [2303.17491](https://arxiv.org/abs/2303.17491) | CMU           |
| TaskMatrix.AI: Completing Tasks by Connecting Foundation Models with Millions of APIs                          | ChatGPT            | [TaskMatrix](https://github.com/microsoft/visual-chatgpt/tree/main/TaskMatrix.AI) |             | [2303.16434](https://arxiv.org/abs/2303.16434) | Microsoft     |
| MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action                                                | ChatGPT            | [MM-REACT](https://github.com/microsoft/MM-REACT)                                 |             | [2303.11381](https://arxiv.org/abs/2303.11381) | Microsoft     |
| ART: Automatic multi-step reasoning and tool-use for large language models                                    | GPT3, Codex        | [Language-Programmes](https://github.com/bhargaviparanjape/language-programmes)   |             | [2303.09014](https://arxiv.org/abs/2303.09014) | Microsoft     |
| Foundation Models for Decision Making: Problems, Methods, and Opportunities                                    | -                  | -                                                                              |             | [2303.04129](https://arxiv.org/abs/2303.04129) | Google        |
| Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback | ChatGPT            | [LLM-Augmenter](https://github.com/pengbaolin/LLM-Augmenter)                      |             | [2302.12813](https://arxiv.org/abs/2302.12813) | Microsoft     |
| Toolformer: Language Models Can Teach Themselves to Use Tools                                                  | GPT-J, OPT, GPT3   | [Toolformer (Unofficial)](https://github.com/lucidrains/toolformer-pytorch)       |             | [2302.04761](https://arxiv.org/abs/2302.04761) | Meta          |
| Visual Programming: Compositional visual reasoning without training                                            | GPT3               | [VisProg](https://github.com/allenai/visprog)                                     | CVPR2023    | [2211.11559](https://arxiv.org/abs/2211.11559) | AI2           |

### Projects

- (2023-05) [Tranformers Agent](https://huggingface.co/docs/transformers/en/transformers_agents), Transformers Agent is an experimental API which is subject to change at any time
- (2023-05)  [闻达](https://github.com/l15y/wenda), 一个LLM调用平台。目标为针对特定环境的高效内容生成，同时考虑个人和中小企业的计算资源局限性，以及知识安全和私密性问题
- (2023-04) [AgentGPT](https://github.com/reworkd/AgentGPT), AgentGPT allows you to configure and deploy Autonomous AI agents
  - [demo](https://agentgpt.reworkd.ai/)
- (2023-04) [Auto-GPT](https://github.com/Torantulino/Auto-GPT), An experimental open-source attempt to make GPT-4 fully autonomous
  - [demo](https://agpt.co/)
- (2023-04) [BabyAGI](https://github.com/yoheinakajima/babyagi), The system uses OpenAI and vector databases such as Chroma or Weaviate to create, prioritize, and execute tasks
  - [blog](https://twitter.com/yoheinakajima/status/1640934493489070080?s=20)

### Applications
- (2023-03) [ChatPaper](https://github.com/kaixindelele/ChatPaper), Use ChatGPT to summarize the arXiv papers. 全流程加速科研，利用chatgpt进行论文总结+润色+审稿+审稿回复
  - [website](https://chatwithpaper.org/)
- (2023-03) [BibiGPT](https://github.com/JimmyLv/BibiGPT), BibiGPT · 1-Click AI Summary for Audio/Video & Chat with Learning Content: Bilibili | YouTube | Tweet丨TikTok丨Local files | Websites丨Podcasts | Meetings | Lectures, etc. 音视频内容 AI 一键总结 & 对话：哔哩哔哩丨YouTube丨推特丨小红书丨抖音丨网页丨播客丨会议丨本地文件等 (原 BiliGPT 省流神器 & 课代表)
  - [website](https://bibigpt.co/)

## Reference

- [ToolLearningPapers](https://github.com/thunlp/ToolLearningPapers)
  > Must-read papers on tool learning with foundation models
  >
- [LLM Tool Use Papers](https://github.com/xlang-ai/llm-tool-use)
  > Paper collection on LLM tool use and code generation covered in the ACL tutorial on complex reasoning
  >
- [Awesome-ALM](https://github.com/pbhu1024/awesome-augmented-language-model)
  > This repo collect research papers about leveraging the capabilities of language models, which can be a good reference for building upper-layer applications
  >
- [LLM-powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/), Overview: panning, memory, tool use
