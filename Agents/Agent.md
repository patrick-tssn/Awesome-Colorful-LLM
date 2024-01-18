# Agents

*This part records LLMs as Agents: Brain, Perception, Tools*

Table of Contents

- [Reading List](#reading-list)
- [Dataset &amp; Benchmarks](#datasets--benchmarks)
- [Projects](#projects)
- [Applications](#applications)

## Reading List

Survey

- (2023-09) The Rise and Potential of Large Language Model Based Agents: A Survey [paper](https://arxiv.org/abs/2309.07864)
- (2023-08) Automatically Correcting Large Language Models: Surveying the landscape of diverse self-correction strategies [paper](https://arxiv.org/abs/2308.03188)
- (2023-06) LLM Powered Autonomous Agents [blog](https://lilianweng.github.io/posts/2023-06-23-agent/)
- (2023-04) Tool Learning with Foundation Models [paper](https://arxiv.org/abs/2304.08354), [BMTools](https://github.com/OpenBMB/BMTools)
- (2023-02) Augmented Language Models: a Survey [paper](https://arxiv.org/abs/2302.07842)

Reading List

*`Correct` stands work that focus on correction of ALM*

| Paper                                                                                                                      | LLM                | Code                                                                           | Publication | Preprint                                                                                        | Affiliation     |
| -------------------------------------------------------------------------------------------------------------------------- | ------------------ | ------------------------------------------------------------------------------ | ----------- | ----------------------------------------------------------------------------------------------- | --------------- |
| LLM Augmented LLMs: Expanding Capabilities through Composition                                                             | PaLM2              | [CALM (non-official)](https://github.com/lucidrains/CALM-pytorch)                 |             | [2401.02412](https://arxiv.org/abs/2401.02412)                                                     | Google          |
| GPT-4V(ision) is a Generalist Web Agent, if Grounded                                                                       | GPT-4              | [SeeAct](https://github.com/OSU-NLP-Group/SeeAct)                                 |             | [2312.github](https://github.com/OSU-NLP-Group/SeeAct/blob/gh-pages/static/paper/SeeAct_Paper.pdf) | OSU             |
| AppAgent: Multimodal Agents as Smartphone Users                                                                            | GPT4               | [AppAgent](https://github.com/mnotgod96/AppAgent)                                 |             | [2312.13771](https://arxiv.org/abs/2312.13771)                                                     | Tencent         |
| An LLM Compiler for Parallel Function Calling                                                                              | GPT, LLaMA2        | [LLMCompiler](https://github.com/SqueezeAILab/LLMCompiler)                        |             | [2312.04511](https://arxiv.org/abs/2312.04511)                                                     | UCB             |
|  ProAgent: From Robotic Process Automation to Agentic Process Automation                                                                                                                          |            GPT-4        |                   [ProAgent](https://github.com/OpenBMB/ProAgent)                                                             |             |     [2311.10751](https://arxiv.org/abs/2311.10751)                                                                                            |  THU               |
| ControlLLM: Augment Language Models with Tools by Searching on Graphs                                                      | ChatGPT, LLaMA     | [ControlLLM](https://github.com/OpenGVLab/ControlLLM)                             |             | [2310.17796](https://arxiv.org/abs/2310.17796)                                                     | Shanghai AI Lab |
| AgentTuning: Enabling Generalized Agent Abilities For LLMs                                                                 | LLaMA2             | [AgentTuning](https://github.com/THUDM/AgentTuning)                               |             | [2310.12823](https://arxiv.org/abs/2310.12823)                                                     | THU             |
| AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation                                                   |                    | [AutoGen](https://github.com/microsoft/autogen)                                   |             | [2308.08155](https://arxiv.org/abs/2308.08155)                                                     | Microsoft       |
| AssistGPT: A General Multi-modal Assistant that can Plan, Execute, Inspect, and Learn                                      | ChatGPT            | [AssistGPT](https://github.com/COOORN/AssistGPT)                                  |             | [2306.08640](https://arxiv.org/abs/2306.08640)                                                     | NUS             |
| ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases                                        | Alpaca             | [ToolAlpaca](https://github.com/tangqiaoyu/ToolAlpaca)                            |             | [2306.05301](https://arxiv.org/abs/2306.05301)                                                     | CAS             |
| GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction                                                | Vicuna-13B         | [GPT4Tools](https://github.com/StevenGrove/GPT4Tools)                             |             | [2305.18752](https://arxiv.org/abs/2305.18752)                                                     | Tencent         |
| AdaPlanner: Adaptive Planning from Feedback with Language Models                                                          | GPT3/3.5           | [AdaPlanner](https://github.com/haotiansun14/AdaPlanner)                          |             | [2305.16653](https://arxiv.org/abs/2305.16653)                                                     | Gatech          |
| `Correct` CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing                               | GPT3/3.5           | [CRITIC](https://github.com/microsoft/ProphetNet/tree/master/CRITIC)              |             | [2305.11738](https://arxiv.org/abs/2305.11738)                                                     | Microsoft       |
| LLM+P: Empowering Large Language Models with Optimal Planning Proficiency                                                  | GPT4               | [LLM-PDDL](https://github.com/Cranial-XIX/llm-pddl)                               |             | [2304.11477](https://arxiv.org/abs/2304.11477)                                                     | UTEXAS          |
| Can GPT-4 Perform Neural Architecture Search?                                                                              | GPT4               | [GENIUS](https://github.com/mingkai-zheng/GENIUS)                                 |             | [2304.10970](https://arxiv.org/abs/2304.10970)                                                     | Cambridge       |
| Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models                                                | GPT4               | [Chameleon](https://github.com/lupantech/chameleon-llm)                           |             | [2304.09842](https://arxiv.org/abs/2304.09842)                                                     | Microsoft       |
| OpenAGI: When LLM Meets Domain Experts                                                                                     | ChatGPT            | [OpenAGI](https://github.com/agiresearch/OpenAGI)                                 |             | [2304.04370](https://arxiv.org/abs/2304.04370)                                                     | Rutgers Univ.   |
| LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models                               | Opensource LLM     | [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters)                   |             | [2304.01933](https://arxiv.org/abs/2304.01933)                                                     | SMU             |
| HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace                                                   | ChatGPT            | [JARVIS](https://github.com/microsoft/JARVIS)                                     |             | [2303.17580](https://arxiv.org/abs/2303.17580)                                                     | Microsoft       |
| Language Models can Solve Computer Tasks                                                                                   | ChatGPT, GPT3, etc | [RCI Agent](https://github.com/posgnu/rci-agent)                                  |             | [2303.17491](https://arxiv.org/abs/2303.17491)                                                     | CMU             |
| TaskMatrix.AI: Completing Tasks by Connecting Foundation Models with Millions of APIs                                      | ChatGPT            | [TaskMatrix](https://github.com/microsoft/visual-chatgpt/tree/main/TaskMatrix.AI) |             | [2303.16434](https://arxiv.org/abs/2303.16434)                                                     | Microsoft       |
| MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action                                                            | ChatGPT            | [MM-REACT](https://github.com/microsoft/MM-REACT)                                 |             | [2303.11381](https://arxiv.org/abs/2303.11381)                                                     | Microsoft       |
| ART: Automatic multi-step reasoning and tool-use for large language models                                                | GPT3, Codex        | [Language-Programmes](https://github.com/bhargaviparanjape/language-programmes)   |             | [2303.09014](https://arxiv.org/abs/2303.09014)                                                     | Microsoft       |
| Foundation Models for Decision Making: Problems, Methods, and Opportunities                                                | -                  | -                                                                              |             | [2303.04129](https://arxiv.org/abs/2303.04129)                                                     | Google          |
| `Correct` Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback | ChatGPT            | [LLM-Augmenter](https://github.com/pengbaolin/LLM-Augmenter)                      |             | [2302.12813](https://arxiv.org/abs/2302.12813)                                                     | Microsoft       |
| Toolformer: Language Models Can Teach Themselves to Use Tools                                                              | GPT-J, OPT, GPT3   | [Toolformer (Unofficial)](https://github.com/lucidrains/toolformer-pytorch)       |             | [2302.04761](https://arxiv.org/abs/2302.04761)                                                     | Meta            |
| Visual Programming: Compositional visual reasoning without training                                                        | GPT3               | [VisProg](https://github.com/allenai/visprog)                                     | CVPR 2023   | [2211.11559](https://arxiv.org/abs/2211.11559)                                                     | AI2             |
| Decomposed Prompting: A Modular Approach for Solving Complex Tasks                                                         | GPT3               | [DecomP](https://github.com/allenai/DecomP)                                       | ICLR 2023   | [2210.02406](https://arxiv.org/abs/2210.02406)                                                     | AI2             |
| TALM: Tool Augmented Language Models                                                                                       | T5                 |                                                                                |             | [2205.12255](https://arxiv.org/abs/2205.12255)                                                     | Google          |

## Datasets & Benchmarks

Datasets

- (2023.07) [ToolBench](https://github.com/OpenBMB/ToolBench), This project (ToolLLM) aims to construct open-source, large-scale, high-quality instruction tuning SFT data to facilitate the construction of powerful LLMs with general tool-use capability.

Benchmarks

- (2024.01) [ToolEyes](https://github.com/Junjie-Ye/ToolEyes),  The system meticulously examines seven real-world scenarios, analyzing five dimensions crucial to LLMs in tool learning: format alignment, intent comprehension, behavior planning, tool selection, and answer organization.
- (2024.01) [AgentBoard](https://github.com/hkust-nlp/AgentBoard)，An Analytical Evaluation Board of Multi-turn LLM Agents 
- (2023.10) [PCA-EVAL](https://github.com/pkunlp-icler/PCA-EVAL), PCA-EVAL is an innovative benchmark for evaluating multi-domain embodied decision-making, specifically focusing on the performance in perception, cognition, and action
- (2023.08) [AgentBench](https://github.com/THUDM/AgentBench), AgentBench is the first benchmark designed to evaluate LLM-as-Agent across a diverse spectrum of different environments. It encompasses 8 distinct environments to provide a more comprehensive evaluation of the LLMs' ability to operate as autonomous agents in various scenarios

## Projects

- (2023-12) [KwaiAgents](https://github.com/KwaiKEG/KwaiAgents), A generalized information-seeking agent system with Large Language Models (LLMs).
- (2023-11) [CrewAI](https://github.com/joaomdmoura/crewAI), Cutting-edge framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence, CrewAI empowers agents to work together seamlessly, tackling complex tasks.
- (2023-11) [WarAgent](https://github.com/agiresearch/WarAgent), Large Language Model-based Multi-Agent Simulation of World Wars
- (2023-08) [MetaGPT](https://github.com/geekan/MetaGPT), The Multi-Agent Framework: Given one line Requirement, return PRD, Design, Tasks, Repo
- (2023-08) [AI-Town](https://github.com/a16z-infra/ai-town), AI Town is a virtual town where AI characters live, chat and socialize
- (2023-08) [XLang](https://github.com/xlang-ai/xlang), An open-source framework for building and evaluating language model agents via executable language grounding
  - [homepage](https://www.xlang.ai/)
- (2023-06) [Gentopia](https://github.com/Gentopia-AI/Gentopia), Gentopia is a lightweight and extensible framework for LLM-driven Agents and [ALM](https://arxiv.org/abs/2302.07842) research
  - [paper](https://arxiv.org/abs/2308.04030)
- (2023-05) [Tranformers Agent](https://huggingface.co/docs/transformers/en/transformers_agents), Transformers Agent is an experimental API which is subject to change at any time
- (2023-05)  [闻达](https://github.com/l15y/wenda), 一个LLM调用平台。目标为针对特定环境的高效内容生成，同时考虑个人和中小企业的计算资源局限性，以及知识安全和私密性问题
- (2023-04) [AgentGPT](https://github.com/reworkd/AgentGPT), AgentGPT allows you to configure and deploy Autonomous AI agents
  - [demo](https://agentgpt.reworkd.ai/)
- (2023-04) [Auto-GPT](https://github.com/Torantulino/Auto-GPT), An experimental open-source attempt to make GPT-4 fully autonomous
  - [demo](https://agpt.co/)
- (2023-04) [BabyAGI](https://github.com/yoheinakajima/babyagi), The system uses OpenAI and vector databases such as Chroma or Weaviate to create, prioritize, and execute tasks
  - [blog](https://twitter.com/yoheinakajima/status/1640934493489070080?s=20)
- (2022-10) [LangChain](https://github.com/langchain-ai/langchain), LangChain is a framework for developing applications powered by language models. It enables applications that: **(i) Are context-aware**: connect a language model to sources of context (prompt instructions, few shot examples, content to ground its response in, etc.) **(ii) Reason**: rely on a language model to reason (about how to answer based on provided context, what actions to take, etc.)
  - [Doc](https://python.langchain.com/docs/get_started/introduction)

## Applications

- (2023-11) [Awesome-AI-GPTs](https://github.com/EmbraceAGI/Awesome-AI-GPTs), 欢迎来到 EmbraceAGI GPTs 开源目录，本项目收录了 OpenAI GPTs 的相关资源和有趣玩法，让我们一起因 AI 而强大
- (2023-04) [众评AI](https://www.zhongpingtechnology.com/#), 全球AI网站排行榜展示了人工智能领域最顶尖的1800+个网站，排行榜每日更新
- (2023-03) [ChatPaper](https://github.com/kaixindelele/ChatPaper), Use ChatGPT to summarize the arXiv papers. 全流程加速科研，利用chatgpt进行论文总结+润色+审稿+审稿回复
  - [website](https://chatwithpaper.org/)
  - Related: [ChatReviewer](https://github.com/nishiwen1214/ChatReviewer)
- (2023-03) [BibiGPT](https://github.com/JimmyLv/BibiGPT), BibiGPT · 1-Click AI Summary for Audio/Video & Chat with Learning Content: Bilibili | YouTube | Tweet丨TikTok丨Local files | Websites丨Podcasts | Meetings | Lectures, etc. 音视频内容 AI 一键总结 & 对话：哔哩哔哩丨YouTube丨推特丨小红书丨抖音丨网页丨播客丨会议丨本地文件等 (原 BiliGPT 省流神器 & 课代表)
  - [website](https://bibigpt.co/)
