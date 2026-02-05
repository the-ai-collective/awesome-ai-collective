# Awesome AI Collective [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A community-curated collection of awesome AI tools, frameworks, learning resources, and research papers for the **AI Collective**.

> **Mission:** To organize the world's AI knowledge for builders, researchers, and hackers.

## Contents

- [Generative AI & LLMs](#generative-ai--llms)
  - [Foundation Models](#foundation-models)
  - [Inference Engines](#inference-engines)
  - [Orchestration](#orchestration)
  - [Agents](#agents)
  - [Development Tools](#development-tools)
  - [Enterprise AI](#enterprise-ai)
- [Computer Vision](#computer-vision)
  - [Image Generation](#image-generation)
  - [Video Generation](#video-generation)
- [Audio](#audio)
  - [Text-to-Speech](#text-to-speech)
  - [Speech-to-Text](#speech-to-text)
- [Infrastructure](#infrastructure)
  - [Vector Databases](#vector-databases)
  - [Evaluation](#evaluation)
  - [Deployment](#deployment)
- [Learning](#learning)
  - [Courses](#courses)
  - [Research Papers](#research-papers)
- [Community](#community)
  - [Discord](#discord)
  - [Events](#events)

---

## Generative AI & LLMs

### Foundation Models
*   [Llama 3](https://ai.meta.com/llama/) - Meta's open weights model.
*   [Mistral](https://mistral.ai/) - High performance open models.

### Inference Engines
*   [llama.cpp](https://github.com/ggerganov/llama.cpp) - Inference of LLaMA models in pure C/C++.
*   [vLLM](https://github.com/vllm-project/vllm) - A high-throughput and memory-efficient inference and serving engine for LLMs.
*   [Ollama](https://github.com/ollama/ollama) - Get up and running with Llama 2, Mistral, Gemma, and other large language models.

### Orchestration
*   [LangChain](https://github.com/langchain-ai/langchain) - Building applications with LLMs through composability.
*   [LlamaIndex](https://github.com/run-llama/llama_index) - Data framework for LLM applications.
*   [GraphRAG](https://github.com/microsoft/graphrag) - Modular graph-based Retrieval-Augmented Generation (RAG) system by Microsoft.

### Agents
*   [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT) - An experimental open-source attempt to make GPT-4 fully autonomous.
*   [OpenClaw](https://openclaw.ai/) - Personal AI assistant that runs locally on user devices (formerly Clawdbot/Moltbot). [GitHub](https://github.com/openclaw/openclaw).
*   [Jules](https://jules.google) - Google's proactive, autonomous coding agent that integrates with your repositories.
*   [Codex CLI](https://developers.openai.com/codex/cli/) - OpenAI's coding agent that runs locally from your terminal.
*   [Claude Cowork](https://claude-cowork.net/) - AI coworker for task automation, workflow execution, and file management.
*   [Fluid](https://www.fluid.sh/) - Secure infrastructure agent that clones production into sandboxes for AI operations.
*   [TradingAgents](https://github.com/TauricResearch/TradingAgents) - Multi-agent financial trading framework (Analysts, Traders, Risk Managers).
*   [browser-use](https://github.com/browser-use/browser-use) - Make websites accessible to AI agents via natural language.
*   [AIHawk](https://github.com/feder-cr/Jobs_Applier_AI_Agent_AIHawk) - Automated job application agent (Auto-Apply).

### Development Tools
*   [LNAI](https://github.com/KrystianJonca/lnai) - Unified AI configuration management CLI. Define once in `.ai/`, sync to Cursor, VS Code, and more.
*   [CoreX](https://github.com/sm0lvoicc/CoreX) - Multi-purpose Discord bot with modular command categories.
*   [Stitch](https://stitch.withgoogle.com/) - Google's AI-powered UI design tool that transforms prompts into designs and code.
*   [GitHub Spec Kit](https://github.com/github/spec-kit) - Spec-Driven Development (SDD) toolkit for defining and generating code from specifications.
*   [Superpowers](https://github.com/obra/superpowers) - Agentic skills framework & software development methodology for coding agents.
*   [Anthropic Skills](https://github.com/anthropics/skills) - Official reference collection of agent skills (computer use, bash, etc.).
*   [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol/servers) - Official reference servers for safe LLM access to local resources.
*   [Claude Code Local Override](https://boxc.net/blog/2026/claude-code-connecting-to-local-models-when-your-quota-runs-out/) - Guide: Connect Claude Code CLI to local OSS models.

### Enterprise AI
*   [Gemini for Work](https://cloud.google.com/ai/gemini-for-work) - AI-powered features for Google Workspace, including automatic note-taking and summaries.
*   [Gemini Enterprise](https://cloud.google.com/gemini-enterprise) - Enterprise-grade AI platform connecting company data with agents and workflows.

## Computer Vision

### Image Generation
*   [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) - Stable Diffusion web UI.
*   [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - The most powerful and modular stable diffusion GUI.

## Infrastructure

### Vector Databases
*   [Pinecone](https://www.pinecone.io/) - The vector database for machine learning.
*   [Weaviate](https://github.com/weaviate/weaviate) - Open-source vector search engine.
*   [Qdrant](https://github.com/qdrant/qdrant) - Vector Similarity Search Engine with extended filtering support.

## Learning

### Courses
*   [Fast.ai](https://course.fast.ai/) - Practical Deep Learning for Coders.
*   [DeepLearning.AI](https://www.deeplearning.ai/) - AI education for everyone.
*   [LLM Agents MOOC](https://llmagents-learning.org/f24) - UC Berkeley's course on Large Language Model Agents.
*   [Anthropic's Prompt Engineering Interactive Tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial) - Hands-on guide to prompting.
*   [Hugging Face Agents Course](https://github.com/huggingface/agents-course) - Comprehensive, free course on building AI Agents (smolagents, LangGraph).

### Research Papers
*   [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The seminal paper introducing the Transformer architecture.
*   [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - Efficient fine-tuning method.
*   [FlashAttention](https://arxiv.org/abs/2205.14135) - Fast and memory-efficient exact attention.
*   [Chinchilla](https://arxiv.org/abs/2203.15556) - Training Compute-Optimal Large Language Models (Scaling Laws).
*   [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - Architecture and training details of the MoE model.
*   [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) - The paper that popularized CoT.
*   [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - Stable alternative to RLHF.
*   [The 2025 AI Engineering Reading List](https://www.latent.space/p/2025-papers) - Curated list of 50+ essential papers for AI Engineers (Latent Space).
*   [Agentic Reasoning for Large Language Models](https://arxiv.org/abs/2601.12538) - Unified roadmap bridging thought and action (Jan 2026).

## Community

### Discord
*   [Claude Code Community](https://claudecode.community/) - Community for Claude Code developers and enthusiasts.

### Events
*   [Claude Community Events](https://luma.com/claudecommunity) - Global events hosted by Claude Community members.

---

## Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.
