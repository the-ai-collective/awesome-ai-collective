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
*   [vLLM](https://github.com/vllm-project/vllm) ⭐69.6k - Industry standard for memory-efficient LLM serving via PagedAttention. Python / C++ / CUDA.
*   [Ollama](https://github.com/ollama/ollama) - Get up and running with Llama 2, Mistral, Gemma, and other large language models.
*   [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA's optimized LLM inference with state-of-the-art GPU performance. Python / C++.
*   [Microsoft BitNet](https://github.com/microsoft/BitNet) - Official 1-bit LLM inference framework with extreme efficiency. Python / C++.
*   [MLC-LLM](https://github.com/mlc-ai/mlc-llm) - Universal LLM deployment with ML compilation for cross-platform execution. Python / Rust.

### Orchestration
*   [LangChain](https://github.com/langchain-ai/langchain) - Building applications with LLMs through composability.
*   [LlamaIndex](https://github.com/run-llama/llama_index) - Data framework for LLM applications.
*   [GraphRAG](https://github.com/microsoft/graphrag) - Modular graph-based Retrieval-Augmented Generation (RAG) system by Microsoft.
*   [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol/servers) ⭐78k - Standardized protocol connecting LLMs with tools and context. TypeScript / Python.
*   [Microsoft MCP Servers](https://github.com/microsoft/mcp) - Official Microsoft MCP implementations for AI data access and tool integration. TypeScript / Python.

### Agents
*   [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT) - An experimental open-source attempt to make GPT-4 fully autonomous.
*   [Microsoft AutoGen](https://github.com/microsoft/autogen) ⭐54k - Event-driven multi-agent framework with distributed execution. Python.
*   [Agent-S](https://github.com/simular-ai/Agent-S) - Open framework using computers like a human via Agent-Computer Interface. Python.
*   [OpenClaw](https://openclaw.ai/) - Personal AI assistant that runs locally on user devices (formerly Clawdbot/Moltbot). [GitHub](https://github.com/openclaw/openclaw).
*   [Jules](https://jules.google) - Google's proactive, autonomous coding agent that integrates with your repositories.
*   [Codex CLI](https://developers.openai.com/codex/cli/) - OpenAI's coding agent that runs locally from your terminal.
*   [Claude Cowork](https://claude-cowork.net/) - AI coworker for task automation, workflow execution, and file management.
*   [Fluid](https://www.fluid.sh/) - Secure infrastructure agent that clones production into sandboxes for AI operations.
*   [TradingAgents](https://github.com/TauricResearch/TradingAgents) - Multi-agent financial trading framework (Analysts, Traders, Risk Managers).
*   [browser-use](https://github.com/browser-use/browser-use) - Make websites accessible to AI agents via natural language.
*   [AIHawk](https://github.com/feder-cr/Jobs_Applier_AI_Agent_AIHawk) - Automated job application agent (Auto-Apply).
*   [smolagents](https://github.com/huggingface/smolagents) - Barebones library for agents that think in code (Python snippets actions).
*   [OpenManus](https://github.com/mannaandpoem/OpenManus) - Open-source implementation of the Manus AI agent concept (Dockerized).
*   [Pi Coding Agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent) - Minimal, opinionated coding agent CLI and unified LLM API.

### Development Tools

#### AI Coding Assistants
*   [Cursor](https://cursor.com/) - AI-first IDE with multi-file context awareness and <320ms rapid prototyping (2026).
*   [Replit Agent v3](https://replit.com/) - 10x autonomous browser-based coding with self-healing bug fixes (Sep 2025).
*   [Augment Code](https://augmentcode.com/) - Enterprise semantic analysis for 400K+ files with ISO 42001 compliance (Jan 2026).
*   [JetBrains AI Assistant](https://jetbrains.com/ai) - AST-aware code understanding with semantic refactoring in JetBrains IDEs (2026).

#### AI App Builders
*   [v0 by Vercel](https://v0.dev/) - Generative UI from prompts/designs. Generates React + Tailwind + shadcn/ui. Token-based pricing, Figma import, direct Vercel deployment.
*   [Lovable](https://lovable.dev/) - Full-stack web apps via natural language. Generates Next.js + React + Supabase. One-click auth/database, GitHub export. $25/mo Pro.
*   [Bolt.new](https://bolt.new/) - Browser-based instant full-stack apps by StackBlitz. React/Next.js + Node.js + Vite. In-browser IDE with live preview. $20/mo Pro.
*   [Builder.io AI](https://builder.io/) - Design-to-code from Figma with design system enforcement and Jira/Slack integration.
*   [Claude Artifacts](https://claude.ai/) - Prompt-to-app in Claude chat. HTML/JS/React output. Free tier + Pro $20/mo (2024 feature).

#### Configuration & Tooling
*   [LNAI](https://github.com/KrystianJonca/lnai) - Unified AI configuration management CLI. Define once in `.ai/`, sync to Cursor, VS Code, and more.
*   [CoreX](https://github.com/sm0lvoicc/CoreX) - Multi-purpose Discord bot with modular command categories.
*   [Stitch](https://stitch.withgoogle.com/) - Google's AI-powered UI design tool that transforms prompts into designs and code.
*   [GitHub Spec Kit](https://github.com/github/spec-kit) - Spec-Driven Development (SDD) toolkit for defining and generating code from specifications.
*   [Superpowers](https://github.com/obra/superpowers) - Agentic skills framework & software development methodology for coding agents.
*   [Anthropic Skills](https://github.com/anthropics/skills) - Official reference collection of agent skills (computer use, bash, etc.).
*   [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol/servers) - Official reference servers for safe LLM access to local resources.
*   [Claude Code Local Override](https://boxc.net/blog/2026/claude-code-connecting-to-local-models-when-your-quota-runs-out/) - Guide: Connect Claude Code CLI to local OSS models.
*   [AionUi](https://github.com/iOfficeAI/AionUi) - Cross-platform GUI for command-line AI agents (Gemini CLI, Claude Code).
*   [AgentOps](https://github.com/AgentOps-AI/agentops) - Observability SDK for AI agent monitoring, cost tracking, and replay analytics.

### Enterprise AI
*   [Gemini for Work](https://cloud.google.com/ai/gemini-for-work) - AI-powered features for Google Workspace, including automatic note-taking and summaries.
*   [Gemini Enterprise](https://cloud.google.com/gemini-enterprise) - Enterprise-grade AI platform connecting company data with agents and workflows.

## Computer Vision

### Image Generation
*   [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) - Stable Diffusion web UI.
*   [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - The most powerful and modular stable diffusion GUI.

### Video Generation
*   [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) - Largest open-source video generation model (13B params).

## Infrastructure

### Vector Databases
*   [Pinecone](https://www.pinecone.io/) - The vector database for machine learning.
*   [Weaviate](https://github.com/weaviate/weaviate) - Open-source vector search engine.
*   [Qdrant](https://github.com/qdrant/qdrant) - Vector Similarity Search Engine with extended filtering support.

### Agent Memory
*   [Memori](https://github.com/MemoriLabs/Memori) - SQL-native memory layer for LLMs (Long-term recall, user preferences).
*   [E2B](https://github.com/e2b-dev/E2B) - Secure cloud sandboxes for AI agents (Code execution environment).

## Learning

### Courses
*   [Fast.ai](https://course.fast.ai/) - Practical Deep Learning for Coders.
*   [DeepLearning.AI](https://www.deeplearning.ai/) - AI education for everyone.
*   [LLM Agents MOOC](https://llmagents-learning.org/f24) - UC Berkeley's course on Large Language Model Agents.
*   [Anthropic's Prompt Engineering Interactive Tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial) - Hands-on guide to prompting.
*   [Hugging Face Agents Course](https://github.com/huggingface/agents-course) - Comprehensive, free course on building AI Agents (smolagents, LangGraph).

### Research Papers

#### Foundation & Architecture
*   [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The seminal paper introducing the Transformer architecture.
*   [Chinchilla](https://arxiv.org/abs/2203.15556) - Training Compute-Optimal Large Language Models (Scaling Laws).

#### Efficiency & Optimization
*   [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - Efficient fine-tuning method.
*   [FlashAttention](https://arxiv.org/abs/2205.14135) - Fast and memory-efficient exact attention.
*   [QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models](https://arxiv.org/abs/2310.16795) - MoE quantization achieving sub-1-bit compression.
*   [MxMoE: Mixed-precision Quantization for MoE](https://github.com/Kai-Liu001/Awesome-Model-Quantization) - Accuracy + performance co-design (ICML 2025).

#### Agentic Reasoning
*   [Agentic Reasoning for Large Language Models](https://arxiv.org/abs/2601.12538) - Comprehensive survey on agentic reasoning paradigms (Jan 2026).
*   [Reasoning RAG via System 1 or System 2](https://arxiv.org/abs/2506.10408) - Survey on reasoning agentic RAG for industry (Jun 2025).
*   [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - Architecture and training details of the MoE model.
*   [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) - The paper that popularized CoT.
*   [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - Stable alternative to RLHF.
*   [The 2025 AI Engineering Reading List](https://www.latent.space/p/2025-papers) - Curated list of 50+ essential papers for AI Engineers (Latent Space).
*   [Agentic Reasoning for Large Language Models](https://arxiv.org/abs/2601.12538) - Unified roadmap bridging thought and action (Jan 2026).

## Community

### Discord
*   [Claude Code Community](https://claudecode.community/) - Community for Claude Code developers and enthusiasts.
*   [The AI Collective Discord](https://discord.gg/aicollective) - 150,000+ AI pioneers, founders, researchers, operators, and investors.

### Events
*   [The AI Collective Events](https://lu.ma/genai-collective) - Global AI meetups in 50+ cities.
*   [Claude Community Events](https://luma.com/claudecommunity) - Global events hosted by Claude Community members.

---

## Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.
