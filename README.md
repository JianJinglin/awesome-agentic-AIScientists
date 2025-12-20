# Multimodal Agentic AI Scientists

> A curated list of papers on **Agentic Multimodal Large Language Models (MLLMs) for Scientific Discovery**

🚀 **Join us in building the AI for Science community!** Know a great paper we missed? [Open an issue](https://github.com/JianJinglin/awesome-agentic-AIScientists/issues) — together, let's accelerate scientific discovery with AI!

This repository accompanies our survey paper: **["Exploring Agentic Multimodal Large Language Models: A Survey for AIScientists"](https://www.techrxiv.org/users/998129/articles/1358530-exploring-agentic-multimodal-large-language-models-a-survey-for-aiscientists)**

### What is an AIScientist?

**AIScientists** are autonomous agents powered by multimodal large language models (MLLMs) that can understand papers, generate hypotheses, plan and conduct experiments, analyze results, and draft manuscripts across the entire scientific research lifecycle ([Lu et al., 2024](http://arxiv.org/abs/2408.06292); [Boiko et al., 2023](https://www.nature.com/articles/s41586-023-06792-0); [Gottweis et al., 2025](http://arxiv.org/abs/2502.18864)). But how do we build one? This survey summarizes a complete pipeline for developing multimodal agentic AIScientists, with representative studies spanning 10+ scientific domains.

### Comparison with Related Surveys

| Paper | Taxonomy | Ag. | DM. | Method | HCI | Ben. | #Dom. |
|:------|:--------:|:---:|:---:|:------:|:---:|:---:|:-----:|
| [Zhang et al. (2024)](https://arxiv.org/abs/2401.00426) | Domain | ✗ | Seq.+ | Train. only | ✗ | ✓ | 6 |
| [Gridach et al. (2025)](https://arxiv.org/abs/2503.01153) | Sci. Workflow | ✓ | ✗ | Infer. only | ✓ | ✗ | 4 |
| [Luo et al. (2025)](https://arxiv.org/abs/2501.04306) | Sci. Workflow | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| [Zhang et al. (2025)](https://arxiv.org/abs/2502.09247) | Sci. Workflow | ✗ | Seq.+ | ✗ | ✗ | ✗ | ✗ |
| [Ren et al. (2025)](https://arxiv.org/abs/2503.03924) | Agent Composition | ✓ | ✗ | Train. & Infer. | ✗ | ✓ | 6+ |
| [Wei et al. (2025)](https://arxiv.org/abs/2503.10794) | Auto. & Domain | ✓ | ✗ | Infer. only | ✗ | ✓ | 4 |
| [Hu et al. (2025)](https://arxiv.org/abs/2503.16894) | Data & Domain | ✓ | ✓ | ✗ | ✗ | ✓ | 6+ |
| **Ours** | **ML Pipeline** | ✓ | ✓ | **Train. & Infer.** | ✓ | ✓ | **10** |

<sub>**Ag.** = Agentic AI; **DM.** = Data Modality; **HCI** = Human-Computer Interaction; **Ben.** = Benchmark; **#Dom.** = Number of domains; **Seq.+** = Sequence and more modalities; **Train.** = Agent Training; **Infer.** = Agent Inference</sub>

### Ours: An End-to-End Developer Pipeline

<img src="assets/figure_overview.png" alt="Overview of the agentic MLLM framework for scientific discovery" width="100%"/>

*Overview of our framework: Starting from diverse **Input & Output** modalities, through **Agent Training** and **Inference** methods, to **Evaluation** benchmarks—with **Human-AI Collaboration** integrated at every stage.*

---

## Table of Contents

- [⚙️ Methods for Scientific MLLM Agents](#️-methods-for-scientific-mllm-agents)
  - [🏋️ Agent Training](#️-agent-training)
  - [🚀 Agent Inference](#-agent-inference)
  - [🤝 Multi-Agent Systems](#-multi-agent-systems)
- [📈 Benchmarks & Evaluation](#-benchmarks--evaluation)
- [🧑‍🔬 Human-AI Collaboration](#-human-ai-collaboration)

---

## ⚙️ Methods for Scientific MLLM Agents

### 🏋️ Agent Training

#### Supervised Fine-Tuning (SFT)

- In-Context Learning with Long-Context Models: An In-Depth Exploration (2025) - *Bertsch et al.*
- RLSF: Fine-tuning LLMs via Symbolic Feedback (2024) - *Jha et al.*
- Efficient Fine-Tuning of Single-Cell Foundation Models Enables Zero-Shot Molecular Perturbation Prediction (2025) - *Maleki et al.*
- Training language models to follow instructions with human feedback (2022) - *Ouyang et al.*
- InstructProtein: Aligning Human and Protein Language via Knowledge Instruction (2023) - *Wang et al.*
- Mavis: Mathematical visual instruction tuning with an automatic data engine (2024) - *Zhang et al.*

#### Reinforcement Learning (RL)

Including RLHF, DPO, and reward-based training methods.

- [AlphaFlow: autonomous discovery and optimization of multi-step chemistry using a self-driven fluidic lab guided by reinforcement learning](https://doi.org/10.1038/s41467-023-37139-y) (2023) - *Volk et al.*
- [DRLinFluids: An open-source Python platform of coupling deep reinforcement learning and OpenFOAM](http://dx.doi.org/10.1063/5.0103113) (2022) - *Wang et al.*
- Improving Targeted Molecule Generation through Language Model Fine-Tuning Via Reinforcement Learning (2025) - *Salma J. Ahmed, Emad A. Mohammed*
- Molecular design in synthetically accessible chemical space via deep reinforcement learning (2020) - *Horwood, Julien, Noutahi, Emmanuel*
- Leanabell-Prover-V2: Verifier-integrated Reasoning for Formal Theorem Proving via Reinforcement Learning (2025) - *Ji et al.*
- The history and risks of reinforcement learning and human feedback (2023) - *Lambert et al.*
- Direct Preference Optimization: Your Language Model is Secretly a Reward Model (2024) - *Rafailov et al.*
- DrugGen: Advancing Drug Discovery with Large Language Models and Reinforcement Learning Feedback (2024) - *Sheikholeslami et al.*
- Reflexion: Language Agents with Verbal Reinforcement Learning (2023) - *Shinn et al.*
- Aligning Large Multimodal Models with Factually Augmented RLHF (2023) - *Sun et al.*
- [Generation of Rational Drug-like Molecular Structures Through a Multiple-Objective Reinforcement Learning Framework](https://www.proquest.com/scholarly-journals/generation-rational-drug-like-molecular/docview/3153791278/se-2) (2025) - *Zhang et al.*
- Bindgpt: A scalable framework for 3d molecular design via language modeling and reinforcement learning (2025) - *Zholus et al.*

#### Contrastive & Adversarial Learning

- [Semi-supervised learning-based virtual adversarial training on graph for molecular property prediction](https://www.sciencedirect.com/science/article/pii/S1110016824015813) (2025) - *Lu et al.*
- [Triplet Contrastive Learning Framework With Adversarial Hard-Negative Sample Generation for Multimodal Remote Sensing Images](https://doi.org/10.1109/TGRS.2024.3354304) (2024) - *Chen et al.*
- Generating mutants of monotone affinity towards stronger protein complexes through adversarial learning (2024) - *Lan et al.*
- Recent advances in generative adversarial networks for gene expression data: a comprehensive review (2023) - *Lee, Minhyeok*
- Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning (2021) - *Li et al.*
- Drug repositioning based on residual attention network and free multiscale adversarial training (2024) - *Li et al.*
- Dinov2: Learning robust visual features without supervision (2023) - *Oquab, Maxime, Darcet, Timothe*
- Robust image representations with counterfactual contrastive learning (2025) - *Roschewitz, Me*
- Improved Techniques for Training GANs (2016) - *Salimans et al.*
- SupReMix: Supervised Contrastive Learning for Medical Imaging Regression with Mixup (2025) - *Wu et al.*

### 🚀 Agent Inference

#### Retrieval-Augmented Generation (RAG)

- ChemCrow: Augmenting large-language models with chemistry tools (2023) - *Bran et al.*
- [ESCARGOT: an AI agent leveraging large language models, dynamic graph of thoughts, and biomedical knowledge graphs for enhanced reasoning](https://academic.oup.com/bioinformatics/article/41/2/btaf031/7972741) (2025) - *Matsumoto et al.*
- RAISE: Enhancing Scientific Reasoning in LLMs via Step-by-Step Retrieval (2025) - *Oh et al.*
- Hypercube-Based Retrieval-Augmented Generation for Scientific Question-Answering (2025) - *Shi et al.*
- MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models (2025) - *Xia et al.*
- [Improving retrieval-augmented generation in medicine with iterative follow-up questions](https://www.worldscientific.com/doi/pdf/10.1142/9789819807024_0015) (2024) - *Xiong et al.*

#### In-Context Learning (ICL)

- Language Models are Few-Shot Learners (2020) - *Brown et al.*
- A Survey on In-context Learning (2024) - *Dong et al.*
- Discovering New Theorems via LLMs with In-Context Proof Learning in Lean (2025) - *Kasaura et al.*
- What Makes Good In-Context Examples for GPT-$3 $? (2021) - *Liu et al.*
- What Makes In-context Learning Effective for Mathematical Reasoning: A Theoretical Analysis (2024) - *Liu et al.*
- Prottex: Structure-in-context reasoning and editing of proteins with large language models (2025) - *Ma et al.*
- Chain-of-thought prompting elicits reasoning in large language models (2022) - *Wei et al.*

#### Planning & Tool Use

- [Automating AI Discovery for Biomedicine Through Knowledge Graphs And LLM Agents](https://www.biorxiv.org/content/10.1101/2025.05.08.652829v2) (2025) - *Aamer et al.*
- [Agent-based learning of materials datasets from the scientific literature](https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00252k) (2024) - *Ansari, Mehrad, Moosavi, Seyed Mohamad*
- AutomaTikZ: Text-Guided Synthesis of Scientific Vector Graphics with TikZ (2024) - *Belouadi et al.*
- Autonomous chemical research with large language models (2023) - *Boiko et al.*
- Comprehensive evaluation of molecule property prediction with ChatGPT (2024) - *Cai et al.*
- ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs (2024) - *Chen et al.*
- Llmphy: Complex physical reasoning using large language models and world models (2024) - *Cherian et al.*
- SynFlowNet: Design of Diverse and Novel Molecules with Synthesis Constraints (2025) - *Cretu et al.*
- AI achieves silver-medal standard solving International Mathematical Olympiad problems (2024) - *AlphaProof, AlphaGeometry teams*
- [MedRAX: Medical Reasoning Agent for Chest X-ray](https://arxiv.org/pdf/2502.02673) (2025) - *Fallahpour et al.*
- Optimizing large language models in digestive disease: strategies and challenges to improve clinical outcomes (2024) - *Giuffre*
- [2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)](https://arxiv.org/abs/2403.06294) (2024) - *Hong et al.*
- Evaluation of large language models for discovery of gene set function (2025) - *Hu et al.*
- [Crispr-GPT: An LLM agent for automated design of gene-editing experiments](http://arxiv.org/abs/2404.18021) (2024) - *Huang et al.*
- [MT-Mol: Multi Agent System with Tool-based Reasoning for Molecular Optimization](http://arxiv.org/abs/2505.20820) (2025) - *Kim et al.*
- [2024 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)](https://doi.org/10.1109/EMBC53108.2024.10781985) (2024) - *Kumar et al.*
- A Survey of Post-Training Scaling in Large Language Models (2025) - *Lai et al.*
- [BioImage. IO Chatbot: a community-driven AI assistant for integrative computational bioimaging](https://www.nature.com/articles/s41592-024-02370-y) (2024) - *Lei, Wanlu, Fuster-Barcelo*
- STMA: A spatio-temporal memory agent for long-horizon embodied task planning (2025) - *Lei et al.*

### 🤝 Multi-Agent Systems

- [Enhancing diagnostic capability with multi-agents conversational large language models](https://www.nature.com/articles/s41746-025-01550-0) (2025) - *Chen et al.*
- [ProtAgents: protein discovery via large language model multi-agent collaborations combining physics and machine learning](https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00013g) (2024) - *Ghafarollahi, Alireza, Buehler, Markus J*
- Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate (2024) - *Liang et al.*
- [Drugagent: Automating ai-aided drug discovery programming through llm multi-agent collaboration](http://arxiv.org/abs/2411.15692) (2024) - *Liu et al.*
- Multi-Agent Collaboration Mechanisms: A Survey of LLMs (2025) - *Tran et al.*
- [ColaCare: Enhancing Electronic Health Record Modeling through Large Language Model-Driven Multi-Agent Collaboration](https://doi.org/10.1145/3696410.3714877) (2025) - *Wang et al.*

---

## 📈 Benchmarks & Evaluation

- Meet Carl: The first AI system to produce academically peer-reviewed research (2025) - *Autoscience Institute*
- [Accelerating Research with Automated Literature Reviews: A Rag-Based Framework](https://api.semanticscholar.org/CorpusID:277095571) (2025) - *Abhinav Balasubramanian*
- [Agent-Based Uncertainty Awareness Improves Automated Radiology Report Labeling with an Open-Source Large Language Model](https://api.semanticscholar.org/CorpusID:276106818) (2025) - *Ben-Atya et al.*
- [Human-artificial interaction in the age of agentic AI: a system-theoretical approach](http://dx.doi.org/10.3389/fhumd.2025.1579166) (2025) - *Borghoff et al.*
- [Uni-SMART: Universal Science Multimodal Analysis and Research Transformer](https://api.semanticscholar.org/CorpusID:268509923) (2024) - *Cai et al.*
- [Detecting and Evaluating Medical Hallucinations in Large Vision Language Models](https://api.semanticscholar.org/CorpusID:270521409) (2024) - *Chen et al.*
- [Have AI-Generated Texts from LLM Infiltrated the Realm of Scientific Writing? A Large-Scale Analysis of Preprint Platforms](https://api.semanticscholar.org/CorpusID:268724393) (2024) - *Cheng et al.*
- [AI in Radiology: Navigating Medical Responsibility](https://api.semanticscholar.org/CorpusID:271168475) (2024) - *Contaldo et al.*
- [Recover: A Neuro-Symbolic Framework for Failure Detection and Recovery](https://api.semanticscholar.org/CorpusID:268819965) (2024) - *Cristina Cornelio, Mohammed Diab*
- [AutoML: A survey of the state-of-the-art](http://dx.doi.org/10.1016/j.knosys.2020.106622) (2021) - *He et al.*
- [Automated Hypothesis Validation with Agentic Sequential Falsifications](https://api.semanticscholar.org/CorpusID:276394614) (2025) - *Huang et al.*
- [Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis](https://api.semanticscholar.org/CorpusID:276482936) (2025) - *Kargupta et al.*
- [Deceptive Semantic Shortcuts on Reasoning Chains: How Far Can Models Go without Hallucination?](https://api.semanticscholar.org/CorpusID:265220880) (2023) - *Li et al.*
- [How to Detect and Defeat Molecular Mirage: A Metric-Driven Benchmark for Hallucination in LLM-based Molecular Comprehension](https://api.semanticscholar.org/CorpusID:277857308) (2025) - *Li et al.*
- [HypoBench: Towards Systematic and Principled Benchmarking for Hypothesis Generation](https://api.semanticscholar.org/CorpusID:277824142) (2025) - *Liu et al.*
- [ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](https://api.semanticscholar.org/CorpusID:277349293) (2025) - *Liu et al.*
- [Foundational Statistical Principles in Medical Research: Sensitivity, Specificity, Positive Predictive Value, and Negative Predictive Value](https://api.semanticscholar.org/CorpusID:235229140) (2021) - *Monaghan et al.*
- [Generation of Scientific Literature Surveys Based on Large Language Models (LLM) and Multi-Agent Systems (MAS)](https://api.semanticscholar.org/CorpusID:274130228) (2024) - *Qi et al.*
- [Improving Scientific Hypothesis Generation with Knowledge Grounded Large Language Models](https://api.semanticscholar.org/CorpusID:273821167) (2024) - *Xiong et al.*
- [Confidence Calibration and Rationalization for LLMs via Multi-Agent Deliberation](https://api.semanticscholar.org/CorpusID:269149545) (2024) - *Yang et al.*
- [LLaMP: Large Language Model Made Powerful for High-fidelity Materials Knowledge Retrieval and Distillation](https://api.semanticscholar.org/CorpusID:267320899) (2024) - *Yuan et al.*
- Maps: A multi-agent framework based on big seven personality and socratic guidance for multimodal scientific problem solving (2025) - *Zhang et al.*
- LitLLMs, LLMs for Literature Review: Are we there yet? (2025) - *Agarwal et al.*
- A Survey on Hypothesis Generation for Scientific Discovery in the Era of Large Language Models (2025) - *Alkan et al.*
- Domain Specific Benchmarks for Evaluating Multimodal Large Language Models (2025) - *Anjuma et al.*
- The challenge of uncertainty quantification of large language models in medicine (2025) - *Atf et al.*
- ChemCrow: Augmenting large-language models with chemistry tools (2023) - *Bran et al.*
- MicroVQA: A Multimodal Reasoning Benchmark for Microscopy-Based Scientific Research (2025) - *Burgess et al.*
- [A benchmark study of simulation methods for single-cell RNA sequencing data](https://www.nature.com/articles/s41467-021-27130-w) (2021) - *Cao et al.*
- Efficient and robust automated machine learning (2015) - *Feurer et al.*
- [ProtAgents: protein discovery via large language model multi-agent collaborations combining physics and machine learning](https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00013g) (2024) - *Ghafarollahi, Alireza, Buehler, Markus J*
- [Robin: A multi-agent system for automating scientific discovery](http://arxiv.org/abs/2505.13400) (2025) - *Ghareeb et al.*
- Accelerating scientific breakthroughs with an AI co-scientist (2025) - *Gottweis, Juraj, Natarajan, Vivek*
- [Towards an AI co-scientist](http://arxiv.org/abs/2502.18864) (2025) - *Gottweis et al.*
- Ds-agent: Automated data science by empowering large language models with case-based reasoning (2024) - *Guo et al.*
- AI Scientist LLM goes rogue: Creators warn of (2024) - *Hamill, Jasper*
- Evolving Code with A Large Language Model (2024) - *Hemberg et al.*
- Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions (2025) - *Hou et al.*
- [Crispr-GPT: An LLM agent for automated design of gene-editing experiments](http://arxiv.org/abs/2404.18021) (2024) - *Huang et al.*
- Discoveryworld: A virtual environment for developing and evaluating automated scientific discovery agents (2024) - *Jansen, Peter, Co*

---

## 🧑‍🔬 Human-AI Collaboration

- Codescientist: End-to-end semi-automated scientific discovery with code-based experimentation (2025) - *Jansen et al.*
- Benchmarking large language models on multiple tasks in bioinformatics nlp with prompting (2025) - *Jiang et al.*
- PubMedQA: A Dataset for Biomedical Research Question Answering (2019) - *Jin et al.*
- PGxQA: A resource for evaluating LLM performance for pharmacogenomic QA tasks (2024) - *Keat et al.*
- Medcalc-bench: Evaluating large language models for medical calculations (2024) - *Khandekar et al.*
- Evaluating Human-Language Model Interaction (2024) - *Lee et al.*
- Scigraphqa: A large-scale synthetic multi-turn question-answering dataset for scientific graphs (2023) - *Li, Shengzhi, Tajbakhsh, Nima*
- Automated statistical model discovery with language models (2024) - *Li et al.*
- Rouge: A package for automatic evaluation of summaries (2004) - *Lin, Chin-Yew*
- [Drugagent: Automating ai-aided drug discovery programming through llm multi-agent collaboration](http://arxiv.org/abs/2411.15692) (2024) - *Liu et al.*
- Toward a Team of AI-made Scientists for Scientific Discovery from Gene Expression Data (2024) - *Liu et al.*
- BioProBench: Comprehensive Dataset and Benchmark in Biological Protocol Understanding and Reasoning (2025) - *Liu et al.*
- A vision for auto research with llm agents (2025) - *Liu et al.*
- [Perspectives for multimessenger astronomy with the next generation of gravitational-wave detectors and high-energy satellites](http://dx.doi.org/10.1051/0004-6361/202243705) (2022) - *Ronchini et al.*
- [The ai scientist: Towards fully automated open-ended scientific discovery](http://arxiv.org/abs/2408.06292) (2024) - *Lu et al.*
- MoleculeQA: A dataset to evaluate factual accuracy in molecular comprehension (2024) - *Lu et al.*
- Benchmarking AI scientists in omics data-driven biological research (2025) - *Luo et al.*
- From intention to implementation: automating biomedical research via LLMs (2025) - *Luo et al.*
- [Augmenting large language models with chemistry tools](https://www.nature.com/articles/s42256-024-00832-8) (2024) - *M. Bran et al.*
- [Directory of Useful Decoys, Enhanced (DUD-E): Better Ligands and Decoys for Better Benchmarking](https://doi.org/10.1021/jm300687e) (2012) - *Mysinger et al.*

---

## Citation

If you find this resource helpful, please cite our survey:

**Cite as:** Jinglin Jian, Yi R. Fung, Denghui Zhang, et al. Exploring Agentic Multimodal Large Language Models: A Survey for AIScientists. TechRxiv. December 01, 2025.
DOI: [10.36227/techrxiv.176344216.60619335/v2](https://doi.org/10.36227/techrxiv.176344216.60619335/v2)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=JianJinglin/awesome-agentic-AIScientists&type=Date)](https://star-history.com/#JianJinglin/awesome-agentic-AIScientists&Date)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
