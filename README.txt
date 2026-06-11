# 🔬 AI Research Pipeline — LangChain + Groq

An automated multi-agent research pipeline built with **LangChain** and **Groq's LLaMA 3.3 70B** model. Given a research topic, five specialized AI agents work sequentially to produce a complete, publication-ready research document exported as a formatted **PDF**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output](#output)
- [Project Structure](#project-structure)
- [Customization](#customization)

---

## Overview

This project replicates a CrewAI-style multi-agent workflow using pure LangChain primitives. Each agent is implemented as a `ChatPromptTemplate | ChatGroq | StrOutputParser` chain. Agents run **sequentially**, with each step passing its output as context to the next.

| Agent | Role |
|---|---|
| 📚 Literature Review Agent | Finds and summarizes 5 relevant peer-reviewed papers |
| 📊 Data Analyst Agent | Performs statistical analysis and generates insights |
| ✍️ Writer Agent | Drafts a full research paper from the above outputs |
| 🗂️ Citation Agent | Formats all references in the specified citation style |
| 🔍 Peer Review Agent | Reviews the draft for gaps, logic, and rigor |

---

## Pipeline Architecture

```
research_topic
      │
      ▼
┌─────────────────────┐
│  Literature Review  │──────────────────────────────┐
└─────────────────────┘                              │
      │                                              │
      ▼                                              ▼
┌─────────────────────┐               ┌─────────────────────┐
│   Data Analysis     │──────────────▶│    Writer Agent     │
└─────────────────────┘               └──────────┬──────────┘
                                                 │
                                                 ▼
                                      ┌─────────────────────┐
                                      │   Citation Agent    │
                                      └──────────┬──────────┘
                                                 │
                                                 ▼
                                      ┌─────────────────────┐
                                      │  Peer Review Agent  │
                                      └──────────┬──────────┘
                                                 │
                                                 ▼
                                          research_output.pdf
```

---

## Prerequisites

- Python 3.9 or higher
- A [Groq API key](https://console.groq.com) (free tier available)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ai-research-pipeline.git
cd ai-research-pipeline

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Open `research_pipeline_langchain.py` and replace the placeholder API key with your own:

```python
GROQ_API_KEY = "your_groq_api_key_here"
```

> **Tip:** For better security, use an environment variable instead:
> ```python
> import os
> GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
> ```
> Then run: `export GROQ_API_KEY=your_key_here` before executing the script.

---

## Usage

**Run with default settings** (topic: *Machine Learning in Healthcare*, style: *APA*):

```bash
python research_pipeline_langchain.py
```

**Customise the topic and citation style** by editing the `__main__` block at the bottom of the file:

```python
results = run_research_pipeline(
    research_topic="Climate Change and Renewable Energy",
    citation_style="MLA",
)

save_results_to_pdf(
    results=results,
    research_topic="Climate Change and Renewable Energy",
    citation_style="MLA",
    output_path="research_output.pdf",
)
```

You can also import and call the pipeline programmatically:

```python
from research_pipeline_langchain import run_research_pipeline, save_results_to_pdf

results = run_research_pipeline(
    research_topic="Quantum Computing in Cryptography",
    citation_style="IEEE",
)

save_results_to_pdf(results, "Quantum Computing in Cryptography", "IEEE", "output.pdf")
```

---

## Output

After running, a `research_output.pdf` is generated in the working directory containing:

1. **Cover page** — topic, citation style, timestamp, and model info
2. **Literature Review** — summaries of 5 peer-reviewed papers with synthesis
3. **Data Analysis** — statistical insights and visualisation descriptions
4. **Draft Paper** — full paper with Introduction, Methodology, Results, and Discussion
5. **References** — formatted bibliography in the chosen citation style
6. **Peer Review Report** — constructive feedback and identified gaps

---

## Project Structure

```
ai-research-pipeline/
├── research_pipeline_langchain.py   # Main pipeline script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Customization

**Change the LLM model** — swap to any model available on Groq:
```python
llm = ChatGroq(model="llama-3.1-8b-instant", ...)   # faster / cheaper
llm = ChatGroq(model="mixtral-8x7b-32768", ...)     # alternative model
```

**Change the number of papers** — edit the literature review task template:
```
2. Select 10 high-quality, relevant papers published within the last 10 years.
```

**Change the output filename** — pass a custom path to `save_results_to_pdf`:
```python
save_results_to_pdf(..., output_path="my_paper.pdf")
```
