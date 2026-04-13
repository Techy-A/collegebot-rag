# CollegeBot  --  End-to-End Production RAG Chatbot
# ===================================================

> "The value of an idea lies in the using of it."  -- Thomas Edison

CollegeBot is a production-grade Retrieval-Augmented Generation (RAG)
chatbot for college information retrieval.  It is built entirely on free
resources: Google Colab Free Tier, Hugging Face Hub, Groq Cloud, and
Streamlit Community Cloud.

---

## Project Structure

```
collegebot-rag/
    app.py                      Streamlit application entry point
    llm_factory.py              Multi-LLM factory (Groq, Phi-3, Mistral)
    ingest.py                   Data ingestion and FAISS vector store builder
    generate_dataset.py         Synthetic Q-A dataset generator for fine-tuning
    requirements.txt            Python dependencies for Streamlit Cloud
    .env.example                Environment variable template
    .streamlit/
        config.toml             Streamlit theme and server config
        secrets.toml            Secret keys template (do not commit)
    data/                       Place your college PDFs here
    faiss_store/                Persistent FAISS index (auto-generated)
    dataset/                    Generated fine-tuning dataset (JSONL)
    evaluation/
        quick_score.py          Heuristic inline scoring (sub-10 ms)
        ragas_eval.py           Full RAGAS evaluation script
        __init__.py
    notebooks/
        CollegeBot_Colab.py     Colab notebook cells as a Python file
```

---

## Phase 0: Pre-requisites (15 minutes)

Before writing a single line of code, gather the following free accounts
and API keys.  Every one of these is genuinely free with no credit card.

| Service              | What it provides               | URL                                        |
|----------------------|--------------------------------|--------------------------------------------|
| Google Account       | Access to Google Colab         | accounts.google.com                        |
| Hugging Face account | Model hosting, free Inference  | huggingface.co/join                        |
| Groq account         | Free LLM API (~500 tok/s)      | console.groq.com                           |
| GitHub account       | Source control, Streamlit CI   | github.com/join                            |
| Streamlit account    | Free cloud deployment          | share.streamlit.io                         |

After registering:

1. Go to https://huggingface.co/settings/tokens
   Create a NEW TOKEN with "Write" permission.  Save it as HF_TOKEN.

2. Go to https://console.groq.com -> API Keys -> Create API Key.
   Save it as GROQ_API_KEY.

3. Create a new public GitHub repository named `collegebot-rag` at https://github.com/Techy-A/collegebot-rag

---

## Phase 1: Data Ingestion and Vector Store

> "Garbage in, garbage out."  -- George Fuechsel (1963)
> This is the most important phase.  Your retrieval quality depends entirely
> on the quality and coverage of your source documents.

### 1.1 Prepare documents

Collect all college information documents you can find:
- Student handbook (PDF)
- Admission brochure (PDF)
- Fee structure circular (PDF)
- Academic calendar (PDF)
- Hostel rules (PDF or DOCX)
- Course catalogue (PDF)
- Examination regulations (PDF)

Place them all in the `data/` folder.

If you do not have real documents yet, run:

    python ingest.py --sample

This creates a synthetic `data/sample_college_handbook.txt` with all the
major topics covered, so you can test the full pipeline immediately.

### 1.2 Build the vector store

    python ingest.py

Expected output:

    CollegeBot -- Data Ingestion Pipeline
    ==================================================
    Scanning ./data -- found 7 files total.
      Loaded  admission_handbook.pdf  (42 pages)
      Loaded  fee_structure_2024.pdf  (8 pages)
      ...
    Total documents loaded before splitting: 50
    Chunks after splitting: 312
    chunk_size=800, overlap=150
    Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
    Building FAISS index at ./faiss_store
      Embedding batch 1/1 (312 chunks)...
    Vector store built in 187.3s.
    Verification -- MMR retrieval smoke test:
      [1] (admission_handbook.pdf) Required documents include 10th and 12th...
      [2] (fee_structure_2024.pdf) The annual fee for B.Tech is Rs 85,000...
      [3] (hostel_rules.pdf) Students may apply for hostel accommodation...
      Smoke test passed.

### 1.3 Optional: tune chunking parameters

If the smoke test returns irrelevant chunks, try:
- Reducing chunk_size to 600 (more precise chunks)
- Increasing chunk_overlap to 200 (better boundary handling)
- Removing short documents (fewer than 500 words) from data/

---

## Phase 2: QLoRA Fine-Tuning (Google Colab)

> "All models are wrong, but some are useful."  -- George E. P. Box
> Fine-tuning makes your model less wrong on the specific domain
> that matters to you.

### 2.1 Generate the fine-tuning dataset

Local mode (no API, fastest):

    python generate_dataset.py --mode local --target 300

Groq-assisted mode (higher quality, needs GROQ_API_KEY in .env):

    python generate_dataset.py --mode groq --target 300

This creates:
    dataset/train.jsonl   (270 records)
    dataset/eval.jsonl    (30 records)

### 2.2 Upload to Google Drive

Upload the dataset/ folder to your Google Drive at:
    My Drive / collegebot / dataset /

Upload your college PDFs to:
    My Drive / collegebot / data /

### 2.3 Open the Colab notebook

1. Go to https://colab.research.google.com
2. File -> New Notebook
3. Runtime -> Change runtime type -> T4 GPU -> Save
4. Copy each cell from notebooks/CollegeBot_Colab.py into Colab cells.
5. Fill in your HF_TOKEN, GROQ_API_KEY, and HF_USERNAME in the relevant cells.
6. Runtime -> Run all

### 2.4 What the notebook does

```
Cell 0  -- Verify T4 GPU is available
Cell 1  -- Install Unsloth, TRL, PEFT, and all dependencies
Cell 2  -- Log in to Hugging Face Hub
Cell 3  -- Set Groq API key
Cell 4  -- Mount Drive, create directory structure
Cell 5  -- [Markdown: explanation]
Cell 6  -- Build FAISS vector store on Colab
Cell 7  -- Generate synthetic Q-A dataset via Groq
Cell 8  -- [Markdown: explanation]
Cell 9  -- Fine-tune Phi-3-mini-4k-instruct (~60-75 min)
Cell 10 -- Merge adapters and push Phi-3 to HF Hub
Cell 11 -- [Markdown: explanation]
Cell 12 -- Fine-tune Mistral-7B-Instruct-v0.3 (~90 min)
Cell 13 -- Run RAGAS evaluation on both models
```

### 2.5 Key hyperparameters

| Parameter              | Value | Reason                                           |
|------------------------|-------|--------------------------------------------------|
| LoRA rank (r)          | 16    | Balances capacity and VRAM usage on T4           |
| LoRA alpha             | 32    | Standard 2x scaling relative to rank            |
| Batch size (Phi-3)     | 2     | Fits in 15 GB T4 VRAM with QLoRA                |
| Batch size (Mistral-7B)| 1     | Mistral-7B is larger; requires smaller batch     |
| Gradient accumulation  | 4 / 8 | Effective batch size = 8 in both cases          |
| Learning rate          | 2e-4  | Standard for LoRA fine-tuning on instruction data|
| Epochs                 | 2     | Sufficient for domain adaptation without overfitting|
| Temperature (train)    | --    | N/A for fine-tuning; applies at inference        |

### 2.6 After training

Both models will appear on your Hugging Face profile:
    https://huggingface.co/YOUR_USERNAME/collegebot-phi3-mini
    https://huggingface.co/YOUR_USERNAME/collegebot-mistral-7b

Update llm_factory.py with your actual username, or set the
PHI3_MODEL_ID and MISTRAL_MODEL_ID environment variables.

---

## Phase 3: Local Development

### 3.1 Set up environment

    git clone https://github.com/Techy-A/collegebot-rag.git
    cd collegebot-rag
    python -m venv venv
    source venv/bin/activate       # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    cp .env.example .env

Edit .env:

    GROQ_API_KEY=gsk_your_key_here
    HF_TOKEN=hf_your_token_here
    PHI3_MODEL_ID=your-username/collegebot-phi3-mini
    MISTRAL_MODEL_ID=your-username/collegebot-mistral-7b
    FAISS_PATH=./faiss_store

### 3.2 Run the app

    streamlit run app.py

The app opens at http://localhost:8501

### 3.3 Test the pipeline

In the browser:
1. Select "groq/llama-3.1-8b-instant" in the sidebar (fastest).
2. Enable "Show retrieved sources".
3. Ask: "What is the last date for admission?"
4. Verify the answer is grounded in your documents.
5. Enable "Show inline eval scores" to see heuristic metrics.

---

## Phase 4: Full RAGAS Evaluation

> "In science, there is only physics; all the rest is stamp collecting."
>   -- Ernest Rutherford
> In ML, there is only evaluation; all the rest is architecture tuning.

Run the full evaluation (requires GROQ_API_KEY and the FAISS store):

    python evaluation/ragas_eval.py

Expected output when all targets are met:

    CollegeBot  --  RAGAS Evaluation Report
    ==============================================================
    [PASS]  Faithfulness            0.9341  (target >= 0.92)  [##################  ]
    [PASS]  Answer Relevance        0.8813  (target >= 0.87)  [#################   ]
    [PASS]  Context Precision       0.9187  (target >= 0.91)  [##################  ]
    ==============================================================
    ALL TARGETS MET.  The system is production ready.

If any metric fails, the script prints targeted optimisation tips.

### Typical iteration cycle to reach targets

Iteration 1 (baseline, often slightly below target):
    Faithfulness:      ~0.88   (3-4 points below target)
    Answer Relevance:  ~0.84   (3 points below target)
    Context Precision: ~0.89   (2 points below target)

Apply fixes:
    - Add "Answer ONLY from context" to QA_PROMPT (faithfulness +3-4%)
    - Reduce temperature from 0.20 to 0.05 (faithfulness +1-2%)
    - Increase fetch_k from 20 to 30 (context precision +1-2%)
    - Add explicit "directly address the question" instruction (relevance +2%)

Iteration 2 (after fixes, typically meets all targets):
    Faithfulness:      ~0.93   PASS
    Answer Relevance:  ~0.89   PASS
    Context Precision: ~0.92   PASS

---

## Phase 5: Deployment on Streamlit Community Cloud

> "Shipping is a feature.  A product that doesn't ship is just a science
>  project."  -- Joel Spolsky

### 5.1 Prepare the GitHub repository

    git init
    git add .
    git commit -m "Initial commit: CollegeBot production RAG"
    git remote add origin https://github.com/Techy-A/collegebot-rag.git
    git push -u origin main

Important: Do NOT commit .env or .streamlit/secrets.toml.
Ensure your .gitignore contains:

    .env
    .streamlit/secrets.toml
    faiss_store/
    __pycache__/
    *.pyc
    venv/
    dataset/

The faiss_store/ directory must be committed if you want the app to work
on Streamlit Cloud without a GPU for re-ingestion.  This is the only
exception: add faiss_store/ to the repository (or upload it separately).

    git add faiss_store/
    git commit -m "Add persistent FAISS vector store"
    git push

### 5.2 Deploy to Streamlit Community Cloud

1. Go to https://share.streamlit.io
2. Click "New app"
3. Select your GitHub repository: your-username/collegebot-rag
4. Set Main file path: app.py
5. Click "Advanced settings"
6. Under Secrets, paste (TOML format):

    GROQ_API_KEY = "gsk_your_key"
    HF_TOKEN     = "hf_your_token"
    FAISS_PATH   = "./faiss_store"

7. Click "Deploy"
8. Wait 3-5 minutes for the first deployment to complete.

Your app is now live at:
    https://your-username-collegebot-rag-app-xxxx.streamlit.app

### 5.3 Optional: HF Inference Endpoints for fine-tuned models

To serve Phi-3 or Mistral-7B on Streamlit Cloud without a GPU:

1. Go to https://huggingface.co/inference-endpoints
2. Create a new endpoint for YOUR_USERNAME/collegebot-phi3-mini
3. Choose "CPU Small" (free tier, ~300 ms latency)
4. Copy the endpoint URL
5. Add to Streamlit Secrets:
    PHI3_ENDPOINT_URL = "https://your-endpoint.aws.endpoints.huggingface.cloud"

The LLM factory will automatically detect and use the endpoint URL
when the "phi3-mini-finetuned" option is selected.

---

## Optimisation Tips Reference

| Metric             | If below target, try...                                                   |
|--------------------|---------------------------------------------------------------------------|
| Faithfulness       | Strengthen grounding instruction in QA_PROMPT                             |
|                    | Reduce temperature to 0.05                                                |
|                    | Increase chunk_overlap to 200                                             |
|                    | Filter chunks shorter than 50 characters                                  |
|                    | Fine-tune with more context-grounded examples                             |
| Answer Relevance   | Add "directly address the question first" to prompt                       |
|                    | Increase k from 6 to 8                                                    |
|                    | Diversify synthetic dataset (more question types)                         |
|                    | Add BM25 hybrid retrieval alongside MMR                                   |
| Context Precision  | Lower lambda_mult from 0.6 to 0.5 in MMR retriever                       |
|                    | Reduce chunk_size from 800 to 600                                         |
|                    | Add metadata filtering by department or document type                     |
|                    | Add a cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L6 on CPU)   |
|                    | Remove noisy or off-topic PDFs from data/                                 |

---

## Troubleshooting

**"FileNotFoundError: No FAISS store found at './faiss_store'"**
    Run: python ingest.py --sample
    Then: streamlit run app.py

**"GROQ_API_KEY is not set"**
    Create a .env file with: GROQ_API_KEY=gsk_your_key_here
    Or set it in shell: export GROQ_API_KEY=gsk_your_key_here

**"CUDA out of memory" in Colab**
    Runtime -> Factory reset runtime
    Set per_device_train_batch_size=1 and gradient_accumulation_steps=8

**RAGAS scores below target after two iterations**
    1. Add at least 50 real Q-A pairs to the evaluation dataset.
    2. Review source documents for OCR errors and noise.
    3. Try running evaluation twice -- Groq outputs vary slightly between runs.
    4. Add a cross-encoder reranker step between retrieval and generation.

**Streamlit app is slow on first query**
    This is the FAISS client cold start and the embedding model load.
    It happens once per session.  Subsequent queries are significantly faster.

---

## License

MIT License.  See LICENSE for details.

> "Make it work, make it right, make it fast."  -- Kent Beck
> This project prioritises making it work (correctness over performance)
> and making it right (clean code over clever code).
> Making it fast is left as an exercise for the reader.
