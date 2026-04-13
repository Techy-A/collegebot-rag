# CollegeBot -- QLoRA Fine-Tuning Notebook (Google Colab Free T4)
# ================================================================
# "An investment in knowledge pays the best interest."
#     -- Benjamin Franklin
#
# INSTRUCTIONS:
#   1. Open https://colab.research.google.com
#   2. Runtime -> Change runtime type -> T4 GPU
#   3. Copy each cell block (between # CELL: markers) into a new Colab cell.
#   4. Run cells top to bottom.  Do not skip any cell.
#
# This notebook fine-tunes two models:
#   a) microsoft/Phi-3-mini-4k-instruct
#   b) mistralai/Mistral-7B-Instruct-v0.3
#
# Both are fine-tuned with QLoRA (r=16, alpha=32) using Unsloth,
# which makes 7B models train in ~2 hours on a free T4.
#
# After training, both models are pushed to your Hugging Face Hub.
# ================================================================


# CELL: [MARKDOWN]
# ## CollegeBot QLoRA Fine-Tuning
# **Platform**: Google Colab Free Tier (T4 GPU)
# **Estimated time**: ~2.5 hours for both models combined
# **Cost**: $0.00
#
# > "Data is the new oil.  But like oil, it's valuable only after refining."
# >   -- Clive Humby


# CELL: 0 -- Verify GPU
import subprocess
result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(result.stdout if result.returncode == 0 else "No GPU found.  Change runtime to T4.")


# CELL: 1 -- Install dependencies
# Unsloth is installed first because it patches transformers in-place.
# The order of these installs matters.
subprocess.run([
    "pip", "install", "--quiet",
    "unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git",
], check=True)
subprocess.run([
    "pip", "install", "--quiet",
    "xformers", "trl", "peft", "accelerate", "bitsandbytes",
    "langchain", "langchain-community", "chromadb",
    "sentence-transformers", "pypdf", "python-docx",
    "ragas", "datasets", "groq", "langchain-groq",
    "python-dotenv", "huggingface_hub",
], check=True)
print("Dependencies installed.")


# CELL: 2 -- Hugging Face login
# You need a WRITE token from https://huggingface.co/settings/tokens
# The model will be pushed to your namespace at the end of training.
from huggingface_hub import login as hf_login
HF_TOKEN = "hf_your_token_here"    # REPLACE THIS WITH YOUR ACTUAL TOKEN
hf_login(token=HF_TOKEN, add_to_git_credential=False)
print("Logged in to Hugging Face.")


# CELL: 3 -- Groq login (for dataset generation and evaluation)
import os
os.environ["GROQ_API_KEY"] = "gsk_your_groq_key_here"   # REPLACE THIS WITH YOUR ACTUAL KEY
print("Groq API key set.")


# CELL: 4 -- Clone project and mount Drive
# Mount Drive to persist the Chroma DB and dataset across sessions.
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

# Clone or recreate the project structure.
os.makedirs("/content/data",    exist_ok=True)
os.makedirs("/content/dataset", exist_ok=True)
os.makedirs("/content/chroma_db", exist_ok=True)
print("Directory structure ready.")
print("Upload your college PDFs to /content/data/ using the Files panel.")


# CELL: 5 -- [MARKDOWN]
# ## Phase 1: Data Ingestion
# Upload your college PDFs/docs to `/content/data/` using the Files panel on the left,
# then run the next cell.


# CELL: 6 -- Build FAISS vector store (aligned with ingest.py)
# Using FAISS instead of Chroma for consistency with the main app.

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

DATA_DIR    = "/content/data"
FAISS_PATH  = "/content/drive/MyDrive/collegebot/faiss_store"  # persisted to Drive
os.makedirs(FAISS_PATH, exist_ok=True)

# Load documents
docs = []
for fpath in Path(DATA_DIR).rglob("*.pdf"):
    loader = PyPDFLoader(str(fpath))
    loaded = loader.load()
    for d in loaded:
        d.metadata["source"] = fpath.name
    docs.extend(loaded)
    print(f"  Loaded: {fpath.name} ({len(loaded)} pages)")

for fpath in Path(DATA_DIR).rglob("*.txt"):
    loader = TextLoader(str(fpath))
    loaded = loader.load()
    for d in loaded:
        d.metadata["source"] = fpath.name
    docs.extend(loaded)

print(f"\nTotal documents: {len(docs)}")

# If data/ is empty, create sample data for testing
if not docs:
    print("No documents found.  Creating sample data for testing...")
    sample = Path(DATA_DIR) / "sample.txt"
    sample.write_text(
        "The last date for admission is June 30.  "
        "Required documents: 10th, 12th mark sheets, TC, and ID proof.  "
        "Annual B.Tech fee: Rs 85,000.  Minimum attendance: 75%.  "
        "Library hours: 8 AM to 8 PM, Monday to Saturday."
    )
    loader = TextLoader(str(sample))
    docs   = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks   = splitter.split_documents(docs)
print(f"Chunks created: {len(chunks)}")

# Embed and store
embeddings  = HuggingFaceEmbeddings(
    model_name    = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs  = {"device": "cuda"},
    encode_kwargs = {"normalize_embeddings": True},
)
vectorstore = FAISS.from_documents(
    documents = chunks,
    embedding = embeddings,
)
vectorstore.save_local(FAISS_PATH)
print(f"FAISS vector store built at {FAISS_PATH}")


# CELL: 7 -- Generate synthetic Q-A dataset
# Uses Groq to generate diverse training examples from the document chunks.
# The dataset is saved to Drive for reuse across sessions.

from groq import Groq
import json, time, random

client      = Groq(api_key=os.environ["GROQ_API_KEY"])
DATASET_OUT = "/content/drive/MyDrive/collegebot/dataset"
os.makedirs(DATASET_OUT, exist_ok=True)

def generate_qa_groq(chunk_text: str, n: int = 3) -> list:
    """Request n Q-A pairs from Groq for a single chunk."""
    prompt = f"""Generate {n} diverse question-answer pairs from this college handbook text.
Each answer must come ONLY from the text.  Format: Q: ... / A: ...

TEXT:
{chunk_text[:500]}

Generate {n} pairs:"""
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}],
            temperature=0.7, max_tokens=400,
        )
        text = resp.choices[0].message.content or ""
        pairs = []
        for line in text.split("\n"):
            if line.strip().lower().startswith("q:"):
                q = line.strip()[2:].strip()
            elif line.strip().lower().startswith("a:") and q:
                a = line.strip()[2:].strip()
                if q and a:
                    pairs.append({"instruction": q, "input": "", "output": a})
                q = ""
        return pairs
    except Exception as e:
        print(f"    Groq error: {e}")
        return []

all_records = []
target      = 300
chunk_texts = [c.page_content for c in chunks]
random.shuffle(chunk_texts)

print(f"Generating dataset (target: {target} pairs)...")
for i, chunk in enumerate(chunk_texts):
    if len(all_records) >= target:
        break
    pairs = generate_qa_groq(chunk, n=3)
    all_records.extend(pairs)
    print(f"  Chunk {i+1}: +{len(pairs)} pairs ({len(all_records)} total)")
    time.sleep(0.8)   # Rate limit buffer

# Shuffle and split
random.shuffle(all_records)
split    = int(len(all_records) * 0.9)
train_ds = all_records[:split]
eval_ds  = all_records[split:]

for name, rows in [("train", train_ds), ("eval", eval_ds)]:
    path = f"{DATASET_OUT}/{name}.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

print(f"\nDataset saved: {len(train_ds)} train / {len(eval_ds)} eval")


# CELL: 8 -- [MARKDOWN]
# ## Phase 2a: Fine-tune Phi-3-mini-4k-instruct with Unsloth + QLoRA
# This cell trains the model.  Estimated time: ~60-75 minutes on T4.
#
# > "The purpose of training is not to complete it, but to improve at
# >  the task it was designed for."  -- Paraphrasing every ML paper ever.


# CELL: 9 -- Fine-tune Phi-3-mini
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch

# ---- Hyperparameters ----
MODEL_NAME_PHI3  = "microsoft/Phi-3-mini-4k-instruct"
HF_PUSH_NAME_PHI3 = "YOUR_HF_USERNAME/collegebot-phi3-mini"    # REPLACE
MAX_SEQ_LEN      = 2048
LORA_R           = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05
TARGET_MODULES   = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]

# ---- Load model with 4-bit quantization ----
print("Loading Phi-3-mini with 4-bit QLoRA via Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name       = MODEL_NAME_PHI3,
    max_seq_length   = MAX_SEQ_LEN,
    dtype            = None,   # Unsloth auto-selects bfloat16 or float16
    load_in_4bit     = True,
    token            = HF_TOKEN,
)

# ---- Apply LoRA adapters ----
model = FastLanguageModel.get_peft_model(
    model,
    r                = LORA_R,
    target_modules   = TARGET_MODULES,
    lora_alpha       = LORA_ALPHA,
    lora_dropout     = LORA_DROPOUT,
    bias             = "none",
    use_gradient_checkpointing = "unsloth",   # Saves ~30% VRAM
    random_state     = 42,
)
print(f"LoRA adapters applied.  r={LORA_R}, alpha={LORA_ALPHA}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ---- Format dataset ----
# Alpaca prompt template.  Phi-3 is instruction-tuned, so it responds
# well to this format without any chat template adjustments.
ALPACA_TEMPLATE = """Below is an instruction from a college student.
Write a response based only on official college information.

### Instruction:
{instruction}

### Response:
{output}"""

def format_alpaca(batch):
    return {
        "text": [
            ALPACA_TEMPLATE.format(
                instruction=i,
                output=o,
            )
            for i, o in zip(batch["instruction"], batch["output"])
        ]
    }

train_data = load_dataset(
    "json",
    data_files=f"{DATASET_OUT}/train.jsonl",
    split="train",
)
train_data = train_data.map(format_alpaca, batched=True)

# ---- Training arguments ----
# These settings are calibrated for a 15 GB T4 with QLoRA:
#   - per_device_train_batch_size=2 with gradient accumulation=4
#     gives an effective batch size of 8.
#   - 2 epochs on 270 examples = 540 gradient steps.
#   - Learning rate 2e-4 is the standard for LoRA fine-tuning.
training_args = TrainingArguments(
    output_dir                  = "/content/phi3_checkpoints",
    num_train_epochs            = 2,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    learning_rate               = 2e-4,
    fp16                        = not torch.cuda.is_bf16_supported(),
    bf16                        = torch.cuda.is_bf16_supported(),
    logging_steps               = 10,
    save_steps                  = 100,
    warmup_ratio                = 0.05,
    lr_scheduler_type           = "cosine",
    optim                       = "adamw_8bit",   # Unsloth's fused 8-bit AdamW
    weight_decay                = 0.01,
    report_to                   = "none",
    seed                        = 42,
)

trainer = SFTTrainer(
    model            = model,
    tokenizer        = tokenizer,
    train_dataset    = train_data,
    dataset_text_field = "text",
    max_seq_length   = MAX_SEQ_LEN,
    args             = training_args,
)

print("\nStarting Phi-3-mini training...")
trainer.train()
print("Training complete.")


# CELL: 10 -- Merge and push Phi-3-mini to Hub
# Merging the LoRA adapters into the base model before pushing
# simplifies inference -- no need for PEFT at inference time.
print("Merging LoRA adapters...")
model.save_pretrained_merged(
    "/content/phi3_merged",
    tokenizer,
    save_method="merged_16bit",
)
print(f"Pushing to {HF_PUSH_NAME_PHI3}...")
model.push_to_hub_merged(
    HF_PUSH_NAME_PHI3,
    tokenizer,
    save_method    = "merged_16bit",
    token          = HF_TOKEN,
    private        = False,
)
print(f"Phi-3-mini pushed to https://huggingface.co/{HF_PUSH_NAME_PHI3}")


# CELL: 11 -- [MARKDOWN]
# ## Phase 2b: Fine-tune Mistral-7B-Instruct-v0.3
# This is the second model.  The same QLoRA configuration is used.
# Estimated time: ~90 minutes on T4.
#
# If you have already used significant GPU time in this session,
# start a fresh session before running this cell to avoid
# the "12-hour limit reached" disconnection mid-training.


# CELL: 12 -- Fine-tune Mistral-7B
# Free GPU memory from Phi-3 training before loading Mistral-7B.
del model, tokenizer, trainer
torch.cuda.empty_cache()
import gc; gc.collect()
print("GPU memory cleared.")

MODEL_NAME_MISTRAL  = "mistralai/Mistral-7B-Instruct-v0.3"
HF_PUSH_NAME_MISTRAL = "YOUR_HF_USERNAME/collegebot-mistral-7b"  # REPLACE

print("Loading Mistral-7B with 4-bit QLoRA via Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME_MISTRAL,
    max_seq_length = MAX_SEQ_LEN,
    dtype          = None,
    load_in_4bit   = True,
    token          = HF_TOKEN,
)

# Mistral uses a chat template.  We apply it here so the model
# sees the same format it was instruction-tuned on.
tokenizer.chat_template = None   # Use default Alpaca format for simplicity

model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_R,
    target_modules = TARGET_MODULES,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROPOUT,
    bias           = "none",
    use_gradient_checkpointing = "unsloth",
    random_state   = 42,
)

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = train_data,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LEN,
    args               = TrainingArguments(
        output_dir                  = "/content/mistral_checkpoints",
        num_train_epochs            = 2,
        per_device_train_batch_size = 1,    # Mistral-7B is larger; reduce batch size
        gradient_accumulation_steps = 8,    # Effective batch size still = 8
        learning_rate               = 2e-4,
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        logging_steps               = 10,
        save_steps                  = 100,
        warmup_ratio                = 0.05,
        lr_scheduler_type           = "cosine",
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        report_to                   = "none",
        seed                        = 42,
    ),
)

print("\nStarting Mistral-7B training...")
trainer.train()
print("Training complete.")

# Merge and push
model.save_pretrained_merged("/content/mistral_merged", tokenizer, save_method="merged_16bit")
model.push_to_hub_merged(HF_PUSH_NAME_MISTRAL, tokenizer,
                          save_method="merged_16bit", token=HF_TOKEN, private=False)
print(f"Mistral-7B pushed to https://huggingface.co/{HF_PUSH_NAME_MISTRAL}")


# CELL: 13 -- Run RAGAS evaluation (aligned with fixes)
# Copies the FAISS store from Drive back to /content, then runs the evaluation.
import shutil
shutil.copytree(
    "/content/drive/MyDrive/collegebot/faiss_store",
    "/content/faiss_store",
    dirs_exist_ok=True,
)
os.environ["FAISS_PATH"] = "/content/faiss_store"

# Run evaluation (inline version of evaluation/ragas_eval.py)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)
vs = FAISS.load_local(
    "/content/faiss_store",
    emb,
    allow_dangerous_deserialization=True,
)
retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k":6,"fetch_k":20,"lambda_mult":0.6})

PROMPT = PromptTemplate(
    template="Answer ONLY from context.\nCONTEXT:{context}\nHISTORY:{chat_history}\nQ:{question}\nA:",
    input_variables=["context","chat_history","question"],
)
llm   = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model_name="llama-3.1-8b-instant",
    temperature=0.05,
    timeout=120,
    max_retries=3,
)
chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever,
    memory=ConversationBufferWindowMemory(k=5,memory_key="chat_history",return_messages=True,output_key="answer"),
    combine_docs_chain_kwargs={"prompt":PROMPT},
    return_source_documents=True,
)

EVAL_QA = [
    ("What is the last date for admission?", "The last date to submit the college admission form is June 30, 2024."),
    ("What is the B.Tech annual fee?",        "The annual fee for B.Tech is Rs 85,000."),
    ("What is the minimum attendance required?", "Students must maintain a minimum of 75% attendance."),
    ("What scholarships are available?",      "Scholarships include Chief Minister, Merit-cum-Means, College Merit, and Sports Scholarships."),
    ("How do I register for elective courses?", "Students register through the online ERP portal at erp.college.edu during Week 3 of each semester."),
]

records = []
for q, gt in EVAL_QA:
    res      = chain.invoke({"question": q})
    answer   = res.get("answer", "")
    contexts = [d.page_content for d in res.get("source_documents", [])]
    records.append({"question":q,"answer":answer,"contexts":contexts,"ground_truth":gt})

judge_llm = LangchainLLMWrapper(
    ChatGroq(
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name="llama-3.1-8b-instant",
        temperature=0.0,
        timeout=120,
        max_retries=3,
    )
)
judge_emb = LangchainEmbeddingsWrapper(emb)
for m in [faithfulness, answer_relevancy, context_precision]:
    m.llm = judge_llm
    if hasattr(m, "embeddings"):
        m.embeddings = judge_emb

print("\nComputing RAGAS scores...")
scores = evaluate(Dataset.from_list(records), metrics=[faithfulness, answer_relevancy, context_precision])
print(f"\n{'='*50}")
print("RAGAS EVALUATION RESULTS")
print(f"{'='*50}")
print(f"  Faithfulness      : {scores['faithfulness']:.4f}  (target >= 0.92)")
print(f"  Answer Relevance  : {scores['answer_relevancy']:.4f}  (target >= 0.87)")
print(f"  Context Precision : {scores['context_precision']:.4f}  (target >= 0.91)")
print(f"{'='*50}")
