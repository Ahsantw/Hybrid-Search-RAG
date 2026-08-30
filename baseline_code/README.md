# Retrieval-Augmented Generation (RAG)

This project demonstrates an end-to-end Retrieval-Augmented Generation (RAG) pipeline. Main features of this repository are:

1. Download the LLaMA model (meta-llama/Llama-3.1-8B-Instruct) from Hugging Face.
2. Convert the LLaMA model to OpenVINO INT4 format.
3. If the model has already been converted, load the existing INT4 version directly from folder.
4. Use a Sentence-Transformers embedding model to generate embeddings and store them using FAISS.
5. Build a question-answering (QA) system using LangChain to complete the RAG pipeline.
6. Logs for all the steps are documented in all_logs folder. So very easy to debug any issues.
### Installation

1. Clone the Repository
```
git clone https://github.com/Ahsantw/RAG
cd RAG
```
2. Install Python 3.10
3. Install required Pakages
```
pip install -r requirements.txt
```
4. Log into to hugginface accout (optional if you do not have HF model i.e. meta-llama/Llama-3.1-8B-Instruct). You will be asked to paste [HF token](https://huggingface.co/docs/hub/en/security-tokens).
```
huggingface-cli login
```
### Model

This repository uses OpenVINO INT4 version of Llama-3.1-8B-Instruct model during inference.

### Full Pipeline Inference
All steps are done with one command. At the end when all steps are done, you can easly type questions.
```
bash run_demo.sh
```
Different parameters/variable can easily be change from [config](https://github.com/Ahsantw/RAG/blob/main/config/config.yaml) file.

### Step by Step Inference
1. Download and Convert llama Model.
```
python src/convert_llama_to_open.py
```
2. Store pdf's embeding using Faiss.
```
python src/vector_db.py
```
3. Answer with reference for queries.
```
python rag_cli.py
```
Different parameters/variable can easily be change from [config](https://github.com/Ahsantw/RAG/blob/main/config/config.yaml) file.

### Sample Output
The output includes a reference from the PDF, followed by the actual answer.
```
Question ('exit'): whats procyon
 - Page 1, File: data/procyon_guide.pdf
 - Page 39, File: data/procyon_guide.pdf
 - Page 13, File: data/procyon_guide.pdf
Answer:  UL Procyon is a suite of benchmark tests for professional users in various industries, designed to measure the performance of computers and devices.
```
```
Question ('exit'): tell me what the context says about Benchmarks at the enterprise IT level
 - Page 0, File: data/procyon_guide.pdf
 - Page 0, File: data/procyon_guide.pdf
 - Page 41, File: data/procyon_guide.pdf
Answer:  Benchmarks at the enterprise IT level support every stage in the life cycle of PC assets, easing PC lifecycle management for IT teams. They provide support for:

• Planning and procurement: Simplify PC performance comparison and cost justification
• Validation and standardization: Test and compare the performance of new PCs against user-defined baselines
• Operations and management: Efficiently automate remote performance testing to provide reliable insights and reporting
• Optimization or replacement: Make informed PC life-cycle decisions based on benchmark results stored in your central database
```

### Hardware Specs.

This RAG pipeline was tested successfully on the following system:

- **OS**: Windows 10/ Ubuntu 22.04 (Tested on Both)
- **Processor**: Intel Core i7 10th Gen
- **RAM**: 32 GB
- **GPU**: NVIDIA RTX 3090
- **HardDrive**: 2TB

### Latency
- **Average response time:** 15-20 seconds

### Common Issues
1. HugginFace login issue.
```
Failed to Run the pipeline : You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```
Solution: Go to huggingface [page](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and request model's access by completing and submitting the form. It takes few minutes and they grant you access.
Then login to your hugginface accout from terminal using following command and then paste [HF token](https://huggingface.co/docs/hub/en/security-tokens).
```
huggingface-cli login
```
