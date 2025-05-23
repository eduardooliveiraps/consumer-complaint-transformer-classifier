# Consumer Complaint Classification – Transformer

## Overview
This project builds on a previous NLP assignment by applying state-of-the-art transformer models to the task of classifying U.S. consumer financial complaints. Leveraging Hugging Face's Transformers library, the project aims to improve classification accuracy and generalizability over traditional machine learning approaches previously explored.

## Project Scope & Methodology
The classification task uses a publicly available dataset from the Consumer Financial Protection Bureau (CFPB), consisting of complaint narratives categorized into five consolidated financial categories: Loans, Credit Reporting, Bank Accounts & Services, Debt Collection, and Credit Card Services.

The key methodological steps include:
- Selection of suitable pre-trained transformer models (e.g., BERT, RoBERTa)
- Fine-tuning models on the CFPB dataset
- Experimentation with domain adaptation and parameter-efficient fine-tuning methods (e.g., LoRA)
- Optional use of prompt-based learning approaches
- Performance comparison with classical ML models developed in the first assignment

## Objectives & Real-World Impact
The main objectives are:
- To evaluate the effectiveness of transformer-based models in complaint classification
- To understand the benefits and limitations of fine-tuning versus traditional ML techniques
- To explore advanced strategies such as domain adaptation and efficient training

## 🔧 Usage / How to Run

Follow the steps below to set up and run the project:

### 1. Clone the Repository

Using **SSH**:
```bash
git clone git@github.com:eduardooliveiraps/consumer-complaint-transformer-classifier.git
```	

Using **HTTPS**:
```bash
git clone https://github.com/eduardooliveiraps/consumer-complaint-transformer-classifier.git
```

Then, navigate to the project directory:
```bash
cd consumer-complaint-transformer-classifier
```

### 2. Download the Dataset

The dataset can be downloaded from the [Consumer Complaint Dataset on Kaggle](https://www.kaggle.com/datasets/namigabbasov/consumer-complaint-dataset).

After downloading, place the file `complaints.csv` in the `/data` folder inside the project directory:

```kotlin
consumer-complaint-transformer-classifier/
│
├── data/
│   └── complaints.csv
```

### 3. Create a Virtual Environment

On Windows:
```bash
python -m venv nlp_env
nlp_env\Scripts\activate
```

On macOS/Linux:
```bash
python3 -m nlp_env nlp_env
source nlp_env/bin/activate
```

### 4. Install Required Packages
```bash
pip install -r requirements.txt
```

### 5. Run the Notebook



