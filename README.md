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
