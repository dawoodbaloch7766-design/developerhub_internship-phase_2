# developerhub_internship-phase_2
Task no 2,4,5

# TASK #2: End-to-End Customer Churn Prediction Pipeline Using Scikit-learn

## Project Overview

This project develops a complete Machine Learning pipeline to predict customer churn in the telecommunications sector. The solution integrates data preprocessing, feature engineering, model training, hyperparameter optimization, evaluation, and deployment into a single automated workflow.

The pipeline is designed to be reusable, scalable, and production-ready, enabling seamless deployment and future predictions on unseen customer data.

## Objectives

* Predict whether a customer is likely to leave the telecom service.
* Build an end-to-end Machine Learning workflow using Scikit-learn Pipeline API.
* Combine preprocessing and model training into a unified pipeline.
* Optimize model performance using GridSearchCV.
* Export the trained model for deployment and reuse.

## Dataset

**Dataset:** IBM Telco Customer Churn Dataset

The dataset contains customer demographic information, service subscriptions, billing details, and churn status.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Pipeline API
* ColumnTransformer
* Random Forest Classifier
* Logistic Regression
* GridSearchCV
* Joblib

## Data Preprocessing

The following preprocessing steps were applied:

* Converted the `TotalCharges` column into numerical format.
* Handled missing and invalid values.
* Encoded the target variable (`Yes = 1`, `No = 0`).
* Removed the non-informative `customerID` feature.

## Feature Engineering

### Numerical Features

* Missing value imputation using median strategy.
* Feature scaling using StandardScaler.

### Categorical Features

* Missing value imputation using most frequent values.
* One-Hot Encoding for categorical variables.

## Machine Learning Pipeline

A unified pipeline was developed using:

* ColumnTransformer for preprocessing
* RandomForestClassifier for prediction

### Benefits

* Automated preprocessing
* Prevention of data leakage
* Simplified training and inference workflow
* Production-ready architecture

## Hyperparameter Optimization

Model performance was improved using GridSearchCV.

### Tuned Parameters

* n_estimators
* max_depth
* min_samples_split

## Model Evaluation

Performance was assessed using:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Classification Report

## Model Persistence

The final trained pipeline was exported using Joblib.

**Saved Model:** `churn_prediction_pipeline.joblib`

The exported file includes:

* Data preprocessing steps
* Feature transformations
* Trained machine learning model

This allows direct deployment without rebuilding the preprocessing workflow.

## Key Learning Outcomes

* End-to-end ML pipeline development
* Data preprocessing and feature engineering
* Handling missing values in real-world datasets
* Hyperparameter tuning with GridSearchCV
* Model serialization and deployment
* Production-ready Machine Learning practices

## Future Enhancements

* Experiment with XGBoost and LightGBM
* Develop a REST API using Flask or FastAPI
* Deploy an interactive Streamlit application
* Create a customer churn analytics dashboard

## Author

Academic Machine Learning Project developed using Python and Scikit-learn.


# TASK #4: Context-Aware RAG Chatbot with Semantic Search

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot capable of answering user queries using information retrieved from a custom knowledge base.

The system leverages semantic search, vector embeddings, and a FAISS vector database to retrieve the most relevant content before generating responses, enabling context-aware and accurate question answering.

## Objectives

* Build a Retrieval-Augmented Generation (RAG) system from scratch.
* Generate dense vector embeddings from textual data.
* Store embeddings efficiently using FAISS.
* Retrieve relevant information through semantic similarity search.
* Develop an interactive chatbot interface.

## Technologies Used

* Python
* FAISS (Facebook AI Similarity Search)
* Sentence Transformers
* all-MiniLM-L6-v2 Embedding Model
* NumPy
* Streamlit

## Knowledge Base

The chatbot uses a custom knowledge repository containing:

* Attendance Policies
* Academic Grading Rules
* Semester Schedules
* Refund Policies

## System Workflow

### 1. Text Chunking

Large documents are divided into smaller chunks to improve retrieval quality and contextual relevance.

### 2. Embedding Generation

Each chunk is transformed into dense vector representations using:

`all-MiniLM-L6-v2`

### 3. Vector Storage

Generated embeddings are stored in a FAISS index for efficient similarity search.

### 4. Query Processing

User questions are:

* Converted into embeddings
* Compared against stored vectors
* Ranked by similarity score
* Retrieved as Top-K relevant chunks

### 5. Response Generation

The retrieved context is used to generate informed and context-aware responses.

## Key Features

* Semantic search instead of keyword matching
* Context-aware question answering
* Fast vector retrieval using FAISS
* Lightweight embedding architecture
* Real-time interaction
* Chat history support

## User Interface

A web-based interface was developed using Streamlit, allowing users to interact with the chatbot in real time.

## Key Learning Outcomes

* Retrieval-Augmented Generation (RAG)
* Text chunking strategies
* Embedding generation
* Semantic search techniques
* Vector databases and indexing
* Conversational AI development
* Streamlit application development

## Future Enhancements

* Integrate advanced LLMs such as GPT or Llama
* Support PDF and document uploads
* Expand to multi-document knowledge bases
* Improve conversational memory
* Deploy on cloud platforms such as Streamlit Cloud or Hugging Face Spaces

## Author

Academic Generative AI Project built using FAISS, Sentence Transformers, and Streamlit.


# TASK #5: Automated Support Ticket Classification Using Large Language Models

## Project Overview

This project automates customer support ticket categorization using Large Language Models (LLMs). The system applies both zero-shot and few-shot learning techniques to classify incoming support requests into relevant categories without requiring extensive model training.

The solution improves ticket routing efficiency and demonstrates practical applications of LLM-based text classification.

## Objectives

* Automatically categorize customer support tickets.
* Evaluate zero-shot and few-shot learning approaches.
* Generate Top-3 predicted categories for each ticket.
* Enhance customer support automation and operational efficiency.

## Dataset

**Dataset:** Customer Support Tickets Dataset

The dataset was sourced from Hugging Face and contains customer support interactions and ticket descriptions.

### Key Features Used

* Ticket Description
* Ticket Type
* Priority
* Status

## Technologies Used

* Python
* Pandas
* Hugging Face Transformers
* BART Large MNLI
* Google Colab

## Methodology

### Data Preparation

* Loaded ticket data using Pandas.
* Selected ticket descriptions as the primary input feature.
* Cleaned and prepared text for classification.

### Zero-Shot Classification

A pre-trained BART model was used to classify tickets without additional training.

Advantages:

* No labeled training dataset required
* Rapid deployment
* Strong baseline performance

### Few-Shot Classification

Representative examples were included within prompts to provide contextual guidance and improve classification quality.

Benefits:

* Better contextual understanding
* Improved prediction accuracy
* Enhanced category relevance

## Candidate Categories

Example ticket labels include:

* Billing
* Technical Issue
* Account Access
* Refund
* Shipping
* Complaint
* Feature Request
* Product Inquiry

## Prediction Output

For every support ticket, the model generates:

* Top 3 predicted categories
* Confidence-based ranking
* Structured output for downstream processing

### Example

**Input:**

"I am unable to log in after resetting my password."

**Predicted Categories:**

1. Account Access
2. Technical Issue
3. Complaint

## Results

* Zero-shot learning delivered strong baseline classification performance.
* Few-shot prompting improved contextual accuracy and relevance.
* Top-3 predictions increased flexibility for automated ticket routing systems.

## Key Learning Outcomes

* Prompt Engineering
* LLM-Based Classification
* Zero-Shot Learning
* Few-Shot Learning
* Multi-Class Prediction
* Ranking and Confidence Scoring
* Applied NLP for Customer Support Automation

## Future Enhancements

* Fine-tune models on domain-specific ticket datasets
* Build a real-time prediction API
* Create a support analytics dashboard
* Integrate with CRM and ticket management platforms

## Author

Academic NLP and Generative AI Project developed using Hugging Face Transformers and Python.
