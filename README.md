# Aspect-Based Sentiment Analysis (ABSA) Project

This project implements a **Transformer-based model (BERT)** for **Aspect-Based Sentiment Analysis** using the **SemEval-2014 Task 4 Restaurant dataset**.

---

## Project Structure

- `data/` : Contains the dataset files (XML)  
  - `Restaurants_Train_v2.xml` : Training set  
  - `Restaurants_Test_Data_PhaseA.xml` : Test set  

- `models/` : Saved trained models (`.pt`)  

- `ABSA_SemEval_2014.ipynb` : Main Jupyter notebook containing:
  - Data extraction
  - Exploratory Data Analysis (EDA)
  - Transformer model training and evaluation
  - Error analysis and robustness tests  

- `requirements.txt` : Python dependencies  

- `README.md` : Project documentation  

---

## Dataset

**SemEval-2014 Task 4 – Restaurant Reviews**  
- Each sentence may contain multiple aspect terms.  
- Polarity labels: `positive`, `negative`, `neutral`  

---

## Model

- Pretrained **BERT-base-uncased**  
- Fine-tuned for **3-class classification**  
- Input format: `[CLS] sentence [SEP] aspect [SEP]`  

---

## Evaluation

- Metrics: Precision, Recall, F1-score  
- Macro-F1 (important for imbalanced classes)  
- Confusion matrix visualization  
- Error analysis (misclassified examples)  
- Robustness tests with challenging sentences  

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt