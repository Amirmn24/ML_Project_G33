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
```


## Overview
This project implements a complete Aspect-Based Sentiment Analysis (ABSA) system that:
1. Identifies aspect terms in sentences
2. Classifies sentiment polarity for each aspect

## Dataset
- **Source**: SemEval 2014 Task 4 (Restaurant reviews)
- **Additional data**: Custom training and test data for restaurants and laptops
- **Task**: Predict sentiment (positive, negative, neutral) for specific aspects in reviews

---

## Project Phases

### Phase 1: Data Exploration and EDA (Exploratory Data Analysis)

**Objective**: Understand the data distribution and characteristics

**Key Steps**:
1. Load and explore the SemEval2014 dataset
2. Analyze aspect categories and terms
3. Create flat dataset structure (sentence-category-polarity pairs)
4. Visualize class distributions
5. Analyze sentence length distributions
6. Generate word clouds for different categories

**Key Findings**:
- Multiple aspects can exist in a single sentence
- Class imbalance exists (more positive reviews)
- Some samples have 'conflict' polarity (removed for 3-class classification)
- Average sentence length helps determine max_length for tokenization

**Deliverables**:
- Data statistics and distributions
- Visualization plots
- Clean dataset for model training

---

### Phase 2: ABSA Model Training (Category-Based Sentiment Classification)

**Objective**: Train a transformer-based model to classify sentiment for aspect categories

**Architecture**:
```
Input: [CLS] sentence [SEP] aspect/category [SEP]
       ↓
BERT/RoBERTa Encoder
       ↓
Classification Head (3 classes)
       ↓
Output: {positive, neutral, negative}
```

**Key Components**:

1. **Dataset Class (ABSADataset)**:
   - Combines sentence and aspect as two text segments
   - Tokenizes using BERT tokenizer
   - Returns input_ids, attention_mask, labels

2. **Model**:
   - Pre-trained BERT-base-uncased
   - Fine-tuned classification head for 3 classes
   - Uses cross-entropy loss

3. **Hyperparameters**:
   ```python
   MODEL_NAME = "bert-base-uncased"
   MAX_LENGTH = 128
   BATCH_SIZE = 16
   EPOCHS = 4
   LEARNING_RATE = 2e-5
   WEIGHT_DECAY = 0.01
   VALIDATION_RATIO = 0.15
   ```

4. **Training Process**:
   - Split training data into train/validation (85%/15%)
   - Train for 4 epochs with validation after each epoch
   - Track loss and accuracy metrics
   - Use AdamW optimizer

**Performance Metrics**:
- Training/Validation Loss curves
- Training/Validation Accuracy curves
- Final test accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score)

**Deliverables**:
- Trained sentiment classification model
- Training history plots
- Evaluation metrics on test set

---

### Phase 3: Aspect Term Extraction (Phase A)

**Objective**: Automatically identify aspect terms in sentences

**Approach**: Sequence Labeling using BIO Tagging Scheme

**BIO Tagging Scheme**:
- **B-ASP**: Beginning of an aspect term
- **I-ASP**: Inside an aspect term (continuation)
- **O**: Outside (not part of any aspect)

**Example**:
```
Sentence: "The food was delicious but the service was slow"
Words:    The  food was delicious but the service was slow
Tags:     O    B-ASP O   O         O   O   B-ASP   O   O
Aspects:  [food, service]
```

**Architecture**:
```
Input: Sentence tokens
       ↓
BERT Token Classifier
       ↓
BIO Tags for each token
       ↓
Extract aspect terms
```

**Key Components**:

1. **Data Preparation**:
   - Convert character-level annotations to word-level BIO tags
   - Handle multi-word aspect terms
   - Create TokenClassificationDataset

2. **Model**:
   - AutoModelForTokenClassification (BERT-based)
   - 3 output classes: O, B-ASP, I-ASP
   - Token-level classification

3. **Training**:
   - Train on restaurant/laptop training data
   - Use -100 label for special tokens and subwords
   - Align labels with tokenized inputs

4. **Inference**:
   - Predict BIO tags for each token
   - Reconstruct aspect terms from consecutive B-ASP and I-ASP tags
   - Handle cases with no detected aspects

**Deliverables**:
- Trained aspect extraction model
- Function to extract aspects from new sentences
- Predictions for Phase A test data

---

### Phase 4: End-to-End Sentiment Classification (Phase B)

**Objective**: Classify sentiment for extracted aspects in test data

**Pipeline**:
```
Test Sentence
      ↓
[Phase A: Extract Aspects]
      ↓
Aspect Terms
      ↓
[Phase B: Classify Sentiment for each aspect]
      ↓
(Aspect, Sentiment) pairs
```

**Process**:

1. **Phase A Processing**:
   - Load test sentences from Phase A files
   - Extract aspect terms using trained NER model
   - Handle sentences with no detected aspects (assign 'NULL')

2. **Phase B Processing**:
   - For each (sentence, aspect) pair
   - Use trained ABSA model to predict sentiment
   - Generate final predictions

**Output Format**:
```
id, sentence, aspect, polarity
1, "The food was great", "food", "positive"
2, "Slow service", "service", "negative"
```

**Deliverables**:
- Phase A results: Extracted aspects
- Phase B results: Sentiment predictions
- CSV files with complete predictions
- Sentiment distribution visualizations

---

## Complete ABSA Pipeline

**End-to-End Usage example**:
```python
# Input
sentence = "The food was delicious but the service was terrible."

# Process
aspects = extract_aspects(sentence, ner_model, ner_tokenizer, device, ID2TAG)
# Output: ['food', 'service']

# Classify sentiment for each aspect
for aspect in aspects:
    sentiment = predict_sentiment(sentence, aspect, model, tokenizer, device, ID2LABEL)
    print(f"{aspect}: {sentiment}")

# Output:
# food: positive
# service: negative
```

---

## Key Improvements and Fixes

### Issues Fixed:

1. **BIO Tagging Logic**:
   - Improved character-to-word alignment
   - Handle overlapping aspects correctly
   - Proper handling of multi-word aspects

2. **Training Stability**:
   - Added validation split
   - Progress bars for better monitoring
   - Proper device handling (CPU/GPU)

3. **Evaluation**:
   - Added confusion matrix
   - Detailed classification report
   - Separate train/val/test evaluation

4. **Phase A & B Integration**:
   - Complete pipeline from raw text to sentiment predictions
   - Handle edge cases (no aspects detected)
   - Proper data flow between phases

### Best Practices Implemented:

1. **Data Handling**:
   - Stratified train/validation split
   - Proper batch processing
   - Handle variable length sequences

2. **Model Training**:
   - Early monitoring with validation set
   - Gradient updates per batch
   - Loss tracking and visualization

3. **Inference**:
   - Efficient batching
   - GPU utilization
   - Result aggregation

4. **Code Organization**:
   - Clear phase separation
   - Reusable functions
   - Comprehensive documentation

---

## Results Summary

### Phase 2 (ABSA Training):
- **Final Test Accuracy**: ~80-85% (depends on training)
- **Classes**: positive, neutral, negative
- **Best performing**: positive class (most common)
- **Challenging**: neutral class (often confused with others)

### Phase 3 (Aspect Extraction):
- **Extracted aspects**: Varies by test set
- **Coverage**: Captures most explicit aspect mentions
- **Limitations**: May miss implicit aspects or paraphrased terms

### Phase 4 (Complete Pipeline):
- **Output**: Sentence-level aspect-sentiment analysis
- **Format**: CSV with aspect terms and polarities
- **Use case**: Restaurant/product review analysis

---

## Model Files Generated

1. **absa_category_model/** - Sentiment classification model
2. **aspect_extraction_model/** - Aspect term extraction model
3. **restaurants_phaseB_predictions.csv** - Restaurant predictions
4. **laptops_phaseB_predictions.csv** - Laptop predictions

---

## Usage Instructions


### Inference on New Data:
```python
# Load models
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import torch

# Load sentiment model
sentiment_model = AutoModelForSequenceClassification.from_pretrained("absa_category_model")
sentiment_tokenizer = AutoTokenizer.from_pretrained("absa_category_model")

# Load aspect extraction model
ner_model = AutoModelForTokenClassification.from_pretrained("aspect_extraction_model")
ner_tokenizer = AutoTokenizer.from_pretrained("aspect_extraction_model")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sentiment_model = sentiment_model.to(device)
ner_model = ner_model.to(device)

# Analyze new sentence
sentence = "The pasta was amazing but the wine was disappointing."
results = complete_absa_pipeline(sentence, ner_model, ner_tokenizer, 
                                sentiment_model, sentiment_tokenizer, 
                                device, ID2TAG, ID2LABEL)
print(results)
```
---

## References

- SemEval 2014 Task 4: Aspect Based Sentiment Analysis
- BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2019)
- Hugging Face Transformers library

---
