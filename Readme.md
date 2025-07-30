
# News Article Classification using BERT

This project classifies news articles into four categories — **World**, **Sports**, **Business**, and **Sci/Tech** — using a fine-tuned `distilbert-base-uncased` model trained on the AG News dataset.

---

## Project Structure

- `data_loading.py`: Loads the AG News dataset and visualizes the class distribution for train and test dataset.
- `tokenizer.py`: Tokenizes the text data for input to the BERT model.
- `model.py`: Loads the pre-trained `distilbert-base-uncased`.
- `training.py`: Train and fine-tune the model, evaluate on validation set, and save the final model.
- `inference.py`: This file loads the saved trained model and tokenizer and classify new articles using the trained model.

---

## Dataset

The model is trained on the **AG News** dataset, which includes:

-`Dataset_Info.png`

- **Training Set**: 120,000 articles  
- **Testing Set**: 7,600 articles  

The dataset is balanced across all four categories. Below are the images for both train and text data per class distribution.

- `class_distribution_train.png`  
- `class_distribution_test.png`  
---

### Prerequisites

- Python 3.8+

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training the Model

Run the training script to fine-tune the model:

```bash
python training.py
```

```
--- Starting Training ---

--- Epoch 1/4 ---
Average Training Loss: 0.2141
Evaluation Accuracy: 0.9416

--- Epoch 2/4 ---
Average Training Loss: 0.1318
Evaluation Accuracy: 0.9476

--- Epoch 3/4 ---
Average Training Loss: 0.0900
Evaluation Accuracy: 0.9424

--- Epoch 4/4 ---
Average Training Loss: 0.0605
Evaluation Accuracy: 0.9439
```

The fine-tuned model will be saved to:

```
./bert_ag_news_model_final
```

---

## Inference Result

```python
sample_text = "Portugal wins the UEFA Nations League title"

sample_text = "Germany provides the world class education in STEM area"

sample_text = "Prof. Andreas Fischer and Zineddine Bettouche are best at teaching Generative AI Frameworks"
```

Then run:

```bash
python inference.py
```

### Inference Output

```
Text to Classify: 'Portugal wins the UEFA Nations League title'
Predicted Category: Sports

Text to Classify: 'Germany provides the world class education in STEM area'
Predicted Category: Sci/Tech

Text to Classify: 'Prof. Andreas Fischer and Zineddine Bettouche are best at teaching Generative AI Frameworks'
Predicted Category: Sci/Tech

```

---

## Model Performance

After 4 training epochs, the model achieves an **overall accuracy of \~94%** on the test set.

- `\Model_Performances.png` 




