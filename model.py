
from transformers import AutoModelForSequenceClassification

# Load pre-trained BERT model
def get_model():
    
    print("\n--- Loading Model ---")
    model = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=4)
    print(f"Model '{model}' loaded successfully")
   
    return model
