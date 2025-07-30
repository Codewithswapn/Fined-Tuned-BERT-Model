import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def classify_text(text, model_path="./bert_ag_news_model_final"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Saved Tokenizer and Model Loading
    print("--- Loading Model and Tokenizer for Inference ---")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval() 

    # Prepare text for the input to the trained BERT model
    inputs = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
    
    # Data moved to Device 
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # The output logits are the raw scores for each class. We get the index of the highest score.
    prediction_index = torch.argmax(outputs.logits, dim=1).item()

    # Maping Index to Labels from the dataset
    label_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    predicted_label = label_names[prediction_index]
    
    return predicted_label


# sample_text = "SpaceX launches a new rocket to the International Space Station with a crew of four astronauts."
sample_text = "Prof. Andreas Fischer and Zineddine Bettouche are best at teaching Generative AI Frameworks"
sample_text = "Portugal wins the UEFA National League title"
# sample_text = "Germany provides the world class education in STEM area"
# sample_text = "Tesla's new factory boosts production"
# sample_text = "China's economic policy affects global markets"

predicted_category = classify_text(sample_text)

print(f"\nText to Classify: '{sample_text}'")
print(f"Predicted Category: {predicted_category}")