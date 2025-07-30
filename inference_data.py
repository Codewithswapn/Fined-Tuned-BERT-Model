import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import confusion_matrix

def evaluate_model(model_path="./bert_ag_news_model_final", num_samples=100):
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print("Loading model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load test dataset
    print("Loading test dataset")
    dataset = load_dataset("ag_news", split="test")
    
    # Labels
    labels = ['World', 'Sports', 'Business', 'Sci/Tech']
    
    # Store results
    all_actual = []
    all_predicted = []
    all_texts = []
    
    print(f"\nEvaluating {num_samples} samples...")
    
    # Make predictions
    for i in range(min(num_samples, len(dataset))):
        text = dataset[i]['text']
        actual = dataset[i]['label']
        
        # Tokenize and predict
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predicted = torch.argmax(outputs.logits, dim=1).item()
        
        all_actual.append(actual)
        all_predicted.append(predicted)
        all_texts.append(text)
    
    # correct
    print("\nCorrect prediction by fine-tuned model")
    count = 0
    for i, (text, actual, pred) in enumerate(zip(all_texts, all_actual, all_predicted)):
        if actual == pred and count < 5:
            print(f"\nExample- {count+1}")
            print(f"Text: {text}")
            print(f"Actual: {labels[actual]}")
            print(f"Predicted: {labels[pred]}")
            count += 1
    
    # Worng Prediction
    print("\nWrong prediction by fine-tuned model")
    count = 0
    for i, (text, actual, pred) in enumerate(zip(all_texts, all_actual, all_predicted)):
        if actual != pred and count < 5:
            print(f"\nExample- {count+1}")
            print(f"Text: {text}")
            print(f"Actual: {labels[actual]}")
            print(f"Predicted: {labels[pred]}")
            count += 1
    
    # Confusion matrix
    print("\n\n=== CONFUSION MATRIX ===")
    cm = confusion_matrix(all_actual, all_predicted)
    print("Rows: Actual, Columns: Predicted")
    print(f"{'':>10}", end='')
    for label in labels:
        print(f"{label:>10}", end='')
    print()
    
    for i, label in enumerate(labels):
        print(f"{label:>10}", end='')
        for j in range(len(labels)):
            print(f"{cm[i][j]:>10}", end='')
        print()

if __name__ == "__main__":
    evaluate_model(num_samples=100)