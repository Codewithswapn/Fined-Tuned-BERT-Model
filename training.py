import torch
from torch.utils.data import DataLoader
from torch.optim import Adam  
from sklearn.metrics import classification_report, accuracy_score
from data_loading import data_loading
from tokenizer import tokenize_data
from model import get_model

def train():

    # GPU Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    raw_dataset = data_loading()
    processed_dataset, tokenizer = tokenize_data(raw_dataset)
    model = get_model()
    model.to(device)
    
    # DataLoaders
    train_dataloader = DataLoader(processed_dataset["train"], shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(processed_dataset["test"], batch_size=8)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=2e-5)

    # Training Loop 
    print("\n--- Starting Training ---")
    num_epochs = 4

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        
        # Training
        model.train()
        total_train_loss = 0

        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # Evaluation 
        model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
        
            for batch in eval_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"Evaluation Accuracy: {accuracy:.4f}")

    # Performance Report
    print("\nModel Performance Report")
   
    label_names = raw_dataset['train'].features['label'].names
    print(classification_report(all_labels, all_predictions, target_names=label_names))

    # Save Model and Tokenizer
    print("\n--- Saving Model and Tokenizer ---")
    model.save_pretrained("./bert_ag_news_model_final")
    tokenizer.save_pretrained("./bert_ag_news_model_final")

    print("Model and tokenizer saved")
   

train()
