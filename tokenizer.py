
from transformers import AutoTokenizer

def tokenize_data(dataset):

    # Loading Tokenizer model
    model = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model)
    print(f"Tokenizer for '{model}' loaded.")

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)
    
    # Applying tokenization
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Preparing data for PyTorch
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    
    return tokenized_datasets, tokenizer
