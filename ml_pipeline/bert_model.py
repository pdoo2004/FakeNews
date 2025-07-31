import torch
import numpy as np
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import os

logger = logging.getLogger(__name__)

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertModel:
    def __init__(self, model_name='bert-base-uncased', max_length=512, output_dir='./bert_output'):
        self.model_name = model_name
        self.max_length = max_length
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading tokenizer: {model_name}")
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None
        self.trainer = None
        
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, train_texts, train_labels, val_texts, val_labels, 
              num_epochs=3, batch_size=16, learning_rate=2e-5):
        """Train BERT model for fake news detection"""
        
        logger.info(f"Initializing BERT model: {self.model_name}")
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2
        ).to(self.device)
        
        # Create datasets
        train_dataset = FakeNewsDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = FakeNewsDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=2,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
            dataloader_pin_memory=False,
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=self.compute_metrics,
        )
        
        logger.info("Starting BERT training...")
        self.trainer.train()
        
        # Evaluate
        eval_results = self.trainer.evaluate()
        logger.info(f"Validation results: {eval_results}")
        
        return eval_results
    
    def predict(self, texts):
        """Make predictions on new texts"""
        if self.model is None:
            raise ValueError("Model must be trained or loaded before making predictions")
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(**encoding)
                prediction = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
                predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, texts):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model must be trained or loaded before making predictions")
        
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(**encoding)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
                probabilities.append(probs)
        
        return np.array(probabilities)
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(filepath, exist_ok=True)
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        logger.info(f"BERT model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        if os.path.exists(filepath):
            self.model = BertForSequenceClassification.from_pretrained(filepath).to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(filepath)
            logger.info(f"BERT model loaded from {filepath}")
        else:
            raise FileNotFoundError(f"Model directory not found: {filepath}") 