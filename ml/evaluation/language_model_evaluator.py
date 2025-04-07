import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from ml.evaluation.metrics import EvaluationTracker


class LanguageModelEvaluator:
    """
    Evaluator for language models in a continual learning setting.
    
    This evaluator can compute:
    - Perplexity
    - Accuracy (next token prediction)
    - BLEU score (for generative models)
    - Cross-entropy loss
    """
    
    def __init__(self, 
                 device: torch.device = torch.device("cpu"),
                 log_to_mlflow: bool = True):
        """
        Initialize the language model evaluator.
        
        Args:
            device: Device to perform evaluation on
            log_to_mlflow: Whether to log metrics to MLflow
        """
        self.device = device
        self.tracker = EvaluationTracker(log_to_mlflow=log_to_mlflow)
        
    def evaluate_task(self, 
                      model: nn.Module, 
                      dataloader: torch.utils.data.DataLoader,
                      task_name: str,
                      criterion: Optional[nn.Module] = None) -> Dict[str, float]:
        """
        Evaluate a model on a specific task.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader containing evaluation data
            task_name: Name of the task
            criterion: Loss function to use (defaults to CrossEntropyLoss if None)
            
        Returns:
            Dictionary containing metrics
        """
        model.eval()
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {task_name}"):
                inputs, targets = self._process_batch(batch)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                
                # Calculate loss
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Some models return multiple outputs
                    
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                total_loss += loss.item() * targets.size(0)
                
                # Calculate accuracy
                preds = torch.argmax(outputs, dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.numel()
                
                # Store predictions and targets for confusion matrix
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
        
        # Calculate metrics
        avg_loss = total_loss / total
        accuracy = correct / total
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Combine predictions and targets
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        # Get vocabulary size for confusion matrix (assuming output dimension = vocab size)
        try:
            vocab_size = outputs.size(-1)
            self.tracker.update_confusion_matrix(task_name, all_preds, all_targets, vocab_size)
        except Exception:
            pass  # Skip confusion matrix if we can't determine vocab size
            
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "perplexity": perplexity
        }
        
        self.tracker.log_task_metrics(task_name, metrics)
        
        return metrics
    
    def evaluate_all_tasks(self, 
                          model: nn.Module, 
                          dataloaders: Dict[str, torch.utils.data.DataLoader],
                          current_task_idx: int,
                          criterion: Optional[nn.Module] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a model on all tasks seen so far.
        
        Args:
            model: Model to evaluate
            dataloaders: Dictionary mapping task names to DataLoaders
            current_task_idx: Index of the current task
            criterion: Loss function to use
            
        Returns:
            Dictionary mapping task names to metric dictionaries
        """
        results = {}
        task_accuracies = {}
        
        for task_name, dataloader in dataloaders.items():
            metrics = self.evaluate_task(model, dataloader, task_name, criterion)
            results[task_name] = metrics
            task_accuracies[task_name] = metrics["accuracy"]
            
        # Update continual learning metrics
        self.tracker.update_cl_metrics(current_task_idx, task_accuracies)
        
        return results
    
    def compute_bleu_score(self, 
                          model: nn.Module, 
                          dataloader: torch.utils.data.DataLoader,
                          task_name: str,
                          tokenizer: Any,
                          max_length: int = 50,
                          beam_size: int = 1) -> float:
        """
        Compute BLEU score for a generative model.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader containing evaluation data
            task_name: Name of the task
            tokenizer: Tokenizer to decode generated text
            max_length: Maximum sequence length to generate
            beam_size: Beam size for beam search (1 = greedy)
            
        Returns:
            BLEU score
        """
        try:
            from nltk.translate.bleu_score import corpus_bleu
        except ImportError:
            print("NLTK not found, install with: pip install nltk")
            return 0.0
            
        model.eval()
        references = []
        hypotheses = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Computing BLEU for {task_name}"):
                input_ids, target_ids = self._process_batch(batch)
                input_ids = input_ids.to(self.device)
                
                # Generate text
                if beam_size > 1 and hasattr(model, "generate"):
                    # Use model's generate method with beam search if available
                    outputs = model.generate(
                        input_ids,
                        max_length=max_length,
                        num_beams=beam_size,
                        early_stopping=True
                    )
                else:
                    # Simple greedy decoding
                    outputs = self._greedy_decode(model, input_ids, max_length)
                    
                # Decode to text
                for i in range(len(outputs)):
                    # Reference is the target
                    if hasattr(tokenizer, "decode"):
                        reference = tokenizer.decode(target_ids[i].tolist(), skip_special_tokens=True)
                        hypothesis = tokenizer.decode(outputs[i].tolist(), skip_special_tokens=True)
                    else:
                        # Fallback if tokenizer doesn't have decode method
                        reference = " ".join([str(x) for x in target_ids[i].tolist()])
                        hypothesis = " ".join([str(x) for x in outputs[i].tolist()])
                        
                    # Convert to tokens for BLEU calculation
                    reference_tokens = reference.split()
                    hypothesis_tokens = hypothesis.split()
                    
                    references.append([reference_tokens])
                    hypotheses.append(hypothesis_tokens)
                    
        # Calculate BLEU score
        bleu_score = corpus_bleu(references, hypotheses)
        
        # Log metric
        self.tracker.log_task_metrics(task_name, {"bleu": bleu_score})
        
        return bleu_score
    
    def _process_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a batch to extract inputs and targets.
        
        Supports different batch formats:
        - tuple/list: (inputs, targets)
        - dict: {'input_ids': inputs, 'labels': targets}
        
        Args:
            batch: Batch of data
            
        Returns:
            Tuple of (inputs, targets)
        """
        if isinstance(batch, (tuple, list)):
            return batch[0], batch[1]
        elif isinstance(batch, dict):
            inputs = batch.get('input_ids', batch.get('inputs', None))
            targets = batch.get('labels', batch.get('targets', None))
            
            if inputs is None or targets is None:
                raise ValueError("Could not extract inputs and targets from batch dictionary")
                
            return inputs, targets
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")
    
    def _greedy_decode(self, model: nn.Module, input_ids: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Perform greedy decoding for text generation.
        
        Args:
            model: Model to use for decoding
            input_ids: Input token IDs
            max_length: Maximum generation length
            
        Returns:
            Generated token IDs
        """
        batch_size = input_ids.size(0)
        output_ids = input_ids.clone()
        
        for _ in range(max_length - input_ids.size(1)):
            outputs = model(output_ids)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            output_ids = torch.cat([output_ids, next_token], dim=1)
            
            # Stop if all sequences have reached end-of-sequence token
            # (this assumes an EOS token ID of 2, adjust as needed)
            if (next_token == 2).all():
                break
                
        return output_ids
    
    def log_final_metrics(self):
        """Log final summary metrics."""
        self.tracker.log_final_metrics()
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all evaluation metrics.
        
        Returns:
            Dictionary containing summary of all metrics
        """
        return self.tracker.get_summary() 