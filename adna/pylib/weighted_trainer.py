import torch
from torch import nn
from transformers import Trainer


class WeightedTrainer(Trainer):
    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        device = "cuda" if torch.has_cuda else "cpu"
        self.weights = torch.tensor(weights, dtype=torch.float32).to(device)
        self.loss_fn = nn.CrossEntropyLoss(weight=self.weights)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_fn(
            logits.view(-1, self.model.config.num_labels), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss
