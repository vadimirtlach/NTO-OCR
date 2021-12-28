from dataset import Dataset
import torch


class Evaluator:
    def __init__(self, model, loss, num_classes, device="cpu"):
        self.model = model
        self.loss = loss
        self.device = device
        self.num_classes = num_classes
    
    def __call__(self, loader):
        loss = 0
        
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for batch in loader:
                batch_inputs, batch_labels = Dataset.collate_batch(batch, inputs_device=self.device, labels_device=self.device)
            
                batch_outputs = self.model(batch_inputs, batch_labels[:, :-1]).contiguous()
                batch_outputs_ = batch_outputs.view(-1, self.num_classes).float()
                batch_labels_ = batch_labels[:, 1:].contiguous().view(-1).long()

                batch_loss = self.loss(batch_outputs_, batch_labels_)
                loss += batch_loss
                
            loss /= len(loader)
            
        return loss