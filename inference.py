from dataset import Dataset
import torch
import numpy as np


class Inference:
    def __init__(self, model, max_length, device="cpu"):
        self.model = model
        self.device = device
        self.max_length = max_length
        
    def __call__(self, image):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():            
            predictions = [Dataset.bos_token_index, ]
            inputs = image.to(self.device)

            for i in range(self.max_length):
                memory = self.model.decoder.transformer.encoder(self.model.encoder(inputs))
                
                tgt = torch.tensor(predictions).unsqueeze(dim=0).long().to(self.device)
                
                tgt = self.model.decoder.tgt_embedding(tgt)
                tgt = self.model.decoder.tgt_pos_encoder(tgt)
                
                outputs = self.model.decoder.transformer.decoder(tgt, memory)
                outputs = self.model.decoder.classifier(outputs).squeeze(dim=0)
                
                token = outputs.argmax(dim=1)[-1].item()
                predictions.append(token)
                
                if token == Dataset.eos_token_index:
                    break

        predictions = np.array(predictions)
        return predictions