from torch import nn
from positional_encoding import PositionalEncoding


class Decoder(nn.Module):
    def __init__(self, d_model=512, num_classes=26, batch_first=False):
        super(Decoder, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, batch_first=batch_first)
        self.tgt_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=d_model)
        self.tgt_pos_encoder = PositionalEncoding(d_model=d_model)
        self.classifier = nn.Linear(in_features=d_model, out_features=num_classes)
        
        self.batch_first = batch_first
        
    def forward(self, src, tgt):
        tgt = self.tgt_embedding(tgt.long()).float()
        tgt = self.tgt_pos_encoder(tgt)
        
        output = self.transformer(src, tgt)
        output = self.classifier(output)
        
        return output