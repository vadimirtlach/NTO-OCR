from torch import nn
from model_components import Encoder, Decoder


class Model(nn.Module):
    def __init__(self, num_classes=26, batch_first=False):
        super(Model, self).__init__()
        self.encoder = Encoder(encoder_name="seresnet34", sequence_length=25, in_channels=1, batch_first=batch_first)
        self.decoder = Decoder(num_classes=num_classes, batch_first=batch_first)
        
    def forward(self, src, tgt):
        output = self.encoder(src)
        output = self.decoder(output, tgt)
        
        return output