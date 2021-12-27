from torch import nn
from map_to_sequence import MapToSequence
from positional_encoding import PositionalEncoding
import timm


class Encoder(nn.Module):
    def __init__(self, encoder_name="resnet18", sequence_length=26, encoder_output_width=7, encoder_output_channels=512, pretrained=True, in_channels=3, batch_first=False, embeddings_index=-1):
        super(Encoder, self).__init__()
        self.feature_extractor = timm.create_model(encoder_name, pretrained=pretrained, in_chans=in_channels, features_only=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 1), stride=1)
        self.map_to_sequence = MapToSequence(in_features=encoder_output_width, out_features=sequence_length, batch_first=batch_first)
        self.pos_encoder = PositionalEncoding(d_model=encoder_output_channels)
        
        self.embeddings_index = embeddings_index
        self.batch_first = batch_first
        
    def forward(self, x):
        output = self.feature_extractor(x)[self.embeddings_index]
        output = self.max_pool(output) 
        output = self.map_to_sequence(output)
        output = self.pos_encoder(output)
        
        return output