class CRNN(nn.Module):
  def __init__(self, encoder_name='seresnet34', num_classes=111):
    super(CRNN, self).__init__()

    self.feature_extractor = timm.create_model(encoder_name, pretrained=True,
                                               in_chans=1, features_only=True)
    self.max_pool = nn.MaxPool2d(kernel_size=(2, 1), stride=1)
    self.map_to_sequence = nn.Sequential(
        nn.Linear(in_features=7, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=1024),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=2048),
        nn.ReLU()
        )

    self.lstm = nn.LSTM(input_size=2048, hidden_size=256, bidirectional=True,
                        num_layers=4, dropout=.1, batch_first=False)
    
    self._weights_init(self.lstm)
    self.head = nn.Sequential(
        nn.Linear(in_features=(self.lstm.hidden_size * (int(self.lstm.bidirectional) + 1)), out_features=self.lstm.hidden_size),
        nn.RReLU(),
        nn.Linear(in_features=self.lstm.hidden_size, out_features=num_classes),
        nn.LogSoftmax(dim=2)
    )
  
  def _weights_init(self, module):
    if isinstance(module, nn.LSTM):
      nn.init.xavier_normal_(module.weight_ih_l0)
      nn.init.xavier_normal_(module.weight_hh_l0)
      if module.bidirectional:
        nn.init.xavier_normal_(module.weight_ih_l0_reverse)
        nn.init.xavier_normal_(module.weight_hh_l0_reverse)
  
  def forward(self, x):
    x = self.feature_extractor(x)[-1]
    x = self.max_pool(x).squeeze(dim=2)
    x = self.map_to_sequence(x).permute(2, 0, 1)
    x, _ = self.lstm(x)
    x = x.permute(1, 0, 2)
    x = self.head(x).permute(1, 0, 2)
    return x
