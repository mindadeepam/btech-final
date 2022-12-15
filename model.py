import torch

class custom_model2(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, num_classes, num_layers):
    super(custom_model2, self).__init__()

    self.num_class = num_classes
    self.num_layers = num_layers
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers = self.num_layers)
    self.fc = torch.nn.Linear(hidden_dim, num_classes)

  def forward(self, inputs):

    # h_0 = Variable(torch.zeros(self.num_layers, inputs.size(0), self.hidden_dim))
    # c_0 = Variable(torch.zeros(self.num_layers, inputs.size(0), self.hidden_dim))
    # Pass the input data through the LSTM layer
    output, (h_out, c_out) = self.lstm(inputs)
    logits = self.fc(output[:,-1])

    # h_out = h_out.view(-1, self.hiden_size)
    return logits