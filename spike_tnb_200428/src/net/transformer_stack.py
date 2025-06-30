from torch import nn

class TransformerStack(nn.Module):

    def __init__(self, n_input, n_layer_list, n_hidden_list, n_output):
        super().__init__()
        assert len(n_layer_list) == len(n_hidden_list)
        n_hidden_list.insert(0, n_input)

        transformer_stack = []
        iter = zip(n_layer_list, n_hidden_list[:-1], n_hidden_list[1:])
        for n_layer, n_hidden_pre, n_hidden_post in iter:
            transformer = nn.TransformerEncoderLayer(
                d_model=n_hidden_pre,
                nhead=8,
                dim_feedforward=n_hidden_post,
                dropout=0.1,
                batch_first=True)
            transformer_stack.append(transformer)
        self.transformer_stack = nn.ModuleList(transformer_stack)
        self.fc = nn.Linear(n_hidden_list[-1], n_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, hidden_list=None):
        if hidden_list is None:
            hidden_list = [None] * len(self.transformer_stack)
        if len(hidden_list) != len(self.transformer_stack):
            raise ValueError()
        
        new_hidden_list = []
        outputs = inputs
        for transformer, hidden in zip(self.transformer_stack, hidden_list):
            outputs = transformer(outputs)
            new_hidden_list.append(hidden)

        x = self.fc(outputs)
        x = self.sigmoid(x)
        return x, new_hidden_list
        