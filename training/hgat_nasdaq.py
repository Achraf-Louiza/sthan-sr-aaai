from torch_geometric import nn
import torch
import torch.nn.functional as F
import torch.nn

class Attention(torch.nn.Module):
    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = torch.nn.Linear(dimensions, dimensions, bias=False).to('cuda')

        self.linear_out = torch.nn.Linear(dimensions * 2, dimensions, bias=False).to('cuda')
        self.softmax = torch.nn.Softmax(dim=-1).to('cuda')
        self.tanh = torch.nn.Tanh().to('cuda')
        self.ae = torch.nn.Parameter(torch.FloatTensor(1026, 1, 1).to('cuda'))
        self.ab = torch.nn.Parameter(torch.FloatTensor(1026, 1, 1).to('cuda'))

    def forward(self, query, context):
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        mix = attention_weights * context.permute(0, 2, 1)

        delta_t = torch.flip(torch.arange(0, query_len), [0]).type(torch.float32).to('cuda')
        delta_t = delta_t.repeat(1026, 1).reshape(1026, 1, query_len).to('cuda')
        bt = torch.exp(-1 * self.ab * delta_t)
        term_2 = F.relu(self.ae * mix * bt)
        mix = torch.sum(term_2 + mix, -1).unsqueeze(1)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights

class gru(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(gru, self).__init__()
        self.gru1 = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True).to('cuda')

    def forward(self, inputs):
        full, last = self.gru1(inputs)
        return full, last

class HGAT(torch.nn.Module):
    def __init__(self, tickers):
        super(HGAT, self).__init__()
        self.tickers = tickers
        self.grup = gru(5, 32)
        self.attention = Attention(32)
        self.hatt1 = nn.HypergraphConv(32, 32, use_attention=True, heads=4, concat=False, negative_slope=0.2, dropout=0.5, bias=True).to('cuda')
        self.hatt2 = nn.HypergraphConv(32, 32, use_attention=True, heads=1, concat=False, negative_slope=0.2, dropout=0.5, bias=True).to('cuda')
        self.linear = torch.nn.Linear(32, 1).to('cuda')

    def forward(self, price_input, e):
        # Ensure tensors are on the correct device
        price_input = price_input.to('cuda')
        e = e.to('cuda')
        context, query = self.grup(price_input)
        query = query.reshape(1026, 1, 32).to('cuda')
        output, weights = self.attention(query, context)
        output = output.reshape((1026, 32))        
        num_edges = e.max().item() + 1
        dummy_edge_attr = torch.ones(num_edges, 32, device='cuda') # Ensure tensor is on the correct device
        x = F.leaky_relu(self.hatt1(output, e, hyperedge_attr=dummy_edge_attr), 0.2)
        x = F.leaky_relu(self.hatt2(x, e, hyperedge_attr=dummy_edge_attr), 0.2)
        return F.leaky_relu(self.linear(x))
