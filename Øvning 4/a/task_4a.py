import torch
import torch.nn as nn


class LongShortTermMemoryModel(nn.Module):

    def __init__(self, encoding_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, encoding_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


char_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0.,],  # 0: ' '
    [0., 1., 0., 0., 0., 0., 0., 0.,],  # 1: 'h'
    [0., 0., 1., 0., 0., 0., 0., 0.,],  # 2: 'e'
    [0., 0., 0., 1., 0., 0., 0., 0.,],  # 3: 'l'
    [0., 0., 0., 0., 1., 0., 0., 0.,],  # 4: 'o'
    [0., 0., 0., 0., 0., 1., 0., 0.,],  # 5: 'w'
    [0., 0., 0., 0., 0., 0., 1., 0.,],  # 6: 'r'
    [0., 0., 0., 0., 0., 0., 0., 1.,],  # 7: 'd'
]
encoding_size = len(char_encodings)

index_to_char = [' ', 'h', 'e', 'l', 'o', 'w', 'r', 'd']

x_train = torch.tensor([[char_encodings[0]], [char_encodings[1]], [char_encodings[2]], [char_encodings[3]], [char_encodings[3]], [char_encodings[4]],
                         [char_encodings[0]], [char_encodings[5]], [char_encodings[4]], [char_encodings[6]], [char_encodings[3]], [char_encodings[7]]])  # ' hello world'
y_train = torch.tensor([char_encodings[1], char_encodings[2], char_encodings[3], char_encodings[3], char_encodings[4], char_encodings[0], char_encodings[5], char_encodings[4], char_encodings[6], char_encodings[3], char_encodings[7], char_encodings[0]])  # 'hello world '

model = LongShortTermMemoryModel(encoding_size)

optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
for epoch in range(1000):
    model.reset()
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 9:
        # Generate characters from the initial characters ' h'
        model.reset()
        text = ' h'
        model.f(torch.tensor([[char_encodings[0]]]))
        y = model.f(torch.tensor([[char_encodings[1]]]))
        text += index_to_char[y.argmax(1)]
        for c in range(50):
            y = model.f(torch.tensor([[char_encodings[y.argmax(1)]]]))
            text += index_to_char[y.argmax(1)]
        print(text)
