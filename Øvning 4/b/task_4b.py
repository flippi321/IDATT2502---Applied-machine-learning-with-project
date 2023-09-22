import numpy as np
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

emojis = {
    'hat': '\U0001F3A9',
    'rat': '\U0001F400',
    'cat': '\U0001F408',
    'flat': '\U0001F3E2',
    'matt': '\U0001F468',
    'cap': '\U0001F9E2',
    'son': '\U0001F466'
}

index_to_emoji =[value for _,value in emojis.items()]

index_to_char = [' ', 'h', 'a', 't', 'r','c', 'f', 'l', 'm', 'p', 's', 'o', 'n']

# Using this is so much simpler than making the ararys directly
char_encodings = np.eye(len(index_to_char))
encoding_size = len(char_encodings)
emojies = np.eye(len(emojis))
emoji_encoding_size=len(emojies)

encoding_size = len(char_encodings)
emoji_encoding_size=len(emojies)


letters ={}

for i, letter in enumerate(index_to_char):
        letters[letter] = char_encodings[i]


x_train = torch.tensor([
        [[letters['h']], [letters['a']], [letters['t']], [letters[' ']]],
        [[letters['r']], [letters['a']], [letters['t']], [letters[' ']]],
        [[letters['c']], [letters['a']], [letters['t']], [letters[' ']]],
        [[letters['f']], [letters['l']], [letters['a']], [letters['t']]],
        [[letters['m']], [letters['a']], [letters['t']], [letters['t']]],
        [[letters['c']], [letters['a']], [letters['p']], [letters[' ']]],
        [[letters['s']], [letters['o']], [letters['n']], [letters[' ']]],
        ], 
        dtype=torch.float)


y_train = torch.tensor([
        [emojies[0], emojies[0], emojies[0], emojies[0]] ,
        [emojies[1], emojies[1], emojies[1], emojies[1]],
        [emojies[2], emojies[2], emojies[2], emojies[2]],
        [emojies[3], emojies[3], emojies[3], emojies[3]],
        [emojies[4], emojies[4], emojies[4], emojies[4]],
        [emojies[5], emojies[5], emojies[5], emojies[5]],
        [emojies[6], emojies[6], emojies[6], emojies[6]]], 
        dtype=torch.float)

model = LongShortTermMemoryModel(encoding_size)

optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
for epoch in range(500):
    for i in range(x_train.size()[0]):
        model.reset()
        model.loss(x_train[i], y_train[i]).backward()
        optimizer.step()
        optimizer.zero_grad()

def generate_emoji(string):
    y = -1
    model.reset()
    for i in range(len(string)):
        char_index = index_to_char.index(string[i])
        y = model.f(torch.tensor([[char_encodings[char_index]]], dtype=torch.float))
    return(index_to_emoji[y.argmax(0)])

print("\n---Testing for all letters---")
for letter in range(1, len(index_to_char)-1):
    text = index_to_char[letter]
    print(f"{text} : {generate_emoji(text)}")

print("\n---rt & rats---")
print(f"rt   : {generate_emoji('rt')}")
print(f"rats : {generate_emoji('rats')}")