import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
Source: https://github.com/FraLotito/pytorch-continuous-bag-of-words/tree/master
"""

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
EMBEDDING_DIM = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()


# By deriving a set from 'raw_text', we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_idx = {word:idx for idx, word in enumerate(vocab)}
idx_to_word = {idx:word for idx, word in enumerate(vocab)}

data = []
for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
    context = (
        [raw_text[i - j - 1] for j in range(CONTEXT_SIZE)]
        +
        [raw_text[i + j + 1] for j in range(CONTEXT_SIZE)]
    )
    target = raw_text[i]
    data.append((context, target))


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim=128):
        super(CBOW, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)

        self.activation_function1 = nn.ReLU()

        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        self.activation_function2 = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        embeds = sum(self.embedding(inputs)).view(1, -1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out

    def get_word_embedding(self, word):
        word = torch.tensor([word_to_idx[word]])
        return self.embedding(word).view(1, -1)


def make_context_vector(context, word_to_idx):
    idxs = [word_to_idx[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long).to(device)


if __name__ == '__main__':
    model = CBOW(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM).to(device)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Train
    model.train()
    for epoch in range(200):
        total_loss = 0.0
        optimizer.zero_grad()

        for context, target in data:
            context_vector = make_context_vector(context, word_to_idx)
            log_probs = model(context_vector)

            loss = loss_function(log_probs, torch.tensor([word_to_idx[target]], dtype=torch.long).to(device))
            loss.backward()
            total_loss += loss.item()

        # optimize at the end of each epoch
        optimizer.step()

        print(f'Epoch {epoch + 1}: Total Loss = {total_loss:.4f}')

    # Test
    model.eval()
    context = ['People', 'create', 'to', 'direct']
    # context = ['conjure', 'the', 'of', 'the']
    context_vector = make_context_vector(context, word_to_idx)
    result = model(context_vector)

    print(f'\nRaw text: {" ".join(raw_text)}\n')
    print(f'Context: {context}\n')
    print(f'Prediction: {idx_to_word[torch.argmax(result[0]).item()]}')

    # model_state = model.state_dict()
    # embedding_params = model_state['embedding.weight']
    # linear1_params = model_state['linear1.weight'], model_state['linear1.bias']
    # torch.save(embedding_params, './model/embedding_params.pth')
    # torch.save(linear1_params, './model/linear1_params.pth')



