import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import time

from .. import Model
from ... import SETTINGS

import os

WORK_PATH = os.getcwd()
__depends__ = {
    "chinese_character": f"{WORK_PATH}/xner/sources/chinese_character.txt"
}

torch.manual_seed(1)
START_TAG = "<START>"
STOP_TAG = "<STOP>"


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = self.log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = self.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = self.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        print(lstm_feats.shape)
        print(lstm_feats)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    @staticmethod
    def argmax(vec):
        # return the argmax as a python int
        _, idx = torch.max(vec, 1)
        return idx.item()

    # Compute log sum exp in a numerically stable way for the forward algorithm
    @classmethod
    def log_sum_exp(cls, vec):
        max_score = vec[0, cls.argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + \
               torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BilstmCrf(Model):
    def __init__(self, embedding_dim=150, hidden_dim=100, lr=0.01, weight_decay=1e-4, epoch=50,
                 expand_words_path=None):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epoch = epoch
        self.expand_words_path = expand_words_path
        if self.expand_words_path is None:
            try:
                self.expand_words_path = __depends__["chinese_character"]
            except:
                print("Do not append other chinese character into the model.")
        self.model = None
        self.word_to_ix = None

        labels = self._prepare_labels()
        self.tag_to_ix = {label: i for i, label in enumerate(labels)}
        self.ix_to_tag = {v: k for k, v in self.tag_to_ix.items()}

    @staticmethod
    def _prepare_labels():
        if SETTINGS["label_type"] == "bmeso":
            label_types = ["B", "M", "E", "S"]
        elif SETTINGS["label_type"] == "biso":
            label_types = ["B", "I", "S"]
        else:
            raise ValueError(f"\"LABEL_TYPE\" should be \"bmeso\" or \"bio\", but currently is \"{SETTINGS['label_type']}\"")

        labels = []
        for label in SETTINGS["labels"]:
            for label_type in label_types:
                if label == "O":
                    labels.append("O")
                    break
                labels.append(f"{label_type}-{label}")
        labels += [START_TAG, STOP_TAG]
        return labels

    @staticmethod
    def _prepare_sequence(seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)

    def train(self, x, y):
        # 更新词表
        self.word_to_ix = {}
        for sentence, tags in zip(x, y):
            for word in sentence:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)

        with open(self.expand_words_path, "r") as f:
            for c in [char for char in f.readlines()[0].strip()]:
                if c not in self.word_to_ix:
                    self.word_to_ix[c] = len(self.word_to_ix)

        self.model = BiLSTM_CRF(len(self.word_to_ix), self.tag_to_ix, self.embedding_dim, self.hidden_dim)
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Make sure prepare_sequence from earlier in the LSTM section is loaded
        self.model.train()
        print("Training:")
        for epoch in range(self.epoch):
            train_start_time = time.time()

            for sentence, tags in zip(x, y):
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.model.zero_grad()

                # Step 2. Get our inputs ready for the network, that is,
                # turn them into Tensors of word indices.
                sentence_in = self._prepare_sequence(sentence, self.word_to_ix)
                targets = torch.tensor([self.tag_to_ix[t] for t in tags], dtype=torch.long)

                # Step 3. Run our forward pass.
                loss = self.model.neg_log_likelihood(sentence_in, targets)

                # Step 4. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()
                loss.backward()
                optimizer.step()

            train_end_time = time.time()

            test_start_time = train_end_time
            train_acc = self.calc_accuracy(self.predict(x), y)
            test_end_time = time.time()

            print(f"Epoch: {epoch}",
                  f"Loss: {round(float(loss.data.numpy()[0]), 3)}",
                  f"TrainACC: {round(train_acc, 3)}",
                  f"TrainTime: {round(train_end_time - train_start_time, 1)}",
                  f"PredTime: {round(test_end_time - test_start_time, 1)}")
        return self

    def predict(self, x):
        test = [x] if isinstance(x[0], str) else x
        self.model.eval()
        with torch.no_grad():
            ret = []
            for t in test:
                torch.manual_seed(1)
                precheck_sent = self._prepare_sequence(t, self.word_to_ix)
                _, y = self.model(precheck_sent)
                ret.append([self.ix_to_tag[i] for i in y])
            return ret

    @staticmethod
    def calc_accuracy(pred, true):
        correct, total = 0, 0
        for pred_, true_ in zip(pred, true):
            for p, t in zip(pred_, true_):
                if true_ != "O":
                    if p == t:
                        correct += 1
                    total += 1
        return correct / total

    def save(self, path):
        torch.save({
            'model': self.model,
            'model_state_dict': self.model.state_dict(),
            'weight_decay': self.weight_decay,
            'lr': self.lr,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'epoch': self.epoch,
            'word_to_ix': self.word_to_ix
        }, path)

    @staticmethod
    def load(path):
        checkpoints = torch.load(path)
        params = {
            "weight_decay": checkpoints["weight_decay"],
            "lr": checkpoints["lr"],
            "embedding_dim": checkpoints["embedding_dim"],
            "hidden_dim": checkpoints["hidden_dim"],
            "epoch": checkpoints["epoch"],
        }
        bcrf = BilstmCrf(**params)
        bcrf.word_to_ix = checkpoints["word_to_ix"]
        bcrf.model = checkpoints["model"]
        return bcrf
