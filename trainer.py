from model import *
from AF_Dataset import *
from utils import *
import numpy as np
import torch
from torch import nn
import pickle
import argparse
import torch.optim as optim
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", default='-0.05', type=float, required=False)
parser.add_argument("--c0", default="2000", type=int, required=False)
parser.add_argument("--train_iters", default="1000000", type=int, required=False)
parser.add_argument("--batch_size", default='50', type=int, required=False)
parser.add_argument("--val_iters", default="10000", type=int, required=False)
parser.add_argument("--similarity_factor", default="32", type=int, required=False)
parser.add_argument("--attention_factor", default="4", type=int, required=False)

args = parser.parse_args()

alpha = args.alpha
c0 = args.c0
train_iters = args.train_iters
batch_size = args.batch_size
val_iters = args.val_iters
similarity_factor = args.similarity_factor
attention_factor = args.attention_factor

# nested dict: [user]['pos'/'neg'][movieId] -> rating
train_data  = pickle.load( open( "./data/user_rating_dict_train.p", "rb" ) )
test_data   = pickle.load( open( "./data/user_rating_dict_test.p", "rb" ) )
# Dict movieId : 'title', 'genres'
movies_dict = pickle.load( open( "./data/movies_dict.p", "rb" ) )
# Dict with movieId : 'size', 'userSet'
pos_mv_dict = pickle.load( open( "./data/pos_movie2user_dict.p", "rb" ) )
neg_mv_dict = pickle.load( open( "./data/neg_movie2user_dict.p", "rb" ) )
# set of all movieId's
movieId_set = pickle.load( open( "./data/movieId_set.p", "rb" ) )

n_movies = len(movieId_set)

model = AttentionFewshot(pos_mv_dict,
                         attention_factor=attention_factor,
                         similarity_factor=similarity_factor,
                         n_movies=n_movies)

train_af_dataset = AttentionFewshotML(pos_mv_dict, neg_mv_dict, movieId_set, train_data)
test_af_dataset = AttentionFewshotML(pos_mv_dict, neg_mv_dict, movieId_set, test_data)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)

batch_train_loss = 0
batch_val_loss = 0
batch_train_losses = []
batch_val_losses = []
print('yolo')
t = trange(train_iters, desc=f"Train Loss: {0}   Val Loss: {0}", leave=True)
for i in t:

    model.train()

    u, query, support_set, label = train_af_dataset.__random_sample__()

    # Loss weighting mechanism
    cu = get_loss_weight(u, support_set, train_data, alpha, c0)

    pred = model(query, support_set)
    loss = criterion(pred, label)

    batch_train_loss += cu * loss.item()

    loss.backward()

    if (i + 1) % batch_size == 0:
        optimizer.step()
        optimizer.zero_grad()
        _batch_train_loss = batch_train_loss / batch_size

        t.set_description(f'Train Loss: {batch_train_loss}   Val Loss: {batch_val_loss}')
        t.update()

    if (i + 1) % val_iters == 0:
        batch_val_loss = 0
        for j in range(batch_size):
            model.eval()

            with torch.no_grad():
                u, query, support_set, label = test_af_dataset.__random_sample__()
                cu = get_loss_weight(u, support_set, test_data, alpha, c0)
                pred = model(query, support_set)
                loss = criterion(pred, label)
                batch_val_loss += cu*loss.item() / batch_size

        batch_val_losses.append(batch_val_loss)
        batch_train_losses.append(_batch_train_loss)

        with open('losses.csv', 'a') as fd:
            fd.write(f'{_batch_train_loss},{batch_val_loss}\n')

        t.set_description(f'Train Loss: {batch_train_loss}   Val Loss: {batch_val_loss}')
        t.update()

    if (i+1) % 100000 == 0:
        torch.save(model.state_dict(), f"./models/xp2_{i}.pt")

    batch_train_loss = 0