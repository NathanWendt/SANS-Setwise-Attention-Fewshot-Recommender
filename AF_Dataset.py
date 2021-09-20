from torch.utils.data import Dataset, DataLoader
import torch
import random
import numpy as np
from tqdm import tqdm

class AttentionFewshotML(Dataset):
    """AttentionFewshot dataset from MovieLens20m."""

    def __init__(self, pos_movie_dict, neg_movie_dict, movieId_set, user_posneg, k=3):
        """
        Args:
            pos_movie_dict (dict): {movieId : {userId's}}
            neg_movie_dict (dict): {movieId : {userId's}}
            data (dict): {userId : {'pos' : {movieId : rating}, 'neg' : {movieId : rating} } }.
        """
        self.pos_movie_dict = pos_movie_dict
        self.neg_movie_dict = neg_movie_dict
        self.movieId_set = movieId_set
        self.user_posneg = user_posneg
        self.um_pairs = []

        for user, posneg_dict in tqdm(self.user_posneg.items()):
            pos_len = len(posneg_dict['pos'])
            if pos_len > k + 1:
                for ke, va in posneg_dict['pos'].items():
                    self.um_pairs.append((user, ke))

        self.len = len(self.um_pairs)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        user, pos_mv = self.um_pairs[idx]
        pos_set = set(self.user_posneg[user]['pos'].keys())
        pos_set.remove(pos_mv)
        support = torch.tensor(random.sample(pos_set, 3))
        pos_set.add(pos_mv)
        rand = np.random.rand()

        # return positive sample
        if rand > 0.5:
            pos_set.remove(pos_mv)

            sample = (user, torch.tensor(pos_mv), support, torch.tensor(1))

        # return negative sample
        else:
            neg_set = set(self.user_posneg[user]['neg'].keys())
            if neg_set:
                # stochastically sample between neg set and random movie
                prob = 1 / len(neg_set)
                prob = min(1 / 2, prob)
                rand = np.random.rand()
                if rand > prob:
                    neg_mv = random.sample(neg_set, 1)

                # This is so ugly right now... divide by 0's and whatnot
                else:
                    random_movie = self.movieId_set.symmetric_difference(pos_set)
                    neg_mv = random.sample(random_movie, 1)
            else:
                random_movie = self.movieId_set.symmetric_difference(pos_set)
                neg_mv = random.sample(random_movie, 1)

            sample = (user, torch.tensor(neg_mv[0]), support, torch.tensor(0))

        return sample

    def __random_sample__(self):
        idx = np.random.randint(self.len)

        user, pos_mv = self.um_pairs[idx]
        pos_set = set(self.user_posneg[user]['pos'].keys())
        pos_set.remove(pos_mv)
        support = torch.tensor(random.sample(pos_set, 3))
        pos_set.add(pos_mv)
        rand = np.random.rand()

        # return positive sample
        if rand > 0.5:
            pos_set.remove(pos_mv)

            sample = (user, torch.tensor([pos_mv]), support, torch.tensor(1.))

        # return negative sample
        else:
            neg_set = set(self.user_posneg[user]['neg'].keys())
            if neg_set:
                # stochastically sample between neg set and random movie
                prob = 1 / len(neg_set)
                prob = min(1 / 2, prob)
                rand = np.random.rand()
                if rand > prob:
                    neg_mv = random.sample(neg_set, 1)

                # This is so ugly right now... divide by 0's and whatnot
                else:
                    random_movie = self.movieId_set.symmetric_difference(pos_set)
                    neg_mv = random.sample(random_movie, 1)
            else:
                random_movie = self.movieId_set.symmetric_difference(pos_set)
                neg_mv = random.sample(random_movie, 1)

            sample = (user, torch.tensor([neg_mv[0]]), support, torch.tensor(0.))

        return sample