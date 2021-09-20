import torch
from torch import nn
import pandas


class Embedder(nn.Module):
    def __init__(self,
                 n_factors=4,
                 n_movies=58958):
        """
        Parameters
        ----------
        n_factors : int
            Size of embedding (latent factors).
        n_movies : int
            Number of movies to consider for user.

        Forward:
            x : tensor(k)
                tensor of movieId's for few-shot support

        Output
        ----------
        embeddings : tensor(k,n_factors)
            embedding tensors for movieId's used for support
        """
        super().__init__()
        self.embed = nn.Embedding(n_movies, n_factors)

    def forward(self, x):
        return self.embed(x)


class AttentionModule(nn.Module):
    def __init__(self, embedder):
        """
        Parameters
        ----------
        n_factors : int
            Size of embedding (latent factors).
        embedder  : class Embedder
            pytorch embedding module

        Forward:
            x : tensor(k)
                tensor of movieId's for few-shot support

        Output
        ----------
        Support set weights : torch.tensor
            stochastic weighting of support set for linear combination of similarity values
        """
        super(AttentionModule, self).__init__()
        self.emb = embedder
        self.sm = nn.Softmax(dim=0)

    def forward(self, x):
        x_T = x.clone()
        embed = self.emb(x)
        embed_T = self.emb(x_T)
        embed_T = torch.transpose(embed_T, 0, 1)
        dot = torch.mm(embed, embed_T)
        dot_sum = torch.sum(dot, dim=1)
        out = self.sm(dot_sum)

        return out


class SimilarityFeeder(nn.Module):
    def __init__(self, embedder, movie_user_dict):
        """
        Parameters
        ----------
        embedder : class Embedder
            generates movie embeddings for similarity network
        movie_user_dict: dict('movieId' : dict ( 'size' : int(set_size), 'userSet' : set (UserId's)))
            nested dict with movieId key and set of users that positively interacted with the movie

        Forward:
            query : Tensor(1)
                movieId for query movie
            support_set: Tensor(k)
                tensor of positive movieId's used for few-shot support

        Output
        ----------

        """
        super(SimilarityFeeder, self).__init__()

        self.movie_user_dict = movie_user_dict
        self.embedder = embedder

    def getIoU(self, query, support):
        """Gets Intersection over Union of positively interacting users between query and support"""

        # In cases where no users have rated a movie positively, IoU = 0
        try:
            query_set = self.movie_user_dict[query.item()]['userSet']
            support_set = self.movie_user_dict[support.item()]['userSet']

            intersect_len = len(query_set.intersection(support_set))
            query_len = self.movie_user_dict[query.item()]['size']
            support_len = self.movie_user_dict[support.item()]['size']

            ratio = intersect_len / (query_len + support_len - intersect_len)
        except:
            ratio = 0

        return ratio

    def makeIoUtensor(self, query, support_set):
        """Builds the tensor of IoU values"""

        iou_tens = torch.zeros(len(support_set))

        for idx, mv_id in enumerate(support_set):
            _IoU = self.getIoU(query, mv_id)
            iou_tens[idx] = _IoU

        return iou_tens

    def forward(self, query, support_set):

        support_embeds = self.embedder(support_set)
        # print("support embeds:", support_embeds.shape)
        query_embed = self.embedder(query)

        query_embeds = query_embed.repeat(1, len(support_set)).view(-1, support_embeds.shape[1])
        # print("query embeds:", query_embeds.shape)

        cat_embeds = torch.cat([support_embeds, query_embeds], dim=1)
        # print("cat embeds:", cat_embeds.shape)

        iou_tens = self.makeIoUtensor(query, support_set)

        return cat_embeds, iou_tens.unsqueeze(1)


# We will do the embedding on the outside so we can pass the whole support set as a batch
class SimilarityModule(nn.Module):
    def __init__(self, n_factors=32):
        """
        Parameters
        ----------
        n_factors : int
            size of movie embeddings
        x   : Tensor(n_factors*2)
            concatenatenated query embedding + support embedding
        IoU : Tensor(1)
            IoU value for users that interact with query and support
        Output
        ----------
        sig_out: Tensor(1)
            similarity value between movies
        """
        super(SimilarityModule, self).__init__()

        self.input_size = 2 * n_factors
        self.hidden_size = n_factors

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, int(self.hidden_size / 2))
        self.fc3 = torch.nn.Linear(int(self.hidden_size / 2), 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, IoU):
        hidden_1 = self.fc1(x)
        relu_1 = self.relu(hidden_1)
        hidden_2 = self.fc2(relu_1)
        relu_2 = self.relu(hidden_2)
        mlp_out = self.fc3(relu_2)
        sig = self.sigmoid(mlp_out)

        output = sig + IoU

        return output


class AttentionFewshot(nn.Module):
    def __init__(self, pos_mv_dict, attention_factor=32, similarity_factor=8, n_movies=58958):
        """
        Parameters
        ----------
        attention_factor  : int
            size of attention embeddings
        similarity_factor : int
            size of similarity embeddings
        n_movies.         : int
            number of movies in data

        Forward:
            query       : torch.tensor(1)
                movieId of the query
            support_set : torch.tensor(k)
                movieId's of the support set

        Output
        ----------
        sig_out: Tensor(1)
            similarity value between movies
        """
        super(AttentionFewshot, self).__init__()

        self.attention_factor = attention_factor
        self.similarity_factor = similarity_factor
        self.n_movies = n_movies

        attention_embedder = Embedder(n_factors=attention_factor, n_movies=n_movies)
        similarity_embedder = Embedder(n_factors=similarity_factor, n_movies=n_movies)

        self.attention_module = AttentionModule(attention_embedder)
        self.similarity_feeder = SimilarityFeeder(similarity_embedder, pos_mv_dict)
        self.similarity_module = SimilarityModule(n_factors=similarity_factor)

        self.sig = torch.nn.Sigmoid()

    def forward(self, query, support_set):
        cat_embs, IoUs = self.similarity_feeder(query, support_set)
        sim_out = self.similarity_module(cat_embs, IoUs)
        att_out = self.attention_module(support_set)

        out = torch.dot(att_out, sim_out.squeeze(1))
        sig_out = self.sig(out)

        return sig_out