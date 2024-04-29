from abc import abstractmethod
import math

import torch
import wandb
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

class FeatureMapMetric:
    def __init__(self, layer: str = "", device: str = "cuda"):
        self.layer = layer
        self.device = device
        self.metric_source = torch.zeros(1).to(device=self.device)
        self.metric_target = torch.zeros(1).to(device=self.device)
        self.metric_across = torch.zeros(1).to(device=self.device)
        self.batch_count = 0
    
    @abstractmethod
    def update(self, source_features: torch.Tensor, target_features: torch.Tensor):
        pass

    def reset(self):
        self.metric_source = torch.zeros(1).to(device=self.device)
        self.metric_target = torch.zeros(1).to(device=self.device)
        self.metric_across = torch.zeros(1).to(device=self.device)
        self.batch_count = 0
    
    def return_metrics(self):
        return {
            self.metric_name + "/source_" + self.layer: self.metric_source / self.batch_count,
            self.metric_name + "/target_" + self.layer: self.metric_target / self.batch_count,
            self.metric_name + "/across_" + self.layer: self.metric_across / self.batch_count
            }


class FeatureMapCosineSimilarity(FeatureMapMetric):
    metric_name = "cos_sim"

    def update(self, source_features: torch.Tensor, target_features: torch.Tensor):
        # Update counts for each training batch
        if source_features.size() != target_features.size():
            raise ValueError(f"source_features.size ({source_features.size()}) and target_features.size ({target_features.size()}) must be equal.")
        batch_size = source_features.shape[0]
        # First dim is the sample within the batch
        flattened_source = source_features.reshape(batch_size,-1)
        flattened_target = target_features.reshape(batch_size,-1)
        flattened_stack = torch.cat((flattened_source,flattened_target))
        flattened_norm_stack = (flattened_stack.T / torch.norm(flattened_stack,dim=1)).T
        cosine_matrix = flattened_norm_stack @ flattened_norm_stack.T
        # Only consider the values above the diagonal in the cosine_matrix. Diagonal values are all 1 and below diag is duplicate of upper.
        upper_tri = cosine_matrix.triu(diagonal=1)    
        # Chop the upper tri into 3 parts: comparing source to source, source to target, and target to target.
        cos_sim_source = upper_tri[:batch_size,:batch_size].sum()/math.factorial(batch_size-1)
        cos_sim_target = upper_tri[batch_size:,batch_size:].sum()/math.factorial(batch_size-1) 
        cos_sim_across = upper_tri[:batch_size,batch_size:].sum()/(batch_size**2)
        self.metric_source += cos_sim_source
        self.metric_target += cos_sim_target
        self.metric_across += cos_sim_across
        self.batch_count += 1


class FeatureMapEuclideanDistance(FeatureMapMetric):     
    metric_name = "euc_dist"
    
    def update(self, source_features: torch.Tensor, target_features: torch.Tensor):
        # Update counts for each training batch
        if source_features.size() != target_features.size():
            raise ValueError(f"source_features.size ({source_features.size()}) and target_features.size ({target_features.size()}) must be equal.")
        batch_size = source_features.shape[0]
        # First dim is the sample within the batch
        flattened_source = source_features.reshape(batch_size,-1)
        flattened_target = target_features.reshape(batch_size,-1)

        # Calculate Euclidean distances: norm(a-b)
        dist_source = [torch.norm(flattened_source[idx_a]-flattened_source[idx_b]) for idx_a in range(batch_size) for idx_b in range(batch_size) if idx_a != idx_b]
        dist_target = [torch.norm(flattened_target[idx_a]-flattened_target[idx_b]) for idx_a in range(batch_size) for idx_b in range(batch_size) if idx_a != idx_b]
        dist_across = [torch.norm(flattened_source[idx_s]-flattened_target[idx_t]) for idx_s in range(batch_size) for idx_t in range(batch_size)]
        
        self.metric_source += torch.Tensor(dist_source).mean()
        self.metric_target += torch.Tensor(dist_target).mean()
        self.metric_across += torch.Tensor(dist_across).mean()
        self.batch_count += 1
    
### MMD Loss ###
# Reference: https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py #
class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None, device='cuda'):
        super().__init__()
        self.bandwidth_multipliers = (mul_factor ** (torch.arange(n_kernels, device=device) - n_kernels // 2))
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)

class MMDLoss(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None, device="cuda"):
        super().__init__()
        self.kernel = RBF(n_kernels=n_kernels, mul_factor=mul_factor, bandwidth=bandwidth, device=device)
        self.device = device
        self.mmd_loss = torch.zeros(1, device=device)
        self.batch_count = 0

    def return_metrics(self):
        return {
            "mmd_loss": self.mmd_loss / self.batch_count
        }

    def reset(self):
        self.mmd_loss = torch.zeros(1, device=self.device)
        self.batch_count = 0

    def forward(self, source_features, target_features):
        source_flattened = source_features.view(source_features.size(0), -1).to(self.device)
        target_flattened = target_features.view(target_features.size(0), -1).to(self.device)

        K = self.kernel(torch.vstack([source_flattened, target_flattened]))

        X_size = source_flattened.shape[0]

        XX = K[:X_size, :X_size].mean()
        YY = K[X_size:, X_size:].mean()
        XY = K[:X_size, X_size:].mean()

        mmd_loss = XX - 2 * XY + YY
        self.mmd_loss += mmd_loss
        self.batch_count += 1
        return mmd_loss

### t-SNE algorithm ###
class TSNEVisualizer:
    def __init__(self, n_components=2, perplexity=30.0, init='pca'):
        """
        Initialize t-SNE visualizer, reference: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

        Parameters:
        - n_components (int): The dimension of the embedded space
        - perplexity (float): The number of nearest neighbors
        - init (string): Initialization of embedding
        """

        self.n_components = n_components
        self.perplexity = perplexity
        self.init = init
        self.divergence = []

        # initialize model
        self.model = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            init=self.init
        )

    def preprocess(self, x: list):
        """
        Flattens the features and normalizes them using a standard scaler.
        For normalization, standard scaler is used which standardizes the features to a mean
        of zero and standard deviation of 1.

        Parameters:
        - x (np.ndarray): feature dataset of shape (n_samples, width, height, n_features)
        """
        # concatenate list of tensors
        x = torch.cat(x, dim=0)

        # convert torch to numpy
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        # flatten the data
        batch_size = x.shape[0]
        w, h = x.shape[2], x.shape[3]
        num_features = x.shape[1]
        flattened_x = x.reshape(batch_size, w*h*num_features)

        # normalize the data
        scaler = StandardScaler()
        x_norm = scaler.fit_transform(flattened_x)

        return x_norm

    def run_tsne(self, x: np.ndarray):
        """
        Runs the t-SNE algorithm.

        Parameters:
        - x (np.ndarray): feature dataset of shape (n_samples, n_features)
        """

        # fit the transform
        x_tsne = self.model.fit_transform(x)

        # calcualate divergence
        self.divergence.append(self.model.kl_divergence_)

        return x_tsne

    @staticmethod
    def plot_tsne(features, labels, step):
        """
        Plots the results from the tSNE algorithm.
        """

        # define colors and markers for each domain and plot
        plt.figure(figsize=(8, 6))
        domain_colors = {0: 'blue', 1: 'red'}
        markers = {0: 'o', 1: '^'}
        domain_labels = {0: 'Source', 1: 'Target'}
        for domain in np.unique(labels):
            idx = np.where(labels == domain)
            plt.scatter(features[idx, 0], features[idx, 1], c=domain_colors[domain],
                        label=f'{domain_labels[domain]} Domain', alpha=0.6, marker=markers[domain])

        plt.legend(title='Domain', loc='best', frameon=False)
        plt.title(f't-SNE Visualization at Step {step}')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')

        # save the plot to a buffer using the canvas directly
        fig = plt.gcf()
        plt.draw()
        image_buf = fig.canvas.tostring_rgb()
        image = np.frombuffer(image_buf, dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # log the image to W&B
        wandb.log({"t-SNE Plot": wandb.Image(image, caption=f"t-SNE at Epoch {step}")}, step=step)
        plt.close(fig)