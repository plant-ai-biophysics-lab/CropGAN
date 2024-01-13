from abc import abstractmethod
import math

import torch
import torch.nn.functional as F

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
    