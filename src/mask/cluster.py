from tqdm import tqdm
import torch
import scipy.cluster.hierarchy as sch
from src.utils.flops import flops_per_head, flops_per_neuron
import numpy as np

def derive_head_tree(config, ATTN_O_input):
    layer_num = config["layer_num"]
    head_num = config["head_num"]
    head_size = config["head_size"]

    distance = ()
    trees = ()
    for layer_idx in tqdm(range(layer_num), desc="Clustering for MHSA"):
        head_input = ()
        # 可优化
        for head_idx in range(head_num):
            head_input = head_input + (ATTN_O_input[layer_idx][:, head_idx*head_size:(head_idx+1)*head_size],)
        
        # Calculate the similarity between each two head
        sim = torch.zeros(head_num, head_num)
        for head_idx1 in range(head_num):
            for head_idx2 in range(head_num):
                if head_idx1 == head_idx2:
                    # The similarity is set to 1.0 between the same head
                    sim[head_idx1, head_idx2] = 1.0
                else:
                    # Calculate the similarity of each two dimensions of head1 and head2
                    sim_dim = torch.zeros(head_size, head_size)
                    for dim_idx in range(head_size):
                        dim_head1 = head_input[head_idx1][:, dim_idx:dim_idx+1].repeat(1, head_size)
                        head2 = head_input[head_idx2]
                        sim_dim[dim_idx, :] = torch.nn.functional.cosine_similarity(dim_head1.t(), head2.t())
                    sim_dim = torch.abs(sim_dim)
                    
                    # attention dimension mapping
                    max_sim_list = []
                    for i in range(head_size):
                        max_sim = torch.max(sim_dim)
                        max_sim_list.append(max_sim)
                        row = torch.argmax(sim_dim) // head_size
                        col = torch.argmax(sim_dim) % head_size
                        sim_dim[row, :] = 0.0
                        sim_dim[:, col] = 0.0
                    mean_cos = sum(max_sim_list) / len(max_sim_list)
                    sim[head_idx1, head_idx2] = mean_cos
        dist = torch.ones(head_num, head_num) - sim
        distance = distance + (dist, )
        tree = sch.linkage(dist, method='complete', metric='chebyshev')
        trees = trees + (tree, )
    return distance, trees

def derive_neuron_tree(config, FFN_2_input):
    layer_num = config["layer_num"]
    d_ffn = config["d_ffn"]
    trees = ()
    distance = ()
    for layer_idx in tqdm(range(layer_num), desc="Clustering for FFN"):
        neuron2 = FFN_2_input[layer_idx]
        sim = torch.zeros(d_ffn, d_ffn)
        for dim_idx1 in range(d_ffn):
            neuron1 = FFN_2_input[layer_idx][:, dim_idx1:dim_idx1 + 1]
            neuron1 = neuron1.repeat(1, d_ffn)
            sim_dim = torch.nn.functional.cosine_similarity(neuron1.t(), neuron2.t())
            sim[dim_idx1, :] = sim_dim
        dist = torch.ones(d_ffn, d_ffn) - torch.abs(sim)
        tree = sch.linkage(dist, method='complete', metric='chebyshev')
        trees = trees + (tree, )
        distance = distance + (dist, )
    return distance, trees
    
def calculate_retained_head(seq_len, config, target_FLOPs, trees, attn_input):
    layer_num = config["layer_num"]
    d_emb = config["d_emb"]
    head_num = config["head_num"]
    head_size = config["head_size"]

    # Determine the threshold to conduct clustering for attention
    th = 0.0
    step = 0.0005
    current_FLOPs = flops_per_head(seq_len, d_emb, head_size) * head_num * layer_num
    while current_FLOPs > target_FLOPs:
        current_FLOPs = 0
        for layer_idx in range(layer_num):
            cluster = sch.fcluster(trees[layer_idx], t=th, criterion='distance')
            current_head_num = np.max(cluster)
            current_FLOPs += flops_per_head(seq_len, d_emb, head_size) * current_head_num
        if th >= 1.0:
            assert False
        th += step
    
    # Conduct clustering based on the threshold and derive the preserved head indices
    head_mask = torch.zeros(layer_num, head_num)
    for layer_idx in range(layer_num):
        cluster = sch.fcluster(trees[layer_idx], t=th, criterion='distance')
        cluster_center = []
        for cluster_idx in range(np.max(cluster)):
            indices_in_cluster = np.where(cluster == cluster_idx + 1)[0]
            if indices_in_cluster.size == 1:
                cluster_center.append(indices_in_cluster[0])
            else:
                mean_mag = torch.zeros(len(indices_in_cluster))
                for i, head_idx in enumerate(range(len(indices_in_cluster))):
                    head = attn_input[layer_idx][:, head_idx*head_size:(head_idx+1)*head_size]
                    mean_mag[i] = torch.mean(torch.mean(torch.abs(head), dim=0), dim=0)
                max_mag_index = torch.argmax(mean_mag)
                cluster_center.append(indices_in_cluster[max_mag_index])

        head_mask[layer_idx][sorted(cluster_center)] = 1
    return th, head_mask

def calculate_retained_neuron(seq_len, config, target_FLOPs, trees, ffn_input):
    layer_num = config["layer_num"]
    d_ffn = config["d_ffn"]
    d_emb = config["d_emb"]

    th = 0.0
    step = 0.0005
    # Determine the threshold to conduct clustering for FFN
    current_FLOPs = flops_per_neuron(seq_len, d_emb) * d_ffn * layer_num
    while current_FLOPs > target_FLOPs:
        current_FLOPs = 0
        for layer_idx in range(layer_num):
            cluster = sch.fcluster(trees[layer_idx], t=th, criterion='distance')
            pruned_d_ffn = np.max(cluster)
            current_FLOPs += flops_per_neuron(seq_len, d_emb) * pruned_d_ffn
        th += step
    
    # conduct clustering based on the threshold and derive the preserved neuron indices
    neuron_mask = torch.zeros(layer_num, d_ffn)
    for layer_idx in range(layer_num):
        cluster = sch.fcluster(trees[layer_idx], t=th, criterion='distance')
        num_cluster = np.max(cluster)
        cluster_center = []
        for i in range(num_cluster):
            indices_in_cluster = np.where(cluster == i + 1)[0]
            if indices_in_cluster.size == 1:
                cluster_center.append(indices_in_cluster[0])
            else:
                neuron = ffn_input[layer_idx][:, indices_in_cluster]
                mean_mag = torch.mean(torch.abs(neuron), dim=0)
                max_mag_index = torch.argmax(mean_mag)
                cluster_center.append(indices_in_cluster[max_mag_index])
        cluster_center = sorted(cluster_center)
        neuron_mask[layer_idx][cluster_center] = 1
    return th, neuron_mask

