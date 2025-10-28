import torch
import torch.nn.functional as F

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx, pairwise_distance

def LGC(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    idx, _ = knn(x, k=k)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    neighbor = x.view(batch_size * num_points, -1)[idx, :]

    neighbor = neighbor.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((neighbor - x, neighbor), dim=3).permute(0, 3, 1, 2)  # local and global all in

    return feature

def PCD(input_points, num_selected_points):
    num_neighbors = 64
    distance_threshold = 0.2
    gaussian_sigma = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = input_points.size(0)
    num_points = input_points.size(2)

    point_features = input_points.view(batch_size, -1, num_points)

    neighbor_indices, neighbor_sq_distances = knn(point_features, k=num_neighbors)

    real_distances = torch.sqrt(torch.abs(neighbor_sq_distances))
    distance_mask = real_distances < distance_threshold

    scaled_sq_dist = neighbor_sq_distances / (gaussian_sigma * gaussian_sigma)
    weights = torch.exp(scaled_sq_dist)
    weights = torch.mul(distance_mask.float(), weights)

    normalization_factor = 1.0 / (torch.sum(weights, dim=1) + 1e-8)

    diag_values = normalization_factor.reshape(batch_size, num_points, 1).repeat(1, 1, num_points)
    identity_matrix = torch.eye(num_points, num_points, device=device).expand(batch_size, num_points, num_points)
    degree_matrix = diag_values * identity_matrix

    normalized_adjacency = torch.matmul(degree_matrix, weights)

    flat_adj_indices_base = torch.arange(0, batch_size * num_points, device=device).view(-1, 1) * num_points
    flat_adj_indices = neighbor_indices.view(batch_size * num_points, -1) + flat_adj_indices_base

    flat_adj_indices = flat_adj_indices.reshape(batch_size * num_points, num_neighbors)[:, 1:num_neighbors]
    flat_adj_indices = flat_adj_indices.reshape(batch_size * num_points * (num_neighbors - 1)).view(-1)

    flat_adjacency = normalized_adjacency.view(-1)
    local_adjacency_values = flat_adjacency[flat_adj_indices].reshape(batch_size, num_points, num_neighbors - 1)

    flat_neighbor_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    flat_neighbor_indices = neighbor_indices + flat_neighbor_base
    flat_neighbor_indices = flat_neighbor_indices.reshape(batch_size * num_points, num_neighbors)[:, 1:num_neighbors]
    flat_neighbor_indices = flat_neighbor_indices.reshape(batch_size * num_points * (num_neighbors - 1))

    _, feature_dimension, _ = point_features.size()

    points_transposed = point_features.transpose(2, 1).contiguous()

    neighbor_features = points_transposed.view(batch_size * num_points, -1)[flat_neighbor_indices, :]
    neighbor_features = neighbor_features.view(batch_size, num_points, num_neighbors - 1, feature_dimension)

    local_adjacency_values = local_adjacency_values.reshape(batch_size, num_points, num_neighbors - 1, 1)
    weighted_neighbors = local_adjacency_values.mul(neighbor_features)
    weighted_avg_neighbors = torch.sum(weighted_neighbors, dim=2)

    variation_scores = torch.norm(points_transposed - weighted_avg_neighbors, dim=-1).pow(2)

    sharp_indices = variation_scores.topk(k=num_selected_points, dim=-1)[1]
    gentle_indices = (-variation_scores).topk(k=num_selected_points, dim=-1)[1]

    batch_offset = torch.arange(0, batch_size, device=device).view(-1, 1) * num_points

    flat_sharp_indices = (sharp_indices + batch_offset).view(-1)
    flat_gentle_indices = (gentle_indices + batch_offset).view(-1)

    all_points_flat = points_transposed.view(batch_size * num_points, -1)

    sharp_points = all_points_flat[flat_sharp_indices, :]
    gentle_points = all_points_flat[flat_gentle_indices, :]

    sharp_points = sharp_points.view(batch_size, num_selected_points, -1)
    gentle_points = gentle_points.view(batch_size, num_selected_points, -1)

    return sharp_points, gentle_points


#得到的是索引
def knn_ssfe(x, k):

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

#LGC
def get_neighbors_ssfe(x, feature, k=20, idx=None):

    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn_ssfe(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_base = idx_base.type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)
    idx = idx.type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)


    _, num_dims, _ = x.size()
    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbor_x = x.view(batch_size * num_points, -1)[idx, :]
    neighbor_x = neighbor_x.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    neighbor_x = torch.cat((neighbor_x - x, x), dim=3).permute(0, 3, 1, 2)


    _, num_dims, _ = feature.size()
    feature = feature.transpose(2,
                                1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbor_feat = feature.view(batch_size * num_points, -1)[idx, :]
    neighbor_feat = neighbor_feat.view(batch_size, num_points, k, num_dims)
    feature = feature.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    neighbor_feat = torch.cat((neighbor_feat - feature, feature), dim=3).permute(0, 3, 1, 2)

    return neighbor_x, neighbor_feat