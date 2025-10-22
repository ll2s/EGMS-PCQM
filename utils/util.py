import torch
import torch.nn.functional as F


def cal_loss(pred, gold, smoothing=True):

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def info(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def square_distance(xyz, center_xyz):
    B, N, _ = center_xyz.shape
    _, M, _ = xyz.shape
    dist = -2 * torch.matmul(center_xyz, xyz.permute(0, 2, 1))
    dist += torch.sum(center_xyz ** 2, -1).view(B, N, 1)
    dist += torch.sum(xyz ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, center_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = center_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(xyz, center_xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(neighbor, xyz, center_xyz):
    sqrdists = square_distance(xyz, center_xyz)
    _, group_idx = torch.topk(sqrdists, neighbor, dim=-1, largest=False, sorted=False)
    return group_idx


def furthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def sample_and_group(npoint, radius, neighbor, xyz, feature):
    feature = feature.permute(0, 2, 1)
    B, N, C = xyz.shape
    S = npoint
    xyz = xyz.contiguous()
    noise = torch.rand(B, N, device=xyz.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :S]
    fps_idx = torch.arange(N, dtype=torch.long).to(xyz.device).unsqueeze(0).repeat(B, 1)
    fps_idx = torch.gather(fps_idx, dim=1, index=ids_keep)

    center_xyz = index_points(xyz, fps_idx)
    center_feature = index_points(feature, fps_idx)
    idx = knn_point(neighbor, xyz, center_xyz)
    grouped_feature = index_points(feature, idx)
    grouped_feature_center = grouped_feature - center_feature.view(B, S, 1, -1)
    res_points = torch.cat([grouped_feature_center, center_feature.view(B, S, 1, -1).repeat(1, 1, neighbor, 1)],
                           dim=-1)
    return center_xyz, res_points