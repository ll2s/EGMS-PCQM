import torch
import torch.nn as nn
import torch.nn.functional as F

from util import sample_and_group

from point_utils import LGC, PCD
from layers import AttentionModule, DualGraphFusion, Local_op

class LSFE_Net(nn.Module):
    def __init__(self):
        super(LSFE_Net, self).__init__()

        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn11 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn12 = nn.BatchNorm1d(64, momentum=0.1)

        self.bn2 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn21 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn22 = nn.BatchNorm1d(64, momentum=0.1)

        self.bn3 = nn.BatchNorm2d(128, momentum=0.1)
        self.bn31 = nn.BatchNorm2d(128, momentum=0.1)
        self.bn32 = nn.BatchNorm1d(128, momentum=0.1)

        self.bn4 = nn.BatchNorm1d(512, momentum=0.1)

        # self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=True),
        #                            self.bn1)
        self.mlp1_1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=True),
                                    self.bn1)
        self.mlp1_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),
                                    self.bn11)
        self.mlp1_3 = nn.Sequential(nn.Conv1d(64 * 2, 64, kernel_size=1, bias=True),
                                    self.bn12)

        # self.conv2 = nn.Sequential(nn.Conv2d(67 * 2, 64, kernel_size=1, bias=True),
        #                            self.bn2)
        self.mlp2_1 = nn.Sequential(nn.Conv2d(70 * 2, 64, kernel_size=1, bias=True),
                                    self.bn2)
        self.mlp2_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),
                                    self.bn21)
        self.mlp2_3 = nn.Sequential(nn.Conv1d(64 * 2, 64, kernel_size=1, bias=True),
                                    self.bn22)

        # self.conv3 = nn.Sequential(nn.Conv2d(131 * 2, 128, kernel_size=1, bias=True),
        #                            self.bn3)
        self.mlp3_1 = nn.Sequential(nn.Conv2d(134 * 2, 128, kernel_size=1, bias=True),
                                    self.bn3)
        self.mlp3_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=True),
                                    self.bn31)
        self.mlp3_3 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=True),
                                    self.bn32)

        self.mlp4 = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, bias=True),
                                  self.bn4)

        self.attention_1_edge = AttentionModule(64)
        self.attention_1_content = AttentionModule(64)
        self.attention_2_edge = AttentionModule(64)
        self.attention_2_content = AttentionModule(64)

        self.linear1 = nn.Linear(1024, 512, bias=True)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.4)
        self.linear2 = nn.Linear(512, 256, bias=True)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.4)

        self.linear3 = nn.Linear(256, 1, bias=True)  # add it myself

    def forward(self, x):
        B, C, N = x.size()



        P_o_f_1_raw = LGC(x, k=30)
        P_o_f_1_raw = F.relu(self.mlp1_1(P_o_f_1_raw))
        P_o_f_1_raw = F.relu(self.mlp1_2(P_o_f_1_raw))
        P_o_f_1 = P_o_f_1_raw.max(dim=-1, keepdim=False)[0]


        P_e_f_1, P_c_f_1 = PCD(P_o_f_1, 256)


        A_e_1 = self.attention_1_edge(P_o_f_1, P_e_f_1.transpose(2, 1))
        A_c_1 = self.attention_1_content(P_o_f_1, P_c_f_1.transpose(2, 1))


        Z_1 = torch.cat([A_e_1, A_c_1], 1)
        F_A_1 = F.relu(self.mlp1_3(Z_1))


        F_A_1_fused = torch.cat((x, F_A_1), dim=1)


        P_o_f_2_raw = LGC(F_A_1_fused, k=30)
        P_o_f_2_raw = F.relu(self.mlp2_1(P_o_f_2_raw))
        P_o_f_2_raw = F.relu(self.mlp2_2(P_o_f_2_raw))
        P_o_f_2 = P_o_f_2_raw.max(dim=-1, keepdim=False)[0]


        P_e_f_2, P_c_f_2 = PCD(P_o_f_2, 256)


        A_e_2 = self.attention_2_edge(P_o_f_2, P_e_f_2.transpose(2, 1))
        A_c_2 = self.attention_2_content(P_o_f_2, P_c_f_2.transpose(2, 1))


        Z_2 = torch.cat([A_e_2, A_c_2], 1)
        F_A_2 = F.relu(self.mlp2_3(Z_2))


        F_A_2_fused = torch.cat((F_A_1_fused, F_A_2), dim=1)


        aggregated_features_raw = LGC(F_A_2_fused, k=30)
        aggregated_features_raw = F.relu(self.mlp3_1(aggregated_features_raw))
        aggregated_features_raw = F.relu(self.mlp3_2(aggregated_features_raw))
        aggregated_features = aggregated_features_raw.max(dim=-1, keepdim=False)[0]
        F_L = F.relu(self.mlp3_3(aggregated_features))


        F_L_prime = torch.cat((F_A_1, F_A_2, F_L), dim=1)
        F_L_prime = F.relu(self.mlp4(F_L_prime))


        max_pooled_features = F.adaptive_max_pool1d(F_L_prime, 1).view(B, -1)
        avg_pooled_features = F.adaptive_avg_pool1d(F_L_prime, 1).view(B, -1)
        global_feature_vector = torch.cat((max_pooled_features, avg_pooled_features), 1)


        global_feature_vector = F.relu(self.bn6(self.linear1(global_feature_vector)))
        global_feature_vector = self.dp1(global_feature_vector)
        global_feature_vector = F.relu(self.bn7(self.linear2(global_feature_vector)))
        global_feature_vector = self.dp2(global_feature_vector)
        quality_score_Q_x_l = self.linear3(global_feature_vector)

        return quality_score_Q_x_l


class SSFE_Net(nn.Module):
    def __init__(self, point_num):
        super(SSFE_Net, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 1024, kernel_size=1, stride=int(point_num / 256), bias=False)
        self.bn2 = nn.BatchNorm1d(1024)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.conv_fuse1 = nn.Sequential(nn.Conv1d(1280, 512, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(512),
                                        nn.LeakyReLU(negative_slope=0.2))
        ######
        self.dual_graph_module = DualGraphFusion(64)

    def forward(self, x):
        xyz = x[:, 0:3, :].permute(0, 2, 1)
        xyz_for_graph = x[:, 0:3, :]
        batch_size, _, _ = x.size()



        f_o = F.relu(self.bn1(self.conv1(x)))


        out_feat = self.dual_graph_module(xyz_for_graph, f_o, 20)

        # B, D, N
        x_skip = F.relu(
            self.bn2(self.conv2(f_o)))


        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, neighbor=32, xyz=xyz,
                                                feature=out_feat)

        feature_0 = self.gather_local_0(
            new_feature)

        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, neighbor=32, xyz=new_xyz, feature=feature_0)
        feature_1 = self.gather_local_1(new_feature)


        feature_1 = torch.cat((feature_1, x_skip), dim=1)  # [32,1024+256,256]
        res = self.conv_fuse1(feature_1)

        return res  # [32,512,256]


class Dual_net(nn.Module):
    def __init__(self, args, final_channels=1):
        super(Dual_net, self).__init__()

        self.Net_b = SSFE_Net(point_num=args.point_num_big)
        self.Net_s = SSFE_Net(point_num=8192)
        self.lsfe = LSFE_Net()

        self.CBR = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(512, 256, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=args.dropout)


        self.linear_mos = nn.Linear(512, 1)

        self.weight_layer = nn.Linear(2, 1)

        self.linear_mymos = nn.Linear(512 * 256 * 2, 1)
        #




        self.linear_final = nn.Linear(2, 1)

    def forward(self, x_lsfe, x_ssfe):
        batch_size = x_lsfe.shape[0]
        x_lsfe_score = self.lsfe(x_lsfe)



        x_ssfe_feat = self.Net_s(x_ssfe)

        x1 = self.CBR(x_ssfe_feat)  # [32, 512, 256]
        x1 = F.adaptive_max_pool1d(x1, 1).view(batch_size, -1)  # [32, 512]

        mos1 = self.linear_mos(x1)  # [32,1]

        mos2 = x_lsfe_score

        mos = torch.cat((mos1, mos2), 1)

        mos = self.linear_final(mos)

        return mos  # [32,1]