import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn import GCNConv, ChebConv
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected
import networkx as nx
from torch_scatter import scatter_mean
import copy
import pandas as pd

# 检查 DGL 是否可用
try:
    import dgl
    from dgl.data.utils import load_graphs

    _HAS_DGL = True
except ImportError:
    _HAS_DGL = False

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# --- 1. 配置类 ---
class AblationConfig:
    def __init__(self, use_gnn=False, use_spectral =True):
        if use_gnn is True and use_spectral is True:
            #空间频谱都用
            use_gnn = True
            use_alignment = True
            use_spectral = True
            self.beta_mlp = 0.35
            self.beta = 1 - self.beta_mlp
        if use_gnn is not True and use_spectral is True:
            # 空间频谱都用，但是不进行GNN对齐
            use_gnn = False
            use_alignment = False
            use_spectral = True
            self.beta_mlp = 0.35
            self.beta = 1 - self.beta_mlp
        if use_gnn is True and use_spectral is not True:
            # 只用空间，不用频谱
            use_gnn = True
            use_alignment = True
            use_spectral = False
            self.beta_mlp = 1
            self.beta = 1 - self.beta_mlp
        if use_gnn is not True and use_spectral is not True:
            # 不用空间，只用频谱
            use_gnn = True
            use_alignment = True
            use_spectral = True
            self.beta_mlp = 0.15
            self.beta = 1 - self.beta_mlp

        self.use_gnn = use_gnn
        self.use_alignment = use_alignment
        self.use_spectral = use_spectral
        # 训练损失权重
        self.lambda_align = 1.0
        self.lambda_spec = 0.1
        # 推理得分权重 (超参数 beta，用于平衡空间与频谱得分)


    # --- 2. 数据加载与划分 (使用你提供的代码) ---


def load_data(data_dir: str, dataset_name: str, device: torch.device):
    path = os.path.join(data_dir, dataset_name)
    if not os.path.exists(path):
        # 兼容性处理：如果是 .mat 但用户没写后缀
        if os.path.exists(path + '.mat'):
            path = path + '.mat'
        else:
            raise FileNotFoundError(f"Dataset not found: {path}")

    if dataset_name in ['AmazonFull', 'YelpChiFull'] or not path.endswith('.mat'):
        if not _HAS_DGL:
            raise RuntimeError("DGL not available, but dataset_name indicates a DGL graph file.")
        graph_list, _ = load_graphs(path)
        graph = graph_list[0]

        features = None
        for k in ["feature", "feat", "features", "x"]:
            if k in graph.ndata:
                features = graph.ndata[k]
                break
        if features is None:
            raise ValueError("DGL graph missing node feature field.")

        labels = None
        for k in ["label", "labels", "y"]:
            if k in graph.ndata:
                labels = graph.ndata[k]
                break
        if labels is None:
            raise ValueError("DGL graph missing node label field.")

        labels_np = labels.cpu().numpy()
        graph = dgl.to_bidirected(graph)
        graph = graph.remove_self_loop()
        nx_graph = dgl.to_networkx(graph)
        adj = nx.to_scipy_sparse_array(nx_graph)

        if sp.issparse(features): features = features.toarray()
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        edge_index, edge_weight = from_scipy_sparse_matrix(adj)
        edge_index = to_undirected(edge_index)

        features = torch.FloatTensor(features).to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.float().to(device) if edge_weight is not None else None
        labels = torch.LongTensor([1 if l > 0 else 0 for l in labels_np]).to(device)
        return features, edge_index, edge_weight, labels

    # 处理 .mat 文件
    mat = sio.loadmat(path)
    adj_key = [k for k in ['Network', 'net', 'homo', 'A'] if k in mat][0]
    feat_key = [k for k in ['Attributes', 'features', 'attr', 'X'] if k in mat][0]
    label_key = [k for k in ['Label', 'label', 'labels'] if k in mat][0]

    adj = sp.coo_matrix(mat[adj_key])
    feat = mat[feat_key]
    if sp.issparse(feat): feat = feat.toarray()
    labels = mat[label_key].flatten()
    labels = np.array([1 if l > 0 else 0 for l in labels], dtype=np.int64)

    scaler = StandardScaler()
    feat = scaler.fit_transform(feat)
    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    edge_index = to_undirected(edge_index)

    features = torch.FloatTensor(feat).to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.float().to(device) if edge_weight is not None else None
    labels = torch.LongTensor(labels).to(device)
    return features, edge_index, edge_weight, labels


def split_dataset(labels: torch.Tensor, train_ratio=0.7, val_ratio=0.1, seed=42, device=None):
    N = len(labels)
    idx = np.arange(N)
    labels_np = labels.detach().cpu().numpy()
    train_val_idx, test_idx = train_test_split(
        idx, test_size=1 - train_ratio, stratify=labels_np, random_state=seed
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio, stratify=labels_np[train_val_idx], random_state=seed
    )
    device = device or labels.device
    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    val_mask = torch.zeros(N, dtype=torch.bool, device=device)
    test_mask = torch.zeros(N, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


# --- 3. 模型定义 ---
class UnsupervisedDetector(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, config):
        super(UnsupervisedDetector, self).__init__()
        self.config = config
        self.mlp_enc = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(hid_channels, out_channels)
        )
        if self.config.use_gnn:
            self.gnn_enc = GCNConv(in_channels, out_channels)
        if self.config.use_spectral:
            self.cheb_filter = ChebConv(out_channels, out_channels, K=2)

    def forward(self, x, edge_index):
        z_mlp = self.mlp_enc(x)
        z_gnn = self.gnn_enc(x, edge_index) if self.config.use_gnn else None

        row, col = edge_index
        z_neighbors_mean = scatter_mean(z_mlp[col], row, dim=0, dim_size=z_mlp.size(0))
        spatial_err = torch.norm(z_mlp - z_neighbors_mean, p=2, dim=1)

        spec_score = torch.zeros(x.size(0), device=x.device)
        if self.config.use_spectral:
            spec_resp = self.cheb_filter(z_mlp, edge_index)
            spec_score = torch.norm(spec_resp, p=2, dim=1)

        return z_mlp, z_gnn, spatial_err, spec_score


# --- 4. 实验核心逻辑 ---
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(model, x, edge_index, labels, mask, beta_mlp, beta):
    model.eval()
    with torch.no_grad():
        _, _, s_err, spec_s = model(x, edge_index)
        final_score = beta_mlp *s_err + beta * spec_s
        y_true = labels[mask].cpu().numpy()
        y_score = final_score[mask].cpu().numpy()
        auc = roc_auc_score(y_true, y_score)
        prc = average_precision_score(y_true, y_score)
    return auc, prc


def run_experiment(seed, features, edge_index, labels, train_mask, val_mask, test_mask):
    set_seed(seed)
    config = AblationConfig()
    model = UnsupervisedDetector(features.size(1), 64, 32, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    best_val_auc = 0
    patience = 20
    counter = 0
    best_state = None

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        z_mlp, z_gnn, _, spec_s = model(features, edge_index)

        loss_align = F.mse_loss(z_mlp, z_gnn.detach()) if config.use_gnn else 0
        # loss_align = F.mse_loss(z_mlp, z_gnn) if config.use_gnn else 0
        loss_spec = torch.mean(spec_s) if config.use_spectral else 0
        total_loss = config.lambda_align * loss_align + config.lambda_spec * loss_spec

        total_loss.backward()
        optimizer.step()

        val_auc, _ = evaluate(model, features, edge_index, labels, val_mask, config.beta_mlp,config.beta)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        if counter >= patience: break

    model.load_state_dict(best_state)
    t_auc, t_prc = evaluate(model, features, edge_index, labels, test_mask, config.beta_mlp,config.beta)
    return t_auc, t_prc, model


from torch_geometric.utils import get_laplacian
from torch_scatter import scatter_add


def export_real_case_study(model, features, edge_index, labels, K=3, filename="real_spectrum_case.csv"):
    """
    真实计算节点在不同切比雪夫阶数（k=0, 1, ..., K-1）下的谱能量分布。
    """
    model.eval()
    with torch.no_grad():
        # 1. 获取节点的基础表示 z (MLP 输出)
        z = model.mlp_enc(features)
        num_nodes = z.size(0)

        # 2. 计算并标准化拉普拉斯矩阵 L_tilde
        # 对应切比雪夫递归公式中的标准化算子
        L_edge_index, L_edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)

        # 3. 定义矩阵乘法函数 (L * z)
        def graph_op(x, edge_index, edge_weight):
            row, col = edge_index
            return scatter_add(edge_weight.view(-1, 1) * x[col], row, dim=0, dim_size=num_nodes)

        # 4. 挑选两个典型代表 (一个 Genuine, 一个 Fake)
        labels_np = labels.cpu().numpy()
        genuine_idx = np.where(labels_np == 0)[0][0]
        fake_idx = np.where(labels_np == 1)[0][0]
        selected_nodes = [genuine_idx, fake_idx]
        node_types = ["Genuine User", "Fake User"]

        # 5. 切比雪夫递归计算 T_k(L)z
        # T_0 = z
        # T_1 = L_tilde * z
        # T_k = 2 * L_tilde * T_{k-1} - T_{k-2}

        node_energies = {name: [] for name in node_types}

        T_k_minus_2 = z
        T_k_minus_1 = graph_op(z, L_edge_index, L_edge_weight)

        # 记录 k=0 和 k=1 的能量
        for i, node_idx in enumerate(selected_nodes):
            node_energies[node_types[i]].append(torch.norm(T_k_minus_2[node_idx], p=2).item())
            node_energies[node_types[i]].append(torch.norm(T_k_minus_1[node_idx], p=2).item())

        # 递归计算更高阶 k >= 2
        for k in range(2, K):
            T_k = 2 * graph_op(T_k_minus_1, L_edge_index, L_edge_weight) - T_k_minus_2
            for i, node_idx in enumerate(selected_nodes):
                node_energies[node_types[i]].append(torch.norm(T_k[node_idx], p=2).item())
            # 更新状态
            T_k_minus_2, T_k_minus_1 = T_k_minus_1, T_k

        # 6. 整理并导出数据
        final_data = []
        for i, name in enumerate(node_types):
            for k in range(K):
                final_data.append({
                    'User_Type': name,
                    'Order_k': f'k={k}',
                    'Energy': node_energies[name][k]
                })

        df = pd.DataFrame(final_data)
        df.to_csv(filename, index=False)
        print(f"Success: Real individual spectrum data exported to {filename}")

# --- 5. 主程序 ---
def main():
    DATA_DIR = '../datasets'
    DATASET_NAME = 'YelpChiFull'

    # 1. 初始读取数据
    features, edge_index, edge_weight, labels = load_data(DATA_DIR, DATASET_NAME, device)

    # 2. 5次实验循环
    seeds = [42, 123, 777, 2024, 999]
    auc_results, prc_results = [], []

    print(f"Dataset: {DATASET_NAME} | Seeds: {seeds}")
    for s in seeds:
        train_m, val_m, test_m = split_dataset(labels, seed=s)
        auc, prc ,model = run_experiment(s, features, edge_index, labels, train_m, val_m, test_m)
        auc_results.append(auc)
        prc_results.append(prc)
        print(f"Seed {s:4d} | AUROC: {auc:.4f} | AUPRC: {prc:.4f}")

    print("\n" + "=" * 40)
    print(f"Results for {DATASET_NAME}:")
    print(f"AUROC: {np.mean(auc_results):.4f} ± {np.std(auc_results):.4f}")
    print(f"AUPRC: {np.mean(prc_results):.4f} ± {np.std(prc_results):.4f}")
    print("=" * 40)
    # export_real_case_study(model, features, edge_index, labels, K=4, filename="my_case_study.csv")


if __name__ == "__main__":
    main()