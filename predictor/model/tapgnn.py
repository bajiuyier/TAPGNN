"""任务对齐传播图神经网络（Task-Aligned Propagation GNN，TAPGNN）。

本文件实现模型结构，计算流程为：
    特征编码 -> 多跳传播 -> 构造节点状态 -> 分量校正 -> 节点级强度 -> 分类。

核心变量（N：节点数，D：隐藏维度，C：类别数，K：传播跳数）：
    h / states[k]：节点自身表示 / 第 k 跳表示，形状均为 [N, D]。
    displacement：基础传播位移 d_i = T(H)_i - h_i，形状为 [N, D]。
    axis：节点级单位校正轴 a_i，形状为 [N, D]。
    kappa：沿校正轴的分量系数，形状为 [N, 1]，理论范围为 (-1, 1)。
    eta：整体更新强度，形状为 [N, 1]，理论范围为 (0, 2)。
    update：eta_i * [d_i + (kappa_i - 1) * <a_i, d_i> * a_i]。

这里的 axis 由可观测节点状态学习得到，并不等于真实任务梯度。
模型前向只接收特征和传播矩阵：不接收标签，也不显式计算任务梯度。
正常训练仍然通过 loss.backward() 对模型参数进行反向传播。

本注释版保留上传文件的计算表达式、接口、初始化和返回字段。
状态拼接及单位轴处理由 utils.tapgnn_utils 中的辅助函数实现。
"""

from __future__ import annotations

import torch
from torch import nn

from utils.tapgnn_utils import build_propagation_state, normalize_axis


class TAPGNN(nn.Module):
    """用多跳状态预测节点级校正行为的全图节点分类模型。

    不同节点使用同一套 correction 网络参数，但输入状态不同，
    因而可以得到不同的 axis、kappa 和 eta。
    gamma 对应的多跳混合权重是全图共享的，不是逐节点的跳数权重。
    """

    def __init__(self, conf):
        """读取 conf.model，创建编码器、共享分类头和校正网络。"""
        super().__init__()
        cfg = conf.model
        in_features = int(cfg["n_feat"])

        # 兼容旧框架用 n_class=1 表示二分类的约定。
        # 当前模型统一输出至少两个 logits，训练使用 softmax 交叉熵。
        # 这不是单输出 sigmoid / BCE 的实现。
        self.num_classes = max(2, int(cfg["n_class"]))
        self.hidden_dim = int(cfg.get("n_hidden", 64))
        self.num_hops = int(cfg.get("num_hops", 2))
        correction_hidden = int(cfg.get("correction_hidden", self.hidden_dim))
        dropout = float(cfg.get("dropout", 0.5))
        self.norm_eps = float(cfg.get("norm_eps", 1e-8))

        # 防止零维线性层、零跳传播和不合法的数值稳定参数。
        if min(in_features, self.hidden_dim, self.num_hops, correction_hidden) < 1:
            raise ValueError("Feature dimensions and num_hops must be positive.")
        if not 0.0 <= dropout < 1.0 or self.norm_eps <= 0:
            raise ValueError("Require dropout in [0, 1) and norm_eps > 0.")

        # f_theta(X)：两层特征编码器。
        # 第一层后使用 GELU 和 Dropout；第二层后不额外添加激活函数。
        # Dropout 只在 model.train() 时生效，model.eval() 时关闭。
        self.encoder = nn.Sequential(
            nn.Linear(in_features, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # 自表示、各跳状态和校正后的表示都使用同一个线性分类头 W,b。
        # 输出为未归一化 logits；计算损失时再进行 log_softmax。
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

        # K 个全局可学习参数经 softmax 得到 alpha_1,...,alpha_K。
        # 初始 gamma 全零，因此初始各跳权重均为 1/K。
        self.gamma = nn.Parameter(torch.zeros(self.num_hops))

        # 节点状态包括：
        #   [H^(0),...,H^(K), p^(0), p^(1)-p^(0),...,p^(K)-p^(0)]。
        # 表示部分占 (K+1)*D，预测状态部分占 (K+1)*C。
        state_dim = (self.num_hops + 1) * (self.hidden_dim + self.num_classes)

        # 校正网络每个节点输出 D+2 个数：
        #   前 D 个数：未经归一化的校正轴；
        #   倒数第 2 个数：生成 kappa 的原始标量；
        #   最后 1 个数：生成 eta 的原始标量。
        self.correction = nn.Sequential(
            nn.Linear(state_dim, correction_hidden),
            nn.GELU(),
            nn.Linear(correction_hidden, self.hidden_dim + 2),
        )
        self.reset_parameters()

    def reset_parameters(self):
        """重置模型参数，并设置接近原传播的初始校正行为。"""
        for module in (self.encoder, self.correction):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()
        self.classifier.reset_parameters()

        # 初始化时直接改写参数，不把这些赋值记录进自动求导图。
        with torch.no_grad():
            self.gamma.zero_()
            output_layer = self.correction[-1]

            # 只把两个标量输出对应的权重行置零，轴向输出仍保留随机权重。
            # 初始 kappa = 1 - 2*sigmoid(-4) ≈ 0.964，eta = 2*sigmoid(0) = 1。
            # 因此开始联合训练时，仅轻微削弱所选轴上的分量。
            # eta=1 只是初始化设置，不代表满足理论的安全步长上界。
            output_layer.weight[-2:].zero_()
            output_layer.bias[-2] = -4.0
            output_layer.bias[-1] = 0.0

    def forward(self, x, adj=None, self_only=False, return_aux=False):
        """计算节点 logits，并按需返回联合训练使用的中间变量。

        参数：
            x：[N, F] 节点特征。
            adj：[N, N] 传播矩阵 P，可为稀疏矩阵或稠密矩阵。
                 P 由 build_row_normalized_adjacency 构造：
                 先删除原对角元素，再添加单位自环，最后按行归一化。
                 这里不接收 [2, E] 的 edge_index。
            self_only：预热时为 True，只运行编码器和自表示分类头。
            return_aux：为 True 时返回字典，否则只返回 logits 张量。

        每次前向都会重新编码 X 并计算 P^k H，传播表示仍可对编码器求导。
        只有 build_propagation_state 中的预测状态分支停止梯度。
        """
        # 自表示和自表示分类结果，也是有害传播惩罚所用的比较基准。
        h = self.encoder(x)
        logits_self = self.classifier(h)

        # 预热阶段不进入传播、跳数混合和校正网络。
        if self_only:
            if return_aux:
                return {"logits": logits_self, "logits_self": logits_self}
            return logits_self
        if adj is None or adj.shape != (x.size(0), x.size(0)):
            raise ValueError("Joint propagation requires an N-by-N matrix P.")

        # H^(0)=H，H^(k)=P H^(k-1)。
        # 逐次矩阵乘法避免显式构造 P 的高次幂。
        # states 保存所有跳数的表示，供混合和节点状态构造共同使用。
        states = [h]
        for _ in range(self.num_hops):
            if adj.layout == torch.strided:
                states.append(adj @ states[-1])
            else:
                states.append(torch.sparse.mm(adj, states[-1]))

        # T(H)=sum_{k=1}^K alpha_k H^(k)。
        # 这里只混合第 1 到 K 跳，H^(0) 不直接进入这个加权和；
        # 不过 P 已有自环，所以各跳传播本身仍包含自身信息。
        alpha = self.gamma.softmax(dim=0)
        mixed = torch.zeros_like(h)
        for k, propagated in enumerate(states[1:]):
            mixed = mixed + alpha[k] * propagated
        displacement = mixed - h

        # 拼接表示与预测变化，描述节点在不同传播范围下的状态。
        # 表示分支保留梯度；概率与概率差分分支只作为状态输入。
        state = build_propagation_state(states, self.classifier, logits_self)

        # 根据每个节点的 state 预测校正轴和两个标量。
        parameters = self.correction(state)
        axis = normalize_axis(parameters[:, :self.hidden_dim], self.norm_eps)

        # kappa 控制轴向分量：
        #   接近 1：近似保留；0：删除；负值：反向并按 |kappa| 缩放。
        # eta 控制校正后整个位移的强度。
        # 对有限实数输入，数学范围分别为 (-1,1) 和 (0,2)；
        # 极端浮点输入下 sigmoid 可能数值饱和到端点。
        # 本实现未使用 g_i、L 或对齐量来对 eta 施加理论步长上界。
        kappa = 1.0 - 2.0 * parameters[:, -2:-1].sigmoid()
        eta = 2.0 * parameters[:, -1:].sigmoid()

        # 单位轴上的正交投影 d_parallel=<a,d>*a。
        # 修正后的位移为 d_perp+kappa*d_parallel：
        # 只改变一条轴上的分量，再由 eta 缩放整个结果。
        # a 和 -a 产生相同投影，因此轴的正负号不影响校正结果。
        parallel = (axis * displacement).sum(dim=-1, keepdim=True) * axis
        update = eta * (displacement + (kappa - 1.0) * parallel)

        # 残差更新：保留当前自表示，再加上学到的传播更新。
        # 若 displacement 恰为零，校正后更新仍为零。
        corrected = h + update
        logits = self.classifier(corrected)
        if not return_aux:
            return logits

        # 联合训练需要 logits_self 来计算自表示损失和传播损失增量。
        # 其余字段用于训练日志、机制分析或可视化，不改变前向结果。
        # update 已经包含 eta；displacement 是未校正的基础位移。
        return {
            "logits": logits,
            "logits_self": logits_self,
            "alpha": alpha,
            "eta": eta,
            "kappa": kappa,
            "axis": axis,
            "displacement": displacement,
            "update": update,
            "self_hidden": h,
            "corrected_hidden": corrected,
            "state": state,
        }
