'''
This file is to split Coauthor/Amazon dataset
'''


from sklearn.model_selection import StratifiedKFold
import random
import numpy as np
import torch
import math

def resolve_imb_ratio(imb_level):
    """
    将类别不平衡等级转换为具体不平衡比例。

    例如：
    imb_level = 1   表示类别平衡；
    imb_level = 10  表示头部类和尾部类约为 10:1；
    imb_level = 20  表示头部类和尾部类约为 20:1；
    imb_level = 100 表示头部类和尾部类约为 100:1。
    """
    if isinstance(imb_level, str):
        return float(imb_level)

    return float(imb_level)


def sort(data):
    """
    按照每个类别的节点数量从多到少排序。

    参数
    ----
    data:
        你的 Dataset 对象，需要包含 data.labels。

    返回
    ----
    class_num_list:
        原始类别顺序下，每个类别的节点总数。

    indices:
        排序后的类别索引。样本数多的类别排在前面。

    inv_indices:
        用于把排序后的长尾数量恢复到原始类别编号顺序。
    """
    labels = data.labels.detach().cpu().long()
    n_classes = int(labels.max().item()) + 1

    class_num_list = torch.bincount(labels, minlength=n_classes).long()

    _, indices = torch.sort(class_num_list, descending=True)

    inv_indices = torch.empty_like(indices)
    inv_indices[indices] = torch.arange(n_classes)

    return class_num_list, indices, inv_indices

#
# def split_lt(class_num_list, indices, inv_indices, imb_ratio, n_cls, n, keep=0):
#     """
#     根据长尾分布计算每个类别应该分配多少训练节点。
#
#     参数
#     ----
#     class_num_list:
#         每个类别的原始节点数量。
#
#     indices:
#         按类别节点数量从多到少排序后的类别索引。
#
#     inv_indices:
#         恢复原始类别顺序用的索引。
#
#     imb_ratio:
#         类别不平衡比例。例如 10、20、100。
#
#     n_cls:
#         类别数。
#
#     n:
#         训练集总节点数，通常设置为总节点数的 10%。
#
#     keep:
#         每类预留节点数，默认 0。
#
#     返回
#     ----
#     class_num_list_train:
#         原始类别编号顺序下，每个类别分配的训练节点数。
#     """
#     class_num_list = class_num_list[indices]
#
#     imb_ratio = float(imb_ratio)
#
#     if imb_ratio == 1:
#         n_max = n / n_cls
#         mu = 1.0
#         inv_mu = 1.0
#     else:
#         mu = np.power(imb_ratio, 1.0 / (n_cls - 1))
#         inv_mu = 1.0 / mu
#         n_max = n / (imb_ratio * mu - 1) * (mu - 1) * imb_ratio
#
#     class_num_list_lt = []
#
#     for i in range(n_cls):
#         target_num = n_max * np.power(inv_mu, i)
#
#         # 理论上每类至少保留 1 个训练节点，避免空类导致训练出错
#         target_num = max(target_num, 1)
#
#         # 当前类别最多只能取已有节点数
#         available_num = int(class_num_list[i].item() - keep)
#
#         if available_num <= 0:
#             final_num = 0
#         else:
#             final_num = round(min(target_num, available_num))
#             final_num = max(final_num, 1)
#
#         class_num_list_lt.append(final_num)
#
#     class_num_list_lt = torch.tensor(class_num_list_lt, dtype=torch.long)
#
#     # 恢复到原始类别编号顺序
#     return class_num_list_lt[inv_indices]

def split_lt(
    class_num_list,
    indices,
    inv_indices,
    imb_ratio,
    n_cls,
    n,
    keep=0,
    debug=False
):
    """
    构造整数长尾训练类别数量。

    严格保证：
    1. 训练节点总数严格等于 n；
    2. 最大类别数 / 最小类别数严格等于 imb_ratio；
    3. 类别数量从多数类到少数类单调不增；
    4. 不超过各类别实际可用节点数。

    尽可能保证：
    5. 最终整数类别数量接近指数长尾分布。

    注意：
    由于类别数量必须为整数，通常无法同时严格满足
    每个相邻类别都具有完全相同的指数比值。
    """

    # ---------------------------------------------------------
    # 1. 参数检查
    # ---------------------------------------------------------
    rho_float = float(imb_ratio)
    rho = int(round(rho_float))

    if not np.isclose(
        rho_float,
        rho,
        rtol=0.0,
        atol=1e-12
    ):
        raise ValueError(
            f"当前实现要求 imb_ratio 为整数，"
            f"当前值为 {imb_ratio}"
        )

    if rho < 1:
        raise ValueError(
            f"imb_ratio 必须大于等于 1，当前为 {rho}"
        )

    if n_cls != len(class_num_list):
        raise ValueError(
            f"n_cls={n_cls}，"
            f"但 class_num_list 长度为 "
            f"{len(class_num_list)}"
        )

    if n <= 0:
        raise ValueError(
            f"训练集总数 n 必须大于 0，当前为 {n}"
        )

    indices_cpu = (
        indices
        .detach()
        .cpu()
        .long()
    )

    inv_indices_cpu = (
        inv_indices
        .detach()
        .cpu()
        .long()
    )

    # ---------------------------------------------------------
    # 2. 按类别原始数量从多到少排序
    # ---------------------------------------------------------
    available_sorted = (
        class_num_list
        .detach()
        .cpu()
        .long()[indices_cpu]
        - int(keep)
    ).clamp_min(0)

    available_np = (
        available_sorted
        .numpy()
        .astype(np.int64)
    )

    if np.any(available_np <= 0):
        raise ValueError(
            "至少有一个类别没有可用节点："
            f"{available_np.tolist()}"
        )

    # ---------------------------------------------------------
    # 3. 平衡情况 rho = 1
    # ---------------------------------------------------------
    if rho == 1:
        # 取不超过目标训练总数 n 的最大严格平衡划分
        per_class = n // n_cls

        if per_class < 1:
            raise ValueError(
                f"训练节点总数过少，无法保证每个类别至少有一个样本："
                f"n={n}, n_cls={n_cls}"
            )

        if np.any(available_np < per_class):
            insufficient_classes = np.where(
                available_np < per_class
            )[0].tolist()

            raise ValueError(
                f"部分类别的可用节点数不足 {per_class}，"
                f"类别位置={insufficient_classes}，"
                f"可用数量={available_np.tolist()}"
            )

        counts_sorted = np.full(
            n_cls,
            per_class,
            dtype=np.int64
        )

        actual_total = int(
            counts_sorted.sum()
        )

        if debug:
            print("\n" + "=" * 100)
            print("严格平衡划分结果")
            print("=" * 100)
            print(f"目标训练节点数: {n}")
            print(f"类别数: {n_cls}")
            print(f"每类训练节点数: {per_class}")
            print(f"实际训练节点总数: {actual_total}")
            print(f"未分配到训练集的节点数: {n - actual_total}")
            print(f"训练类别数量: {counts_sorted.tolist()}")
            print("实际不平衡比: 1.0")
            print("=" * 100 + "\n")

        counts_tensor = torch.tensor(
            counts_sorted,
            dtype=torch.long
        )

        return counts_tensor[inv_indices_cpu]

    # ---------------------------------------------------------
    # 4. 理论指数衰减系数
    #
    # 从多数类到少数类：
    # n_i = n_max * r^i
    #
    # r^(n_cls-1) = 1 / rho
    # ---------------------------------------------------------
    decay_ratio = np.power(
        rho,
        -1.0 / (n_cls - 1)
    )

    # 少数类数量的可行上限
    max_minority_count = min(
        int(available_np[-1]),
        int(available_np[0] // rho)
    )

    if max_minority_count < 1:
        raise ValueError(
            f"无法构造严格 {rho}:1 的划分。"
            f"多数类可用数量={available_np[0]}，"
            f"少数类可用数量={available_np[-1]}"
        )

    best_solution = None
    best_score = None

    # ---------------------------------------------------------
    # 5. 枚举整数少数类数量
    # ---------------------------------------------------------
    for minority_count in range(
        1,
        max_minority_count + 1
    ):
        majority_count = (
            rho * minority_count
        )

        if majority_count > available_np[0]:
            continue

        # 所有类别至少都要能容纳 minority_count
        if np.any(
            available_np < minority_count
        ):
            continue

        # 头尾固定以后，中间类别需要承担的总数
        middle_total = (
            n
            - majority_count
            - minority_count
        )

        if n_cls == 2:
            if middle_total != 0:
                continue

            counts_sorted = np.array(
                [
                    majority_count,
                    minority_count
                ],
                dtype=np.int64
            )

            score = 0.0

        else:
            if middle_total < 0:
                continue

            # -------------------------------------------------
            # 5.1 原始连续指数序列
            # -------------------------------------------------
            ideal_counts = np.array(
                [
                    majority_count
                    * np.power(
                        decay_ratio,
                        i
                    )
                    for i in range(n_cls)
                ],
                dtype=np.float64
            )

            ideal_middle_sum = float(
                ideal_counts[1:-1].sum()
            )

            if ideal_middle_sum <= 0:
                continue

            # -------------------------------------------------
            # 5.2 统一缩放中间类别
            #
            # 保持中间类别之间仍为指数关系，
            # 同时令连续目标总数严格等于 n。
            # -------------------------------------------------
            middle_scale = (
                middle_total
                / ideal_middle_sum
            )

            if middle_scale <= 0:
                continue

            target_counts = (
                ideal_counts.copy()
            )

            target_counts[0] = (
                majority_count
            )

            target_counts[-1] = (
                minority_count
            )

            target_counts[1:-1] *= (
                middle_scale
            )

            # -------------------------------------------------
            # 5.3 连续目标必须单调递减
            # -------------------------------------------------
            if np.any(
                target_counts[:-1]
                < target_counts[1:]
                - 1e-12
            ):
                continue

            # 不能低于 1
            if np.any(
                target_counts < 1
            ):
                continue

            # 不能超过每个类别的可用数量
            if np.any(
                target_counts
                > available_np
                + 1e-12
            ):
                continue

            # -------------------------------------------------
            # 5.4 先向下取整
            # -------------------------------------------------
            counts_sorted = np.floor(
                target_counts + 1e-12
            ).astype(np.int64)

            # 重新固定头尾
            counts_sorted[0] = (
                majority_count
            )

            counts_sorted[-1] = (
                minority_count
            )

            remaining = int(
                n - counts_sorted.sum()
            )

            if remaining < 0:
                continue

            # -------------------------------------------------
            # 5.5 最大余数法确定性补齐
            #
            # 每次选择增加 1 后，
            # 对连续目标平方误差增加最小的类别。
            # -------------------------------------------------
            while remaining > 0:
                candidates = []

                for i in range(
                    1,
                    n_cls - 1
                ):
                    # 不超过该类别可用数量
                    if (
                        counts_sorted[i] + 1
                        > available_np[i]
                    ):
                        continue

                    # 增加后不能超过前一个类别
                    if (
                        counts_sorted[i] + 1
                        > counts_sorted[i - 1]
                    ):
                        continue

                    # 为保持最接近 target，
                    # 一个类别最多增加到 ceil(target)
                    target_upper = int(
                        math.ceil(
                            target_counts[i]
                            - 1e-12
                        )
                    )

                    if (
                        counts_sorted[i] + 1
                        > target_upper
                    ):
                        continue

                    old_error = (
                        counts_sorted[i]
                        - target_counts[i]
                    ) ** 2

                    new_error = (
                        counts_sorted[i] + 1
                        - target_counts[i]
                    ) ** 2

                    error_increase = (
                        new_error - old_error
                    )

                    candidates.append(
                        (
                            error_increase,
                            i
                        )
                    )

                if len(candidates) == 0:
                    break

                # 误差增加最小者优先；
                # 相同时优先较靠前的类别
                _, selected_class = min(
                    candidates,
                    key=lambda item: (
                        item[0],
                        item[1]
                    )
                )

                counts_sorted[
                    selected_class
                ] += 1

                remaining -= 1

            if remaining != 0:
                continue

            # -------------------------------------------------
            # 5.6 最终可行性检查
            # -------------------------------------------------
            if int(
                counts_sorted.sum()
            ) != n:
                continue

            if (
                counts_sorted[0]
                != rho
                * counts_sorted[-1]
            ):
                continue

            if np.any(
                counts_sorted[:-1]
                < counts_sorted[1:]
            ):
                continue

            if np.any(
                counts_sorted
                > available_np
            ):
                continue

            # -------------------------------------------------
            # 5.7 评价最终整数分布与指数分布的偏差
            #
            # 在对数空间评价相邻比例，
            # 因为指数序列在 log 空间为线性。
            # -------------------------------------------------
            log_target_ratio = math.log(
                decay_ratio
            )

            log_ratio_error = 0.0

            for i in range(
                n_cls - 1
            ):
                current_ratio = (
                    counts_sorted[i + 1]
                    / counts_sorted[i]
                )

                log_ratio_error += (
                    math.log(current_ratio)
                    - log_target_ratio
                ) ** 2

            # middle_scale 越接近 1，
            # 对原始指数序列的修改越小
            scale_error = abs(
                math.log(middle_scale)
            )

            score = (
                log_ratio_error,
                scale_error,
                -minority_count
            )

        # 选择最接近指数长尾的可行方案
        if (
            best_solution is None
            or score < best_score
        ):
            best_solution = (
                counts_sorted.copy()
            )
            best_score = score

    # ---------------------------------------------------------
    # 6. 没有可行整数解
    # ---------------------------------------------------------
    if best_solution is None:
        raise ValueError(
            f"无法同时满足：\n"
            f"1. 训练总数严格等于 {n}；\n"
            f"2. 最大类/最小类严格等于 {rho}；\n"
            f"3. 类别数量单调递减；\n"
            f"4. 不超过类别可用数量。\n"
            f"类别可用数量（多→少）："
            f"{available_np.tolist()}"
        )

    counts_sorted = best_solution

    # ---------------------------------------------------------
    # 7. 最终严格验证
    # ---------------------------------------------------------
    final_total = int(
        counts_sorted.sum()
    )

    final_max = int(
        counts_sorted[0]
    )

    final_min = int(
        counts_sorted[-1]
    )

    if final_total != n:
        raise RuntimeError(
            f"训练总数错误："
            f"actual={final_total}, target={n}"
        )

    if final_max != rho * final_min:
        raise RuntimeError(
            f"头尾比例错误："
            f"max={final_max}, min={final_min}, "
            f"actual={final_max / final_min}, "
            f"target={rho}"
        )

    if np.any(
        counts_sorted[:-1]
        < counts_sorted[1:]
    ):
        raise RuntimeError(
            "类别数量不是单调递减"
        )

    if np.any(
        counts_sorted > available_np
    ):
        raise RuntimeError(
            "类别训练数量超过可用节点数"
        )

    if np.any(
        counts_sorted <= 0
    ):
        raise RuntimeError(
            "训练集中存在空类别"
        )

    if debug:
        adjacent_ratios = [
            counts_sorted[i + 1]
            / counts_sorted[i]
            for i in range(n_cls - 1)
        ]

        print("\n" + "=" * 100)
        print("严格总数和头尾比例的整数长尾划分")
        print("=" * 100)

        print(
            f"类别可用数量（多数→少数）: "
            f"{available_np.tolist()}"
        )

        print(
            f"最终训练数量（多数→少数）: "
            f"{counts_sorted.tolist()}"
        )

        print(
            f"最终训练数量（少数→多数）: "
            f"{counts_sorted[::-1].tolist()}"
        )

        print(
            f"训练总数: "
            f"{final_total}"
        )

        print(
            f"头尾比例: "
            f"{final_max}/{final_min}="
            f"{final_max / final_min:.6f}"
        )

        print(
            f"理论相邻衰减系数: "
            f"{decay_ratio:.6f}"
        )

        print(
            "实际相邻类别比例: "
            f"{[round(x, 6) for x in adjacent_ratios]}"
        )

        print("=" * 100 + "\n")

    # ---------------------------------------------------------
    # 8. 恢复原始类别编号顺序
    # ---------------------------------------------------------
    counts_sorted_tensor = torch.tensor(
        counts_sorted,
        dtype=torch.long
    )

    class_num_list_train = (
        counts_sorted_tensor[
            inv_indices_cpu
        ]
    )

    return class_num_list_train


def get_classnum_imb_split(target_data, imb_level, shuffle_seed, debug=True):
    """
    对 Dataset 对象重新进行类别不平衡划分。

    划分规则
    ----
    1. 训练集占总节点数 10%；
    2. 验证集占总节点数 10%；
    3. 测试集为剩余节点，约占 80%；
    4. 训练集按照长尾类别不平衡划分；
    5. 验证集尽量按类别均衡划分；
    6. 测试集使用剩余节点。

    注意
    ----
    这个函数会覆盖 target_data 中原来的：
    train_masks / val_masks / test_masks
    idx_train / idx_val / idx_test
    mask_train / mask_val / mask_test
    unlabel_masks / idx_unlabel / mask_unlabel
    """
    random.seed(shuffle_seed)
    np.random.seed(shuffle_seed)

    device = target_data.device

    labels_cpu = target_data.labels.detach().cpu().long()
    total_items = labels_cpu.shape[0]
    n_classes = int(labels_cpu.max().item()) + 1

    shuffled_indices = list(range(total_items))
    random.shuffle(shuffled_indices)
    shuffled_indices = torch.tensor(shuffled_indices, dtype=torch.long)

    imb_ratio = resolve_imb_ratio(imb_level)

    class_num_list, indices, inv_indices = sort(data=target_data)

    n_train = int(total_items * 0.1)
    n_val = int(total_items * 0.1)

    class_num_list_train = split_lt(
        class_num_list=class_num_list,
        indices=indices,
        inv_indices=inv_indices,
        imb_ratio=imb_ratio,
        n_cls=n_classes,
        n=n_train
    )

    # 验证集每类尽量均衡
    n_val_per_class = n_val // n_classes
    class_num_list_val = torch.full((n_classes,), n_val_per_class, dtype=torch.long)

    remainder = n_val % n_classes
    if remainder > 0:
        class_num_list_val[:remainder] += 1

    train_indices = []
    val_indices = []
    test_indices = []

    for cls in range(n_classes):
        class_mask = (labels_cpu[shuffled_indices] == cls)
        class_indices = shuffled_indices[class_mask]

        num_samples_in_class = len(class_indices)

        # 当前类别训练样本数
        n_train_samples = int(class_num_list_train[cls].item())
        n_train_samples = min(n_train_samples, num_samples_in_class)

        train_class_indices = class_indices[:n_train_samples]
        train_indices.extend(train_class_indices.tolist())

        remaining_class_indices = class_indices[n_train_samples:]
        num_remaining = len(remaining_class_indices)

        # 当前类别验证样本数
        n_val_samples = int(class_num_list_val[cls].item())
        n_val_samples = min(n_val_samples, num_remaining)

        val_class_indices = remaining_class_indices[:n_val_samples]
        val_indices.extend(val_class_indices.tolist())

        # 剩余样本全部作为测试集
        test_class_indices = remaining_class_indices[n_val_samples:]
        test_indices.extend(test_class_indices.tolist())

    train_indices = np.array(train_indices, dtype=np.int64)
    val_indices = np.array(val_indices, dtype=np.int64)
    test_indices = np.array(test_indices, dtype=np.int64)

    # 检查三个集合互不重叠
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    assert len(set(train_indices) & set(val_indices)) == 0
    assert len(set(train_indices) & set(test_indices)) == 0
    assert len(set(val_indices) & set(test_indices)) == 0

    # 保存 numpy 形式索引
    target_data.train_masks = train_indices
    target_data.val_masks = val_indices
    target_data.test_masks = test_indices

    # 保存 torch 形式索引
    target_data.idx_train = torch.from_numpy(train_indices).long().to(device)
    target_data.idx_val = torch.from_numpy(val_indices).long().to(device)
    target_data.idx_test = torch.from_numpy(test_indices).long().to(device)

    # 当前划分使用了全部节点，因此未标注节点为空
    all_idx = np.arange(total_items)
    used_idx = np.unique(np.concatenate([train_indices, val_indices, test_indices]))
    unlabel_idx = np.setdiff1d(all_idx, used_idx, assume_unique=True)

    target_data.unlabel_masks = unlabel_idx
    target_data.idx_unlabel = torch.from_numpy(unlabel_idx).long().to(device)

    # 保存 bool mask
    target_data.mask_train = torch.zeros(total_items, dtype=torch.bool, device=device)
    target_data.mask_val = torch.zeros(total_items, dtype=torch.bool, device=device)
    target_data.mask_test = torch.zeros(total_items, dtype=torch.bool, device=device)
    target_data.mask_unlabel = torch.zeros(total_items, dtype=torch.bool, device=device)

    target_data.mask_train[target_data.idx_train] = True
    target_data.mask_val[target_data.idx_val] = True
    target_data.mask_test[target_data.idx_test] = True

    if len(unlabel_idx) > 0:
        target_data.mask_unlabel[target_data.idx_unlabel] = True

    # 额外保存类别数量，方便实验记录
    target_data.class_num_list_train = class_num_list_train

    if debug:
        train_counts = torch.bincount(
            target_data.labels[target_data.mask_train].detach().cpu().long(),
            minlength=n_classes
        ).tolist()

        val_counts = torch.bincount(
            target_data.labels[target_data.mask_val].detach().cpu().long(),
            minlength=n_classes
        ).tolist()

        test_counts = torch.bincount(
            target_data.labels[target_data.mask_test].detach().cpu().long(),
            minlength=n_classes
        ).tolist()

        print("\n" + "=" * 100)
        print(f"类别不平衡划分结果 | imb_ratio={imb_ratio}")
        print("=" * 100)
        print(f"训练集每类数量: {train_counts}")
        print(f"验证集每类数量: {val_counts}")
        print(f"测试集每类数量: {test_counts}")
        print(f"训练集总数: {sum(train_counts)}")
        print(f"验证集总数: {sum(val_counts)}")
        print(f"测试集总数: {sum(test_counts)}")
        print("=" * 100 + "\n")

    return target_data


def sample_per_class(labels, num_examples_per_class, forbidden_indices=None):
    num_samples = len(labels)
    num_classes = labels.max() + 1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [np.random.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])

def get_split(labels, train_examples_per_class=None, val_examples_per_class=None, test_examples_per_class=None,
              train_size=None, val_size=None, test_size=None):
    num_samples = len(labels)
    num_classes = labels.max() + 1
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = np.random.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = np.random.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))

    if test_examples_per_class is not None:
        test_indices = sample_per_class(labels, test_examples_per_class, forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = np.random.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    return train_indices, val_indices, test_indices


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=6789)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]   # 这一步可能不严谨

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices

# Local author-published partitions. No random generation or network fallback.
import hashlib
import json
from functools import lru_cache
from pathlib import Path

_GEOM_SOURCE = 'https://github.com/bingzhewei/geom-gcn/tree/master/splits'
_HETERO_SOURCE = 'https://github.com/yandex-research/heterophilous-graphs/tree/main/data'
_LINKX_SOURCE = 'https://github.com/CUAI/Non-Homophily-Large-Scale/tree/master/data/splits'

DEFAULT_SPLIT_PROTOCOLS = {}
for _name, _nodes in (
    ('cora', 2708), ('citeseer', 3327), ('pubmed', 19717),
    ('texas', 183), ('cornell', 183), ('wisconsin', 251), ('actor', 7600),
    ('chameleon', 2277), ('squirrel', 5201),
):
    DEFAULT_SPLIT_PROTOCOLS[_name] = dict(
        kind='local_author_masks', official=True, num_splits=10,
        num_nodes=_nodes, ratios=(0.48, 0.32, 0.20),
        reference='Geom-GCN public partitions (not regenerated 60/20/20)',
        source=_GEOM_SOURCE,
    )
for _name, _nodes in (('squirrel-f', 2223), ('chameleon-f', 890)):
    DEFAULT_SPLIT_PROTOCOLS[_name] = dict(
        kind='local_author_masks', official=True, num_splits=10,
        num_nodes=_nodes, ratios=None,
        reference='Platonov et al., ICLR 2023: filtered inherited masks',
        source=_HETERO_SOURCE,
    )
for _name, _nodes in (
    ('roman-empire', 22662), ('amazon-ratings', 24492),
    ('minesweeper', 10000), ('tolokers', 11758), ('questions', 48921),
):
    DEFAULT_SPLIT_PROTOCOLS[_name] = dict(
        kind='local_author_masks', official=True, num_splits=10,
        num_nodes=_nodes, ratios=(0.50, 0.25, 0.25),
        reference='Platonov et al., ICLR 2023: original benchmark masks',
        source=_HETERO_SOURCE,
    )
for _name, _nodes in (('penn94', 41554), ('genius', 421961), ('arxiv-year', 169343)):
    DEFAULT_SPLIT_PROTOCOLS[_name] = dict(
        kind='local_author_masks', official=True, num_splits=5,
        num_nodes=_nodes, ratios=(0.50, 0.25, 0.25),
        reference='LINKX benchmark, NeurIPS 2021: original fixed partitions',
        source=_LINKX_SOURCE,
    )
# Optional: use the OGB loader's temporal masks for ogbn-arxiv, a different task.
DEFAULT_SPLIT_PROTOCOLS['ogbn-arxiv'] = dict(
    kind='ogb_temporal_masks', official=True, num_splits=1,
    num_nodes=169343, ratios=None,
    reference='OGB official temporal split; distinct from arxiv-year',
    source='https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv',
)

_DEFAULT_ALIASES = {
    'film': 'actor', 'chameleon-filtered': 'chameleon-f',
    'chameleon_filtered': 'chameleon-f', 'squirrel-filtered': 'squirrel-f',
    'squirrel_filtered': 'squirrel-f', 'roman_empire': 'roman-empire',
    'amazon_ratings': 'amazon-ratings', 'arxiv_year': 'arxiv-year',
    'fb100-penn94': 'penn94', 'obgn-arxiv': 'ogbn-arxiv',
}


def canonical_split_name(name):
    name = str(name).strip().lower()
    return _DEFAULT_ALIASES.get(name, name)


def get_num_default_splits(dataset_name):
    name = canonical_split_name(dataset_name)
    if name not in DEFAULT_SPLIT_PROTOCOLS:
        raise ValueError(f'No public split protocol is registered for {dataset_name!r}.')
    return DEFAULT_SPLIT_PROTOCOLS[name]['num_splits']


def _split_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _split_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'{name} must be an integer, received {value!r}.')
    return int(value)


def _split_labels_sha256(labels):
    return hashlib.sha256(np.asarray(labels).reshape(-1).astype('<i8').tobytes()).hexdigest()


def _split_fingerprint(parts):
    digest = hashlib.sha256()
    for key, indices in zip(('train', 'valid', 'test'), parts):
        indices = np.sort(np.asarray(indices, dtype=np.int64).reshape(-1))
        digest.update(key.encode('ascii'))
        digest.update(np.asarray([len(indices)], dtype='<i8').tobytes())
        digest.update(indices.astype('<i8').tobytes())
    return digest.hexdigest()


def _default_split_directory(split_dir=None, cache_dir=None):
    # Resolve the bundled files relative to this module, not the shell's cwd.
    location = split_dir if split_dir is not None else cache_dir
    if location is None:
        location = Path(__file__).resolve().parents[1] / 'data' / 'official_splits'
    return Path(location).expanduser().resolve()


@lru_cache(maxsize=16)
def _read_split_manifest(path_string, modified_ns, size):
    with Path(path_string).open(encoding='utf-8') as stream:
        manifest = json.load(stream)
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('datasets'), dict):
        raise ValueError(f'Unsupported official split manifest: {path_string}')
    return manifest


@lru_cache(maxsize=4)
def _read_local_mask_bank(path_string, modified_ns, size, expected_sha256):
    path = Path(path_string)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f'Split-file checksum mismatch: {path}. Restore the downloaded file.')
    with np.load(path, allow_pickle=False) as archive:
        if int(archive['schema_version'].item()) != 1:
            raise ValueError(f'Unsupported mask-file schema: {path}.')
        matrices = tuple(archive[key].copy() for key in
                         ('train_masks', 'val_masks', 'test_masks'))
        name = str(archive['dataset'].item())
        n_nodes = int(archive['num_nodes'].item())
        count = int(archive['num_splits'].item())
        for matrix in matrices:
            if matrix.dtype != np.bool_ or matrix.shape != (n_nodes, count):
                raise ValueError(f'Invalid mask shape or dtype in {path}.')
            matrix.setflags(write=False)
        if np.any((matrices[0] & matrices[1]) | (matrices[0] & matrices[2]) |
                  (matrices[1] & matrices[2])):
            raise ValueError(f'Overlapping training/validation/test masks in {path}.')
    return name, n_nodes, count, matrices


def _validate_official_parts(parts, labels):
    result = []
    for key, indices in zip(('train', 'valid', 'test'), parts):
        indices = _split_numpy(indices).reshape(-1)
        if indices.dtype.kind not in 'iu':
            raise TypeError(f'{key} indices must have an integer dtype.')
        indices = indices.astype(np.int64, copy=False)
        if np.any(indices < 0) or np.any(indices >= len(labels)):
            raise ValueError(f'{key} node indices are out of range.')
        if len(indices) == 0 or len(np.unique(indices)) != len(indices):
            raise ValueError(f'{key} indices are empty or contain duplicates.')
        if np.any(labels[indices] < 0):
            raise ValueError(f'{key} contains unlabeled nodes. Check labels and node ordering; '
                             'the original author partition will not be silently changed.')
        result.append(indices.copy())
    if (np.intersect1d(result[0], result[1]).size or
            np.intersect1d(result[0], result[2]).size or
            np.intersect1d(result[1], result[2]).size):
        raise ValueError('Training, validation and test sets overlap.')
    return tuple(result)


def get_default_split(dataset_name, labels, data=None, split_id=0,
                      split_seed=None, cache_dir=None, return_info=False,
                      split_dir=None):
    """Read an original public partition and return NumPy index arrays.

    All 19 requested datasets read local safe NPZ files from the downloaded
    bundle. This function never downloads, generates, or substitutes masks.
    ``split_id`` selects a partition independently of the training seed.
    ``split_seed`` is accepted only for backward compatibility and is ignored.
    ``cache_dir`` is an old-name alias for ``split_dir``.

    The graph must preserve the source node numbering. Node count and file
    checksums are always checked; label ordering is checked when the manifest
    includes source labels. For LINKX datasets, use the unmodified original
    PyG/LINKX/OGB node order (count alone cannot certify arbitrary reindexing).
    Optional ogbn-arxiv reads the OGB wrapper's official temporal masks.
    """
    name = canonical_split_name(dataset_name)
    if name not in DEFAULT_SPLIT_PROTOCOLS:
        raise ValueError(f'No default split is registered for {dataset_name!r}.')
    protocol = DEFAULT_SPLIT_PROTOCOLS[name]
    split_id = _split_integer(split_id, 'split_id')
    count = protocol['num_splits']
    if not 0 <= split_id < count:
        raise ValueError(f'{name}: split_id must be in [0, {count - 1}], not {split_id}.')
    labels = _split_numpy(labels).reshape(-1)
    if labels.dtype.kind not in 'iu':
        raise TypeError('labels must contain integer class IDs, with -1 for unlabeled nodes.')
    labels = labels.astype(np.int64, copy=False)
    if len(labels) != protocol['num_nodes']:
        raise ValueError(f'{name} requires {protocol["num_nodes"]} original nodes, '
                         f'not {len(labels)}. Check dataset version and node numbering.')
    n_labeled = int((labels >= 0).sum())
    if not n_labeled:
        raise ValueError('No labeled nodes are available.')
    file_path = None
    record = {}
    if protocol['kind'] == 'ogb_temporal_masks':
        if data is None:
            raise ValueError('ogbn-arxiv requires the official OGB data wrapper.')
        parts = []
        for key in ('train_mask', 'val_mask', 'test_mask'):
            value = data.get(key) if isinstance(data, dict) else getattr(data, key, None)
            if value is None:
                raise ValueError(f'The official OGB loader did not supply {key}.')
            mask = _split_numpy(value)
            if mask.dtype != np.bool_ or mask.shape not in ((len(labels),), (len(labels), 1)):
                raise ValueError(f'Invalid OGB mask: {key}.')
            parts.append(np.flatnonzero(mask.reshape(-1)))
    else:
        root = _default_split_directory(split_dir, cache_dir)
        manifest_path = root / 'manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(f'Official split manifest not found: {manifest_path}. '
                                    'Extract the downloaded data/official_splits directory, '
                                    'or pass its location as split_dir.')
        stat = manifest_path.stat()
        manifest = _read_split_manifest(str(manifest_path), stat.st_mtime_ns, stat.st_size)
        if name not in manifest['datasets']:
            raise ValueError(f'{name} is missing from the local split manifest.')
        record = manifest['datasets'][name]
        if record.get('num_nodes') != len(labels) or record.get('num_splits') != count:
            raise ValueError(f'Node/split count mismatch in the {name} manifest record.')
        file_path = root / f'{name}.npz'
        if not file_path.is_file():
            raise FileNotFoundError(f'Official split file not found: {file_path}.')
        stat = file_path.stat()
        bank_name, n_nodes, n_splits, matrices = _read_local_mask_bank(
            str(file_path), stat.st_mtime_ns, stat.st_size, record['file_sha256'])
        if bank_name != name or n_nodes != len(labels) or n_splits != count:
            raise ValueError(f'Mask bank does not match {name}.')
        expected_labels = record.get('labels_sha256')
        if expected_labels and _split_labels_sha256(labels) != expected_labels:
            raise ValueError(f'{name}: labels or node order differ from the source dataset. '
                             'Use the original labels and node indexing.')
        parts = tuple(np.flatnonzero(matrix[:, split_id]) for matrix in matrices)
    parts = _validate_official_parts(parts, labels)
    sizes = tuple(len(indices) for indices in parts)
    assigned = sum(sizes)
    fingerprint = _split_fingerprint(parts)
    if record and fingerprint != record['splits'][split_id]['fingerprint']:
        raise ValueError(f'{name}: selected indices differ from the author partition.')
    info = dict(
        dataset=name, split_id=split_id, num_splits=count,
        official=True, kind=protocol['kind'],
        reference=protocol['reference'], source=protocol['source'],
        split_seed=None, cache_path=str(file_path) if file_path else None,
        num_nodes=len(labels), num_labeled=n_labeled, sizes=sizes,
        ratios=tuple(size / n_labeled for size in sizes),
        ratios_of_assigned=tuple(size / assigned for size in sizes),
        nominal_ratios=protocol['ratios'], removed_unlabeled=0,
        assigned_labeled=assigned, unassigned_labeled=n_labeled - assigned,
        labels_verified=bool(record.get('labels_sha256')), fingerprint=fingerprint,
    )
    return (*parts, info) if return_info else parts
