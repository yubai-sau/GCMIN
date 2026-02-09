import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from matplotlib import cm
import spectral as spy
from scipy.io import loadmat
from sklearn import metrics
import time
from sklearn import preprocessing
import torch
import model
import slic_simple
import numpy as np


np.float = float  # 强制让 np.float 指向 Python 的 float
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 为DataLoader工作进程设置种子的函数
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


Seed_List = [0,1,2,3,4,5,6,7,8,9]  # Random seed points

torch.cuda.empty_cache()
# 画图
def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    numlabel = numlabel.astype(np.int16)
    plt.imshow(numlabel, cmap='gray')
    #v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    pass

#可视化
def GT_To_One_Hot(gt, class_count):  # 独热编码
    GT_One_Hot = []
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            temp = np.zeros(class_count, dtype=np.float32)
            if gt[i, j] != 0:
                temp[int(gt[i, j]) - 1] = 1
            GT_One_Hot.append(temp)
    GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
    return GT_One_Hot


def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
    real_labels = reallabel_onehot
    we = -torch.mul(real_labels, torch.log(predict + 1e-10))
    we = torch.mul(we, reallabel_mask)
    pool_cross_entropy = torch.sum(we)
    return pool_cross_entropy


def compute_crossentropy(index, data, gt, criteon):
    data_new = torch.cat([data[ind, :].unsqueeze(0) for ind in index], dim=0)
    return criteon(data_new, gt.long())


def SAM_vector(H_i, H_j):
    SAM_value = np.math.sqrt(torch.dot(H_i, H_i)) * np.math.sqrt(torch.dot(H_j, H_j))
    SAM_value = torch.tensor(SAM_value)
    SAM_value = torch.dot(H_i, H_j) / SAM_value
    if SAM_value > 1 or SAM_value < -1:
        SAM_value = 1
    SAM_value = np.math.acos(SAM_value)
    SAM_value = torch.tensor(SAM_value)
    return SAM_value

# 计算超像素初始值
def cross_superpixel_init(Q, A, net_input, idx_temp):
    # Calculate the initial value of the superpixel
    I = torch.eye(A.shape[0], A.shape[0], requires_grad=False).to(device)
    A = A + I

    [h, w, c] = net_input.shape

    # print(self.Q.shape)
    norm_col_Q = torch.sum(Q, 0, keepdim=True)
    x_HSI_flatten = net_input.reshape([h * w, -1])
    superpixels_flatten_HSI = torch.mm(Q.t(), x_HSI_flatten)

    V_HSI = superpixels_flatten_HSI / norm_col_Q.t().to(device)
    Z_HSI = x_HSI_flatten

    Q = Q.cpu().numpy()
    A = A.cpu().numpy()
    P_HSI = torch.zeros([h * w, superpixels_flatten_HSI.shape[0]]).to(device)
    for i in range(h * w):
        j = np.argwhere(Q[i])  # Find which superpixel block node i is in (j)
        index = np.argwhere(A[j].reshape(1, A.shape[0]))[:, 1]  # 1-Order neighbors of the jth superpixel block
        for k in range(len(index)):
            # print(index[k])
            P_HSI[i, index[k]] = torch.exp(-0.2 * SAM_vector(Z_HSI[i, :], V_HSI[k, :]))
            # P_HSI[i, index[k]] = torch.exp(-0.2 * torch.pow(torch.norm(Z_HSI[i, :] - V_HSI[k, :]), 2))

    P_H = P_HSI.cpu().numpy()
    sio.savemat('P_H_' + str(idx_temp) + '.mat', {'P_H': P_H})

    norm_col_P_HSI = torch.sum(P_HSI, 0, keepdim=True)

    H_HSI = torch.mm(P_HSI.t(), x_HSI_flatten)
    H_HSI = H_HSI / norm_col_P_HSI.t().to(device)

    return H_HSI, P_HSI


def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, require_AA_KPP=False,printFlag=True):
    if False == require_AA_KPP:
        with torch.no_grad():
            available_label_idx = (train_samples_gt != 0).float()
            available_label_count = available_label_idx.sum()
            correct_prediction = torch.where(
                torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),
                available_label_idx, zeros).sum()
            OA = correct_prediction.cpu() / available_label_count

            return OA
    else:
        with torch.no_grad():
            # OA
            available_label_idx = (train_samples_gt != 0).float()
            available_label_count = available_label_idx.sum()
            correct_prediction = torch.where(
                torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),
                available_label_idx, zeros).sum()
            OA = correct_prediction.cpu() / available_label_count
            OA = OA.cpu().numpy()

            zero_vector = np.zeros([class_count])
            output_data = network_output.cpu().numpy()
            train_samples_gt = train_samples_gt.cpu().numpy()
            train_samples_gt_onehot = train_samples_gt_onehot.cpu().numpy()

            output_data = np.reshape(output_data, [m * n, class_count])
            idx = np.argmax(output_data, axis=-1)
            for z in range(output_data.shape[0]):
                if ~(zero_vector == output_data[z]).all():
                    idx[z] += 1
            # idx = idx + train_samples_gt
            count_perclass = np.zeros([class_count])
            correct_perclass = np.zeros([class_count])
            for x in range(len(train_samples_gt)):
                if train_samples_gt[x] != 0:
                    count_perclass[int(train_samples_gt[x] - 1)] += 1
                    if train_samples_gt[x] == idx[x]:
                        correct_perclass[int(train_samples_gt[x] - 1)] += 1
            test_AC_list = correct_perclass / count_perclass
            test_AA = np.average(test_AC_list)
            # ================ 新增的 F1 计算部分 ================
            # 初始化混淆矩阵统计量
            TP_perclass = np.zeros(class_count)  # 真正例 (True Positive)
            FP_perclass = np.zeros(class_count)  # 假正例 (False Positive)
            FN_perclass = np.zeros(class_count)  # 假反例 (False Negative)

            # 遍历所有样本（包含有效样本）
            for x in range(len(train_samples_gt)):
                true_class = int(train_samples_gt[x])
                pred_class = int(idx[x])

                # 只处理有效样本（跳过背景类0）
                if true_class != 0:
                    true_index = true_class - 1  # 转换为0-based索引

                    # 统计TP和FN
                    if true_class == pred_class:
                        TP_perclass[true_index] += 1
                    else:
                        FN_perclass[true_index] += 1

                    # 统计FP（如果预测类别有效且预测错误）
                    if pred_class != 0:  # 只考虑有效类别的预测
                        pred_index = pred_class - 1  # 转换为0-based索引
                        if pred_class != true_class:
                            FP_perclass[pred_index] += 1

            # 计算每类的Precision, Recall和F1
            precision_perclass = np.zeros(class_count)
            recall_perclass = np.zeros(class_count)
            f1_perclass = np.zeros(class_count)

            for i in range(class_count):
                # 跳过没有真实样本的类别
                if count_perclass[i] == 0:
                    continue

                # 计算精确率 (Precision)
                if TP_perclass[i] + FP_perclass[i] > 0:
                    precision_perclass[i] = TP_perclass[i] / (TP_perclass[i] + FP_perclass[i])
                else:
                    precision_perclass[i] = 0.0

                # 计算召回率 (Recall)
                recall_perclass[i] = TP_perclass[i] / count_perclass[i]  # count_perclass是真实样本数

                # 计算F1分数
                if precision_perclass[i] + recall_perclass[i] > 0:
                    f1_perclass[i] = 2 * (precision_perclass[i] * recall_perclass[i]) / (
                                precision_perclass[i] + recall_perclass[i])
                else:
                    f1_perclass[i] = 0.0

            # 计算宏平均F1 (排除没有样本的类别)
            valid_classes = (count_perclass > 0)
            F1 = np.mean(f1_perclass[valid_classes]) if np.any(valid_classes) else 0.0
            # ================ F1 计算结束 ================
            # Kappa
            test_pre_label_list = []
            test_real_label_list = []
            output_data = np.reshape(output_data, [m * n, class_count])
            idx = np.argmax(output_data, axis=-1)
            idx = np.reshape(idx, [m, n])
            for ii in range(m):
                for jj in range(n):
                    if Test_GT[ii][jj] != 0:
                        test_pre_label_list.append(idx[ii][jj] + 1)
                        test_real_label_list.append(Test_GT[ii][jj])
            test_pre_label_list = np.array(test_pre_label_list)
            test_real_label_list = np.array(test_real_label_list)
            kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                              test_real_label_list.astype(np.int16))
            test_kpp = kappa

            # print
            if printFlag:
                print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                print('acc per class:')
                print(test_AC_list)

            OA_ALL.append(OA)
            AA_ALL.append(test_AA)
            F1_ALL.append(F1)
            KPP_ALL.append(test_kpp)
            AVG_ALL.append(test_AC_list)

            # save data
            os.makedirs("results", exist_ok=True)
            f = open('results/' + dataset_name + '_results.txt', 'a+')
            str_results = '\n======================' \
                          + " learning rate=" + str(learning_rate) \
                          + " epochs=" + str(max_epoch) \
                          + " train ratio=" + str(train_ratio) \
                          + " val ratio=" + str(val_ratio) \
                          + " ======================" \
                          + "\nOA=" + str(OA) \
                          + "\nAA=" + str(test_AA) \
                          + '\nkpp=' + str(test_kpp) \
                          + '\nacc per class:' + str(test_AC_list) + "\n"
            f.write(str_results)
            f.close()
            return OA


for (FLAG, curr_train_ratio, Scale) in [(0, 200, 400)]:
    torch.cuda.empty_cache()
    OA_ALL = []
    AA_ALL = []
    F1_ALL = []
    KPP_ALL = []
    AVG_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []

    import os
    import scipy.io as sio

    # 数据集根目录（无需重复定义 data_path1）
    data_path = os.path.join(r'/home/dell/data/baiyu/srl/SRGM/mygcn/dataset/data/Farmland/')

    # 加载数据（直接拼接文件名）
    #data1 = sio.loadmat(os.path.join(data_path, 'Farm1.mat'))['imgh']
    #data2 = sio.loadmat(os.path.join(data_path, 'Farm2.mat'))['imghl']
    #gt = sio.loadmat(os.path.join(data_path, 'GTChina1.mat'))['label']

    #data1 = sio.loadmat(os.path.join(data_path, 'USA_Change_Dataset.mat'))['T1']
    #data2 = sio.loadmat(os.path.join(data_path, 'USA_Change_Dataset.mat'))['T2']
    #gt = sio.loadmat(os.path.join(data_path, 'USA_Change_Dataset.mat'))['Binary']
    data1 = sio.loadmat(os.path.join(data_path, 'Farm1.mat'))['imgh']
    data2 = sio.loadmat(os.path.join(data_path, 'Farm2.mat'))['imghl']
    gt = sio.loadmat(os.path.join(data_path, 'GTChina1.mat'))['label']
    """
    plt.figure(figsize=(gt.shape[1] * 4 / 400, gt.shape[0] * 4 / 400), dpi=400)
    plt.imshow(gt, cmap='gray')  # 显示灰度图像
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('ground_truth6.png', format='png', transparent=True, dpi=400)
    """
    #gt = gt + 1

    samples_type = ['ratio', 'same_num'][FLAG]  # ratio or number

    # parameter preset
    val_ratio = 0.001
    class_count = 2  # class
    learning_rate = 0.0001  # learning rate
    max_epoch = 300  # iterations
    dataset_name = "hermiston"  # dataset name
    train_ratio = 0.1 if samples_type == "ratio" else curr_train_ratio
    superpixel_scale = Scale
    train_samples_per_class = curr_train_ratio
    val_samples = class_count
    m, n, d = data1.shape  # shape of dataset

    # standardization 归一化
    height, width, bands = data1.shape
    data1 = np.reshape(data1, [height * width, bands])
    minMax = preprocessing.StandardScaler()
    data1 = minMax.fit_transform(data1)
    data1 = np.reshape(data1, [height, width, bands])
    # orig_data = data2
    height, width, bands = data2.shape
    data2 = np.reshape(data2, [height * width, bands])
    minMax = preprocessing.StandardScaler()
    data2 = minMax.fit_transform(data2)
    data2 = np.reshape(data2, [height, width, bands])
    original_gt = gt.copy()
    print(gt.shape)
    print(data1.shape)
    # print the number of samples per class
    gt_reshape = np.reshape(gt, [-1])
    for i in range(class_count):
        idx = np.where(gt_reshape == i + 1)[-1]
        samplesCount = len(idx)
        print(i)
        print("标签数量")
        print(samplesCount)

    for curr_seed in Seed_List:
        random.seed(curr_seed)
        gt = original_gt.copy()
        gt_reshape = np.reshape(gt, [-1])
        train_rand_idx = []
        val_rand_idx = []
        if samples_type == 'ratio':
            train_data_index = []  # 初始化一维列表
            for i in range(class_count):
                idx = np.where(gt_reshape == i + 1)[-1]
                samplesCount = len(idx)
                if samplesCount == 0:
                    continue  # 跳过空类别
                rand_list = list(range(samplesCount))
                # 确保至少选择一个样本（避免 ceil(0) = 0）
                n_samples = max(1, np.ceil(samplesCount * train_ratio).astype('int32'))
                print(n_samples)
                rand_idx = random.sample(rand_list, n_samples)
                rand_real_idx_per_class = idx[rand_idx]
                train_data_index.extend(rand_real_idx_per_class.tolist())  # 直接合并到一维列表

            # 转换为 NumPy 数组并保存
            train_data_index = np.array(train_data_index)
            sio.savemat('train_index.mat', {'index': train_data_index})

            # 生成验证集和测试集索引
            train_set = set(train_data_index)
            all_data_index = set(range(len(gt_reshape)))
            background_idx = set(np.where(gt_reshape == 0)[-1])
            test_data_index = list(all_data_index - train_set - background_idx)
            val_data_count = int(val_ratio * (len(test_data_index) + len(train_data_index)))
            val_data_index = random.sample(test_data_index, val_data_count)
            test_data_index = list(set(test_data_index) - set(val_data_index))
            sio.savemat('test_index.mat', {'index': test_data_index})

            train_data_index = set(train_data_index)
            all_data_index = [i for i in range(len(gt_reshape))]
            all_data_index = set(all_data_index)

            # the index of the background
            background_idx = np.where(gt_reshape == 0)[-1]
            background_idx = set(background_idx)
            test_data_index = all_data_index - train_data_index - background_idx

            # the validation set
            val_data_count = int(val_ratio * (len(test_data_index) + len(train_data_index)))
            val_data_index = random.sample(list(test_data_index), val_data_count)  # 关键修改：list()转换
            val_data_index = set(val_data_index)
            test_data_index = test_data_index - val_data_index

            test_data_index = list(test_data_index)
            train_data_index = list(train_data_index)
            val_data_index = list(val_data_index)
            sio.savemat('test_index.mat', {'index': test_data_index})


        # train set 训练集标签
        train_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(train_data_index)):
            train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
            pass

        # test set 测试集标签
        test_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(test_data_index)):
            test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
            pass

        Test_GT = np.reshape(test_samples_gt, [m, n])  # 测试样本图

        # validation set 验证集标签
        val_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(val_data_index)):
            val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
            pass

        train_samples_gt = np.reshape(train_samples_gt, [height, width])
        nonzero_count = np.count_nonzero(train_samples_gt)

        print(f"Number of non-zero labels: {nonzero_count}")
        test_samples_gt = np.reshape(test_samples_gt, [height, width])
        val_samples_gt = np.reshape(val_samples_gt, [height, width])

        print(f"修正后的类别数: {class_count}")
        print(f"标签统计 - 最小值: {test_samples_gt.min()}, 最大值: {test_samples_gt.max()}")
        print(f"唯一标签值: {np.unique(test_samples_gt)}")

        # 检查是否有超出类别范围的标签
        out_of_range = np.sum((test_samples_gt < 0) | (test_samples_gt >= class_count))
        print(f"超出范围的标签数量: {out_of_range}")

        train_samples_gt_onehot = GT_To_One_Hot(train_samples_gt, class_count)
        test_samples_gt_onehot = GT_To_One_Hot(test_samples_gt, class_count)
        val_samples_gt_onehot = GT_To_One_Hot(val_samples_gt, class_count)

        train_samples_gt_onehot = np.reshape(train_samples_gt_onehot, [-1, class_count]).astype(int)
        test_samples_gt_onehot = np.reshape(test_samples_gt_onehot, [-1, class_count]).astype(int)
        val_samples_gt_onehot = np.reshape(val_samples_gt_onehot, [-1, class_count]).astype(int)

        # one-hot
        # train set
        train_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        train_samples_gt = np.reshape(train_samples_gt, [m * n])
        print("train_samples_gt:", train_samples_gt.shape)
        for i in range(m * n):
            if train_samples_gt[i] != 0:
                train_label_mask[i] = temp_ones
        train_label_mask = np.reshape(train_label_mask, [m * n, class_count])

        # test set
        test_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        test_samples_gt = np.reshape(test_samples_gt, [m * n])
        for i in range(m * n):
            if test_samples_gt[i] != 0:
                test_label_mask[i] = temp_ones
        test_label_mask = np.reshape(test_label_mask, [m * n, class_count])

        # validation set
        val_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        val_samples_gt = np.reshape(val_samples_gt, [m * n])
        for i in range(m * n):
            if val_samples_gt[i] != 0:
                val_label_mask[i] = temp_ones
        val_label_mask = np.reshape(val_label_mask, [m * n, class_count])

        ls = slic_simple.SP_SLIC(data1, data2, np.reshape(train_samples_gt, [height, width]), class_count - 1)
        tic0 = time.time()
        ##### Q1 = Q2
        Q1, S1, A1, Q2, S2, A2, Seg = ls.simple_superpixel(scale=superpixel_scale)
        #Q1, S1, A1, Q2, S2, A2, Seg = ls.simple_superpixel_no_LDA(scale=800)

        toc0 = time.time()
        SLIC_Time = toc0 - tic0

        print("SLIC costs time: {}".format(SLIC_Time))
        Q1 = torch.from_numpy(Q1).to(device)
        A1 = torch.from_numpy(A1).to(device)
        Q2 = torch.from_numpy(Q2).to(device)
        A2 = torch.from_numpy(A2).to(device)


        # transform to GPU
        train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
        test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
        val_samples_gt = torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)

        train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
        test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
        val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)

        train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
        test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
        val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)

        net_input_1 = np.array(data1, np.float32)
        net_input_1 = torch.from_numpy(net_input_1.astype(np.float32)).to(device)
        net_input_2 = np.array(data2, np.float32)
        net_input_2 = torch.from_numpy(net_input_2.astype(np.float32)).to(device)

        zeros = torch.zeros([m * n]).to(device).float()
        #kernel_size = 1  # 可在此处定义或从配置读取

        #net = model.Net(height, width, bands, class_count, Q1, A1, Q2, A2, Q3, A3, Q4, A4)
        net = model.Net(height, width, bands, class_count, Q1, A1, Q2, A2)
        net.to(device)

        # training 训练
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        best_loss = 99999
        net.train()
        tic1 = time.time()
        gt = torch.from_numpy(gt_reshape).to(device) - torch.ones_like(torch.from_numpy(gt_reshape)).to(device)
        gt_new = torch.cat([gt[ind].unsqueeze(0) for ind in train_data_index], dim=0)
        for i in range(max_epoch + 1):
            optimizer.zero_grad()  # zero the gradient buffers
            output = net(net_input_1, net_input_2)
            loss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
            loss.backward(retain_graph=False)
            optimizer.step()  # Does the update

            if i % 10 == 0:
                with torch.no_grad():
                    net.eval()
                    trainloss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
                    trainOA = evaluate_performance(output, train_samples_gt, train_samples_gt_onehot)
                    valloss = compute_loss(output, val_samples_gt_onehot, val_label_mask)
                    valOA = evaluate_performance(output, val_samples_gt, val_samples_gt_onehot)
                    print(
                        "{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(i + 1), trainloss, trainOA,
                                                                                         valloss, valOA))

                    if valloss < best_loss:
                        best_loss = valloss
                        os.makedirs("model", exist_ok=True)
                        torch.save(net.state_dict(), "model/best_model.pt")
                        print('save model...')
                torch.cuda.empty_cache()
                net.train()
        toc1 = time.time()
        print("\n\n====================training done. starting evaluation...========================\n")
        training_time = toc1 - tic1 + SLIC_Time
        Train_Time_ALL.append(training_time)

        # testing
        torch.cuda.empty_cache()
        with torch.no_grad():
            net.load_state_dict(torch.load("model/best_model.pt"))
            net.eval()
            tic2 = time.time()
            output = net(net_input_1, net_input_2)
            toc2 = time.time()
            testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask)
            testOA = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot, require_AA_KPP=True,
                                          printFlag=False)
            print("{}\ttest loss={}\t test OA={}".format(str(i + 1), testloss, testOA))


            testing_time = toc2 - tic2
            Test_Time_ALL.append(testing_time)

            classification_map = torch.argmax(output, 1) + 1
            background_idx = list(background_idx)
            classification_map[background_idx] = 0
            classification_map = classification_map.reshape([height, width]).cpu()
            Draw_Classification_Map(classification_map, "results/" + dataset_name + str(testOA))
            pred = np.array(classification_map)
            sio.savemat("results/pred.mat", {'pred': pred})



    torch.cuda.empty_cache()
    del net

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    F1_ALL = np.array(F1_ALL)
    KPP_ALL = np.array(KPP_ALL)
    AVG_ALL = np.array(AVG_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)
    print("\ntrain_ratio={}".format(curr_train_ratio),
          "\n==============================================================================")
    print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
    print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
    print('F1=', np.mean(F1_ALL), '+-', np.std(F1_ALL))
    print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
    print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
    print("Average training time:{}".format(np.mean(Train_Time_ALL)))
    print("Average testing time:{}".format(np.mean(Test_Time_ALL)))

    f = open('results/' + dataset_name + '_results.txt', 'a+')
    str_results = '\n\n************************************************' \
                  + "\ntrain_ratio={}".format(curr_train_ratio) \
                  + '\nOA=' + str(np.mean(OA_ALL)) + '+-' + str(np.std(OA_ALL)) \
                  + '\nAA=' + str(np.mean(AA_ALL)) + '+-' + str(np.std(AA_ALL)) \
                  + '\nKpp=' + str(np.mean(KPP_ALL)) + '+-' + str(np.std(KPP_ALL)) \
                  + '\nAVG=' + str(np.mean(AVG_ALL, 0)) + '+-' + str(np.std(AVG_ALL, 0)) \
                  + "\nAverage training time:{}".format(np.mean(Train_Time_ALL)) \
                  + "\nAverage testing time:{}".format(np.mean(Test_Time_ALL))
    f.write(str_results)
    f.close()