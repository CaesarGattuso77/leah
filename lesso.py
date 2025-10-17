from math import log

def cal_shannon_ent(dataset):
    num_entries = len(dataset)
    labels_counts = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        if current_label not in labels_counts.keys():
            labels_counts[current_label] = 0
            labels_counts[current_label] += 1
        print("类别统计：", labels_counts)  #YES=2,NO=3
    shannon_ent = 0.0
    for key in labels_counts:  #key有两个：yes和no
        prob = float(labels_counts[key])/num_entries  
        shannon_ent -= prob*log(prob, 2)  
    return shannon_ent


def create_dataSet():
    """
    熵接近 1，说明“yes”和“no”两个类别的比例比较接近，数据集的不确定性较高。
    熵接近 0,类别越集中，数据集越“纯”或“确定性越强”
    """
    dataset = [[1, 1, 'yes'],[1,1, 'yes'],[1, 0, 'no'],[0, 1, 'no'],[0, 1, 'no']]
    labels = ['no suerfacing', 'flippers']
    return dataset, labels


dataset, labels = create_dataSet()
#print(cal_shannon_ent(dataset))


def split_dataset(dataset, axis, value):
    """
    按照指定特征(axis)的某个取值(value)划分数据集。
    会选出所有该特征等于 value 的样本，
    并且返回时会去掉这一列特征。

    参数：
        dataset: 原始数据集（二维列表，每一行是一个样本，每一列是一个特征，最后一列通常是标签）
        axis: 要划分的特征列索引（例如 0 表示第 1 个特征）
        value: 特征的目标取值（例如 'sunny'）

    返回：
        ret_dataset: 划分后的子数据集（不包含 axis 那一列）
    """
    ret_dataset = [] 
    for feat_vec in dataset:   #[0, 'sunny', 'yes'],    我想以天气情况来划分   axis=1,value=sunny
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]    
            reduced_feat_vec.extend(feat_vec[axis+1:]) 
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset   #[[1,'yes'],[0,'yes']]


dataset_test = [
    [1, 'sunny', 'yes'],
    [1, 'rainy', 'no'],
    [0, 'sunny', 'yes']
]
result = split_dataset(dataset_test, 0, 1)   #[['sunny', 'yes']  ,  [ 'rainy', 'no']],
#print(result)


#dataset_test = [
#    [1, 'sunny', 'yes'],
#    [1, 'rainy', 'no'],
#    [0, 'sunny', 'yes']
##



def choose_best_feature_split(dataset):
    """
    选择信息增益最大的特征索引，作为本轮划分的最优特征。

    参数：
        dataset: 数据集（二维列表，每行一条样本，最后一列是标签）
    返回：
        best_feature: 最优特征的索引位置
    """

    num_features = len(dataset[0])-1    
    base_entropy = cal_shannon_ent(dataset)
    best_info_gain = 0.0
    best_feature = 1
    for i in range(num_features):    #num_features=2,range(2)=(0,1)
        feat_list = [example[i] for example in dataset]    # [1, 'sunny', 'yes'] i=0  
        print(feat_list)
        unique_val = set(feat_list) #[sunny,rainy,sunny]
        print(unique_val)  #(sunny,rainy)
        new_entropy = 0.0
        for value in unique_val:
            sub_dataset = split_dataset(dataset, i, value)  
            prob = len(sub_dataset)/float(len(dataset))
            new_entropy += prob*cal_shannon_ent(sub_dataset)
        info_gain = base_entropy-new_entropy
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    return best_feature

loan_data = [

    ['高', '有工作', '良好', 'yes'],
    ['高', '无工作', '一般', 'no'],
    ['中', '有工作', '良好', 'yes'],
    ['低', '有工作', '差', 'no'],
    ['低', '无工作', '一般', 'no'],
    ['高', '有工作', '差', 'yes'],
    ['中', '无工作', '良好', 'yes'],
    ['低', '有工作', '良好', 'yes']
]

print(choose_best_feature_split(loan_data))

