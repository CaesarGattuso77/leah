from math import log
import operator

def cal_shannon_ent(dataset):
    num_entries = len(dataset)
    labels_counts = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        if current_label not in labels_counts.keys():
            labels_counts[current_label] = 0
        labels_counts[current_label] += 1

        #print("类别统计：", labels_counts)
 
    shannon_ent = 0.0
    for key in labels_counts:
        prob = float(labels_counts[key])/num_entries
        shannon_ent -= prob*log(prob, 2)
    return shannon_ent


def create_dataSet():
    dataset = [['晴','弱','否'],
               ['晴','强','否'],
              ['阴','弱','是'],
               ['雨','弱','是'],
               ['雨','强','否']]
    labels = ['否', '是']
    return dataset, labels


dataset, labels = create_dataSet()
print(cal_shannon_ent(dataset))


def split_dataset(dataset, axis, value):

    ret_dataset = []  
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]    
            reduced_feat_vec.extend(feat_vec[axis+1:])  
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset



dataset_test = [
    [1, 'sunny', 'yes'],
    [1, 'rainy', 'no'],
    [0, 'sunny', 'yes']
]


result = split_dataset(dataset_test, 0, 1)
print(result)


def choose_best_feature_split(dataset):

    num_features = len(dataset[0])-1
    base_entropy = cal_shannon_ent(dataset)
    best_info_gain = 0.0
    best_feature = 1
    for i in range(num_features):
        feat_list = [example[i] for example in dataset]
        #feat_list = []
        #for example in dataset:
        #    feat_list.append(example[i])
        unique_val = set(feat_list)
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

#print(choose_best_feature_split(loan_data))

def majority_cnt(class_list):
    class_count={}
    for vote in class_list:
        if vote not in class_count.keys():class_count[vote]=0
        class_count[vote]+=1
    sorted_class_count=sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    return sorted_class_count


def creat_tree(dataset,labels):
    class_list=[example[-1] for example in dataset]  
    if class_list.count(class_list[0])==len(class_list):  
        return class_list[0]
    if len(dataset[0])==1:  
        return majority_cnt(class_list)
    best_feat=choose_best_feature_split(dataset)   #best_feat=0
    best_feat_label=labels[best_feat] 
    my_tree={best_feat_label:{}}   #my_tree={'晴':'否'}
    del(labels[best_feat])
    feat_values=[example[best_feat] for example in dataset]  
    unique_vals=set(feat_values)  
    for value in unique_vals:  
        sub_labels=labels[:]   
        my_tree[best_feat_label][value]=creat_tree(split_dataset(dataset,best_feat,value),sub_labels)
    return my_tree

my_data,labels=create_dataSet()
my_tree=creat_tree(my_data,labels)
