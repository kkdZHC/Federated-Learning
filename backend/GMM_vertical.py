import numpy as np
import pandas as pd
import federated_learning_GMM as fl

def GMM_vertical(start_feature,end_feature):
    # 创建边缘设备或本地服务器
    filename1 = "vertical1.csv"
    filename2 = "vertical2.csv"
    # start_feature = 1
    # end_feature = 3
    data1 = pd.read_csv(filename1)
    data2 = pd.read_csv(filename2)
    feature1 = data1.iloc[:, start_feature:end_feature].values
    feature2 = data2.iloc[:, start_feature:end_feature].values
    np.random.shuffle(feature1)
    np.random.shuffle(feature2)
    feature1 = feature1[:len(feature1)//2]
    feature2 = feature2[:len(feature2)//2]

    local_device_1 = fl.LocalDevice(feature1)
    local_device_2 = fl.LocalDevice(feature2)
    local_devices = [local_device_1, local_device_2]

    # 运行联邦学习过程
    num_epochs = 10
    final_global_means,final_global_covariances,scores = fl.vertically_federated_learning(local_devices, num_epochs)

    # 纵向联邦学习下聚类结果
    print("纵向联邦学习下最终的全局聚类中心:\n", final_global_means)
    print("纵向联邦学习下最终的全局协方差矩阵:\n", final_global_covariances)
    print("纵向联邦学习下每一轮的GMM聚类算法下的轮廓系数：")
    for i in range(len(scores)):
        print("第",i+1,"轮:",scores[i])

    # 设备1下聚类结果
    label1 = local_device_1.train()
    device1_means = local_device_1.get_means()
    device1_covariances = local_device_1.get_covariances()
    device1_score = fl.calculate_silhouette_score(local_device_1.get_data(),label1)
    print("设备1下聚类中心:\n", device1_means)
    print("设备1下协方差矩阵:\n", device1_covariances)
    print("设备1下的GMM聚类算法下的轮廓系数：\n",device1_score)

    # 设备2下聚类结果
    label2 = local_device_2.train()
    device2_means = local_device_2.get_means()
    device2_covariances = local_device_2.get_covariances()
    device2_score = fl.calculate_silhouette_score(local_device_2.get_data(),label2)
    print("设备2下聚类中心:\n", device2_means)
    print("设备2下协方差矩阵:\n", device2_covariances)
    print("设备2下的GMM聚类算法下的轮廓系数：\n",device2_score)

    return final_global_means,final_global_covariances,scores,device1_means,device1_covariances,device1_score,device2_means,device2_covariances,device2_score

GMM_vertical(1,3)
