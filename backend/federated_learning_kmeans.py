import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
def calculate_silhouette_score(data, labels):
    score = silhouette_score(data, labels)
    return score
#轮廓系数 取值范围在[-1, 1]之间，值越接近1表示聚类效果越好。

# 边缘设备或本地服务器定义
class LocalDevice:
    def __init__(self, data):
        self.data = data
        self.centroids = None  # 本地模型的聚类中心

    def train(self):
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(self.data)
        self.centroids = kmeans.cluster_centers_
        return kmeans.labels_

    def get_centroids(self):
        return self.centroids # 获取本地模型的聚类中心

    def get_data(self):
        return self.data

# 横向联邦学习
def horizontally_federated_learning(local_devices, num_epochs):
    num_clusters = 3
    num_features = 4
    global_centroids = np.zeros((num_clusters, num_features))
    best_centroids = np.zeros((num_clusters, num_features))
    score = []
    last_score = 0

    for epoch in range(num_epochs):
        # 每轮训练前，将全局聚类中心发送到所有边缘设备
        for device in local_devices:
            device.centroids = global_centroids

        # 在每个边缘设备上进行本地模型训练
        temp_score = 0
        for device in local_devices:
            label_predict = device.train()
            temp_score += calculate_silhouette_score(device.get_data(),label_predict)
        temp_score = temp_score / len(local_devices)

        # 聚合本地聚类中心到全局聚类中心
        total_centroids = np.zeros_like(global_centroids)
        for device in local_devices:
            total_centroids += device.get_centroids()
        global_centroids = total_centroids / len(local_devices)

        if temp_score >= last_score:
            best_centroids = global_centroids
        score.append(temp_score)
        last_score = temp_score
    return best_centroids,score

# 纵向联邦学习
def vertically_federated_learning(local_devices, num_epochs):
    num_clusters = 3
    num_features = 1000
    deminsions = []
    for device in local_devices:
        deminsions.append(len(device.get_data()[0]))
        if num_features > len(device.get_data()[0]):
            num_features = len(device.get_data()[0])
    combined_data = local_devices[0].get_data()
    for i in range(len(local_devices) - 1):
        combined_data = np.hstack((combined_data, local_devices[i + 1].get_data()))
    global_centroids = np.zeros((num_clusters, len(combined_data[0])))
    best_centroids = np.zeros((num_clusters, len(combined_data[0])))
    score = []
    last_score = 0

    for epoch in range(num_epochs):
        # 每轮训练前，将全局聚类中心发送到所有边缘设备
        rest = 0
        for i in range(len(local_devices)):
            local_devices[i].centroids = [row[rest:deminsions[i]] for row in global_centroids]
            rest += deminsions[i]

        # 将每个边缘设备上数据集进行融合
        combined_data = local_devices[0].get_data()
        for i in range(len(local_devices) - 1):
            combined_data = np.hstack((combined_data, local_devices[i + 1].get_data()))

        # 在主设备上进行本地模型训练
        temp_score = 0
        main_device = LocalDevice(combined_data)
        label_predict = main_device.train()
        temp_score += calculate_silhouette_score(main_device.get_data(),label_predict)
        global_centroids = main_device.get_centroids()

        if temp_score >= last_score:
            best_centroids = global_centroids
        score.append(temp_score)
        last_score = temp_score
    return best_centroids,score


# 迁移学习联邦学习
def transfer_learning_federated_learning(local_devices, num_epochs):
    num_clusters = 3
    num_features = 1000
    deminsions = []
    for device in local_devices:
        deminsions.append(len(device.get_data()[0]))
        if num_features > len(device.get_data()[0]):
            num_features = len(device.get_data()[0])

    score = []
    last_score = 0
    best_centroids = []
    for epoch in range(num_epochs):

        # 在每个边缘设备上进行本地模型训练
        temp_score = 0
        for device in local_devices:
            label_predict = device.train()
        combined_centroids = local_devices[0].get_centroids()
        for i in range(len(local_devices) - 1):
            combined_centroids = np.vstack((combined_centroids, local_devices[i + 1].get_centroids()))

        # 在主设备上进行本地模型训练
        temp_score = 0
        main_device = LocalDevice(combined_centroids)
        label_predict = main_device.train()
        temp_score += calculate_silhouette_score(main_device.get_data(),label_predict)
        if temp_score > last_score:
            best_centroids = main_device.get_centroids()
        score.append(temp_score)
        last_score = temp_score

        # 聚合参数并广播给数据源
        rest = 0
        for i in range(len(local_devices)):
            local_devices[i].centroids = [row[rest:deminsions[i]] for row in main_device.get_centroids()]
            rest += deminsions[i]
    return best_centroids,score



