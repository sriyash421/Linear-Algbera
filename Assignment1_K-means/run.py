import numpy as np
from keras.datasets import mnist
from tqdm import trange
import matplotlib.pyplot as plt
np.random.seed(42)


def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train.reshape(
        x_train.shape[0], -1)/255.0, x_test.reshape(x_test.shape[0], -1)/255.0
    temp = []
    temp_target = []
    for i in range(10):
        temp.extend(x_train[np.where(y_train == i)[
                    0][np.random.permutation(100)]])
        temp_target.extend([i]*100)
    train = np.hstack((np.vstack(temp), np.vstack(temp_target)))
    test = np.hstack((x_test[:50], y_test[:50, None]))
    print(f"Train: {train.shape} Test: {test.shape}")
    np.random.shuffle(train)
    np.random.shuffle(test)
    return train, test


def plot_line(title, x, y):
    fig = plt.figure()
    fig.suptitle(title, fontsize=20)
    plt.plot(x, y)
    plt.show()


class KMeans():
    def __init__(self, data, num_clusters=20, init='choose'):
        self.data = data[:, :-1]
        self.target = data[:, -1]
        self.num_clusters = num_clusters
        self.feature_size = self.data.shape[1]
        self.train_size = self.data.shape[0]
        if init == 'random':
            self.centers = np.random.uniform(size=(self.num_clusters, self.feature_size))
        else:
            self.centers = self.data[np.random.randint(
                0, self.train_size, self.num_clusters)]
        self.class_labels = np.zeros(self.train_size)
        print(
            f"Data: {self.data.shape} Target: {self.target.shape} Centers: {self.centers.shape}")
        self.loss = []

    def calculate_distance(self, a, b):
        return np.linalg.norm(a-b, 2)

    def calculate_loss(self):
        loss = np.array([self.calculate_distance(self.data[i, :],
                                                 self.centers[int(self.class_labels[i]), :]) for i in range(self.train_size)])
        return np.mean(loss)

    def assign_labels(self, data, labels):
        for i in range(data.shape[0]):
            labels[i] = int(np.argmin(
                np.array([self.calculate_distance(data[i, :], x) for x in self.centers])))
        return labels

    def update_centers(self):
        for i in range(self.num_clusters):
            index = np.where(self.class_labels == i)[0]
            if index.shape[0] > 0 :
                self.centers[i, :] = np.mean(self.data[index], axis=0)
            else:
                self.centers[i,:] = np.zeros(self.feature_size)

    def get_cluster_target(self):
        cluster_target = -1*np.ones(self.num_clusters)
        for i in range(self.num_clusters):
            temp = np.bincount(np.array(self.target[np.where(self.class_labels == i)],dtype=np.int))
            if temp.shape[0] > 0:
                cluster_target[i] = int(np.argmax(temp, axis=0))
        return cluster_target

    def calculate_accuracy(self, data, target):
        self.cluster_target = self.get_cluster_target()
        cluster_labels = np.zeros(data.shape[0], dtype=np.int)
        cluster_labels = self.assign_labels(data, cluster_labels)
        predictions = self.cluster_target[cluster_labels]
        accuracy = np.mean(
            np.array(np.equal(predictions, target), dtype=np.float32))
        print(f"\tAccuracy: {accuracy}\n")

    def plot(self):
        image_size = int(np.sqrt(self.feature_size))
        Z = self.centers.reshape((self.num_clusters, image_size, image_size))
        fig, axs = plt.subplots(5, 4, figsize=(14, 14))
        plt.gray()
        for i, ax in enumerate(axs.flat):
            ax.matshow(Z[i])
            ax.axis('off')
        fig.suptitle("Cluster Representatives", fontsize=25)
        plt.show()

    def convergence_criterion(self):
        if len(self.loss) < 2:
            return False
        if self.loss[-1] == self.loss[-2]:
            return True
        return False

    def train(self, iterations, plot=False, test=None):
        i = None
        for i in trange(iterations):
            self.class_labels = self.assign_labels(self.data, self.class_labels)
            self.update_centers()
            self.loss.append(self.calculate_loss())
            if self.convergence_criterion():
                break
        if plot:
            print(f"\n\tConverged at iteration: {i+1}")
            self.plot()
            print(f"\tJ_clust: {self.loss[-1]}")
            plot_line("J_clust", range(len(self.loss)), self.loss)
            self.calculate_accuracy(test[:, :-1], test[:, -1])
            return None
        else:
            return self.loss[-1]


def search_k(data, init, kmin=5, kmax=20):
    j_clust = []
    k_range = range(kmin, kmax)
    for k in k_range:
        kmeans = KMeans(data, k, init)
        j_clust.append(kmeans.train(100))
    j_clust = np.array(j_clust)
    print(
        f"\n\tMin J_clust value: {np.min(j_clust)} at k= {k_range[np.argmin(j_clust)]}\n")
    plot_line("J_Cluster vs k", k_range, j_clust)


def run_k(data, test, k, init):
    kmeans = KMeans(data, k, init)
    kmeans.train(100, True, test)


if __name__ == "__main__":
    data, test = get_mnist_data()
    for i in ["random","choose"] :
        run_k(data, test, 20, i)
        search_k(data, i)
