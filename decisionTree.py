from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import numpy as np
class DecisionTree:
    def __init__(self, dataset):
        self.dataset = dataset
        self.df_encoded = None
        self.X = None
        self.y = None
        self.label_encoder = None
        self.dtc = None

    def preprocess_data(self):
        self.label_encoder = LabelEncoder()
        self.df_encoded = self.dataset.apply(self.label_encoder.fit_transform)

        self.X = self.df_encoded.iloc[:, :-1].values

        print(self.X)

        self.y = self.df_encoded.iloc[:, -1].values
        print(self.y)
    def calculate_entropy(self, data):
        classes, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def splitting_criteria(self):
        num_features = self.X.shape[1]
        entropies = []

        for feature_idx in range(num_features):
            values = np.unique(self.X[:, feature_idx])
            entropy_sum = 0

            for value in values:
                subset_indices = np.where(self.X[:, feature_idx] == value)
                subset_labels = self.y[subset_indices]
                entropy_subset = self.calculate_entropy(subset_labels)
                weight = len(subset_labels) / len(self.y)
                entropy_sum += weight * entropy_subset

            entropies.append(entropy_sum)

        information_gains = [self.calculate_entropy(self.y) - entropy for entropy in entropies]
        best_feature_idx = np.argmax(information_gains)

        return best_feature_idx, information_gains

    def train(self):
        self.preprocess_data()

        self.dtc = tree.DecisionTreeClassifier(criterion="entropy")
        self.dtc.fit(self.X, self.y)

    def predict(self, data):
        # encoded_data = self.label_encoder.transform(data)
        y_pred = self.dtc.predict([data])
        return self.label_encoder.inverse_transform(y_pred)