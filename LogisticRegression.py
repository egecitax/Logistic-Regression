import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate  # Öğrenme oranı
        self.max_iterations = max_iterations  # Maksimum iterasyon sayısı
        self.tolerance = tolerance  # Durdurma eşiği

    def fit(self, X, y, initial_params): #Gradient Yükselişi Algoritması
        params = initial_params
        num_params = len(initial_params)
        likelihood_history = []

        for _ in range(self.max_iterations):
            # Tahmin
            predictions = self.predict(X, params)

            # Log-likelihood hesaplama
            likelihood = self.log_likelihood(y, predictions)
            likelihood_history.append(likelihood)

            # Gradyan hesaplama
            gradient = self.compute_gradient(X, y, predictions)

            # Parametre güncelleme
            params += self.learning_rate * gradient

            # Durdurma koşulu
            if all(abs(grad) < self.tolerance for grad in gradient):
                break

        return params, likelihood_history

    def predict(self, X, params):
        # Tahmin fonksiyonu (logistic fonksiyon)
        return 1 / (1 + np.exp(-X @ params))

    def log_likelihood(self, y_true, y_pred):
        # Log-likelihood hesaplama
        return np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def compute_gradient(self, X, y_true, y_pred):
        # Gradyan hesaplama
        return X.T @ (y_true - y_pred) / len(y_true)