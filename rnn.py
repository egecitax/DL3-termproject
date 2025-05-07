import numpy as np
import matplotlib.pyplot as plt

class RNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01 
        self.Why = np.random.randn(output_size, hidden_size) * 0.01 
        self.bh = np.zeros((hidden_size, 1)) * 0.01
        self.by = np.zeros((output_size, 1)) * 0.01

    def tanh(self, x):
        return np.tanh(x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def bce(self, y_hat, y_true):
        y_hat = np.clip(y_hat, 1e-8, 1 - 1e-8)
        return -(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        h_list = [h]
        for x in inputs:
            x = np.asarray(x).reshape(self.input_size, 1)
            h = self.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            h_list.append(h)
        y = np.dot(self.Why, h) + self.by
        y_hat = self.sigmoid(y)
        return y_hat, h, h_list

    def backward(self, inputs, y_hat, y_true, h_list, h):
        dwxh = np.zeros_like(self.Wxh)
        dwhh = np.zeros_like(self.Whh)
        dwhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dLdy = y_hat - y_true
        dwhy += np.dot(dLdy, h.T)
        dby += dLdy
        dh = np.dot(self.Why.T, dLdy)

        for t in reversed(range(len(inputs))):
            x_t = np.asarray(inputs[t]).reshape(self.input_size, 1)
            h_t = h_list[t + 1]
            h_prev = h_list[t]
            dh_raw = dh * (1 - h_t ** 2)
            dwxh += np.dot(dh_raw, x_t.T)
            dwhh += np.dot(dh_raw, h_prev.T)
            dbh += dh_raw
            dh = np.dot(self.Whh.T, dh_raw)

        self.Wxh -= self.lr * dwxh
        self.Whh -= self.lr * dwhh
        self.Why -= self.lr * dwhy
        self.bh -= self.lr * dbh
        self.by -= self.lr * dby

    def train(self, X_train, y_train, epochs):
        train_losses = []
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            for x, y_true in zip(X_train, y_train):
                y_hat, h, h_list = self.forward(x)
                loss = self.bce(y_hat, y_true)
                total_loss += loss

                pred = 1 if y_hat > 0.5 else 0
                if pred == y_true:
                    correct += 1

                self.backward(x, y_hat, np.array([[y_true]]), h_list, h)

            avg_loss = total_loss / len(X_train)
            train_losses.append(avg_loss[0][0])
            accuracy = correct / len(X_train)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss[0][0]:.4f} - Accuracy: {accuracy:.2%}")

        plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def predict(self, encoded_sentence):
        y_hat, _, _ = self.forward(encoded_sentence)
        return 1 if y_hat > 0.5 else 0
