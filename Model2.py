import torch
import torch.nn as nn
import torch.optim as optim

class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TorchRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        h0 = torch.zeros(1, x.size(0), self.hidden_size)  # [num_layers, batch, hidden]
        out, _ = self.rnn(x, h0)  # out: [batch, seq_len, hidden]
        out = self.fc(out[:, -1, :])  # sadece son zaman adımının çıktısı
        out = self.sigmoid(out)
        return out

    def train_torch_rnn(model, X_train, y_train, epochs=20, learning_rate=0.01):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model.train()

        train_losses = []

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0

            for x_vecs, y_true in zip(X_train, y_train):
                # Veriyi tensöre çevir
                x_tensor = torch.stack([
                    torch.tensor(vec, dtype=torch.float32).squeeze(1) for vec in x_vecs
                ])
                x_tensor = x_tensor.unsqueeze(0)  # [1, seq_len, input_size]
                y_tensor = torch.tensor([[y_true]], dtype=torch.float32)

                # İleri geçiş
                output = model(x_tensor)
                loss = criterion(output, y_tensor)

                # Geri yayılım
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pred = 1 if output.item() > 0.5 else 0
                if pred == y_true:
                    correct += 1

            avg_loss = total_loss / len(X_train)
            accuracy = correct / len(X_train)
            train_losses.append(avg_loss)

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2%}")

        return train_losses
