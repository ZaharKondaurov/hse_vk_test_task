import numpy as np

from typing import List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score

from scipy.signal import butter, filtfilt

def generate_signal(f_low: float, f_high: float) -> (np.ndarray, int):
    duration = 1
    sr = 16_000

    ts = np.linspace(0, duration, duration * sr, endpoint=False)
    freqs = np.random.uniform(100, 4_000, 3)
    signals = np.sin(2 * np.pi * freqs[:, None] * ts[None, :])    # sine_wave = sin(2pi*f*t)

    output = signals.sum(axis=0)
    output += np.random.normal(0.0, np.sqrt(0.01), sr * duration)

    label = 0
    if np.any((f_low <= freqs) & (freqs <= f_high)):
        label = 1

    return output, label

def butterworth_filter(signal: np.ndarray, f_low: float, f_high: float, sr: int, order: int=4):
    normalized_low = f_low * 2 / sr     # Отнормируем частоты границы частот, как требуется в фильтре
    normalized_high = f_high * 2 / sr
    numerator, denominator = butter(order, [normalized_low, normalized_high], btype="band", )
    return filtfilt(numerator, denominator, signal)

def feature_extraction(signal: np.ndarray, sr: int) -> List:
    x = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.size, 1 / sr)

    spectral_centroid = (freqs * x).sum() / x.sum()
    spectral_bandwidth = np.sqrt(np.sum(x * (freqs - spectral_centroid) ** 2))
    return [spectral_centroid, spectral_bandwidth]

def process_audio_signal(f_low: float, f_high: float) -> tuple[np.ndarray, np.ndarray]:
    signal, label = generate_signal(f_low, f_high)
    filtered_signal = butterworth_filter(signal, f_low, f_high, 16_000)
    extracted_features = feature_extraction(filtered_signal, 16_000)
    # Вернём сигнал, и массив с фичами и меткой
    return signal, np.array(extracted_features + [int(label), ])


class FrequenciesDetector(nn.Module):

    def __init__(self):
        super(FrequenciesDetector, self).__init__()

        self.fc1 = nn.Linear(2, 64)
        self.relu1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        return x

class SignalDataset(Dataset):

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        super(SignalDataset, self).__init__()

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model(model, data_train, data_val, criterion, optimizer, epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train = SignalDataset(data_train[..., :-1], data_train[..., -1].type(torch.LongTensor))
    ds_val = SignalDataset(data_val[..., :-1], data_val[..., -1].type(torch.LongTensor))

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    model = model.to(device)
    criterion = criterion.to(device)
    optimizer = optimizer

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    for epoch in tqdm(range(epochs), desc="Training model"):
        model.train()
        total_train_loss = 0
        total_val_loss = 0

        train_acc = []
        val_acc = []
        for features, target in dl_train:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(features)

            preds = torch.argmax(output, dim=1).cpu().detach().numpy()
            train_acc.append(accuracy_score(target.cpu().detach().numpy(), preds))

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.detach().item()

        avg_train_acc = np.stack(train_acc, axis=0).mean()

        avg_train_loss = total_train_loss / len(ds_train)
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)

        model.eval()

        with torch.no_grad():
            for features, target in dl_val:
                features, target = features.to(device), target.to(device)

                output = model(features)

                preds = torch.argmax(output, dim=1).cpu().detach().numpy()
                val_acc.append(accuracy_score(target.cpu().detach().numpy(), preds))
                loss = criterion(output, target)
                total_val_loss += loss.detach().item()

        avg_val_acc = np.stack(val_acc, axis=0).mean()
        avg_val_loss = total_val_loss / len(ds_val)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)

        if epoch % 10 == 0:
            tqdm.write(
                f"Epoch {epoch}: train loss: {avg_train_loss:.3f}, train acc: {avg_train_acc:.2f}, test loss: {avg_val_loss:.3f}, test acc: {avg_val_acc:.2f}, lr: {optimizer.param_groups[0]['lr']}")

    return model, train_loss_list, val_loss_list, train_acc_list, val_acc_list


def test_model(model, test_data):
    model = model.to("cpu")

    model.eval()
    preds = []
    with torch.no_grad():
        for row in test_data:
            features, target = row[..., :-1], int(row[..., -1])
            label = model(features[None, :]).to("cpu").detach().numpy().squeeze()
            preds.append(label.argmax(axis=-1))
    preds = np.array(preds)

    print(f"Accuracy: {accuracy_score(test_data[..., -1], preds)}")
    print(f"F1 Score: {f1_score(test_data[..., -1], preds)}")
    print(f"Recall: {recall_score(test_data[..., -1], preds)}")
    print(f"Precision: {precision_score(test_data[..., -1], preds)}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(test_data[..., -1], preds)}")
    return preds
