import numpy as np
from sklearn.model_selection import train_test_split
from scipy.signal import spectrogram
from scipy.interpolate import griddata
from skimage.transform import resize
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

# Nhập dữ liệu từ file
def load_data(data_path):
    X_train = np.load(f'{data_path}/A01T_X.npy')
    y_train = np.load(f'{data_path}/A01T_Y.npy')
    X_test = np.load(f'{data_path}/A01E_X.npy')
    y_test = np.load(f'{data_path}/A01E_Y.npy')

    for i in range(2, 10, 1):
        xt = np.load(f'{data_path}/A0{i}T_X.npy')
        yt = np.load(f'{data_path}/A0{i}T_Y.npy')
        xv = np.load(f'{data_path}/A0{i}E_X.npy')
        yv = np.load(f'{data_path}/A0{i}E_Y.npy')

        X_train = np.concatenate([X_train, xt])
        y_train = np.concatenate([y_train, yt])
        X_test = np.concatenate([X_test, xv])
        y_test = np.concatenate([y_test, yv])

    return X_train, X_test, y_train, y_test

# Nhập dữ liệu từ file rồi gộp lại
def load_data_and_mix(data_path):
    X_train = np.load(f'{data_path}/A01T_X.npy')
    y_train = np.load(f'{data_path}/A01T_Y.npy')
    X_test = np.load(f'{data_path}/A01E_X.npy')
    y_test = np.load(f'{data_path}/A01E_Y.npy')

    for i in range(2, 10, 1):
        xt = np.load(f'{data_path}/A0{i}T_X.npy')
        yt = np.load(f'{data_path}/A0{i}T_Y.npy')
        xv = np.load(f'{data_path}/A0{i}E_X.npy')
        yv = np.load(f'{data_path}/A0{i}E_Y.npy')

        X_train = np.concatenate([X_train, xt])
        y_train = np.concatenate([y_train, yt])
        X_test = np.concatenate([X_test, xv])
        y_test = np.concatenate([y_test, yv])
    
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    return X, y


# Hàm tính năng lượng phổ
def compute_band_energy(signal, fs, band):
    freqs, _, Sxx = spectrogram(signal, fs=fs, nperseg=256)
    band_energy = Sxx[(freqs >= band[0]) & (freqs <= band[1]), :].sum(axis=0)
    return band_energy.mean()
    # return Sxx

# Chuyển đổi tín hiệu thành ảnh
def create_spatial_map(trial_data, fs, bands, output_size=(32, 32)):
    spatial_maps = []
    for band_name, band_range in bands.items():
        # Tính năng lượng phổ cho mỗi kênh
        band_energies = [compute_band_energy(trial_data[channel], fs, band_range) for channel in range(trial_data.shape[0])]

        # Nội suy để tạo bản đồ không gian 32x32
        spatial_map = np.stack(band_energies, axis=-1)
        spatial_map = resize(spatial_map, (output_size[0], output_size[1]), mode='reflect')
        
        # Chuẩn hóa
        min_val = np.min(spatial_map)
        max_val = np.max(spatial_map)
        spatial_map = (spatial_map - min_val) / (max_val - min_val)  
        spatial_map = (spatial_map * 255).astype(np.uint8)  # Chuyển đổi sang uint8

        spatial_maps.append(spatial_map)

    # Ghép thành 3 kênh RGB-like
    return np.stack(spatial_maps[:3], axis=0)  # Chỉ lấy 3 băng tần (Theta, Alpha, Beta)

# Áp dụng cho toàn bộ dữ liệu
def process_all_trials(X, fs, bands, output_size=(32, 32)):
    processed_data = []
    
    for trial in tqdm(X):
        processed_data.append(create_spatial_map(trial, fs, bands, output_size))
    
    return np.array(processed_data)


class EEGSpectralDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Lấy ảnh và nhãn
        image = self.X[idx]
        label = self.y[idx]

        # Áp dụng biến đổi
        if self.transform:
            image = self.transform(image)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    

class EEGDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Lấy ảnh và nhãn
        image = self.X[idx]
        label = self.y[idx]

        # Áp dụng biến đổi
        if self.transform:
            image = self.transform(image)

        # Thêm chiều
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        # Chuyển kiểu dữ liệu cho phù hợp với CELoss
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label