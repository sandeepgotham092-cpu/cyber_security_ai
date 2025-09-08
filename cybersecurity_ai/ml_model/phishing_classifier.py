
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle
import io
import mysql.connector
from urllib.parse import urlparse
import os

# MySQL configuration (update with your credentials)
MYSQL_CONFIG = {
    'host': 'DESKTOP-FJB12PK',  # Update for Render deployment
    'user': 'root',
    'password': 'Sandeep#@161',
    'database': 'phishing_detector'
}

# Extract features from URLs
def extract_url_features(urls):
    features = []
    for url in urls:
        if not isinstance(url, str) or not url.strip():
            url = "http://example.com"
        parsed = urlparse(url)
        domain = parsed.netloc or "example.com"
        path = parsed.path or ""
        query = parsed.query or ""
        feat = [
            len(url),
            len(domain),
            url.count('.'),
            url.count('-'),
            url.count('/'),
            len(query),
            int('http://' in url),
            len(domain.split('.')),
            sum(c.isdigit() for c in url),
            sum(c.isalpha() for c in url) / max(len(url), 1),
        ]
        features.append(feat)
    return np.array(features, dtype=np.float64)

# Load dataset based on mode
def load_dataset(mode):
    if mode == 'email':
        file_path = 'cybersecurity_ai/data/phishing_email.csv'
        data = pd.read_csv(file_path)
        data['text'] = data['text'].astype(str).fillna('')
        texts = data['text'].values
        labels = pd.to_numeric(data['label'], errors='coerce').fillna(0).astype(np.int64).to_numpy()
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(texts).toarray()
        return X, labels, vectorizer
    elif mode == 'phonecall':
        file_path = 'cybersecurity_ai/data/sms_spam.csv'
        data = pd.read_csv(file_path, encoding='latin-1')
        data['v2'] = data['v2'].astype(str).fillna('')
        texts = data['v2'].values
        labels = (data['v1'] == 'spam').astype(np.int64).to_numpy()
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(texts).toarray()
        return X, labels, vectorizer
    elif mode == 'url':
        file_path = 'cybersecurity_ai/data/phishing_urls.csv'
        data = pd.read_csv(file_path)
        data['URL'] = data['URL'].astype(str).fillna('http://example.com')
        urls = data['URL'].values
        labels = pd.to_numeric(data['Label'], errors='coerce').fillna(0).astype(np.int64).to_numpy()
        X = extract_url_features(urls)
        return X, labels, None

# Custom Dataset
class ThreatDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Neural Network Model
class ThreatClassifier(nn.Module):
    def __init__(self, input_size):
        super(ThreatClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Save model and preprocessor to MySQL
def save_to_mysql(mode, model, preprocessor=None):
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()
    
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    model_weights = buffer.getvalue()
    
    preprocessor_data = None
    if preprocessor:
        buffer = io.BytesIO()
        pickle.dump(preprocessor, buffer)
        preprocessor_data = buffer.getvalue()
    
    cursor.execute("""
        INSERT INTO models (mode, model_weights, preprocessor)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
        model_weights = %s, preprocessor = %s, created_at = CURRENT_TIMESTAMP
    """, (mode, model_weights, preprocessor_data, model_weights, preprocessor_data))
    
    conn.commit()
    cursor.close()
    conn.close()

# Load model and preprocessor from MySQL
def load_from_mysql(mode, input_size):
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()
    
    cursor.execute("SELECT model_weights, preprocessor FROM models WHERE mode = %s", (mode,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not result:
        raise FileNotFoundError(f"Model for {mode} not found in database. Run train_model(mode='{mode}') first.")
    
    model_weights, preprocessor_data = result
    
    model = ThreatClassifier(input_size)
    buffer = io.BytesIO(model_weights)
    model.load_state_dict(torch.load(buffer))
    
    preprocessor = None
    if preprocessor_data:
        buffer = io.BytesIO(preprocessor_data)
        preprocessor = pickle.load(buffer)
    
    return model, preprocessor

def train_model(mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, labels, vectorizer = load_dataset(mode)
    input_size = X.shape[1]
    scaler = StandardScaler() if mode == 'url' else None
    if scaler:
        X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    train_dataset = ThreatDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = ThreatClassifier(input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"{mode.capitalize()} Epoch {epoch+1}, Loss: {loss.item():.2f}")
    preprocessor = scaler if mode == 'url' else vectorizer
    save_to_mysql(mode, model, preprocessor)

    test_dataset = ThreatDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"{mode.capitalize()} Accuracy: {correct / total:.4f}")

def detect_threat(input_text, mode):
    if not isinstance(input_text, str) or not input_text.strip():
        raise ValueError("Input must be a non-empty string")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode in ['email', 'phonecall']:
        model, vectorizer = load_from_mysql(mode, input_size=1000)
        features = vectorizer.transform([input_text]).toarray()
    elif mode == 'url':
        model, scaler = load_from_mysql(mode, input_size=10)
        features = extract_url_features([input_text])
        features = scaler.transform(features)

    model = model.to(device)
    model.eval()

    if features.ndim == 1:
        features = features.reshape(1, -1)
    input_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred = torch.argmax(probabilities, dim=1).item()
        conf = probabilities[0][pred].item()
    return pred, conf

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        train_model(mode=sys.argv[1])
    else:
        train_model(mode='email')
        train_model(mode='phonecall')
        train_model(mode='url')
