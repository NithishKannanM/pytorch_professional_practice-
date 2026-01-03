import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("dataset/diabetes.csv")

X = df[['Pregnancies','Glucose','Insulin','BMI','Age']]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42  #train and test sets maintain the same class distribution 
)

scalar = StandardScaler() #normalise the feature mean = 0, standard-deviation = 1
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32) #convert numpy array to tensors 
y_train_t = torch.tensor(y_train.to_numpy(), dtype=torch.long)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test.to_numpy(), dtype=torch.long)

class_counts = torch.bincount(y_train_t) #calculate the output class count
class_weights = 1.0 / class_counts.float() #calculate the weights

print("Class counts:", class_counts)
print("Class weights:", class_weights)


sample_weights = class_weights[y_train_t] #assigning each sample its weights 

sampler = WeightedRandomSampler( #Rebalances sampling
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

count_neg = (y_train == 0).sum()
count_pos = (y_train == 1).sum()

pos_weight = torch.tensor([count_neg / count_pos], dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_t) # TensorDataset pairs X and y together so DataLoader can load them in batches
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader( train_dataset, batch_size=32, sampler=sampler)
test_loader = DataLoader( test_dataset, batch_size=32, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.net = nn.Sequential(
            # nn.Linear(X_train.shape[1],32),
            # nn.ReLU(),
            # nn.Linear(32,16),
            # nn.ReLU(),
            # nn.Linear(16,1)

            
            nn.Linear(X_train.shape[1], 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1)
        )
        
    def forward(self, X):
        return self.net(X)
model = Net()

#criterion = nn.CrossEntropyLoss(weight=class_weights)  # to punish the model in the mistakes
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) #

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 500   #training loop
model.train()

train_loss = []


for epoch in range(epochs):
    epoch_loss = 0

    for Xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(Xb).squeeze()
        loss = criterion(out,yb.float())
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    train_loss.append(epoch_loss)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, loss = {epoch_loss:.4f}")

model.eval()
y_pred_list = []
val_losses = []
val_epoch_loss = 0

with torch.no_grad():
    for Xb, yb in test_loader:
        # outputs = model(Xb)   #normal way
        # probs = torch.softmax(outputs, dim=1)
        # preds = torch.argmax(probs, dim=1)
        # y_pred_list.extend(preds.cpu().numpy())

        outputs = model(Xb).squeeze()
        preds = (outputs > 0).long()  # For binary classification with BCEWithLogitsLoss
        y_pred_list.extend(preds.cpu().numpy())
        loss = criterion(outputs, yb.float())
        val_epoch_loss += loss.item()

val_losses.append(val_epoch_loss)
model.train()



print("Accuracy :", accuracy_score(y_test, y_pred_list))
print("Precision:", precision_score(y_test, y_pred_list))
print("Recall   :", recall_score(y_test, y_pred_list))
print("F1 Score :", f1_score(y_test, y_pred_list))

plt.figure(figsize=(10,5))
plt.plot(train_loss, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.show()
