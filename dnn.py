import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from config.base_config import OUTPUT_CONFIG
import matplotlib.pyplot as plt

class DNN(nn.Module):
    """DNN Architecture"""
    def __init__(self, input_dim=11):
        super(DNN, self).__init__()
        self.network = nn.Sequential(
            # First hidden layer: 11 -> 128
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Second hidden layer: 128 -> 64
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output layer: 64 -> 1
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

class DNNModel(BaseModel):
    """PyTorch DNN Model for Healthcare Fraud Detection"""
    def __init__(self):
        super().__init__("DNN")
        self.device = torch.device('cpu')
        self.build()
        
    def build(self):
        """Build DNN model with specified architecture"""
        self.model = DNN().to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the DNN model"""
        self.logger.start_timer()
        
        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=512)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(50):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            if X_val is not None and y_val is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 
                             os.path.join(OUTPUT_CONFIG['model_dir'], 'best_dnn.pth'))
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
                    
                self.logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, "
                               f"Val Loss = {val_loss:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load(
            os.path.join(OUTPUT_CONFIG['model_dir'], 'best_dnn.pth')
        ))
        
        # Plot training history
        self._plot_training_history(train_losses, val_losses)
        
        # Evaluate on validation set
        if X_val is not None and y_val is not None:
            self.evaluate(X_val, y_val, "Validation Set")
            
        return self.model
    
    def predict(self, X):
        """Predict probabilities"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            y_pred = self.model(X_tensor)
        return y_pred.cpu().numpy()
    
    def _plot_training_history(self, train_losses, val_losses):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('DNN Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        save_path = os.path.join(OUTPUT_CONFIG['figure_dir'], 'dnn_training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() 