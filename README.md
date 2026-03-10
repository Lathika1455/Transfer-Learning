# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset

Develop a binary classification model using a pretrained VGG19 to distinguish between defected and non-defected capacitors by modifying the last layer to a single neuron.
Train the model on a dataset containing images of various defected and non-defected capacitors to improve defect detection accuracy.
Optimize and evaluate the model to ensure reliable classification for capacitor quality assessment in manufacturing.



## DESIGN STEPS
## STEP 1:
Import required libraries.Then dataset is loaded and define the training and testing dataset.

## STEP 2:
initialize the model,loss function,optimizer. CrossEntropyLoss for multi-class classification and Adam optimizer for efficient training.

## STEP 3:
Train the model with training dataset.

## STEP 4:
Evaluate the model with testing dataset.

## STEP 5:
Make Predictions on New Data.

## PROGRAM
Include your code here
```python
# Load Pretrained Model and Modify for Transfer Learning

model = models.vgg19(weights = models.VGG19_Weights.DEFAULT)

for param in model.parameters():
  param.requires_grad = False

# Modify the final fully connected layer to match the dataset classes

num_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(num_features,1)


# Include the Loss function and optimizer

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

# Train the model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:LATHIKA SREE R")
    print("Register Number:212224040169")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()



```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot

<img width="888" height="845" alt="image" src="https://github.com/user-attachments/assets/b387764c-1e49-4313-baaf-6915c9df54e9" />


### Confusion Matrix

<img width="812" height="729" alt="image" src="https://github.com/user-attachments/assets/aedccec2-78aa-4ff6-b8c4-04d83fcd21bf" />


### Classification Report

<img width="611" height="235" alt="image" src="https://github.com/user-attachments/assets/95a9b66c-abd7-45cd-8fff-e1eff2dc599a" />

### New Sample Prediction

<img width="422" height="488" alt="image" src="https://github.com/user-attachments/assets/ca055168-f1ca-4222-b2c6-cadf5c5c3215" />


## RESULT

The VGG-19 model was successfully trained and optimized to classify defected and non-defected capacitors.

