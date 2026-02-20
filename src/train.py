import torch
import torch.nn as nn
import torch.optim as optim
import os


def fit(model, loaders, cfg):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    train_loader, validation_loader = loaders

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in range(cfg.epochs):
        # Train
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        validation_acc = correct / total 

        print(f"Epoch {epoch+1}done - validation_acc={validation_acc:.3f}")

    save_path = os.path.join("outputs", "runs", cfg.exp_name)
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))