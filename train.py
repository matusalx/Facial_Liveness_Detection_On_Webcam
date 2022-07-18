import numpy as np
import torch


classes = ('spoof', 'real')
idx_to_class = {i: j for i, j in enumerate(classes)}
class_to_idx = {value: key for key, value in idx_to_class.items()}


def train(model, optimizer, criterion, trainloader, validloader, checkpoint_path, n_epochs=1):
    min_val_loss = np.Inf
    n_epochs = n_epochs
    # Main loop
    for epoch in range(n_epochs):
        total_trues_predicts = 0
        all_predicts = 0
        # Initialize validation loss for epoch
        val_loss = 0
        running_loss = 0
        model.train()
        # Training loop
        for data in trainloader:
            # pass
            image = data['image']
            target = torch.tensor([class_to_idx[c] for c in data['target']])
            optimizer.zero_grad()
            # Generate predictions
            out = model(image)
            # Calculate loss
            loss = criterion(out, target)
            running_loss += loss
            # Backpropagation
            loss.backward()
            # Update model parameters
            optimizer.step()
            print('batch loss: ', loss)
        print("Training Loss: {:.6f}".format(running_loss/len(trainloader)))
        # Validation loop
        with torch.no_grad():
            model.eval()
            for data in validloader:
                image = data['image']
                target = torch.tensor([class_to_idx[c] for c in data['target']])

                # Generate predictions
                out = model(image)
                # Calculate loss
                loss = criterion(out, target)
                val_loss += loss

                total_trues_predicts += torch.sum(target==out.argmax(dim=1)).item()
                all_predicts += len(data['target'])
                print("batch_accuracy_score: {:.6f}".format(total_trues_predicts / all_predicts))
            print("validation Loss: {:.6f}".format(val_loss / len(validloader)))
            print("accuracy_score: {:.6f}".format(total_trues_predicts / all_predicts))
            # Average validation loss
            val_loss = val_loss / len(validloader)
            # If the validation loss is at a minimum
            if val_loss < min_val_loss:
                # Save the model
                torch.save(model.state_dict(), checkpoint_path)
                min_val_loss = val_loss
