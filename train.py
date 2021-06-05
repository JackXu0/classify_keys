import torch
import torchvision.models as models

import config
import sample
from dataset import myDataset
from model import myModel


def collate_fn(batch):

    images = []
    labels = []
    for i, l in batch:
        images.append(i)
        labels.append(l)

    return torch.stack(images), torch.stack(labels)


def train(device, loader, model, criterion, optimizer):

    loss_epoch = 0
    model.train()
    for (step, (image, label)) in enumerate(loader):
        optimizer.zero_grad()

        image = image.to(device)
        label = label.to(device)

        output = model(image)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}", flush=True)

    return loss_epoch


if __name__ == '__main__':

    print("Sampling training videos")
    
    sample.sample_training_videos()
    
    print("Finishes sampling")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')

    train_dataset = myDataset("dataset/train/")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=16,
        collate_fn=collate_fn,
        num_workers=0,
    )

    start_epoch = 1
    encoder = models.resnet34(pretrained=True)
    n_features = encoder.fc.out_features  # get dimensions of fc layer

    model = myModel(encoder, n_features, config.class_count)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    criterion = torch.nn.CrossEntropyLoss()

    epochs = 100
    print("Starts training")
    for epoch in range(start_epoch, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(device, train_loader, model, criterion, optimizer)

        # save every 5 epochs
        # if epoch % 5 == 0:
        torch.save(model.state_dict(), "checkpoints/" + str(epoch) + ".pth.tar")

        print(f"Epoch [{epoch}/{epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}", flush=True)

    print("Finishes training")