import torch

import config
import sample
from dataset import myDataset
from train import collate_fn
from model import myModel
import torchvision.models as models

print("Sampling testing videos")

sample.sample_testing_videos()

print("Finished sampling")

encoder = models.resnet34(pretrained=True)
n_features = encoder.fc.out_features  # get dimensions of fc layer

model = myModel(encoder, n_features, config.class_count)
model.load_state_dict(torch.load("checkpoints/1.pth.tar"))

test_dataset = myDataset("dataset/test/")
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=True,
    batch_size=8,
    collate_fn=collate_fn,
    num_workers=0,
)

model.eval()
total, correct = 0, 0

print("Starts testing")
for (step, (image, label)) in enumerate(test_loader):
    model.zero_grad()

    output = model(image)
    _, output = output.max(1)
    print(output)
    print(label)
    check = [1 if output[i] == label[i] else 0 for i in range(len(output))]

    total += len(output)
    correct += sum(check)

print("Finishes testing")
print(correct, 'in', total, 'samples are correct')