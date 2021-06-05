from collections import defaultdict
import torch
import config
import sample
from dataset import myDataset, myPredictionDataset
from model import myModel
import torchvision.models as models

video_url = "dataset/prediction_videos/random.mp4"

print("Sampling prediction videos")

sample.sample_prediction_videos(video_url)

print("Finished sampling")

encoder = models.resnet34(pretrained=True)
n_features = encoder.fc.out_features  # get dimensions of fc layer

model = myModel(encoder, n_features, config.class_count)
model.load_state_dict(torch.load("checkpoints/1.pth.tar"))

prediction_dataset = myPredictionDataset("dataset/prediction/")
prediction_loader = torch.utils.data.DataLoader(
    prediction_dataset,
    shuffle=True,
    batch_size=8,
    num_workers=0,
)

model.eval()
total = 0
mmap = defaultdict(int)
for (step, image) in enumerate(prediction_loader):
    model.zero_grad()

    output = model(image)
    _, output = output.max(1)
    for i in output:
        mmap[i.item()] += 1

    total += len(output)

res = [[k, v / total] for k, v in mmap.items()]
res.sort(key=lambda x: x[1])

print(res)