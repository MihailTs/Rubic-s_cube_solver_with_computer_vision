from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils import data

train_transforms = transforms.Compose([
  transforms.ToTensor(), # convert the object into a tensor
  transforms.Resize((20, 20)), # resize the images to be of size 128x128
])

dataset_train = ImageFolder(
  'data/colors_train',
  transform=train_transforms,
)

dataloader_train = data.DataLoader(
  dataset_train,
  shuffle=True,
  batch_size=1,
)

image, label = next(iter(dataloader_train))
print(image.shape)