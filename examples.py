import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

from torchvision.datasets.oxford_iiit_pet import OxfordIIITPet

from transformer import build_encoder_decoder_model
from vision import Patchify
from utils import subsequent_mask
from training import LabelSmoothing


def encoder_decoder_inference_example():
    test_model = build_encoder_decoder_model(11, 11, n_layers=2, n_heads=4)
    test_model.eval()

    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for _ in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.head(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print('Examplary untrained model prediction: ', ys)


def label_smoothing_example():
    criterion = LabelSmoothing(5, 0, 0.1)
    preds = torch.Tensor([
        [0, 0.2, 0.7, 0.1, 0],
        [0, 0.2, 0.7, 0.1, 0]])
    
    criterion(x=preds, target=torch.LongTensor([1, 2]))
    print(criterion.true_dist)


def patchify_example():
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        Patchify((256, 256), 16)
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((256, 256))
    ])

    train_ds = OxfordIIITPet('./data', download=True, split='trainval', 
                       target_types='category', 
                       transform=train_transforms)
    test_ds = OxfordIIITPet('./data', split='trainval', 
                       target_types='category', 
                       transform=test_transforms)

    test_img = test_ds[0][0]
    plt.imshow(test_img)
    plt.show()

    train_img = train_ds[0][0]
    plt.figure(figsize=(10, 10))
    for i, flat_patch in enumerate(train_img):
        ax = plt.subplot(16, 16, i+1)
        ax.axis('off')
        ax.imshow(flat_patch.reshape((16, 16, 3)))
    plt.show()


if __name__ == '__main__':
    # encoder_decoder_inference_example()
    # label_smoothing_example()
    patchify_example()

