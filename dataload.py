import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import os



def rgb_jittering(im):
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:, :, ch] += np.random.randint(-2, 2)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')

image_transformer = transforms.Compose([
    transforms.Resize((30,30), Image.BILINEAR)])

augment_tile = transforms.Compose([
    #transforms.RandomCrop(64),
    #transforms.Resize((75, 75), Image.BILINEAR),
    #transforms.Lambda(rgb_jittering),
    transforms.ToTensor(),])

permutations = np.load('./permutations_1000.npy')

class DataLoader(datasets.ImageFolder):
        def __getitem__(self, index):
            #try:
                #filename = self.data_dir + '/' + self.file_list[index]
                path,_=self.imgs[index]
                img = Image.open(path).convert('RGB')
                if np.random.rand() < 0.30:
                    img = img.convert('LA').convert('RGB')
                #print(img.size)
                #if img.size[0] != 255:
                img = image_transformer(img)
                s = float(img.size[0]) / 3
                a = s / 2
                #print(a)
                tiles = [None] * 9
                for n in range(9):
                #    print(n)
                    i = n / 3
                    j = n % 3
                    c = [a * i * 2 + a, a * j * 2 + a]
                    c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
                    tile = img.crop(c.tolist())
                    tile = augment_tile(tile)
                    # Normalize the patches indipendently to avoid low level features shortcut
                    m, s = tile.view(3, -1).mean(dim = 1).numpy(), tile.view(3, -1).std(dim = 1).numpy()
                    s[s == 0] = 1
                    norm = transforms.Normalize(mean = m.tolist(), std = s.tolist())
                    tile = norm(tile)
                    tiles[n] = tile
                order = np.random.randint(len(permutations))
                #print(data.size)
                data = [tiles[permutations[order][t]] for t in range(9)]
                
                #print(len(data),len(data[0]),len(data[0][0]),len(data[0][0][0]))#,len(data[0][0][0][0]))
                data = torch.stack(data, 0)
                #print(order)
                return data, int(order), tiles
            #except:
             #   pass
        def __len__(self):
            return len(self.imgs)
