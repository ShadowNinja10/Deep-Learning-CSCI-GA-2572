import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import VGG13_BN_Weights, vgg13_bn
from tqdm import tqdm

import torchvision.transforms.functional as d

DEVICE = "cuda"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def save_img(image, path):
    # Push to CPU, convert from (1, 3, H, W) into (H, W, 3)
    image = image[0].permute(1, 2, 0)
    image = image.clamp(min=0, max=1)
    image = (image * 255).cpu().detach().numpy().astype(np.uint8)
    # opencv expects BGR (and not RGB) format
    cv.imwrite(path, image[:, :, ::-1])


def main():
    model = vgg13_bn(VGG13_BN_Weights.IMAGENET1K_V1).to(DEVICE)
    print(model)
    for label in [0, 12, 954]:
        image = torch.randn(1, 224, 224, 3).to(DEVICE)
        image = (image * 8 + 128) / 255  # background color = 128,128,128
        image = image.permute(0, 3, 1, 2)
        image.requires_grad_()
        image = gradient_descent(image, model, lambda tensor: tensor[0, label].mean(),)
        save_img(image, f"./img_{label}.jpg")
        out = model(image)
        print(f"ANSWER_FOR_LABEL_{label}: {out.softmax(1)[0, label].item()}")


# DO NOT CHANGE ANY OTHER FUNCTIONS ABOVE THIS LINE FOR THE FINAL SUBMISSION


def normalize_and_jitter(img, step=8):
    # You should use this as data augmentation and normalization,
    # convnets expect values to be mean 0 and std 1
    dx, dy = np.random.randint(-step, step - 1, 2)
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(
        img.roll(dx, -1).roll(dy, -2)
    )


IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225], dtype=np.float32)
min_bound = torch.tensor((-IMAGENET_MEAN_1 / IMAGENET_STD_1).reshape(1, -1, 1, 1)).to(DEVICE)
max_bound = torch.tensor(((1 - IMAGENET_MEAN_1) / IMAGENET_STD_1).reshape(1, -1, 1, 1)).to(DEVICE)

def normalize(img):

    img = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(img)

    return img

def jitter(img, step=8):
    dx, dy = np.random.randint(-step, step - 1, 2)
    img = img.roll(dx, -1).roll(dy, -2)
    return img

def get_new_shape(original_shape, curr_level):
    ratio = 1.8
    scales = 4
    exponent = curr_level - scales + 1
    new_shape = np.round(np.float32(original_shape) * (ratio**exponent)).astype(np.int32)
    return new_shape



def blur(img_tensor):
  device = img_tensor.device
  dejittered_img = d.gaussian_blur(img_tensor, kernel_size=[5, 5], sigma=[0.45, 0.45])

  return dejittered_img


def gradient_ascent(input, model, loss, pyramid_level, iterations=1200,lr = 0.05):
  # input.data = normalize(input.data)

  pl=pyramid_level

  # if pl==0:
  #   lr=0.00005
  # if pl==1:
  #   lr=0.0005
  # if pl==2:
  #   lr=0.005
  # if pl==3:
  #   lr=0.055



  for i in tqdm(range(iterations)):
    input.data = jitter(input.data)
    out = model(input)
    target_loss = loss(out)
    target_loss.backward()

    grad = input.grad.data

    # smooth_grad = d.gaussian_blur(grad,kernel_size=[5, 5], sigma=[0.48, 0.48])

    smooth_grad = grad

    g_std = torch.std(smooth_grad)
    g_mean = torch.mean(smooth_grad)
    smooth_grad = smooth_grad - g_mean
    smooth_grad = smooth_grad / g_std

    input.data += lr * smooth_grad

    input.data = blur(input.data)
    input.grad.data.zero_()
    input.data = torch.max(torch.min(input, max_bound), min_bound)
  # input = jitter_2(input)
  with torch.no_grad():
    input_no = input.detach()
  # print(target_loss)



  return input_no

def gradient_descent(input, model, loss,lr = 0.06):
  original_shape=(224,224)
  input.data = normalize(input.data)
  for pyramid_level in range(4):
    new_shape = get_new_shape(original_shape, pyramid_level)

    input.data = torch.nn.functional.interpolate(input.data, (new_shape[0],new_shape[1]), mode='bilinear', align_corners=False)  # resize depending on the current pyramid level

    input = gradient_ascent(input, model, loss,pyramid_level)
    input.requires_grad = True

  return input




def forward_and_return_activation(model, input, module):
    """
    This function is for the extra credit. You may safely ignore it.
    Given a module in the middle of the model (like `model.features[20]`),
    it will return the intermediate activations.
    Try setting the modeul to `model.features[20]` and the loss to `tensor[0, ind].mean()`
    to see what intermediate activations activate on.
    """
    features = []

    def hook(model, input, output):
        features.append(output)

    handle = module.register_forward_hook(hook)
    model(input)
    handle.remove()

    return features[0]


if __name__ == "__main__":
    main()
