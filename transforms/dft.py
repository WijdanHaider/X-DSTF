import torch
import torchvision.transforms as T

class DFTTransform:
    def __call__(self, image_tensor):
        if image_tensor.shape[0] > 1:
            grayscale_tensor = T.functional.rgb_to_grayscale(image_tensor)
        else:
            grayscale_tensor = image_tensor

        dft = torch.fft.fft2(grayscale_tensor.squeeze(0))
        dft = torch.fft.fftshift(dft)

        raw_magnitude = torch.abs(dft)
        log_magnitude = torch.log(raw_magnitude + 1e-8)
        phase = torch.angle(dft)

        return torch.stack([log_magnitude, phase, raw_magnitude], dim=0)


def get_transforms():
    spatial_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    freq_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        DFTTransform()
    ])

    return spatial_transform, freq_transform
