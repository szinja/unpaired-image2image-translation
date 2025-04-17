# from models.unet import UNet
# import torch, cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def predict(img_path, model_path):
#     model = UNet()
#     model.load_state_dict(torch.load(model_path))
#     model.eval()

#     image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
#     image = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()
#     with torch.no_grad():
#         pred = model(image).squeeze().numpy()
#         pred = (pred > 0.5).astype(np.uint8)

#     return pred