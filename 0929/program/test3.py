import os

# Set cache paths before importing transformers, including in existing IDE sessions.
os.environ['HF_HOME'] = r'E:\huggingface'
os.environ['HF_HUB_CACHE'] = r'E:\huggingface\hub'
# Use HTTP downloads because the Xet CDN fails TLS on this network.
os.environ['HF_HUB_DISABLE_XET'] = '1'

from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from transformers import pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent

IMAGE_PATH = DATA_DIR / "face.jpg"
OUTPUT_PATH = DATA_DIR / "face_depth.png"


if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"找不到图片：{IMAGE_PATH}")


print("正在加载深度估计模型...")

depth_estimator = pipeline(
    "depth-estimation",
    model="Intel/dpt-large"
)

print("模型加载成功")

image = Image.open(IMAGE_PATH).convert("RGB")

print("正在估计深度...")

result = depth_estimator(image)

depth = np.array(result["depth"]).astype(np.float32)

# 归一化
depth = (
    (depth - depth.min())
    / (depth.max() - depth.min() + 1e-8)
)

depth_img = (depth * 255).astype(np.uint8)

cv2.imwrite(
    str(OUTPUT_PATH),
    depth_img
)

print("深度图生成完成：")
print(OUTPUT_PATH)

cv2.imshow("Depth Map", depth_img)
cv2.waitKey(0)
cv2.destroyAllWindows()