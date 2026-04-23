import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import time
import os, socket, torch
import nvtx

class AnyresCLIPWrapper:
    def __init__(self, model_path="../models/clip-vit-large-patch14-336", device="cuda", patch_size=56):
        self.device = device
        self.patch_size = patch_size

        # 加载模型和预处理器
        self.model = CLIPModel.from_pretrained(model_path).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_path)

        # 图像 transform（不 resize，只做 normalize）
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processor.image_processor.image_mean,
                                 std=self.processor.image_processor.image_std),
        ])

    def split_into_patches(self, image: Image.Image):
        w, h = image.size
        patch_size = self.patch_size
        grid_w = (w + patch_size - 1) // patch_size
        grid_h = (h + patch_size - 1) // patch_size

        patches = []
        for i in range(grid_h):
            for j in range(grid_w):
                left = j * patch_size
                top = i * patch_size
                right = min((j + 1) * patch_size, w)
                bottom = min((i + 1) * patch_size, h)

                patch = image.crop((left, top, right, bottom))
                patch = patch.resize((patch_size, patch_size))  # resize to ViT input
                patches.append(patch)
        return patches, (grid_h, grid_w)

    def encode_image(self, image: Image.Image):
        # 切分图像为多个 patch
        patches, grid_shape = self.split_into_patches(image)

        pixel_values = torch.stack([self.normalize(patch) for patch in patches]).to(self.device)

        # 使用 CLIP 编码每个 patch
        with torch.no_grad():
            outputs = self.model.vision_model(pixel_values=pixel_values)
            patch_embeds = outputs.pooler_output  # shape: [num_patches, hidden_dim]

        return patch_embeds, grid_shape

    def match_text_image(self, image: Image.Image, texts):
        patch_embeds, _ = self.encode_image(image)
    
        # # 使用 projection 将 image 映射到 text 空间（768 dim）
        # patch_embeds = self.model.visual_projection(patch_embeds)  # [num_patches, 768]
        # image_embed = patch_embeds.mean(dim=0, keepdim=True)       # [1, 768]
    
        # # 文本编码
        # inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        # with torch.no_grad():
        #     text_embeds = self.model.text_model(**inputs).pooler_output  # [num_texts, ?]
        #     text_embeds = self.model.text_projection(text_embeds)        # [num_texts, 768]
    
        # # 归一化 & 相似度
        # image_embed = F.normalize(image_embed, dim=-1)
        # text_embeds = F.normalize(text_embeds, dim=-1)
    
        # logits_per_image = image_embed @ text_embeds.T
        # probs = logits_per_image.softmax(dim=-1)
        # return probs


if __name__ == "__main__":
    model = AnyresCLIPWrapper(model_path="../models/clip-vit-large-patch14-336", device="cuda", patch_size=336)

    # 读取任意分辨率图像
    image = Image.open("../images/dogs.png").convert("RGB")
    texts = ["a photo of a cat", "a photo of a flower", "a beautiful landscape"]

    print("我是man！")

    print("开始")
    for _ in range(1):
        # torch.cuda.synchronize()
        # host = os.environ["START_HOST"]
        # port = int(os.environ["START_PORT"])
        # s = socket.create_connection((host, port))
        # _ = s.recv(1)   # 阻塞等待“开跑”信号
        # s.close()
        # start = time.time()
        time.sleep(13)
        with nvtx.annotate("extra_vit", color="green"):
            probs = model.match_text_image(image, texts)
        print("hello")
        # torch.cuda.synchronize()
        # end = time.time()
        # # time.sleep(2)
        # print(f"总耗时: {end - start:.4f} 秒")

