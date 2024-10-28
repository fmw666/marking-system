# 实现了给一个文件夹内图片给定prompt评分的功能
import base64
import time
import torch
import functools

from io import BytesIO
from PIL import Image
from typing import List
from transformers import AutoProcessor, AutoModel, CLIPModel, CLIPProcessor


# 设置使用不同的实验结果
# 训练好的模型 eg: /root/autodl-tmp/output/bottle_shape/best.pth"
trained_ckpt_template = "/root/autodl-tmp/output/{item_type}_{mark_type}/best.pth"

# 预训练的模型位置
model_pretrained_name_or_path = "/root/.cache/huggingface/hub/models--yuvalkirstain--PickScore_v1/snapshots/a4e4367c6dfa7288a00c550414478f865b875800"  # 预训练模型
processor_name_or_path = "/root/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/de081ac0a0ca8dc9d1533eed1ae884bb8ae1404b"  # 预处理器

device = "cuda"


ITEM_TYPE_MAP = {
    "shoes": "shoes",
    "luggage": "bags",
    "perfume_bottle": "toilette",
    "bluetooth_mouse": "mouse",
    "cup": "cup",
    "sound_box": "soundBox",
}

MODEL_CACHE = {}


@functools.lru_cache(maxsize=None)
def get_processor() -> CLIPProcessor:
    processor: CLIPProcessor = AutoProcessor.from_pretrained(processor_name_or_path)
    return processor


@functools.lru_cache(maxsize=None)
def get_model() -> CLIPModel:
    """
    不用缓存，单独加载时间为 1s
    """
    model: CLIPModel = (
        AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)
    )
    return model


@functools.lru_cache(maxsize=None)
def load_model(item_type: str, mark_type: str):
    """
    加载训练好的模型
    不用缓存，单独加载时间为 5s
    """
    trained_ckpt = trained_ckpt_template.format(
        item_type=ITEM_TYPE_MAP.get(item_type), mark_type=mark_type
    )
    return torch.load(trained_ckpt, map_location="cpu")


def init_model():
    """
    初始化模型，提前加载的内存中
    """
    print("正在加载模型到内存中...")
    start_time_total = time.time()
    get_processor()
    get_model()

    for item_type in ITEM_TYPE_MAP.keys():
        for mark_type in ["object", "shape"]:
            start_time_single = time.time()
            model = get_model()
            # load_state_dict 耗时 6s+
            model.load_state_dict(load_model(item_type, mark_type))
            MODEL_CACHE[f"{item_type}_{mark_type}"] = model
            print(
                f"加载 {item_type}_{mark_type} 模型成功. 耗时 {time.time() - start_time_single}s"
            )

    print(f"总耗时：{time.time() - start_time_total}s")


def calc_probs(model: CLIPModel, prompt: str, images: List[Image.Image]) -> List[float]:
    """
    计算图片和 prompt 的相似度
    """
    processor = get_processor()
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        # get probabilities if you have multiple images to choose from
        probs = torch.softmax(scores, dim=-1)

    return probs.cpu().tolist()


def image_rating(prompt: str, item_type: str, img_datas: List[str]) -> List[str]:
    """
    评分模块
    给定 prompt 和 item_type，给定一组图片 url，返回每张图片的评分
    根据评分总和排序，返回按评分降序排列的图片 url 列表

    输入参数：
    prompt: str, 给定的 prompt
    item_type: str, 给定的 item_type
    img_urls: List[str], 给定的图片 base64 列表

    输出参数：
    List[str], 按评分降序排列的图片 base64 列表

    {
      0: 1.2323,
      ...
    }
    """
    # base64 转 PIL.Image
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} - 开始处理图片...")
    images = [Image.open(BytesIO(base64.b64decode(img_data))) for img_data in img_datas]
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} - 图片处理完成")

    # key 为下标位置，否则存储 base64 内存压力太大
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} - 开始评分...")
    index_probs = {}
    for mark_type in ["object", "shape"]:
        model = MODEL_CACHE.get(f"{item_type}_{mark_type}")
        if model is None:
            model = get_model()
            model.load_state_dict(load_model(item_type, mark_type))
            MODEL_CACHE[f"{item_type}_{mark_type}"] = model
        probs = calc_probs(model, prompt, images)
        for i in range(len(probs)):
            index_probs[i] = index_probs.get(i, 0) + probs[i]
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} - 评分完成")

    sorted_index_probs = sorted(index_probs.items(), key=lambda x: x[1], reverse=True)
    # return [img_datas[i] for i, _ in sorted_index_probs]
    return [i for i, _ in sorted_index_probs]


if __name__ == "__main__":
    prompt = "bags"
    item_type = "luggage"

    init_model()

    import time

    MODEL_DICT = {}

    for i in range(5):
        print(f"开始第 {i} 次")
        start_time = time.time()

        for item_type in ITEM_TYPE_MAP.keys():
            for mark_type in ["object", "shape"]:
                if f"{item_type}_{mark_type}" in MODEL_DICT:
                    model = MODEL_DICT[f"{item_type}_{mark_type}"]
                else:
                    model = get_model()
                    model.load_state_dict(load_model(item_type, mark_type))
                    MODEL_DICT[f"{item_type}_{mark_type}"] = model
                print(f"{item_type}_{mark_type} 模型加载成功")

        print(f"第 {i} 次 耗时 {time.time() - start_time} s\n\n")

    import requests

    def url_to_base64(img_url):
        # 下载图像
        response = requests.get(img_url)
        # 确保请求成功
        if response.status_code == 200:
            # 将图像数据编码为Base64
            img_base64 = base64.b64encode(response.content).decode("utf-8")
            return img_base64
        else:
            raise Exception(f"Failed to retrieve image from URL: {img_url}")

    img_urls = [
        "https://img1.comixai.online/innoverse/works/202405132005_task331_result1.jpg",
        "https://img1.comixai.online/innoverse/works/202405132005_task331_result2.jpg",
        "https://img1.comixai.online/innoverse/works/202405132005_task331_result3.jpg",
        "https://img1.comixai.online/innoverse/works/202405132005_task331_result4.jpg",
        "https://img1.comixai.online/innoverse/works/202405141729_task342_result1.jpg",
        "https://img1.comixai.online/innoverse/works/202405141729_task342_result2.jpg",
        "https://img1.comixai.online/innoverse/works/202405141729_task342_result3.jpg",
        "https://img1.comixai.online/innoverse/works/202405141729_task342_result4.jpg",
    ]
    img_datas = []
    for img_url in img_urls:
        img_datas.append(url_to_base64(img_url))

    image_rating(prompt=prompt, item_type=item_type, img_datas=img_datas)
