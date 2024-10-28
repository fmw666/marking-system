import base64
import requests


url = "https://u43695-8d7c-a1f29f86.westc.gpuhub.com:8443/api/rating"


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


prompt = "bags"
item_type = "luggage"

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

import sys


def task():
    print("start")
    img_datas = []
    for img_url in img_urls:
        base64_data = url_to_base64(img_url)
        # print(sys.getsizeof(base64_data) / (1024 * 1024))
        img_datas.append(base64_data)

    response = requests.post(
        url, json={"prompt": prompt, "item_type": item_type, "img_datas": img_datas}
    )

    # img_datas = response.json().get("data").get("img_datas")
    print(response.json())


# 用多线程跑
import concurrent.futures


if __name__ == "__main__":
    # 八个线程一起跑
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        # 等 midjourney 多实例再扩展
        for _ in range(8):
            futures.append(executor.submit(task))
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result(timeout=300)
            except concurrent.futures.TimeoutError:
                raise TimeoutError
