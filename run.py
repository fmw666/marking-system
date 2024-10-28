from fastapi import FastAPI
from pydantic import BaseModel
from vis_result_api import image_rating, init_model


class RatingRequest(BaseModel):
    prompt: str
    item_type: str
    img_datas: list[str]


# 创建 FastAPI 应用
app = FastAPI()


# 创建一个路由，当访问根路径时返回 "Hello, World!"
@app.get("/api")
async def api_view():
    return {"code": 0, "message": "Hello, World!"}


# 检查节点是否初始化完成（好像又不需要）
@app.get("/check")
async def check_view():
    pass


# 评分系统接口
@app.post("/api/rating")
async def rating_view(req_data: RatingRequest):
    # img_datas = image_rating(req_data.prompt, req_data.item_type, req_data.img_datas)
    # return {"code": 0, "message": "", "data": {"img_datas": img_datas}}
    sorted_index = image_rating(req_data.prompt, req_data.item_type, req_data.img_datas)
    return {"code": 0, "message": "", "data": {"sorted_index": sorted_index}}


# 运行 FastAPI 应用
if __name__ == "__main__":
    init_model()

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
