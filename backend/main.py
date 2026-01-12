import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import service
import training_api

app = FastAPI(title="Battery Analysis API")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有源，生产环境请修改
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录，用于访问生成的图片
# 图片存储在 algorithm_training_platform/results
# main.py 在 algorithm_training_platform/backend
# 所以 results 目录是 ../results
results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

app.mount("/results", StaticFiles(directory=results_dir), name="results")
app.include_router(training_api.router)
app.include_router(training_api.predict_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to Battery Analysis API"}

@app.get("/api/battery/{battery_id}")
async def get_battery_stats(battery_id: int, plots: bool = False):
    """
    获取指定电池组的分析结果。
    - plots: 是否在后端生成静态图片（默认 False，建议前端使用 raw_data 自行绘图）
    """
    # 调用 service 中的 analyze_battery
    result = service.analyze_battery(battery_id, create_plots=plots)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    # 为图片添加完整的访问URL
    if "plots" in result:
        base_url = "/results/"
        for key in result["plots"]:
            filename = result["plots"][key]
            result["plots"][key] = base_url + filename
            
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
