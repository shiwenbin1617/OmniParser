import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
import torch
from PIL import Image
import uvicorn

from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

# 创建FastAPI应用
app = FastAPI(
    title="OmniParser API",
    description="OmniParser是一个用于将GUI屏幕转换为结构化元素的屏幕解析工具",
    version="1.0.0"
)

# 初始化模型
# 加载YOLO检测模型
yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
# 加载图标描述模型
caption_model_processor = get_caption_model_processor(model_name="florence2",
                                                      model_name_or_path="weights/icon_caption_florence")

# 设置设备
DEVICE = torch.device('mps')


class ProcessResponse(BaseModel):
    """响应模型定义"""
    image: str  # Base64编码的结果图像
    parsed_content: str  # 解析出的屏幕元素内容


@app.post("/process", response_model=ProcessResponse)
async def process_image(
        image: UploadFile = File(...),  # 上传的图像文件
        box_threshold: float = Form(0.05),  # 边界框置信度阈值
        iou_threshold: float = Form(0.1),  # IOU阈值，用于移除重叠的边界框
        use_paddleocr: bool = Form(True),  # 是否使用PaddleOCR
        imgsz: int = Form(640)  # 图标检测的图像大小
) -> ProcessResponse:
    """
    处理上传的图像并返回解析结果

    参数:
    - image: 要处理的图像文件
    - box_threshold: 边界框置信度阈值，用于移除低置信度的边界框
    - iou_threshold: IOU阈值，用于移除重叠度较大的边界框
    - use_paddleocr: 是否使用PaddleOCR进行文本识别
    - imgsz: 图标检测的图像尺寸

    返回:
    - 处理后的图像和解析出的屏幕元素内容
    """
    try:
        # 读取上传的图像
        contents = await image.read()
        image_input = Image.open(io.BytesIO(contents))

        # 计算边界框覆盖比例
        box_overlay_ratio = image_input.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        # 执行OCR检测，获取文本区域
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            image_input,
            display_img=False,
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold': 0.9},
            use_paddleocr=use_paddleocr
        )
        text, ocr_bbox = ocr_bbox_rslt

        # 获取标注后的图像和解析内容
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image_input,
            yolo_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption_model_processor,
            ocr_text=text,
            iou_threshold=iou_threshold,
            imgsz=imgsz,
        )

        # 格式化解析内容
        parsed_content = '\n'.join([f'icon {i}: ' + str(v) for i, v in enumerate(parsed_content_list)])

        # 返回处理结果
        return ProcessResponse(
            image=dino_labled_img,  # 已经是base64编码的图像
            parsed_content=parsed_content
        )

    except Exception as e:
        # 异常处理
        raise HTTPException(status_code=500, detail=f"处理图像时发生错误: {str(e)}")


@app.get("/")
async def root():
    """API根路径，返回欢迎信息"""
    return {
        "message": "欢迎使用OmniParser API",
        "description": "这是一个用于解析GUI屏幕的API，将GUI屏幕转换为结构化元素",
        "usage": "发送POST请求到/process端点，上传图像进行处理"
    }


# 启动服务器
if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0", port=6006)
