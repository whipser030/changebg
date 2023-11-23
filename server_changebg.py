
import uvicorn
import base64
import os, re
from rembg import remove,new_session
from PIL import Image
from io import BytesIO
from pydantic import BaseModel
import json
import requests
from box import Box
from fastapi import FastAPI
import time

class InputConfig(BaseModel):
    img_pth: str = ""
    img_base64: str = ""
    input_prompt:str = ""
    mode: int = 1      # [0 1 2] for low middle high   
    mask_content: int = 0  # ["fill", "original"]   
    output_nums: int = 1
    rembg_model:str = "general-s"
    style_select:str = ""

def img_to_base64(filename):
    with open(filename, "rb") as file:
        data = file.read()

    base64_str = str(base64.b64encode(data), "utf-8")
    # return "data:image/png;base64," + base64_str
    return  base64_str


def read_json(json_file, to_obj=False):
    """加载json的列表文件"""
    dic = json.load(open(json_file,'r',encoding='utf8'))
    obj = Box(dic) if to_obj else dic
    return obj

def legal_base64(string_x):
    """
    判定base64编码是否合法
    """
    try:
        base64.b64decode(string_x)
        return True
    except:
        return False


def image_to_base64(image: Image.Image, fmt='png'):  # PIL  -> base64
    """
    输入PIL.Image信息，输出图片base64
    """
    output_buffer = BytesIO()
    image.save(output_buffer, format=fmt)
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return f'data:image/{fmt};base64,' + base64_str


def base64_to_image(base64_str: str) -> Image.Image:  # base64  -> PIL
    """
    输入图片base64，输出PIL.Image信息
    """
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    return img

def get_img64_maskimg(img_base64:str, model_type="silueta"):
    """
    输入图片base64信息，计算图片的后景抠图，以base64形式返回，并返回图片size
    """
    if not legal_base64(img_base64):
        raise ValueError("Input image is not correctly encoded in base64 mode.")

    img_PIL = base64_to_image(img_base64)
    session=new_session(model_type)
     
    out_PIL = remove(img_PIL, session=session, post_process_mask=True, only_mask = True)  # Image
    out_base64 = image_to_base64(out_PIL)
    ori_base64 = image_to_base64(img_PIL)
    return ori_base64, out_base64, img_PIL.size



def get_img_maskimg(img_pth:str, model_type="silueta"):
    """
    输入图片路径，计算图片的后景抠图，以base64形式返回，并返回图片size
    """
    assert os.path.exists(cfg.img_pth), 'Input image does not exist.'
    img_PIL = Image.open(img_pth)
    session=new_session(model_type)
     
    out_PIL = remove(img_PIL, session=session, post_process_mask=True, only_mask = True)  # Image
    out_base64 = image_to_base64(out_PIL)
    ori_base64 = image_to_base64(img_PIL)
    return ori_base64, out_base64, img_PIL.size

def gen_styled_prompt(usr_prompt, negprompt_plus, style_select:str, styles:dict):
    """
    选取style prompt中的模板描述，替换正反prompts
    """
    for template in styles:
        if 'name' not in template or 'prompt' not in template or 'negative_prompt' not in template:
            raise ValueError(
                "Invalid template. Missing 'name' or 'prompt' or 'negative_prompt' field.")
        if template["name"] == style_select:
            pos_prompt =  template.get("prompt").replace('{prompt}', usr_prompt) # pos做替换
            sug_neg_prompt =  template.get("negative_prompt")  # neg 做合并
            neg_prompt = f"{sug_neg_prompt}, {negprompt_plus}"

            return pos_prompt, neg_prompt
    raise ValueError(f"No template found with name {style_select}.")


def request_img2img(
                    url: str,
                    prompt:str,
                    negprompt:str,
                    payload: dict,
                    ):
    """
    调用API获取生成的图片base64，和info信息
    """
    payload["prompt"] = prompt
    payload["negative_prompt"] = negprompt
    # with open("./log.json","w") as f:  # 打印参数
    #     json.dump(payload,f,indent=2, separators=(',', ': '))

    headers = {'Content-Type': 'application/json'}  
    response = requests.post(url=f"{url}/sdapi/v1/img2img",headers=headers, json=payload)
    res = response.json()

    # 仅记录第一张图片的info返回
    png_payload = {"image": "data:image/png;base64," + res['images'][0]}
    res_info = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)
    imgs_info = res_info.json().get("info")

    return res['images'], imgs_info


def change_bg(cfg:InputConfig, pms:dict, styles:dict):
    assert cfg.mask_content in [0, 1], "Wrong, 'mask_content' should be chosen in [0, 1]."
    assert cfg.mode in [0, 1, 2], "Wrong, 'mode' should be chosen in [0, 1, 2]."
    assert cfg.rembg_model in pms.rembg_model_dict, "Wrong, 'rembg_model' should be chosen in['general-1', 'general-2', 'general-s', 'human', 'cloth', 'anime']."
    # 设置 CFG_Scale, Denoising_strength 参数
    if cfg.mask_content == 0:
        detail_obj = pms.inpaint_mode.fill
    else:
        detail_obj = pms.inpaint_mode.original

    if cfg.mode == 0:
        strength = detail_obj.low
    elif cfg.mode == 1:
        strength = detail_obj.detail.middle
    else:
        strength = detail_obj.detail.high

    # 计算 iter batch 
    batch_size = cfg.output_nums

    # 查找 style，并整理正反prompt内容
    if not cfg.input_prompt.strip():
        raise ValueError("Please input prompt.")
    if cfg.style_select:
        pos_prompt, neg_prompt = gen_styled_prompt(cfg.input_prompt, pms.payload.negative_plus, cfg.style_select, styles)
    else:
        pos_prompt = f"{cfg.input_prompt}, {pms.payload.prompt_plus}"
        neg_prompt = pms.payload.negative_plus
    
    # 处理图片 转base64  获得mask  计算照片大小
    rembg_sess = pms.rembg_model_dict[cfg.rembg_model] 

    if cfg.img_pth:
        img_str, mask_str, img_size = get_img_maskimg(cfg.img_pth, model_type=rembg_sess)
    elif cfg.img_base64:
        img_str, mask_str, img_size = get_img64_maskimg(cfg.img_base64, model_type=rembg_sess)
    else:
        raise ValueError("Unable to find the input image, please input image path or base64 info.")
    width, height = img_size

    # 整理请求
    add_payload = {
        "init_images": [img_str],
        "cfg_scale": strength.CFG_Scale,
        "denoising_strength": strength.Denoising_strength,
        "batch_size": batch_size,
        "n_iter": 1,
        "mask": mask_str,
        "width": width,
        "height": height,
        "inpaint_full_res": True, 
        "save_images": False,
        "do_not_save_samples": True,
        "inpainting_fill":cfg.mask_content
    }
    payload = pms.payload.copy()
    payload.update(add_payload)
    # 发送请求，获得返回的图片以及info
    imgs_base64, imgs_info = request_img2img(pms.url, pos_prompt, neg_prompt, payload)

    return {"images_base64":imgs_base64, "images_info": imgs_info}

app = FastAPI()

@app.post('/change_bg')
async def change_img_bg(cfg: InputConfig):
    pms = read_json("./files/params.json", to_obj=True)
    styles = read_json("./files/sdxl_styles.json")

    res_dict = change_bg(cfg, pms, styles) # keys = [images_info, images_base64]
    responses = {"status": "success"}
    responses.update(res_dict)
    return responses

if __name__ == "__main__":
    uvicorn.run(app="server_changebg:app", host="0.0.0.0", port=8894, reload=True)

