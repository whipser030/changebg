
1. 输入和输出
输入json：
{
    "img_base64" : "XXXXX",
    "input_prompt" : "A serene forest floor in the early morning, covered with wet moss and soft sunlight, casting warm rays on fallen leaves",
    "mode" : 1,
    "mask_content" : 0, 
    "output_nums" : 1,
    "style_select" : "Anime"
}
其中：
img_pth 和 input_prompt 是必填项，其余的可以选填

输出json：
{
    "status": "success",
    "images_base64": ["XXXXX","XXXXX"],
    "images_info": "XXX"
}

2. docker images创建，Dockerfile文件内容：
FROM python:3.10.6
# 设置容器时间
RUN cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo 'Asia/Shanghai' >/etc/timezone
ENV LANG C.UTF-8      
# 设置code文件夹为工作目录
WORKDIR /test/code/
COPY . . 
# 安装依赖包
RUN pip3 install -r requirements.txt

创建命令：docker build -t changebg . 

3. docker 容器创建命令：

DOCKER_IMAGE_NAME=change_bg:latest
docker run -dit \
        --name change_bg_server \
        --gpus "device=2" \
        -p8893:8894 \
        --shm-size=512g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --privileged \
        ${DOCKER_IMAGE_NAME} \
        /bin/bash /test/code/start_server.sh

4. post访问  http://10.178.13.79:8893/change_bg
