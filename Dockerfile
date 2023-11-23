FROM python:3.10.6
# 设置容器时间
RUN cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo 'Asia/Shanghai' >/etc/timezone
ENV LANG C.UTF-8      
#设置code文件夹为工作目录
WORKDIR /test/code/
COPY . . 
# 安装依赖包
RUN pip3 install -r requirements.txt
# CMD python3 run_img2img.py $PARAMS




# v2 
# FROM python:3.10.6

# # 设置容器时间
# RUN cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo 'Asia/Shanghai' >/etc/timezone
# ENV LANG C.UTF-8      
# ENV PARAMS="init_p"  

# #设置code文件夹为工作目录
# WORKDIR /test/code/

# # #代码添加到code文件夹
# # ADD ./run_img2img.py /test/code/
# # ADD ./u2net /test/code/
# # ADD ./files /test/code/
# # ADD ./requirements.txt /test/code/
# COPY . . 
# # 安装依赖包
# RUN pip3 install -r requirements.txt

# CMD python3 run_img2img.py $PARAMS


# docker build -t changebg .   或者   docker build --no-cache -t changebg .
# docker run -it -d --name tmp_del   changebg  # my_image 放最后  这里hahaha 加不加引号 无所谓
# docker logs -f --tail 200 my_container
