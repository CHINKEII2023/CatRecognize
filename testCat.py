import os
import shutil
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 图像的尺寸
IMG_HEIGHT = 150
IMG_WIDTH = 150

def preprocess_images(image_directory, batch_size):
    datagen = ImageDataGenerator(rescale=1./255)

    image_generator = datagen.flow_from_directory(
        directory=image_directory,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='rgb',
        class_mode=None,  # 不需要标签
        batch_size=1,  # 处理一个图片
        shuffle=False,  # 不打乱顺序
    )
    return image_generator

# 加载模型
model = load_model(r'catModel.keras')

# 测试数据目录
test_directory = r'IMGs\testIMGs'

# 创建测试数据生成器
test_generator = preprocess_images(test_directory, 1)

# 创建结果目录
result_directory = r'IMGs\ResultIMGs'
os.makedirs(result_directory, exist_ok=True)

# 初始化计数器
correct_count = 0
total_count = 0

# 遍历测试集中的每一张图片
for i in range(len(test_generator)):
    # 获取图片路径
    image_path = test_generator.filepaths[i]
    
    # 获取图片
    image = test_generator[i]
    
    # 使用模型进行预测
    prediction = model.predict(image)
    
    # 根据预测结果移动图片
    if prediction[0][0] < 0.35:  # 预测为猫
        new_directory = os.path.join(result_directory, 'isCat')
        # 如果图片实际为猫，增加正确计数
        if 'isCat' in image_path:
            correct_count += 1
    else:  # 预测为非猫
        new_directory = os.path.join(result_directory, 'notCat')
        # 如果图片实际为非猫，增加正确计数
        if 'notCat' in image_path:
            correct_count += 1
        
    # 总计数器增加
    total_count += 1

    # 创建新的目录
    os.makedirs(new_directory, exist_ok=True)
    
    # 构建新的文件路径
    new_path = os.path.join(new_directory, os.path.basename(image_path))
    
    # 移动图片
    shutil.copy(image_path, new_path)

# 计算精度
accuracy = correct_count / total_count

print(f'Test accuracy: {accuracy:.2%}')
