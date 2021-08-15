import os
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import json
import numpy as np
import cv2

logging = tf.compat.v1.logging
logging.set_verbosity(logging.INFO)

flags = tf.compat.v1.flags
args = flags.FLAGS

# dataset
flags.DEFINE_string(name='local_train_data', default='./train_data.json', help='本地训练数据集')
flags.DEFINE_string(name='local_test_data', default='./test_data.json', help='本地测试集')
flags.DEFINE_float(name='train_rate', default='0.8', help='数据集中训练占比')
flags.DEFINE_integer(name='train_samples', default=0, help='训练集样本总数')
flags.DEFINE_integer(name='val_samples', default=0, help='验证机样本总数')
flags.DEFINE_integer(name='img_height', default=28, help='input image height')
flags.DEFINE_integer(name='img_width', default=28, help='input image width')
flags.DEFINE_integer(name='img_channel', default=3, help='input image width')
flags.DEFINE_integer(name='classes', default=2, help='num of classes')
# train
flags.DEFINE_integer(name='batch_size', default=4, help='batch size')
flags.DEFINE_integer(name='epochs', default=10, help='训练总的epoch数量')
flags.DEFINE_string(name='logdir', default='./logs', help='tensorboard 日志目录')
flags.DEFINE_string(name='modeldir', default='./models', help='模型保存目录')

# 构建训练数据集
dataset_mean = [0.5, 0.5, 0.5]
dataset_std = [0.5, 0.5, 0.5]


def read_image_pyfunc(file_name):
    # 读取图片
    img_path = bytes.decode(file_name.numpy(), encoding="utf8")
    img_decode = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # 将图片缩放为网络输入尺寸
    img_decode = cv2.resize(img_decode, (args.img_width, args.img_height))
    # 将图片归一化到[-1, 1]之间的浮点数
    img_decode = img_decode / img_decode.max()  # scale to [0,1]
    img_decode = (img_decode - np.array(dataset_mean).reshape(1, 1, 3)) / np.array(dataset_std).reshape(1, 1, 3)
    img_decode = tf.cast(img_decode, tf.float32)
    # 这里只对图像做了缩放、归一化等必须的操作，如果需要其他的如随机裁剪、颜色变换、灰度等等都在这里实现
    return img_decode


def preload(filename, label):
    # 该函数是tf.data.Dataset内部自动调用，调用一次操作一张图片
    # 该函数内部只能使用tf定义的函数，不能使用其他模块的函数
    image_data = tf.py_function(read_image_pyfunc, [filename], [tf.float32])
    return image_data, label


with open(args.local_train_data, 'r', encoding='utf-8') as f:
    info_dict = json.loads(f.readline().strip())
    # data_list 为 [ ['xxx.jpg', '0'], ['xxx.jpg', '1'], ['xxx.jpg', '0']... ]
    # list每个元素包含了 图片文件名 及 对应的类别
    data_list = info_dict['data_lst']
train_samples = int(np.floor(len(data_list) * args.train_rate))
args.val_samples = len(data_list) - train_samples
args.train_samples = train_samples

train_lst = data_list[:train_samples]
val_lst = data_list[train_samples:]

# 分别取出文件名、对应的label
train_lst, train_labels = zip(*[(l[0], int(l[1])) for l in train_lst])
# 将label变为one-hot形式
train_one_shot_labels = keras.utils.to_categorical(train_labels, args.classes).astype(dtype=np.int32)

# 示例化训练数据集对象
train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(train_lst), tf.constant(train_one_shot_labels)))
# 使用 map 实现对数据的预处理、设置预加载
train_ds = train_ds.map(preload, num_parallel_calls=1).prefetch(10)
# 设置 repeat次数、batch size大小、加载数据时的shuffle buffer大小
train_ds = train_ds.repeat().shuffle(10).batch(args.batch_size)
# 示例化验证集对象，操作和训练集类似
val_lst, val_labels = zip(*[(l[0], int(l[1])) for l in val_lst])
val_one_shot_labels = keras.utils.to_categorical(val_labels, args.classes).astype(dtype=np.int32)
valid_ds = tf.data.Dataset.from_tensor_slices((tf.constant(val_lst), tf.constant(val_one_shot_labels)))
# 这里 preload 函数使用同一个，如果训练集和验证集的数据预处理流程不一样，可以实现不同的preload就行了
valid_ds = valid_ds.map(preload, num_parallel_calls=1).prefetch(10)
valid_ds = valid_ds.repeat().batch(args.batch_size)

# 构建迭代器，循环训练的时候使用
train_iter = iter(train_ds)
val_iter = iter(valid_ds)


# 定义模型
class CLS_MODEL:
    def __init__(self, args):
        self.args = args

    def build_model(self):
        input = layers.Input(shape=(1, self.args.img_height, self.args.img_width, self.args.img_channel), name='input')
        net_input = layers.Reshape((self.args.img_height, self.args.img_width, 3), name='reshape')(input)
        x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', name='conv1')(net_input)
        x = layers.MaxPool2D(pool_size=2, strides=2, name='pool1')(x)

        x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', name='conv2')(x)
        x = layers.MaxPool2D(pool_size=2, strides=2, name='pool2')(x)

        x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', name='conv3')(x)
        x = layers.MaxPool2D(pool_size=2, strides=2, name='pool3')(x)

        x = layers.Flatten(name='flatten')(x)

        x = layers.Dense(units=256, name='dense')(x)

        x = layers.Dense(units=self.args.classes, name='logit')(x)
        logit = layers.Softmax(name='softmax')(x)

        return models.Model(inputs=input, outputs=logit)


# 建立模型
cls_model = CLS_MODEL(args)
model = cls_model.build_model()
model.summary()

# 定义损失函数
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

# 定义优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# 构建summary记录tensorboard
train_summary_writer = tf.summary.create_file_writer(os.path.join(args.logdir, 'train'))
val_summary_writer = tf.summary.create_file_writer(os.path.join(args.logdir, 'val'))

# 开始循环训练
train_steps_per_epoch = args.train_samples // args.batch_size + 1
val_steps_per_epoch = args.val_samples // args.batch_size + 1
accuracy_fn = tf.keras.metrics.CategoricalAccuracy()
step = 0
for epoch in range(args.epochs):
    for i in range(train_steps_per_epoch):
        step += 1
        # 取一个batch数据
        train_x, train_y = next(train_iter)
        with tf.GradientTape() as tape:
            # 前向运算，得到网络输出
            logit = model(train_x)
            # 求出损失
            train_loss = loss_fn(y_true=train_y, y_pred=logit)
            # 求出梯度
            grad = tape.gradient(train_loss, model.trainable_variables)
        # 反向传输优化参数
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        train_acc = accuracy_fn(y_true=train_y, y_pred=logit)
        # 记录损失和精度
        print('train ---> epoch: %d, step: %d, train_loss: %.5f, train_acc: %.5f' % (epoch, step, train_loss, train_acc))
        with train_summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss, step=step)
            tf.summary.scalar('train_acc', train_acc, step=step)
        # 训练20步，跑一轮验证集
        if step % 20 == 0:
            val_los = []
            val_acc = []
            for a in range(val_steps_per_epoch):
                val_x, val_y = next(val_iter)
                logit = model(val_x)
                loss = loss_fn(y_true=val_y, y_pred=logit)
                val_los.append(loss)
                acc = accuracy_fn(y_true=val_y, y_pred=logit)
                val_acc.append(acc)
            # 整个验证集上求平均
            val_los = tf.reduce_mean(val_los)
            val_acc = tf.reduce_mean(val_acc)
            print("val ---> epoch: %d, step: %d, val_loss: %.5f, val_acc: %.5f" % (epoch, step, val_los, val_acc))
            with val_summary_writer.as_default():
                tf.summary.scalar('val_loss', val_los, step=step)
                tf.summary.scalar('val_acc', val_acc, step=step)
    # 保存模型
    if (epoch+1) % 5 == 0:
        model.save(os.path.join(args.modeldir, str(epoch)))
