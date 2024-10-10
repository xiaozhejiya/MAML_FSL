import tensorflow as tf
import numpy as np
from PIL import Image
import random
import os
import glob
from tqdm import tqdm

import ResNet

train_data_path = r'./omniglot/images_background'
test_data_path = r'./omniglot/images_evaluation'

n_way = 5
k_shot = 1
q_query = 5
outer_lr = 0.001
inner_lr = 0.04
meta_batch_size = 32
train_inner_step = 1
eval_inner_step = 3
num_iterations = 1000
num_workers = 0
valid_size = 0.2
random_seed = 42
display_gap = 50

random.seed(42)


def get_supportSet_and_querySet(file_list, n_way, k_shot, q_query):
    # 从文件列表中随机选择 n_way 个类目录
    img_dirs = random.sample(file_list, n_way)
    support_data = []
    query_data = []

    support_image = []
    support_label = []
    query_image = []
    query_label = []

    for label, img_dir in enumerate(img_dirs):
        # 查找该类目录下所有 .png 图像
        img_list = [f for f in glob.glob(img_dir + "**/*.png", recursive=True)]
        # 从该类中随机抽取 k_shot + q_query 个图像
        images = random.sample(img_list, k_shot + q_query)

        # 读取支持集
        for img_path in images[:k_shot]:
            image = Image.open(img_path).convert('L')
            image = image.resize((28, 28))
            image = np.array(image, dtype=np.float32) / 255.0
            image = np.expand_dims(image, axis=-1)  # 增加通道维度
            support_data.append((image, label))

        # 读取查询集
        for img_path in images[k_shot:]:
            image = Image.open(img_path).convert('L')
            image = image.resize((28, 28))
            image = np.array(image, dtype=np.float32) / 255.0
            image = np.expand_dims(image, axis=-1)
            query_data.append((image, label))

    # 随机打乱支持集数据
    random.shuffle(support_data)
    for data in support_data:
        support_image.append(data[0])
        support_label.append(data[1])

    # 随机打乱查询集数据
    random.shuffle(query_data)
    for data in query_data:
        query_image.append(data[0])
        query_label.append(data[1])

    return np.array(support_image), np.array(support_label), np.array(query_image), np.array(query_label)


# 创建任务生成函数，生成多个任务
def generate_task_dataset(file_list, n_tasks, n_way, k_shot, q_query):
    tasks = []
    for _ in range(n_tasks):
        support_images, support_labels, query_images, query_labels = get_supportSet_and_querySet(file_list, n_way,
                                                                                                 k_shot, q_query)
        tasks.append((support_images, support_labels, query_images, query_labels))
    return tasks


def split_train_valid(file_list, valid_set_size, shuffle=False):
    if shuffle:
        np.random.shuffle(file_list)
    valid_len = int(len(file_list) * valid_set_size)
    train_file_list = file_list[:-valid_len]
    valid_file_list = file_list[-valid_len:]
    return train_file_list, valid_file_list


def lode_dataset(data_dir, n_way, k_shot, q_query, split_dataset=False, valid_size=None, n_train_tasks=None,
                 n_valid_tasks=None, n_tasks=None):
    file_list = [f for f in glob.glob(os.path.join(data_dir, '**/character*'), recursive=True)]

    if split_dataset:
        train_file_list, valid_file_list = split_train_valid(file_list, valid_set_size=valid_size, shuffle=True)
        print(len(train_file_list))
        print(len(valid_file_list))
        train_tasks = generate_task_dataset(train_file_list, n_train_tasks, n_way, k_shot, q_query)
        valid_tasks = generate_task_dataset(valid_file_list, n_valid_tasks, n_way, k_shot, q_query)
        return train_tasks, valid_tasks
    else:
        print(len(file_list))
        tasks = generate_task_dataset(file_list, n_tasks, n_way, k_shot, q_query)
        return tasks


# 定义任务数量
n_train_tasks = 772
n_valid_tasks = 192
n_test_tasks = 659

# 加载训练集和验证集任务
train_tasks, valid_tasks = lode_dataset(train_data_path, n_way, k_shot, q_query,
                                        split_dataset=True, valid_size=valid_size,
                                        n_train_tasks=n_train_tasks, n_valid_tasks=n_valid_tasks)

# 加载测试集任务
test_tasks = lode_dataset(test_data_path, n_way, k_shot, q_query, n_tasks=n_test_tasks)


# 创建任务生成器函数
def task_generator(tasks):
    for task in tasks:
        support_images, support_labels, query_images, query_labels = task
        yield support_images, support_labels, query_images, query_labels


# 创建训练数据集
train_dataset = tf.data.Dataset.from_generator(
    lambda: task_generator(train_tasks),
    output_types=(tf.float32, tf.int32, tf.float32, tf.int32),
    output_shapes=(
        (n_way * k_shot, 28, 28, 1),
        (n_way * k_shot,),
        (n_way * q_query, 28, 28, 1),
        (n_way * q_query,)
    )
)

# 创建验证数据集
valid_dataset = tf.data.Dataset.from_generator(
    lambda: task_generator(valid_tasks),
    output_types=(tf.float32, tf.int32, tf.float32, tf.int32),
    output_shapes=(
        (n_way * k_shot, 28, 28, 1),
        (n_way * k_shot,),
        (n_way * q_query, 28, 28, 1),
        (n_way * q_query,)
    )
)

# 创建测试数据集
test_dataset = tf.data.Dataset.from_generator(
    lambda: task_generator(test_tasks),
    output_types=(tf.float32, tf.int32, tf.float32, tf.int32),
    output_shapes=(
        (n_way * k_shot, 28, 28, 1),
        (n_way * k_shot,),
        (n_way * q_query, 28, 28, 1),
        (n_way * q_query,)
    )
)

# 创建数据加载器，应用批处理和预取
train_loader = train_dataset.shuffle(len(train_tasks)).batch(meta_batch_size).prefetch(tf.data.experimental.AUTOTUNE)
valid_loader = valid_dataset.shuffle(len(valid_tasks)).batch(meta_batch_size).prefetch(tf.data.experimental.AUTOTUNE)
test_loader = test_dataset.batch(meta_batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# 测试数据加载器
for batch in train_loader.take(1):
    support_images_batch, support_labels_batch, query_images_batch, query_labels_batch = batch
    print("Support Images Batch Shape:", support_images_batch.shape)
    print("Support Labels Batch Shape:", support_labels_batch.shape)
    print("Query Images Batch Shape:", query_images_batch.shape)
    print("Query Labels Batch Shape:", query_labels_batch.shape)


def get_initial_weights(model):
    weights = {}
    # 遍历模型每一层
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            # 获取当前层的训练参数和偏执项
            for var in layer.trainable_variables:
                # 获取变量名并去除 ':0'
                # 在 TensorFlow 中，变量名通常以 ':0' 结尾，如 'dense/kernel:0'。我们需要去掉这个后缀，使变量名更简洁
                var_name = var.name.split(':')[0]
                # 去除模型前缀
                var_name = var_name.replace(f"{model.name}/", "")
                weights[var_name] = tf.identity(var)
    return weights


def update_weights(weights, gradients, inner_lr):
    update_weights = {}
    for key in weights.keys():
        update_weights[key] = weights[key] - inner_lr * gradients[key]
    return update_weights


def maml_train(
        model,
        support_images,  # 支持集图像
        support_labels,  # 支持集标签
        query_images,  # 查询集图像
        query_labels,  # 查询集标签
        train_inner_step,  # 内部更新步数（在支持集上进行快速调整的步数）
        inner_lr,  # 内循环的学习率
        optimizer,  # 优化器
        loss_fn,  # 损失函数

        is_train=True
):
    # 获取批次大小
    support_set_size = support_images.shape[0]
    # 损失
    meta_loss = []
    # 准确率
    meta_acc = []

    # 外循环的梯度记录器
    with tf.GradientTape() as outer_tape:
        # 遍历子任务并进行学习
        for i in range(support_set_size):
            support_image = support_images[i]
            support_label = support_labels[i]
            query_image = query_images[i]
            query_label = query_labels[i]

            # 获取模型的初始权重
            weights = get_initial_weights(model)

            # 内循环
            for _ in range(train_inner_step):
                with tf.GradientTape() as inner_tape:
                    inner_tape.watch(weights.values())
                    # 前向传播
                    support_logits = model.call(support_image, weights)
                    # 计算损失
                    support_loss = loss_fn(support_label, support_logits)

                # 计算梯度
                gradients = inner_tape.gradient(support_loss, weights.values())
                # 将梯度和权重的键名对应起来
                gradients = dict(zip(weights.keys(), gradients))

                # 因为字典是乱序的，所以想进行梯度下降，就需要遍历取出对应的数据
                # 更新权重
                weights = update_weights(weights, gradients, inner_lr)

            # 使用更新后的权重在查询集上进行评估
            query_logits = model.call(query_image, weights)
            # 计算损失值
            query_loss = loss_fn(query_label, query_logits)
            meta_loss.append(query_loss)

            # 计算准确率
            query_pre = tf.argmax(query_logits, axis=1, output_type=tf.int32)
            query_acc = tf.reduce_mean(tf.cast(tf.equal(query_label, query_pre), tf.float32))
            meta_acc.append(query_acc)

        # 计算平均损失
        meta_loss = tf.reduce_mean(meta_loss)
        # 计算平均准确率
        meta_acc = tf.reduce_mean(meta_acc)

        # 外循环更新
        if is_train:
            # 计算元损失关于模型初始参数的梯度
            gradients = outer_tape.gradient(meta_loss, model.trainable_variables)
            # 更新模型参数
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return meta_loss, meta_acc


train_iter = iter(train_loader)
valid_iter = iter(valid_loader)

model = ResNet.ResNet(input_shape=(28, 28, 1), n_way=n_way)
optimizer = tf.keras.optimizers.Adam(learning_rate=outer_lr)
# 定义损失函数
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# **添加以下代码，使用一个虚拟输入构建模型**
dummy_input = tf.zeros([1, 28, 28, 1])  # 根据输入形状创建一个全零张量
_ = model(dummy_input)  # 调用模型，构建变量
# 训练1000轮
for iteration in range(1, num_iterations + 1):
    # ========================= 训练模型 =====================
    try:
        # 从训练迭代器中获取支持集和查询集的数据
        support_images, support_labels, query_images, query_labels = next(train_iter)
    except StopIteration:
        # 如果迭代器结束，则重新创建一个训练迭代器并继续获取数据
        train_iter = iter(train_loader)
        support_images, support_labels, query_images, query_labels = next(train_iter)

    # 使用MAML方法在验证集上评估模型
    # 在支持集上进行模型更新，然后在查询集上评估模型性能
    loss, acc = maml_train(
        model,
        support_images,  # 支持集图像
        support_labels,  # 支持集标签
        query_images,  # 查询集图像
        query_labels,  # 查询集标签
        train_inner_step,  # 内部更新步数（在支持集上进行快速调整的步数）
        inner_lr,  # 内循环的学习率
        optimizer,  # 优化器
        loss_fn,  # 损失函数
        is_train=True  # 训练模式设置为False，因为这是验证过程
    )

    # ========================= 验证模型 =====================
    try:
        # 从训练迭代器中获取支持集和查询集的数据
        support_images, support_labels, query_images, query_labels = next(valid_iter)
    except StopIteration:
        # 如果迭代器结束，则重新创建一个训练迭代器并继续获取数据
        valid_iter = iter(valid_loader)
        support_images, support_labels, query_images, query_labels = next(valid_iter)

    test_loss, test_acc = maml_train(
        model,
        support_images,  # 支持集图像
        support_labels,  # 支持集标签
        query_images,  # 查询集图像
        query_labels,  # 查询集标签
        train_inner_step,  # 内部更新步数（在支持集上进行快速调整的步数）
        inner_lr,  # 内循环的学习率
        optimizer,  # 优化器
        loss_fn,  # 损失函数
        is_train=False  # 训练模式设置为False，因为这是验证过程
    )

    print(f'Epoch {iteration}, Meta Loss: {loss.numpy()}, Meta Accuracy: {acc.numpy()}, Test Loss: {test_loss.numpy()}, Test Accuracy: {test_acc.numpy()}')
# 保存模型
model.save(filepath="./MAML_FSL_omniglot_model2", save_format="tf")

# model.save(filepath=f"./MAML_FSL_omniglot_model.h5")
# ====================== 评估模型 ====================


test_acc = []
test_loss = []


test_bar = tqdm(test_loader, desc="Testing")

for support_images, support_labels, query_images, query_labels in test_bar:
    # 确保数据类型正确
    support_images = tf.cast(support_images, tf.float32)
    support_labels = tf.cast(support_labels, tf.int32)
    query_images = tf.cast(query_images, tf.float32)
    query_labels = tf.cast(query_labels, tf.int32)

    # 在评估过程中，设置 is_train=False
    loss, acc = maml_train(
        model,
        support_images,
        support_labels,
        query_images,
        query_labels,
        eval_inner_step,
        inner_lr,
        optimizer,
        loss_fn,
        is_train=False  # 评估模式，不更新模型参数
    )

    # 记录损失和准确率
    test_loss.append(loss.numpy())
    test_acc.append(acc)

    # 更新进度条信息
    test_bar.set_postfix(loss=loss.numpy(), acc=acc)

# 计算平均损失和准确率
test_loss = np.mean(test_loss)
test_acc = np.mean(test_acc)
print('Meta Test Loss: {:.3f}, Meta Test Acc: {:.2f}%'.format(test_loss, 100 * test_acc))

