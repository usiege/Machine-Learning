#

def inference(X):
    # 计算推断模型在数据X上的输出，并将结果返回
    # example
    return tf.matmul(X, W) + b

def loss(X, Y):
    # 依据训练数据X及其期望输出Y计算损失

def inputs():
    # 读取或生成训练数据X及其期望输出Y

def train(total_loss):
    # 依据计算的总损失训练或调整模型参数

def evaluate(sess, X, Y):
    # 对训练得到的模型进行评估


def main():

    # 模型定义，创建一个Saver对象
    saver = tf.train.Saver()


    # 在一个会话对象中启动数据流图， 搭建流程
    with tf.Session() as sess:

        tf.initialize_all_variables().run()

        X, Y = inputs()

        total_loss = loss(X, Y)
        train_op = train(total_loss)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 模型设置

        # 训练闭环
        training_steps = 1000
        for step in range(training_steps):
            sess.run([train_op])



            # 查看损失在训练过程中递减的情况
            if step % 10 == 0:
                print 'loss: ', sess.run([total_loss])

            if step % 1000 == 0:
                saver.save(sess, 'my-model', global_step=step)

        evaluate(sess, X, Y)

        coord.request_stop()
        coord.join(threads)

        saver.save(sess, 'my-model', global_step=training_steps)

        sess.closed()

def restore():

    # 模型设置

    initial_step = 0

    # 验证之前是否已经保存了检查点文件
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))
    if ckpt and ckpt.model_checkpoint_path:
        # 从检查点恢复模型参数
        saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])

    # 实际的训练闭环
    for step in range(initial_step, training_steps):
        # bababa
