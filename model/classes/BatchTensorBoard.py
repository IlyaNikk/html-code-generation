import tensorflow as tf

# Кастомный callback для логирования метрик после каждого батча
class BatchTensorBoard(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super(BatchTensorBoard, self).__init__()
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        # Используем количество итераций оптимизатора, если доступно, иначе номер батча
        step = self.model.optimizer.iterations.numpy() if hasattr(self.model.optimizer, 'iterations') else batch
        with self.writer.as_default():
            for key, value in logs.items():
                tf.summary.scalar(key, value, step=step)
            self.writer.flush()
