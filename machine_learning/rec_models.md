

[toc]

# 必要基础知识

## tensorflow

[TensorFlow基础 &mdash; 简单粗暴 TensorFlow 2 0.4 beta 文档](https://tf.wiki/zh_hans/basic/basic.html)

[tf.tensordot &nbsp;|&nbsp; TensorFlow Core v2.9.1](https://www.tensorflow.org/api_docs/python/tf/tensordot)

https://github.com/Amberlan1001/eat_tensorflow2_in_30_days_ipynb

# 常见模型

## 早期模型

### BPR

### 贝叶斯平滑



## 特征交叉模型

### FM

https://zhuanlan.zhihu.com/p/342803984

![img](https://pic4.zhimg.com/80/v2-a59d65d4ccb9b570c80b2ea861d60de3_720w.png)

![img](https://pic4.zhimg.com/80/v2-f00d1b2cf8da721305c6fdf3a772d95f_720w.jpg)

```python
# layer.py
class FMLayer(layers.Layer):
    def __init__(self, k, w_reg, v_reg):
        super(FMLayer, self).__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
    
    def build(self, input_shape):
        self.w0 = self.add_weight("w0", shape=(1,),
                                initializer=tf.zeros_initializer(),
                                trainable=True)
        self.w = self.add_weight("w", shape=(input_shape[-1], 1),
                                initializer=tf.random_normal_initializer(),
                                trainable=True,
                                regularizer=regularizers.L2(self.w_reg))
        self.v = self.add_weight("v", shape=(input_shape[-1], self.k),
                                initializer=tf.random_normal_initializer(),
                                trainable=True,
                                regularizer=regularizers.L2(self.v_reg))
    
    def call(self, inputs):
        if K.ndim(inputs) != 2:
            raise ValueError("Unexpected inputs dimension %d, expect to be 2 dimensions" % (K.ndim(inputs)))
        
        linear_part = tf.matmul(inputs, self.w) + self.w0
        inter_part1 = tf.pow(tf.matmul(inputs, self.v), 2)
        inter_part2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2))
        inter_part = 0.5 * tf.reduce_sum(inter_part1 - inter_part2, axis=-1, keepdims=True)
        output = linear_part + inter_part
        return output
    
# model.py
class FM(Model):
    def __init__(self, k, w_reg=1e-4, v_reg=1e-4) -> None:
        super().__init__()
        self.fm = FMLayer(k, w_reg, v_reg)
    
    def call(self, inputs):
        output = self.fm(inputs)
        output = tf.nn.sigmoid(output)
        return output
```



### CCPM

### FFM

https://zhuanlan.zhihu.com/p/348596108

![img](https://pic4.zhimg.com/80/v2-1fec232b8134c2d5f1692b7dbaa6febf_720w.jpg)

```python
# layer.py
class FFMLayer(layers.Layer):
    def __init__(self, feature_columns, k, w_reg, v_reg):
        super(FFMLayer, self).__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.feature_num = sum([feat['feat_onehot_dim'] for feat in self.sparse_feature_columns]) \
                            + len(self.dense_feature_columns)
        self.field_num = len(self.dense_feature_columns) + len(self.sparse_feature_columns)
    
    def build(self, input_shape):
        self.w0 = self.add_weight("w0", shape=(1,),
                                initializer=tf.zeros_initializer(),
                                trainable=True)
        self.w = self.add_weight("w", shape=(self.feature_num, 1),
                                initializer=tf.random_normal_initializer(),
                                trainable=True,
                                regularizer=regularizers.L2(self.w_reg))
        # for FM, every feature has only one hidden vector
        # but for FFM, every feature has `field_num` hidden vectors
        self.v = self.add_weight("v", shape=(self.feature_num, self.field_num, self.k),
                                initializer=tf.random_normal_initializer(),
                                trainable=True,
                                regularizer=regularizers.L2(self.v_reg))
    
    def call(self, inputs):
        dense_inputs = inputs[:, :13]
        sparse_inputs = inputs[:, 13:]

        # onehot-encoding
        x = tf.cast(dense_inputs, dtype=tf.float32)
        for i in range(sparse_inputs.shape[1]):
            x = tf.concat([x, tf.one_hot(tf.cast(sparse_inputs[:, i], dtype=tf.int32),
                                      depth=self.sparse_feature_columns[i]['feat_onehot_dim'])], axis=1)
        # for i in range(sparse_inputs.shape[1]):
        #     x = tf.concat(
        #         [x, tf.one_hot(tf.cast(sparse_inputs[:, i], dtype=tf.int32),
        #                            depth=self.sparse_feature_columns[i]['feat_onehot_dim'])], axis=1)
        
        linear_part = tf.matmul(x, self.w) + self.w0
        inter_part = 0
        # 1. calculate VX
            # process of tensordot: https://blog.csdn.net/tjh1998/article/details/123563159
            # for b in batch_size:
            #     tmp = x[b, :] matmul v = [1, 2291] matmul [2291, 39, 8] = [1, 39, 8]
            # res = concat all tmp
        field_f = tf.tensordot(x, self.v, axes=1) # [batch_size, 2291] x [2291, 39, 8] = [batch_size, 39, 8]
        # 2. calculate <V_i_fj, V_j_fi> * x_i * x_j for every (i, j)
        for i in range(self.field_num):
            for j in range(i + 1, self.field_num):
                inter_part += tf.reduce_sum(
                    tf.multiply(field_f[:, i], field_f[:, j]), # [None, 8]
                    axis=1, keepdims=True
                )
        
        output = linear_part + inter_part
        return output
    
# model.py
class FFM(Model):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4) -> None:
        super().__init__()
        self.ffm = FFMLayer(feature_columns, k, w_reg, v_reg)
    
    def call(self, inputs):
        output = self.ffm(inputs)
        output = tf.nn.sigmoid(output)
        return output
```



### Deep Crossing

![img](https://ask.qcloudimg.com/http-save/7175224/tokxm2i3zh.png?imageView2/2/w/1620)

![img](E:\learning_home\learning_records\machine_learning\imgs\DeepCrossing.png)

```python

```

### DCN & DCN-M

![img](E:\learning_home\learning_records\machine_learning\imgs\DCN.png)

![img](E:\learning_home\learning_records\machine_learning\imgs\DCN-M.png)





## MTL模型

#### ESMM



#### MMOE

#### PLE



## 图模型



# 召回模型

## 倒排索引



## i2i

### itemcf



## u2i

### MIND



# 资料收集

https://n02r1t64ol.feishu.cn/docs/doccnYGRpz4a4epbGZ8bTLNPjRf