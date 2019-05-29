import tensorflow as tf 
import sonnet as snt 


class unit2d(snt.AbstractModule):

    def __init__(self,output_channels=256,
                kernel_shape=(1,1),
                stride = (1,1),
                activation_fn = tf.nn.relu,
                use_bias = False,
                use_bn = True,
                name='unit2d'):
        super(unit2d,self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._actvation_fn = activation_fn
        self._use_bias = use_bias
        self._use_bn = use_bn


    def _build(self,inputs,is_training):
        net = snt.Conv2D(output_channels=self._output_channels, 
                        kernel_shape=self._kernel_shape, stride=self._stride, 
                        use_bias=self._use_bias,padding=snt.SAME)(inputs)
        if self._use_bn:
            bn = snt.BatchNormV2()
            net = bn(net,is_training=is_training,test_local_stats=False)
        
        if self._actvation_fn is not None:
            net = self._actvation_fn(net)

        return net


class OctaveUnit2d(snt.AbstractModule):

    def __init__(self,output_channels=256,
                kernel_shape=(1,1),
                ratio = 0.5,
                stride = (1,1),
                activation_fn = tf.nn.relu,
                use_bias = False,
                use_bn = True,
                name='OctaveUnit2d'):
        super(OctaveUnit2d,self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._actvation_fn = activation_fn
        self._use_bias = use_bias
        self._use_bn = use_bn

        self._ratio = ratio

    def _build(self,inputs,is_training):

        x_h, x_l = inputs if type(inputs) is tuple else (inputs, None)

        if self._stride == 2 or self._stride == (2,2) or self._stride == [2,2]:
            x_h = tf.nn.avg_pool(x_h,(1,2,2,1),(1,2,2,1),'SAME') if x_h is not None else None
            x_l = tf.nn.avg_pool(x_l,(1,2,2,1),(1,2,2,1),'SAME') if x_l is not None else None

        _ , h,w, _ = x_h.shape.as_list()

        l_out = self._output_channels * self._ratio
        h_out = self._output_channels - l_out

        x_h2h , x_h2l, x_l2l, x_l2h = None, None, None, None
        if x_h is not None:
            x_h2h = snt.Conv2D(output_channels=h_out,kernel_shape=self._kernel_shape,padding=snt.SAME,name='h2h')(x_h)
            if l_out > 0:
                x_h2l = tf.nn.avg_pool(x_h, (1,2,2,1), (1,2,2,1), padding=snt.SAME)
                x_h2l = snt.Conv2D(output_channels=l_out,kernel_shape=self._kernel_shape,padding=snt.SAME,name='h2l')(x_h2l)

        if x_l is not None:
            if l_out > 0:
                x_l2l = snt.Conv2D(output_channels=l_out,kernel_shape=self._kernel_shape,padding=snt.SAME,name='l2l')(x_l)
            x_l2h = snt.Conv2D(output_channels=h_out,kernel_shape=self._kernel_shape,padding=snt.SAME,name='l2h')(x_l)
            x_l2h = tf.image.resize_nearest_neighbor(x_l2h,(h,w))

        y_h = x_h2h + x_l2h if x_l2h is not None else x_h2h
        y_l = x_h2l + x_l2l if x_l2l is not None else x_h2l

        if self._use_bn:
            bn1 = snt.BatchNormV2(name='h_bn')
            bn2 = snt.BatchNormV2(name='l_bn')
            y_h = bn1(y_h,is_training=is_training,test_local_stats=False) if y_h is not None else None
            y_l = bn2(y_l,is_training=is_training,test_local_stats=False) if y_l is not None else None 

        if self._actvation_fn is not None:
            y_h = self._actvation_fn(y_h) if y_h is not None else None
            y_l = self._actvation_fn(y_l) if y_l is not None else None

        return y_h if y_l is None else (y_h, y_l)


if __name__ == "__main__":
    a = tf.placeholder(tf.float32,[None,224,224,3])
    b = tf.placeholder(tf.float32,[None,112,112,3])

    c = OctaveUnit2d()((a,b),True)
    d = OctaveUnit2d()(a,True)
    print(c,d)

