import tensorflow as tf 
import sonnet as snt 

from octave_conv import OctaveUnit2d
from octave_conv import unit2d

class basicblock(snt.AbstractModule):

    expansion = 1
    def __init__(self,depth=64,stride=(1,1),ratio=0.5, is_first = False,name='bottleneck'):
        super(basicblock,self).__init__(name=name)
        self._depth = depth
        self._stride = stride
        self._ratio = ratio
        self._is_first = is_first

    def _build(self,inputs,is_training):
        
        if self._is_first is False:
            shortcut = inputs
        else:
            shortcut = OctaveUnit2d(output_channels=self._depth,kernel_shape=(1,1),ratio=self._ratio,stride=self._stride,activation_fn=None,name='shortcut')(inputs,is_training=is_training)

        residual = OctaveUnit2d(output_channels=self._depth,kernel_shape=(3,3),ratio=self._ratio,stride=self._stride,name='conv1')(inputs,is_training=is_training)
        residual = OctaveUnit2d(output_channels=self._depth,kernel_shape=(3,3),ratio=self._ratio,activation_fn=None,name='conv2')(residual,is_training=is_training)

        shortcut_h, shortcut_l = shortcut if type(shortcut) is tuple else (shortcut, None)
        residual_h, residual_l = shortcut if type(residual) is tuple else (residual, None)

        out_h = tf.nn.relu(shortcut_h + residual_h)
        out_l = tf.nn.relu(shortcut_l + residual_l) if shortcut_l is not None and residual_l is not None else None

        return out_h if out_l is None else (out_h, out_l)

class bottleneck(snt.AbstractModule):

    expansion = 4
    def __init__(self,depth=64,tride=(1,1),ratio=0.5,is_first = False,name='bottleneck'):
        super(bottleneck,self).__init__(name=name)
        self._depth = depth
        self._stride = stride
        self._ratio = ratio
        self._is_fisrt = is_first

    def _build(self,inputs,is_training):
        
        if self._is_fisrt is False:
            shortcut = inputs
        else:
            shortcut = OctaveUnit2d(output_channels=self._depth * self.expansion,kernel_shape=(1,1),ratio=self._ratio,stride=self._stride,activation_fn=None,name='shortcut')(inputs,is_training=is_training)

        residual = OctaveUnit2d(output_channels=self._depth,kernel_shape=(1,1),ratio=self._ratio,stride=self._stride,name='conv1')(inputs,is_training=is_training)
        residual = OctaveUnit2d(output_channels=self._depth,kernel_shape=(3,3),ratio=self._ratio,name='conv2')(residual,is_training=is_training)
        residual = OctaveUnit2d(output_channels=self._depth * self.expansion,kernel_shape=(1,1),ratio=self._ratio,activation_fn=None,name='conv3')(residual,is_training=is_training)

        out = tf.nn.relu(shortcut + residual)
        return out

class Resnet(snt.AbstractModule):

    VALID_ENDPOINTS = (
        'conv1',
        'pool1',
        'block1',
        'block2',
        'block3',
        'block4',
        'logits',
        'Predictions'
    )

    def  __init__(self,num_classes = 1000, block = basicblock, ratio=0.25,spatia_squeeze = True,unit_num = [2,2,2,2],
                 final_endpoint = 'logits',name = 'resnet_v1_18'):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(Resnet, self).__init__(name = name)
        self._num_classes = num_classes
        self._spatia_squeeze = spatia_squeeze
        self._final_endpoint = final_endpoint
        self._unit_num = unit_num
        self._block = block
        self._ratio = ratio

    def _build(self, inputs ,is_training ,dropout_keep_prob = 0.5):
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        net = inputs
        end_points = {}
        end_point = 'conv1'
        net = unit2d(output_channels=64,kernel_shape=[7,7],
                     stride=[2,2],name = end_point)(net,is_training=is_training)

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points
        end_point = 'pool1'
        net = tf.nn.max_pool(net,ksize=(1,3,3,1),strides=(1,2,2,1),
                               padding=snt.SAME,name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points
        
        end_point = 'block1'
        with tf.variable_scope(end_point):
            num_units = self._unit_num[0]
            for i in range(num_units):
                with tf.variable_scope('unit_%d' % (i+1)):
                    if i != 0:
                        net = self._block(depth=64,stride=1,ratio=self._ratio )(net,is_training=is_training )

                    else:
                        net = self._block(depth=64,stride=1,ratio=self._ratio, is_first=True )(net,is_training=is_training )

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net,end_points

        end_point = 'block2'
        with tf.variable_scope(end_point):
            num_units = self._unit_num[1]
            for i in range (num_units):
                with tf.variable_scope ('unit_%d' % (i + 1)):
                    if i != 0:
                        net = self._block (depth=128, stride=1, ratio=self._ratio ) (net, is_training=is_training )
                        
                    else:
                        net = self._block (depth=128, stride=2,ratio=self._ratio, is_first=True ) (net, is_training=is_training )
                        
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'block3'
        with tf.variable_scope(end_point):
            num_units = self._unit_num[2]
            for i in range (num_units):
                with tf.variable_scope ('unit_%d' % (i + 1)):
                    if i != 0:
                        net = self._block (depth=256,stride=1,ratio=self._ratio ) (net, is_training=is_training )
                        
                    else:
                        net = self._block (depth=256, stride=2,ratio=self._ratio, is_first=True ) (net, is_training=is_training )
                        
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'block4'
        with tf.variable_scope(end_point):
            num_units = self._unit_num[3]
            for i in range (num_units):
                with tf.variable_scope ('unit_%d' % (i + 1)):
                    if i != 0:
                        net = self._block (depth=512, stride=1, ratio=0 ) (net, is_training=is_training )
                        
                    else:
                        net = self._block (depth=512, stride=2, is_first=True, ratio=0 ) (net, is_training=is_training )
                        
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points
       
        
        _,h,w,_ = net.shape.as_list()
        net = tf.nn.avg_pool(net,(1,h,w,1),strides=(1,1,1,1),padding=snt.VALID)

        with tf.variable_scope('logits'):
            logits = snt.Conv2D(self._num_classes,kernel_shape=(1,1),use_bias=True)(net)
        
        logits = tf.squeeze(logits,axis=[1,2])

        return logits, end_points

def Resnet18(num_classes = 1000, block = basicblock, ratio=0.25,unit_num = [2,2,2,2],name='resnet_v1_18'):

    return Resnet(num_classes=num_classes,block=block,ratio=ratio,unit_num=unit_num,name=name)

def Resnet18(num_classes = 1000, block = basicblock, ratio=0.25,unit_num = [3,4,6,3],name='resnet_v1_34'):

    return Resnet(num_classes=num_classes,block=block,ratio=ratio,unit_num=unit_num,name=name)

def Resnet50(num_classes = 1000, block = bottleneck, ratio=0.25,unit_num = [3,4,6,3],name='resnet_v1_50'):

    return Resnet(num_classes=num_classes,block=block,ratio=ratio,unit_num=unit_num,name=name)

def Resnet101(num_classes = 1000, block = bottleneck, ratio=0.25,unit_num = [3,4,23,3],name='resnet_v1_101'):

    return Resnet(num_classes=num_classes,block=block,ratio=ratio,unit_num=unit_num,name=name)

def Resnet152(num_classes = 1000, block = bottleneck, ratio=0.25,unit_num = [3,8,36,3],name='resnet_v1_152'):

    return Resnet(num_classes=num_classes,block=block,ratio=ratio,unit_num=unit_num,name=name)

if __name__ == "__main__":
    a = tf.placeholder(tf.float32, [None,224,224,3])
    Resnet()(a,True)

    for i in tf.global_variables():
        print(i)