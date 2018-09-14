from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import traceback

import numpy as np
import tensorflow as tf

from enas.cifar10.models import Model
from enas.cifar10.image_ops import conv
from enas.cifar10.image_ops import fully_connected
from enas.cifar10.image_ops import norm
from enas.cifar10.image_ops import batch_norm_with_mask
from enas.cifar10.image_ops import relu
from enas.cifar10.image_ops import max_pool
from enas.cifar10.image_ops import drop_path
from enas.cifar10.image_ops import global_max_pool

from enas.utils import count_model_params
from enas.utils import get_train_ops
from enas.common_ops import create_weight
import keras

import grasp_metrics


class MicroChild(Model):
    def __init__(self,
                 images,
                 labels,
                 use_aux_heads=False,
                 cutout_size=None,
                 fixed_arc=None,
                 num_layers=2,
                 num_cells=5,
                 out_filters=24,
                 keep_prob=1.0,
                 drop_path_keep_prob=None,
                 batch_size=32,
                 clip_mode=None,
                 grad_bound=None,
                 l2_reg=1e-4,
                 lr_init=0.1,
                 lr_dec_start=0,
                 lr_dec_every=10000,
                 lr_dec_rate=0.1,
                 lr_cosine=False,
                 lr_max=None,
                 lr_min=None,
                 lr_T_0=None,
                 lr_T_mul=None,
                 num_epochs=None,
                 optim_algo=None,
                 sync_replicas=False,
                 num_aggregate=None,
                 num_replicas=None,
                 data_format="NHWC",
                 name="child",
                 valid_set_size=32,
                 image_shape=(32, 32, 3),
                 translation_only=False,
                 rotation_only=False,
                 stacking_reward=False,
                 use_root=False,
                 one_hot_encoding=False,
                 dataset="cifar",
                 data_base_path="",
                 output_dir="",
                 pool_distance=2,
                 use_msle=False,
                 **kwargs
                 ):

        super(self.__class__, self).__init__(
            images,
            labels,
            cutout_size=cutout_size,
            batch_size=batch_size,
            clip_mode=clip_mode,
            grad_bound=grad_bound,
            l2_reg=l2_reg,
            lr_init=lr_init,
            lr_dec_start=lr_dec_start,
            lr_dec_every=lr_dec_every,
            lr_dec_rate=lr_dec_rate,
            keep_prob=keep_prob,
            optim_algo=optim_algo,
            sync_replicas=sync_replicas,
            num_aggregate=num_aggregate,
            num_replicas=num_replicas,
            data_format=data_format,
            name=name,
            valid_set_size=valid_set_size,
            image_shape=image_shape,
            translation_only=translation_only,
            rotation_only=rotation_only,
            stacking_reward=stacking_reward,
            data_base_path=data_base_path,
            use_root=use_root,
            one_hot_encoding=one_hot_encoding,
            dataset=dataset)

        if self.data_format == "NHWC":
            self.actual_data_format = "channels_last"
        elif self.data_format == "NCHW":
            self.actual_data_format = "channels_first"
        else:
            raise ValueError(
                "Unknown data_format '{0}'".format(self.data_format))

        self.use_aux_heads = use_aux_heads
        self.use_root = use_root
        self.num_epochs = num_epochs
        self.num_train_steps = self.num_epochs * self.num_train_batches
        self.drop_path_keep_prob = drop_path_keep_prob
        self.lr_cosine = lr_cosine
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_T_0 = lr_T_0
        self.lr_T_mul = lr_T_mul
        self.out_filters = out_filters
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.fixed_arc = fixed_arc
        self.translation_only = translation_only
        self.rotation_only = rotation_only
        self.stacking_reward = stacking_reward
        self.data_base_path = data_base_path
        self.verbose = 0
        self.output_dir = output_dir
        self.one_hot_encoding = one_hot_encoding
        self.use_msle = use_msle

        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name="global_step")

        if self.drop_path_keep_prob is not None:
            assert num_epochs is not None, "Need num_epochs to drop_path"

        self.pool_distance = pool_distance
        # pool_distance was originally based on the number of layers
        # pool_distance = self.num_layers // 3
        # self.pool_layers = [pool_distance, 2 * pool_distance + 1]

        self.pool_layers = []
        for layer_num in range(self.num_layers):
            if layer_num != 0 and layer_num % pool_distance == 0:
                self.pool_layers += [layer_num]

        if self.use_aux_heads:
            if len(self.pool_layers) > 2:
                pool_index = int(len(self.pool_layers) / 2)
                self.aux_head_indices = [self.pool_layers[pool_index] + 1]
            else:
                self.aux_head_indices = [self.pool_layers[-1] + 1]

    def _factorized_reduction(self, x, out_filters, stride, is_training):
        """Reduces the shape of x without information loss due to striding."""
        assert out_filters % 2 == 0, (
            "Need even number of filters when using this factorized\
                reduction.")
        if stride == 1:
            with tf.variable_scope("path_conv"):
                inp_c = self._get_C(x)
                w = create_weight("w", [1, 1, inp_c, out_filters])
                x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                                 data_format=self.data_format)
                x = norm(x, is_training=is_training, data_format=self.data_format, norm_type="batch")
                return x

        stride_spec = self._get_strides(stride)
        # Skip path 1
        path1 = tf.nn.max_pool(
            x, [1, 1, 1, 1], stride_spec, "VALID",
            data_format=self.data_format)
        with tf.variable_scope("path1_conv"):
            inp_c = self._get_C(path1)
            w = create_weight("w", [1, 1, inp_c, out_filters // 2])
            path1 = tf.nn.conv2d(path1, w, [1, 1, 1, 1], "VALID",
                                 data_format=self.data_format)

        # Skip path 2
        # First pad with 0"s on the right and bottom, then shift the filter to
        # include those 0"s that were added.
        if self.data_format == "NHWC":
            pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
            path2 = tf.pad(x, pad_arr)[:, 1:, 1:, :]
            concat_axis = 3
        else:
            pad_arr = [[0, 0], [0, 0], [0, 1], [0, 1]]
            path2 = tf.pad(x, pad_arr)[:, :, 1:, 1:]
            concat_axis = 1

        path2 = tf.nn.max_pool(
            path2, [1, 1, 1, 1], stride_spec, "VALID",
            data_format=self.data_format)
        with tf.variable_scope("path2_conv"):
            inp_c = self._get_C(path2)
            w = create_weight("w", [1, 1, inp_c, out_filters // 2])
            path2 = tf.nn.conv2d(path2, w, [1, 1, 1, 1], "VALID",
                                 data_format=self.data_format)

        # Concat and apply BN
        final_path = tf.concat(values=[path1, path2], axis=concat_axis)
        final_path = norm(final_path, is_training=is_training,
                          data_format=self.data_format, norm_type="batch")

        return final_path

    def _get_C(self, x):
        """
        Args:
          x: tensor of shape [N, H, W, C] or [N, C, H, W]
        """
        if self.data_format == "NHWC":
            assert x.get_shape().as_list()[3] is not None
            return x.get_shape()[3].value
        elif self.data_format == "NCHW":
            assert x.get_shape().as_list()[1] is not None
            return x.get_shape()[1].value
        else:
            raise ValueError(
                "Unknown data_format '{0}'".format(self.data_format))

    def _get_HW(self, x):
        """
        Args:
          x: tensor of shape [N, H, W, C] or [N, C, H, W]
        """
        assert x.get_shape().as_list()[2] is not None
        return x.get_shape()[2].value

    def _get_strides(self, stride):
        """
        Args:
          x: tensor of shape [N, H, W, C] or [N, C, H, W]
        """
        if self.data_format == "NHWC":
            return [1, stride, stride, 1]
        elif self.data_format == "NCHW":
            return [1, 1, stride, stride]
        else:
            raise ValueError(
                "Unknown data_format '{0}'".format(self.data_format))

    def _apply_drop_path(self, x, layer_id):
        drop_path_keep_prob = self.drop_path_keep_prob

        layer_ratio = float(layer_id + 1) / (self.num_layers + 2)
        drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)

        step_ratio = tf.to_float(self.global_step + 1) / \
            tf.to_float(self.num_train_steps)
        step_ratio = tf.minimum(1.0, step_ratio)
        drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)

        x = drop_path(x, drop_path_keep_prob)
        return x

    def _maybe_calibrate_size(self, layers, out_filters, is_training):
        """Makes sure layers[0] and layers[1] have the same shapes."""

        hw = [self._get_HW(layer) for layer in layers]
        c = [self._get_C(layer) for layer in layers]

        with tf.variable_scope("calibrate"):
            x = layers[0]
            if hw[0] != hw[1]:
                assert hw[0] == 2 * hw[1]
                with tf.variable_scope("pool_x"):
                    x = tf.nn.elu(x)
                    x = self._factorized_reduction(
                        x, out_filters, 2, is_training)
            elif c[0] != out_filters:
                with tf.variable_scope("pool_x"):
                    w = create_weight("w", [1, 1, c[0], out_filters])
                    x = tf.nn.elu(x)
                    x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                                     data_format=self.data_format)
                    x = norm(
                        x, is_training=is_training, data_format=self.data_format, norm_type="batch")

            y = layers[1]
            if c[1] != out_filters:
                with tf.variable_scope("pool_y"):
                    w = create_weight("w", [1, 1, c[1], out_filters])
                    y = tf.nn.elu(y)
                    y = tf.nn.conv2d(y, w, [1, 1, 1, 1], "SAME",
                                     data_format=self.data_format)
                    y = norm(
                        y, is_training=is_training, data_format=self.data_format, norm_type="batch")
        return [x, y]

    def concat_images_with_tiled_vector(images, vector):
        """Combine a set of images with a vector, tiling the vector at each pixel in the images and concatenating on the channel axis.

        # Params

            images: list of images with the same dimensions
            vector: vector to tile on each image. If you have
                more than one vector, simply concatenate them
                all before calling this function.

        # Returns

        """
        with tf.variable_scope('concat_images_with_tiled_vector'):
            if not isinstance(images, list):
                images = [images]
            image_shape = K.int_shape(images[0])
            tiled_vector = tile_vector_as_image_channels(vector, image_shape)
            images.append(tiled_vector)
            combined = K.concatenate(images)

            return combined

    def _model(self, images, is_training, reuse=False):
        """Compute the logits given the images."""

        # TODO(ahundt) this line doesn't seem correct, because if doing eval with fixed arcs, training should definitely be false
        # if self.fixed_arc is None:
        #     is_training = True

        with tf.variable_scope(self.name, reuse=reuse):
            # Conv for 2 seperate stacking images
            if self.dataset == "stacking" and self.use_root is True:
                # input_channels_1 = self._get_C(images[0])
                # input_channels_2 = self._get_C(images[1])
                with tf.variable_scope("init_root"):
                    w_1 = create_weight(
                        "w_1", [3, 3, 3, 64])
                    x_1 = tf.nn.conv2d(
                        images[:, :, :, :3], w_1, [1, 1, 1, 1], "SAME")
                    x_1 = norm(x_1, is_training=is_training, data_format=self.data_format, norm_type="batch", name="x_1_norm")
                    x_1 = tf.nn.elu(x_1, name='elu_x_1')
                    w_2 = create_weight(
                        "w_2", [3, 3, 3, 64])
                    x_2 = tf.nn.conv2d(
                        images[:, :, :, 3:6], w_2, [1, 1, 1, 1], "SAME")
                    x_2 = norm(x_2, is_training=is_training, data_format=self.data_format, norm_type="batch", name="x_2_norm")
                    x_2 = tf.nn.elu(x_2, name='elu_x_2')
                    x_3 = tf.layers.dense(images[:, :, :, 6:], units=2048, activation=tf.nn.relu)
                    # dropout
                    x_3 = tf.nn.dropout(x_3, 0.25)
                    # x_3 = tf.layers.dense(x_3, units=64, activation=tf.nn.relu)

                    # dense_layer
                # tiling of images
                print("shape of x_1--", x_1.shape)
                image = [x_1, x_2]
                print("shape of x_3--", len(image))
                x = tf.concat([x_1, x_2, x_3], axis=-1)
                print("shape after concat", x.shape)

            # the first two inputs
            if self.dataset == "stacking" and self.use_root is True:
                input_channels = self._get_C(x)
            else:
                input_channels = self._get_C(images)
            print("channels--------------------------", input_channels)
            with tf.variable_scope("stem_conv"):
                w = create_weight(
                    "w", [3, 3, input_channels,
                          self.out_filters * 3])
                if self.use_root is True:
                    x = tf.nn.conv2d(
                        x, w, [1, 1, 1, 1], "SAME",
                        data_format=self.data_format)
                else:
                    x = tf.nn.conv2d(
                        images, w, [1, 1, 1, 1], "SAME",
                        data_format=self.data_format)
                x = norm(x, is_training=is_training, data_format=self.data_format, norm_type="batch")
            if self.data_format == "NHWC":
                split_axis = 3
            elif self.data_format == "NCHW":
                split_axis = 1
            else:
                raise ValueError(
                    "Unknown data_format '{0}'".format(self.data_format))
            layers = [x, x]

            # building layers in the micro space
            out_filters = self.out_filters
            for layer_id in range(self.num_layers + 2):
                with tf.variable_scope("layer_{0}".format(layer_id)):
                    if layer_id not in self.pool_layers:
                        if self.fixed_arc is None:
                            x = self._enas_layer(
                                    layer_id, layers, self.normal_arc, out_filters,
                                    is_training=is_training)
                        else:
                            x = self._fixed_layer(
                                layer_id, layers, self.normal_arc, out_filters,
                                1, is_training=is_training,
                                normal_or_reduction_cell="normal")
                    else:
                        out_filters *= 2
                        if self.fixed_arc is None:
                            x = self._factorized_reduction(
                                x, out_filters, 2, is_training)
                            layers = [layers[-1], x]
                            x = self._enas_layer(
                                layer_id, layers, self.reduce_arc, out_filters,
                                is_training=is_training)
                        else:
                            x = self._fixed_layer(
                                layer_id, layers, self.reduce_arc, out_filters,
                                2, is_training=is_training,
                                normal_or_reduction_cell="reduction")
                    print("Layer {0:>2d}: {1}".format(layer_id, x))
                    layers = [layers[-1], x]

                # auxiliary heads
                self.num_aux_vars = 0
                if (self.use_aux_heads and
                    layer_id in self.aux_head_indices
                        and is_training):
                    print("Using aux_head at layer {0}".format(layer_id))
                    with tf.variable_scope("aux_head"):
                        aux_logits = tf.nn.elu(x)
                        aux_logits = tf.layers.average_pooling2d(
                            aux_logits, [5, 5], [3, 3], "VALID",
                            data_format=self.actual_data_format)
                        with tf.variable_scope("proj"):
                            inp_c = self._get_C(aux_logits)
                            w = create_weight("w", [1, 1, inp_c, 128])
                            aux_logits = tf.nn.conv2d(aux_logits, w,
                                                      [1, 1, 1, 1], "SAME",
                                                      data_format=self.data_format)
                            aux_logits = norm(aux_logits,
                                              is_training=is_training,
                                              data_format=self.data_format, norm_type="batch")
                            aux_logits = tf.nn.elu(aux_logits)

                        with tf.variable_scope("avg_pool"):
                            inp_c = self._get_C(aux_logits)
                            hw = self._get_HW(aux_logits)
                            w = create_weight("w", [hw, hw, inp_c, 768])
                            aux_logits = tf.nn.conv2d(aux_logits, w, [1, 1, 1, 1], "SAME",
                                                      data_format=self.data_format)
                            aux_logits = norm(aux_logits,  is_training=is_training,
                                              data_format=self.data_format, norm_type="batch")
                            aux_logits = tf.nn.elu(aux_logits)

                        with tf.variable_scope("fc"):
                            aux_logits = global_max_pool(aux_logits,
                                                         data_format=self.data_format)
                            inp_c = aux_logits.get_shape()[1].value
                            w = create_weight("w", [inp_c, self.num_classes])
                            aux_logits = tf.matmul(aux_logits, w)
                            self.aux_logits = aux_logits

                    aux_head_variables = [
                        var for var in tf.trainable_variables() if (
                            var.name.startswith(self.name) and "aux_head" in var.name)]
                    self.num_aux_vars = count_model_params(aux_head_variables)
                    print("Aux head uses {0} params".format(self.num_aux_vars))

            x = tf.nn.elu(x)
            x = global_max_pool(x, data_format=self.data_format)
            if is_training and self.keep_prob is not None and self.keep_prob < 1.0:
                x = tf.nn.dropout(x, self.keep_prob)
            with tf.variable_scope("fc"):
                inp_c = x.get_shape()[1]
                # print("inp_c--------------",inp_c)
                # print("shape x model --------------", x.shape)
                w = create_weight("w", [inp_c, self.num_classes])
                x = tf.matmul(x, w)
        return x

    def _fixed_conv(self, x, f_size, out_filters, stride, is_training,
                    stack_convs=2):
        """Apply fixed convolution.

        Args:
          stacked_convs: number of separable convs to apply.
        """

        for conv_id in range(stack_convs):
            inp_c = self._get_C(x)
            if conv_id == 0:
                strides = self._get_strides(stride)
            else:
                strides = [1, 1, 1, 1]

            with tf.variable_scope("sep_conv_{}".format(conv_id)):
                w_depthwise = create_weight(
                    "w_depth", [f_size, f_size, inp_c, 1])
                w_pointwise = create_weight(
                    "w_point", [1, 1, inp_c, out_filters])
                x = tf.nn.elu(x)
                x = tf.nn.separable_conv2d(
                    x,
                    depthwise_filter=w_depthwise,
                    pointwise_filter=w_pointwise,
                    strides=strides, padding="SAME", data_format=self.data_format)
                x = norm(x, is_training=is_training, data_format=self.data_format, norm_type="batch")

        return x

    def _fixed_combine(self, layers, used, out_filters, is_training,
                       normal_or_reduction_cell="normal"):
        """Adjust if necessary.

        Args:
          layers: a list of tf tensors of size [NHWC] of [NCHW].
          used: a numpy tensor, [0] means not used.
        """

        out_hw = min([self._get_HW(layer)
                      for i, layer in enumerate(layers) if used[i] == 0])
        out = []

        with tf.variable_scope("final_combine"):
            for i, layer in enumerate(layers):
                if used[i] == 0:
                    hw = self._get_HW(layer)
                    if hw > out_hw:
                        assert hw == out_hw * \
                            2, ("i_hw={0} != {1}=o_hw".format(hw, out_hw))
                        with tf.variable_scope("calibrate_{0}".format(i)):
                            x = self._factorized_reduction(
                                layer, out_filters, 2, is_training)
                    else:
                        x = layer
                    out.append(x)

            if self.data_format == "NHWC":
                out = tf.concat(out, axis=3)
            elif self.data_format == "NCHW":
                out = tf.concat(out, axis=1)
            else:
                raise ValueError(
                    "Unknown data_format '{0}'".format(self.data_format))

        return out

    def _fixed_layer(self, layer_id, prev_layers, arc, out_filters, stride,
                     is_training, normal_or_reduction_cell="normal"):
        """
        Args:
          prev_layers: cache of previous layers. for skip connections
          is_training: for batch_norm
        """

        assert len(prev_layers) == 2
        layers = [prev_layers[0], prev_layers[1]]
        layers = self._maybe_calibrate_size(layers, out_filters,
                                            is_training=is_training)

        with tf.variable_scope("layer_base"):
            x = layers[1]
            inp_c = self._get_C(x)
            w = create_weight("w", [1, 1, inp_c, out_filters])
            x = tf.nn.elu(x)
            x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                             data_format=self.data_format)
            x = norm(x, is_training=is_training, data_format=self.data_format, norm_type="batch")
            layers[1] = x

        used = np.zeros([self.num_cells + 2], dtype=np.int32)
        f_sizes = [3, 5]
        for cell_id in range(self.num_cells):
            with tf.variable_scope("cell_{}".format(cell_id)):
                x_id = arc[4 * cell_id]
                used[x_id] += 1
                x_op = arc[4 * cell_id + 1]
                x = layers[x_id]
                x_stride = stride if x_id in [0, 1] else 1
                with tf.variable_scope("x_conv"):
                    if x_op in [0, 1]:
                        f_size = f_sizes[x_op]
                        x = self._fixed_conv(
                            x, f_size, out_filters, x_stride, is_training)
                    elif x_op in [2, 3]:
                        inp_c = self._get_C(x)
                        if x_op == 2:
                            x = tf.layers.average_pooling2d(
                                x, [3, 3], [x_stride, x_stride], "SAME",
                                data_format=self.actual_data_format)
                        else:
                            x = tf.layers.max_pooling2d(
                                x, [3, 3], [x_stride, x_stride], "SAME",
                                data_format=self.actual_data_format)
                        if inp_c != out_filters:
                            w = create_weight("w", [1, 1, inp_c, out_filters])
                            x = tf.nn.elu(x)
                            x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                                             data_format=self.data_format)
                            x = norm(
                                x, is_training=is_training, data_format=self.data_format, norm_type="batch")
                    else:
                        inp_c = self._get_C(x)
                        if x_stride > 1:
                            assert x_stride == 2
                            x = self._factorized_reduction(
                                x, out_filters, 2, is_training)
                        if inp_c != out_filters:
                            w = create_weight("w", [1, 1, inp_c, out_filters])
                            x = tf.nn.elu(x)
                            x = tf.nn.conv2d(
                                x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
                            x = norm(
                                x, is_training=is_training, data_format=self.data_format, norm_type="batch")
                    if (x_op in [0, 1, 2, 3] and
                        self.drop_path_keep_prob is not None and
                            is_training):
                        x = self._apply_drop_path(x, layer_id)

                y_id = arc[4 * cell_id + 2]
                used[y_id] += 1
                y_op = arc[4 * cell_id + 3]
                y = layers[y_id]
                y_stride = stride if y_id in [0, 1] else 1
                with tf.variable_scope("y_conv"):
                    if y_op in [0, 1]:
                        f_size = f_sizes[y_op]
                        y = self._fixed_conv(
                            y, f_size, out_filters, y_stride, is_training)
                    elif y_op in [2, 3]:
                        inp_c = self._get_C(y)
                        if y_op == 2:
                            y = tf.layers.average_pooling2d(
                                y, [3, 3], [y_stride, y_stride], "SAME",
                                data_format=self.actual_data_format)
                        else:
                            y = tf.layers.max_pooling2d(
                                y, [3, 3], [y_stride, y_stride], "SAME",
                                data_format=self.actual_data_format)
                        if inp_c != out_filters:
                            w = create_weight("w", [1, 1, inp_c, out_filters])
                            y = tf.nn.elu(y)
                            y = tf.nn.conv2d(y, w, [1, 1, 1, 1], "SAME",
                                             data_format=self.data_format)
                            y = norm(
                                y, is_training=is_training, data_format=self.data_format, norm_type="batch")
                    else:
                        inp_c = self._get_C(y)
                        if y_stride > 1:
                            assert y_stride == 2
                            y = self._factorized_reduction(
                                y, out_filters, 2, is_training)
                        if inp_c != out_filters:
                            w = create_weight("w", [1, 1, inp_c, out_filters])
                            y = tf.nn.elu(y)
                            y = tf.nn.conv2d(y, w, [1, 1, 1, 1], "SAME",
                                             data_format=self.data_format)
                            y = norm(
                                y, is_training=is_training, data_format=self.data_format, norm_type="batch")

                    if (y_op in [0, 1, 2, 3] and
                        self.drop_path_keep_prob is not None and
                            is_training):
                        y = self._apply_drop_path(y, layer_id)

                out = x + y
                layers.append(out)
        out = self._fixed_combine(layers, used, out_filters, is_training=is_training,
                                  normal_or_reduction_cell=normal_or_reduction_cell)

        return out

    def _enas_cell(self, x, curr_cell, prev_cell, op_id, out_filters, is_training):
        """Performs an enas operation specified by op_id."""

        num_possible_inputs = curr_cell + 1

        with tf.variable_scope("avg_pool"):
            avg_pool = tf.layers.average_pooling2d(
                x, [3, 3], [1, 1], "SAME", data_format=self.actual_data_format)
            avg_pool_c = self._get_C(avg_pool)
            if avg_pool_c != out_filters:
                with tf.variable_scope("conv"):
                    w = create_weight(
                        "w", [num_possible_inputs, avg_pool_c * out_filters])
                    w = w[prev_cell]
                    w = tf.reshape(w, [1, 1, avg_pool_c, out_filters])
                    avg_pool = tf.nn.elu(avg_pool)
                    avg_pool = tf.nn.conv2d(avg_pool, w, strides=[1, 1, 1, 1],
                                            padding="SAME", data_format=self.data_format)
                    avg_pool = norm(avg_pool, is_training=is_training,
                                    data_format=self.data_format, norm_type="batch")

        with tf.variable_scope("max_pool"):
            max_pool = tf.layers.max_pooling2d(
                x, [3, 3], [1, 1], "SAME", data_format=self.actual_data_format)
            max_pool_c = self._get_C(max_pool)
            if max_pool_c != out_filters:
                with tf.variable_scope("conv"):
                    w = create_weight(
                        "w", [num_possible_inputs, max_pool_c * out_filters])
                    w = w[prev_cell]
                    w = tf.reshape(w, [1, 1, max_pool_c, out_filters])
                    max_pool = tf.nn.elu(max_pool)
                    max_pool = tf.nn.conv2d(max_pool, w, strides=[1, 1, 1, 1],
                                            padding="SAME", data_format=self.data_format)
                    max_pool = norm(max_pool, is_training=is_training,
                                    data_format=self.data_format, norm_type="batch")

        x_c = self._get_C(x)
        if x_c != out_filters:
            with tf.variable_scope("x_conv"):
                w = create_weight(
                    "w", [num_possible_inputs, x_c * out_filters])
                w = w[prev_cell]
                w = tf.reshape(w, [1, 1, x_c, out_filters])
                x = tf.nn.elu(x)
                x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME",
                                 data_format=self.data_format)
                x = norm(x, is_training=is_training,
                         data_format=self.data_format, norm_type="batch")

        out = [
            self._enas_conv(x, curr_cell, prev_cell, 3, out_filters, is_training=is_training),
            self._enas_conv(x, curr_cell, prev_cell, 5, out_filters, is_training=is_training),
            avg_pool,
            max_pool,
            x,
        ]

        out = tf.stack(out, axis=0)
        if self.verbose > 0:
            print('-' * 80)
            shape_list = out.get_shape().as_list()
            print('_enas_cell::cell op_id: ' + str(op_id) + ' out shape: ' + str(shape_list) + ' data_format: ' + str(self.data_format))
            for line in traceback.format_stack():
                print(line.strip())
        out = out[op_id, :, :, :, :]
        return out

    def _enas_conv(self, x, curr_cell, prev_cell, filter_size, out_filters, is_training,
                   stack_conv=2, norm_type='group'):
        """Performs an enas convolution specified by the relevant parameters."""

        with tf.variable_scope("conv_{0}x{0}".format(filter_size)):
            num_possible_inputs = curr_cell + 2
            for conv_id in range(stack_conv):
                with tf.variable_scope("stack_{0}".format(conv_id)):
                    # create params and pick the correct path
                    inp_c = self._get_C(x)
                    w_depthwise = create_weight(
                        "w_depth", [num_possible_inputs, filter_size * filter_size * inp_c])
                    w_depthwise = w_depthwise[prev_cell, :]
                    w_depthwise = tf.reshape(
                        w_depthwise, [filter_size, filter_size, inp_c, 1])

                    w_pointwise = create_weight(
                        "w_point", [num_possible_inputs, inp_c * out_filters])
                    w_pointwise = w_pointwise[prev_cell, :]
                    w_pointwise = tf.reshape(
                        w_pointwise, [1, 1, inp_c, out_filters])

                    # the computations
                    x = tf.nn.elu(x)
                    x = tf.nn.separable_conv2d(
                        x,
                        depthwise_filter=w_depthwise,
                        pointwise_filter=w_pointwise,
                        strides=[1, 1, 1, 1], padding="SAME",
                        data_format=self.data_format)
                    x = norm(x, is_training=is_training, norm_type="batch")
        return x

    def _enas_layer(self, layer_id, prev_layers, arc, out_filters, is_training):
        """
        Args:
          layer_id: current layer
          prev_layers: cache of previous layers. for skip connections
          start_idx: where to start looking at. technically, we can infer this
            from layer_id, but why bother...
        """

        assert len(prev_layers) == 2, "need exactly 2 inputs"
        layers = [prev_layers[0], prev_layers[1]]
        layers = self._maybe_calibrate_size(
            layers, out_filters, is_training=is_training)
        used = []
        for cell_id in range(self.num_cells):
            prev_layers = tf.stack(layers, axis=0)
            with tf.variable_scope("cell_{0}".format(cell_id)):
                with tf.variable_scope("x"):
                    x_id = arc[4 * cell_id]
                    x_op = arc[4 * cell_id + 1]
                    x = prev_layers[x_id, :, :, :, :]
                    x = self._enas_cell(x, cell_id, x_id, x_op, out_filters, is_training=is_training)
                    x_used = tf.one_hot(
                        x_id, depth=self.num_cells + 2, dtype=tf.int32)

                with tf.variable_scope("y"):
                    y_id = arc[4 * cell_id + 2]
                    y_op = arc[4 * cell_id + 3]
                    y = prev_layers[y_id, :, :, :, :]
                    y = self._enas_cell(y, cell_id, y_id, y_op, out_filters, is_training=is_training)
                    y_used = tf.one_hot(
                        y_id, depth=self.num_cells + 2, dtype=tf.int32)

                out = x + y
                used.extend([x_used, y_used])
                layers.append(out)
                if self.verbose > 0:
                    print('-' * 80)
                    shape_list = out.get_shape().as_list()
                    print('_enas_layer::cell cell_id: ' + str(cell_id) + ' out shape: ' + str(shape_list) + ' data_format: ' + str(self.data_format))
                    for line in traceback.format_stack():
                        print(line.strip())

        used = tf.add_n(used)
        indices = tf.where(tf.equal(used, 0))
        indices = tf.to_int32(indices)
        indices = tf.reshape(indices, [-1])
        num_outs = tf.size(indices)
        out = tf.stack(layers, axis=0)
        out = tf.gather(out, indices, axis=0)

        inp = prev_layers[0]
        # get shape as an integer list,
        # this is necessary to prevent some shape information being lost
        # in the transpose/reshape below
        inp_shape_list = inp.get_shape().as_list()
        if self.verbose > 0:
            print('-' * 80)
            print('_enas_layer::inp tensor: ' + str(inp) + ' shape: ' + str(inp_shape_list) + ' data_format: ' + str(self.data_format))
            out_shape_list = out.get_shape().as_list()
            print('_enas_layer::out tensor: ' + str(out) + ' shape: ' + str(out_shape_list) + ' data_format: ' + str(self.data_format))
            print('_enas_layer::num_outs: ' + str(num_outs) + ' _enas_layer::out_filters: ' + str(out_filters))
            for line in traceback.format_stack():
                print(line.strip())
        if self.data_format == "NHWC":
            N = tf.shape(inp)[0]
            H = inp_shape_list[1]
            W = inp_shape_list[2]
            C = inp_shape_list[3]
            out = tf.transpose(out, [1, 2, 3, 0, 4])
            out = tf.reshape(out, [N, H, W, num_outs * out_filters])
        elif self.data_format == "NCHW":
            N = tf.shape(inp)[0]
            C = inp_shape_list[1]
            H = inp_shape_list[2]
            W = inp_shape_list[3]
            out = tf.transpose(out, [1, 0, 2, 3, 4])
            out = tf.reshape(out, [N, num_outs * out_filters, H, W])
        else:
            raise ValueError(
                "Unknown data_format '{0}'".format(self.data_format))

        with tf.variable_scope("final_conv"):
            if self.verbose > 0:
                print('-' * 80)
                shape_list = out.get_shape().as_list()
                print('_enas_layer::final_conv out shape: ' + str(shape_list) + ' data_format: ' + str(self.data_format))
                for line in traceback.format_stack():
                    print(line.strip())
            w = create_weight(
                "w", [self.num_cells + 2, out_filters * out_filters])
            w = tf.gather(w, indices, axis=0)
            w = tf.reshape(w, [1, 1, num_outs * out_filters, out_filters])
            out = tf.nn.elu(out)
            out = tf.nn.conv2d(out, w, strides=[1, 1, 1, 1], padding="SAME",
                               data_format=self.data_format)
            out = norm(out, is_training=is_training,
                       data_format=self.data_format, norm_type="batch")

        out = tf.reshape(out, tf.shape(prev_layers[0]))

        return out

    # override
    def eval_once(self, sess, eval_set, feed_dict=None, verbose=False):
        """Expects self.acc and self.global_step to be defined.

        Args:
          sess: tf.Session() or one of its wrap arounds.
          feed_dict: can be used to give more information to sess.run().
          eval_set: "valid" or "test"
        """

        assert self.global_step is not None
        global_step = sess.run(self.global_step)
        print("Eval {} set at {}".format(eval_set, global_step))

        if eval_set == "valid":
            assert self.x_valid is not None
            assert self.valid_acc is not None
            num_examples = self.num_valid_examples
            num_batches = self.num_valid_batches
            acc_op = self.valid_acc
            acc_op_5mm_7_5deg = self.valid_acc_5mm_7_5deg
            acc_op_1cm_15deg = self.valid_acc_1cm_15deg
            acc_op_2_30 = self.valid_acc_2cm_30deg
            acc_op_4_60 = self.valid_acc_4cm_60deg
            acc_op_8_120 = self.valid_acc_8cm_120deg
            acc_op_16cm_240deg = self.valid_acc_16cm_240deg
            acc_op_32cm_360deg = self.valid_acc_32cm_360deg
            loss_secondary_op = self.valid_loss_secondary
            cart_op = self.valid_cart_error
            ang_er_op = self.valid_angle_error
            loss_op = self.valid_loss
            mae_op = self.valid_mae
            csvfile = self.output_dir + "/valid_metrics.csv"
        elif eval_set == "test":
            assert self.test_acc is not None
            num_examples = self.num_test_examples
            num_batches = self.num_test_batches
            acc_op = self.test_acc
            acc_op_5mm_7_5deg = self.test_acc_5mm_7_5deg
            acc_op_1cm_15deg = self.test_acc_1cm_15deg
            acc_op_2_30 = self.test_acc_2cm_30deg
            acc_op_4_60 = self.test_acc_4cm_60deg
            acc_op_8_120 = self.test_acc_8cm_120deg
            acc_op_16cm_240deg = self.test_acc_16cm_240deg
            acc_op_32cm_360deg = self.test_acc_32cm_360deg
            loss_secondary_op = self.test_loss_secondary
            ang_er_op = self.test_angle_error
            cart_op = self.test_cart_error
            loss_op = self.test_loss
            mae_op = self.test_mae
            csvfile = self.output_dir + "/test_metrics.csv"
        else:
            raise NotImplementedError("Unknown eval_set '{}'".format(eval_set))

        total_acc = 0
        total_acc_5mm_7_5deg = 0
        total_acc_1cm_15deg = 0
        total_acc_2_30 = 0
        total_acc_4_60 = 0
        total_acc_8_120 = 0
        total_acc_16cm_240deg = 0
        total_acc_32cm_360deg = 0
        total_cart_error = 0
        total_mae = 0
        total_loss = 0
        total_exp = 0
        total_angle_error = 0
        total_loss_sec = 0
        normal_arc = []
        reduce_arc = []
        for batch_id in range(num_batches):
            # if batch_id == 0:
            #     if feed_dict is None:
            #         feed_dict = {}
            #     # print the arc if we're on batch 0
            #     feed_dict['print_arc'] = self.print_arc
            # elif batch_id == 1 and feed_dict is not None and 'print_arc' in feed_dict:
            #     # remove the print arc tensor if we're on batch 1
            #     feed_dict.pop('print_arc', None)
            if self.fixed_arc is None:
                acc, acc_5_7_5, acc_1_15, acc_2_30, acc_4_60, acc_8_120, acc_16_240, acc_32_360, cart_error, angle_error, mse, mae, loss_sec = sess.run(
                    [acc_op, acc_op_5mm_7_5deg, acc_op_1cm_15deg, acc_op_2_30, acc_op_4_60, acc_op_8_120, acc_op_16cm_240deg, acc_op_32cm_360deg, cart_op, ang_er_op, loss_op, mae_op, loss_secondary_op], feed_dict=feed_dict)
            else:
                acc, acc_5_7_5, acc_1_15, acc_2_30, acc_4_60, acc_8_120, acc_16_240, acc_32_360, cart_error, angle_error, mse, mae, loss_sec = sess.run(
                    [acc_op, acc_op_5mm_7_5deg, acc_op_1cm_15deg, acc_op_2_30, acc_op_4_60, acc_op_8_120, acc_op_16cm_240deg, acc_op_32cm_360deg, cart_op, ang_er_op, loss_op, mae_op, loss_secondary_op], feed_dict=feed_dict)
            total_acc += acc
            total_acc_5mm_7_5deg += acc_5_7_5
            total_acc_1cm_15deg += acc_1_15
            total_acc_2_30 += acc_2_30
            total_acc_4_60 += acc_4_60
            total_acc_8_120 += acc_8_120
            total_acc_16cm_240deg += acc_16_240
            total_acc_32cm_360deg += acc_32_360
            total_cart_error += cart_error
            total_angle_error += angle_error
            total_loss += mse
            total_mae += mae
            total_loss_sec += loss_sec
            total_exp += self.eval_batch_size
            if verbose:
                sys.stdout.write(
                    "\r{:<5d}/{:>5d}".format(total_acc, total_exp))
        if verbose:
            print("")
        print("{}_accuracy: {:<6.4f}".format(
            eval_set, float(total_acc) / total_exp))
        print("{}_accuracy_5mm_7_5deg: {:<6.4f}".format(
            eval_set, float(total_acc_5mm_7_5deg) / total_exp))
        print("{}_accuracy_1cm_15deg: {:<6.4f}".format(
            eval_set, float(total_acc_1cm_15deg) / total_exp))
        print("{}_accuracy_2cm_30deg: {:<6.4f}".format(
            eval_set, float(total_acc_2_30) / total_exp))
        print("{}_accuracy_4cm_60deg: {:<6.4f}".format(
            eval_set, float(total_acc_4_60) / total_exp))
        print("{}_accuracy_8cm_120deg: {:<6.4f}".format(
            eval_set, float(total_acc_8_120) / total_exp))
        print("{}_accuracy_16cm_240deg: {:<6.4f}".format(
            eval_set, float(total_acc_16cm_240deg) / total_exp))
        print("{}_accuracy_32cm_360deg: {:<6.4f}".format(
            eval_set, float(total_acc_32cm_360deg) / total_exp))
        if self.rotation_only is False and self.stacking_reward is False:
            print("{}_cart_error: {:<6.4f}".format(
                eval_set, float(total_cart_error) / num_batches))
        if self.translation_only is False and self.stacking_reward is False:
            print("{}_angle_error: {:<6.4f}".format(
                eval_set, float(total_angle_error) / num_batches))
        print("{}_loss_1: {:<6.4f}".format(
            eval_set, float(total_loss) / num_batches))
        print("{}_loss_2: {:<6.4f}".format(
            eval_set, float(total_loss_sec) / num_batches))
        print("{}_mae: {:<6.4f}".format(
            eval_set, float(total_mae) / num_batches))
        if self.fixed_arc is None:
            print(eval_set, end=" ")
            print('Eval Architecture:')
            # print(np.reshape(normal_arc, [-1]))
            # print(np.reshape(reduce_arc, [-1]))
            # self.global_step = tf.Print(self.global_step, [self.normal_arc, self.reduce_arc], 'connect_controller(): [normal_arc, reduce_arc]: ', summarize=20)
        if os.path.exists(csvfile):
            file_mode = 'a'
        else:
            file_mode = 'w+'
        with open(csvfile, file_mode) as fp:
            fp.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(
                total_acc, total_acc_5mm_7_5deg, total_acc_1cm_15deg, total_acc_2_30, total_acc_4_60, total_acc_8_120, total_acc_16cm_240deg, total_acc_32cm_360deg, total_loss, total_mae, total_angle_error, total_cart_error, total_loss_sec))

    # override
    def _build_train(self):
        print("-" * 80)
        print("Build train graph")
        # print("xtrshape-----------------------",self.x_train.shape)
        logits = self._model(self.x_train, is_training=True)
        # tf.Print(logits,[tf.shape(logits),"-----------log"])
        # print("ytrshape-----------", self.y_train)
        if self.dataset == "stacking":
            log_probs = tf.nn.sigmoid(logits)
            if self.use_msle is False:
                self.loss = tf.losses.mean_squared_error(
                    labels=self.y_train, predictions=log_probs)
                self.loss_secondary = tf.reduce_mean(keras.losses.msle(
                    self.y_train, log_probs))
            else:
                self.loss = tf.reduce_mean(keras.losses.msle(
                    self.y_train, log_probs))
                self.loss_secondary = tf.losses.mean_squared_error(
                    labels=self.y_train, predictions=log_probs)
        else:
            activation_fn = tf.nn.sparse_softmax_cross_entropy_with_logits
            log_probs = activation_fn(
                logits=logits, labels=self.y_train)
            self.loss = tf.reduce_mean(log_probs)

        if self.use_aux_heads:
            if self.dataset == "stacking":
                # Check
                log_probs = tf.losses.mean_squared_error(
                    labels=self.y_train, predictions=log_probs)
            else:
                log_probs = activation_fn(
                    logits=self.aux_logits, labels=self.y_train)
            self.aux_loss = tf.reduce_mean(log_probs)
            train_loss = self.loss + 0.4 * self.aux_loss
        else:
            train_loss = self.loss

        if self.dataset == "stacking":
            cast_type = tf.to_float
        else:
            cast_type = tf.to_int32

        if self.dataset == "stacking":
            self.train_preds = tf.nn.sigmoid(logits)
            self.train_acc = grasp_metrics.grasp_acc(
                self.y_train, self.train_preds)
            print("train_acc--------------", self.train_acc)
            self.train_acc = self.train_acc
            self.train_acc = tf.reduce_mean(self.train_acc)

            self.train_acc_5mm_7_5deg = grasp_metrics.grasp_acc_5mm_7_5deg(
                self.y_train, self.train_preds)
            self.train_acc_5mm_7_5deg = tf.reduce_mean(self.train_acc_5mm_7_5deg)

            self.train_acc_1cm_15deg = grasp_metrics.grasp_acc_1cm_15deg(
                self.y_train, self.train_preds)
            self.train_acc_1cm_15deg = tf.reduce_mean(self.train_acc_1cm_15deg)

            self.train_acc_2cm_30deg = grasp_metrics.grasp_acc_2cm_30deg(
                self.y_train, self.train_preds)
            self.train_acc_2cm_30deg = tf.reduce_mean(self.train_acc_2cm_30deg)

            self.train_acc_4cm_60deg = grasp_metrics.grasp_acc_4cm_60deg(
                self.y_train, self.train_preds)
            self.train_acc_4cm_60deg = tf.reduce_mean(self.train_acc_4cm_60deg)

            self.train_acc_8cm_120deg = grasp_metrics.grasp_acc_8cm_120deg(
                self.y_train, self.train_preds)
            self.train_acc_8cm_120deg = tf.reduce_mean(self.train_acc_8cm_120deg)

            self.train_acc_16cm_240deg = grasp_metrics.grasp_acc_16cm_240deg(
                self.y_train, self.train_preds)
            self.train_acc_16cm_240deg = tf.reduce_mean(self.train_acc_16cm_240deg)

            self.train_acc_32cm_360deg = grasp_metrics.grasp_acc_32cm_360deg(
                self.y_train, self.train_preds)
            self.train_acc_32cm_360deg = tf.reduce_mean(self.train_acc_32cm_360deg)

            self.train_cart_error = grasp_metrics.cart_error(
                self.y_train, self.train_preds)
            if self.rotation_only is True or self.stacking_reward is True:
                self.train_cart_error = tf.zeros([1])
            else:
                self.train_cart_error = tf.reduce_mean(self.train_cart_error)
            if self.translation_only is True or self.stacking_reward is True:
                self.train_angle_error = tf.zeros([1])
            else:
                self.train_angle_error = grasp_metrics.angle_error(
                    self.y_train, self.train_preds)
                self.train_angle_error = tf.reduce_mean(self.train_angle_error)
            self.train_mae = tf.metrics.mean_absolute_error(
                self.y_train, self.train_preds)
            self.train_mae = tf.reduce_mean(self.train_mae)

        else:
            self.train_preds = tf.argmax(logits, axis=1)
            self.train_preds = cast_type(self.train_preds)
            # tf.Print(self.train_preds,[tf.shape(self.train_preds),"trainpreds----"])
            # tf.Print(self.y_train,[tf.shape(self.y_train),"ytra==-------------"])
            self.train_acc = tf.equal(self.train_preds, self.y_train)
            self.train_acc = cast_type(self.train_acc)
            self.train_acc = tf.reduce_mean(self.train_acc)
            self.train_cart_error = tf.zeros([1])
            self.train_angle_error = tf.zeros([1])
            self.train_mae = tf.zeros([1])

        tf_variables = [
            var for var in tf.trainable_variables() if (
                var.name.startswith(self.name) and "aux_head" not in var.name)]
        self.num_vars = count_model_params(tf_variables)
        print("Model has {0} params".format(self.num_vars))

        self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
            train_loss,
            tf_variables,
            self.global_step,
            clip_mode=self.clip_mode,
            grad_bound=self.grad_bound,
            l2_reg=self.l2_reg,
            lr_init=self.lr_init,
            lr_dec_start=self.lr_dec_start,
            lr_dec_every=self.lr_dec_every,
            lr_dec_rate=self.lr_dec_rate,
            lr_cosine=self.lr_cosine,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            lr_T_0=self.lr_T_0,
            lr_T_mul=self.lr_T_mul,
            num_train_batches=self.num_train_batches,
            optim_algo=self.optim_algo,
            sync_replicas=self.sync_replicas,
            num_aggregate=self.num_aggregate,
            num_replicas=self.num_replicas)

    # override
    def _build_valid(self):
        if self.x_valid is not None:
            print("-" * 80)
            print("Build valid graph")
            logits = self._model(
                self.x_valid, is_training=True, reuse=True)
            if self.dataset == "stacking":
                logits = tf.nn.sigmoid(logits)
                cast_type = tf.to_float
                self.valid_preds = logits
                self.valid_acc = grasp_metrics.grasp_acc(
                    self.y_valid, self.valid_preds)
                self.valid_acc = tf.reduce_sum(self.valid_acc)

                self.valid_acc_5mm_7_5deg = grasp_metrics.grasp_acc_5mm_7_5deg(
                    self.y_valid, self.valid_preds)
                self.valid_acc_5mm_7_5deg = tf.reduce_sum(self.valid_acc_5mm_7_5deg)

                self.valid_acc_1cm_15deg = grasp_metrics.grasp_acc_1cm_15deg(
                    self.y_valid, self.valid_preds)
                self.valid_acc_1cm_15deg = tf.reduce_sum(self.valid_acc_1cm_15deg)

                self.valid_acc_2cm_30deg = grasp_metrics.grasp_acc_2cm_30deg(
                    self.y_valid, self.valid_preds)
                self.valid_acc_2cm_30deg = tf.reduce_sum(self.valid_acc_2cm_30deg)

                self.valid_acc_4cm_60deg = grasp_metrics.grasp_acc_4cm_60deg(
                    self.y_valid, self.valid_preds)
                self.valid_acc_4cm_60deg = tf.reduce_sum(self.valid_acc_4cm_60deg)

                self.valid_acc_8cm_120deg = grasp_metrics.grasp_acc_8cm_120deg(
                    self.y_valid, self.valid_preds)
                self.valid_acc_8cm_120deg = tf.reduce_sum(self.valid_acc_8cm_120deg)

                self.valid_acc_16cm_240deg = grasp_metrics.grasp_acc_16cm_240deg(
                    self.y_valid, self.valid_preds)
                self.valid_acc_16cm_240deg = tf.reduce_sum(self.valid_acc_16cm_240deg)

                self.valid_acc_32cm_360deg = grasp_metrics.grasp_acc_32cm_360deg(
                    self.y_valid, self.valid_preds)
                self.valid_acc_32cm_360deg = tf.reduce_sum(self.valid_acc_32cm_360deg)

                if self.use_msle is False:
                    self.valid_loss = tf.losses.mean_squared_error(
                        labels=self.y_valid, predictions=self.valid_preds)
                    self.valid_loss_secondary = tf.reduce_mean(keras.losses.msle(
                        self.y_valid, self.valid_preds))
                else:
                    self.valid_loss = tf.reduce_mean(keras.losses.msle(
                        self.y_valid, self.valid_preds))
                    self.valid_loss_secondary = tf.losses.mean_squared_error(
                        labels=self.y_valid, predictions=self.valid_preds)

                self.valid_cart_error = grasp_metrics.cart_error(
                  self.y_valid, self.valid_preds)
                if self.rotation_only is True or self.stacking_reward is True:
                    self.valid_cart_error = tf.zeros([1])
                else:
                    self.valid_cart_error = tf.reduce_mean(self.valid_cart_error)
                if self.translation_only is True or self.stacking_reward is True:
                    self.valid_angle_error = tf.zeros([1])
                else:
                    self.valid_angle_error = grasp_metrics.angle_error(
                        self.y_valid, self.valid_preds)
                    self.valid_angle_error = tf.reduce_mean(self.valid_angle_error)
                self.valid_mae = tf.metrics.mean_absolute_error(
                    self.y_valid, self.valid_preds)
                self.valid_mae = tf.reduce_mean(self.valid_mae)

            else:
                cast_type = tf.to_int32
                self.valid_preds = tf.argmax(logits, axis=1)
                self.valid_preds = cast_type(self.valid_preds)
                self.valid_acc = tf.equal(self.valid_preds, self.y_valid)
                self.valid_acc = cast_type(self.valid_acc)
                self.valid_acc = tf.reduce_sum(self.valid_acc)

    # override
    def _build_test(self):
        print("-" * 80)
        print("Build test graph")
        logits = self._model(self.x_test, is_training=False, reuse=True)
        if self.dataset == "stacking":
            logits = tf.nn.sigmoid(logits)
            cast_type = tf.to_float
            self.test_preds = logits
            self.test_acc = grasp_metrics.grasp_acc(
                self.y_test, self.test_preds)
            self.test_acc = tf.reduce_sum(self.test_acc)

            self.test_acc_5mm_7_5deg = grasp_metrics.grasp_acc_5mm_7_5deg(
                self.y_test, self.test_preds)
            self.test_acc_5mm_7_5deg = tf.reduce_sum(self.test_acc_5mm_7_5deg)

            self.test_acc_1cm_15deg = grasp_metrics.grasp_acc_1cm_15deg(
                self.y_test, self.test_preds)
            self.test_acc_1cm_15deg = tf.reduce_sum(self.test_acc_1cm_15deg)

            self.test_acc_2cm_30deg = grasp_metrics.grasp_acc_2cm_30deg(
                    self.y_test, self.test_preds)
            self.test_acc_2cm_30deg = tf.reduce_sum(self.test_acc_2cm_30deg)

            self.test_acc_4cm_60deg = grasp_metrics.grasp_acc_4cm_60deg(
                self.y_test, self.test_preds)
            self.test_acc_4cm_60deg = tf.reduce_sum(self.test_acc_4cm_60deg)

            self.test_acc_8cm_120deg = grasp_metrics.grasp_acc_8cm_120deg(
                self.y_test, self.test_preds)
            self.test_acc_8cm_120deg = tf.reduce_sum(self.test_acc_8cm_120deg)

            self.test_acc_16cm_240deg = grasp_metrics.grasp_acc_16cm_240deg(
                self.y_test, self.test_preds)
            self.test_acc_16cm_240deg = tf.reduce_sum(self.test_acc_16cm_240deg)

            self.test_acc_32cm_360deg = grasp_metrics.grasp_acc_32cm_360deg(
                self.y_test, self.test_preds)
            self.test_acc_32cm_360deg = tf.reduce_sum(self.test_acc_32cm_360deg)

            self.test_cart_error = grasp_metrics.cart_error(
                self.y_test, self.test_preds)
            if self.rotation_only is True or self.stacking_reward is True:
                self.test_cart_error = tf.zeros([1])
            else:
                self.test_cart_error = tf.reduce_mean(self.test_cart_error)
            if self.translation_only is True or self.stacking_reward is True:
                self.test_angle_error = tf.zeros([1])
            else:
                self.test_angle_error = grasp_metrics.angle_error(
                    self.y_test, self.test_preds)
                self.test_angle_error = tf.reduce_mean(self.test_angle_error)
            self.test_mae = tf.metrics.mean_absolute_error(
                self.y_test, self.test_preds)
            self.test_mae = tf.reduce_mean(self.test_mae)
            if self.use_msle is False:
                self.test_loss = tf.losses.mean_squared_error(
                        labels=self.y_test, predictions=self.test_preds)
                self.test_loss_secondary = tf.reduce_mean(keras.losses.msle(
                        self.y_test, self.test_preds))
            else:
                self.test_loss = tf.reduce_mean(keras.losses.msle(
                    self.y_test, self.test_preds))
                self.test_loss_secondary = tf.losses.mean_squared_error(
                    labels=self.y_test, predictions=self.test_preds)

        else:
            cast_type = tf.to_int32
            self.test_preds = tf.argmax(logits, axis=1)
            self.test_preds = cast_type(self.test_preds)
            self.test_acc = tf.equal(self.test_preds, self.y_test)
            self.test_acc = cast_type(self.test_acc)
            self.test_acc = tf.reduce_sum(self.test_acc)

    # override
    def build_valid_rl(self, shuffle=False):
        print("-" * 80)
        print("Build valid graph on shuffled data")
        if self.dataset == "stacking":
            with tf.device("/cpu:0"):
                if not shuffle:
                    self.x_valid_shuffle, self.y_valid_shuffle = self.x_valid, self.y_valid
                else:
                    raise NotImplementedError(
                        'This portion of the code is not correctly implemented, '
                        'so it must be fixed before running it. '
                        'see models.py::__init__() for reference code using the '
                        'CostarBlockStackingSequence().')
                    data_features = ['image_0_image_n_vec_xyz_aaxyz_nsc_15']
                    label_features = ['grasp_goal_xyz_aaxyz_nsc_8']
                    validation_shuffle_generator = CostarBlockStackingSequence(
                        self.validation_data, batch_size=self.batch_size, verbose=0,
                        label_features_to_extract=label_features,
                        data_features_to_extract=data_features, output_shape=self.image_shape, shuffle=True)
                    validation_enqueuer = OrderedEnqueuer(
                                  validation_generator,
                                  use_multiprocessing=False,
                                  shuffle=True)
                    validation_enqueuer.start(workers=10, max_queue_size=100)

                    def validation_generator(): return iter(train_enqueuer.get())
                    validation_dataset = Dataset.from_generator(validation_generator, (tf.float32, tf.float32), (tf.TensorShape([None, self.image_shape[0], self.image_shape[1], self.data_features_len]), tf.TensorShape([None, None])))
                    x_valid_shuffle, y_valid_shuffle = validation_dataset.make_one_shot_iterator().get_next()

        else:
            with tf.device("/cpu:0"):
                # shuffled valid data: for choosing validation model
                if not shuffle and self.data_format == "NCHW":
                    self.images["valid_original"] = np.transpose(
                        self.images["valid_original"], [0, 3, 1, 2])
                self.x_valid_shuffle, self.y_valid_shuffle = tf.train.shuffle_batch(
                    [self.images["valid_original"], self.labels["valid_original"]],
                    batch_size=self.batch_size,
                    capacity=25000,
                    enqueue_many=True,
                    min_after_dequeue=0,
                    num_threads=16,
                    seed=self.seed,
                    allow_smaller_final_batch=True,
                )

                def _pre_process(x):
                    x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
                    x = tf.random_crop(x, [32, 32, 3], seed=self.seed)
                    x = tf.image.random_flip_left_right(x, seed=self.seed)
                    if self.data_format == "NCHW":
                        x = tf.transpose(x, [2, 0, 1])
                    return x

                if shuffle:
                    x_valid_shuffle = tf.map_fn(
                        _pre_process, x_valid_shuffle, back_prop=False)

        # TODO(ahundt) should is_training really be true here? this looks like a validation step... but it is in the controller so maybe some training does happen...
        logits = self._model(
            self.x_valid_shuffle, is_training=True, reuse=True)
        if self.dataset == "stacking":
            logits = tf.nn.sigmoid(logits)
            cast_type = tf.to_float
            self.valid_shuffle_preds = logits
            self.valid_shuffle_acc = grasp_metrics.grasp_acc(
                self.y_valid_shuffle, self.valid_shuffle_preds)
            self.valid_shuffle_acc = tf.reduce_sum(self.valid_shuffle_acc)

            self.valid_shuffle_acc_5mm_7_5deg = grasp_metrics.grasp_acc_5mm_7_5deg(
                self.y_valid_shuffle, self.valid_shuffle_preds)
            self.valid_shuffle_acc_5mm_7_5deg = tf.reduce_sum(self.valid_shuffle_acc_5mm_7_5deg)

            self.valid_shuffle_acc_1cm_15deg = grasp_metrics.grasp_acc_1cm_15deg(
                self.y_valid_shuffle, self.valid_shuffle_preds)
            self.valid_shuffle_acc_1cm_15deg = tf.reduce_sum(self.valid_shuffle_acc_1cm_15deg)

            self.valid_shuffle_acc_2cm_30deg = grasp_metrics.grasp_acc_2cm_30deg(
                    self.y_valid_shuffle, self.valid_shuffle_preds)
            self.valid_shuffle_acc_2cm_30deg = tf.reduce_sum(self.valid_shuffle_acc_2cm_30deg)

            self.valid_shuffle_acc_4cm_60deg = grasp_metrics.grasp_acc_4cm_60deg(
                self.y_valid_shuffle, self.valid_shuffle_preds)
            self.valid_shuffle_acc_4cm_60deg = tf.reduce_sum(self.valid_shuffle_acc_4cm_60deg)

            self.valid_shuffle_acc_8cm_120deg = grasp_metrics.grasp_acc_8cm_120deg(
                self.y_valid_shuffle, self.valid_shuffle_preds)
            self.valid_shuffle_acc_8cm_120deg = tf.reduce_sum(self.valid_shuffle_acc_8cm_120deg)

            self.valid_shuffle_acc_16cm_240deg = grasp_metrics.grasp_acc_16cm_240deg(
                self.y_valid_shuffle, self.valid_shuffle_preds)
            self.valid_shuffle_acc_16cm_240deg = tf.reduce_sum(self.valid_shuffle_acc_16cm_240deg)

            self.valid_shuffle_acc_32cm_360deg = grasp_metrics.grasp_acc_32cm_360deg(
                self.y_valid_shuffle, self.valid_shuffle_preds)
            self.valid_shuffle_acc_32cm_360deg = tf.reduce_sum(self.valid_shuffle_acc_32cm_360deg)

            if self.use_msle is False:
                self.valid_shuffle_loss = tf.reduce_mean(tf.losses.mean_squared_error(
                        labels=self.y_valid_shuffle, predictions=self.valid_shuffle_preds))
                self.valid_shuffle_loss_secondary = tf.reduce_mean(keras.losses.msle(
                    self.y_valid_shuffle, self.valid_shuffle_preds))
            else:
                self.valid_shuffle_loss = tf.reduce_mean(keras.losses.msle(
                    self.y_valid_shuffle, self.valid_shuffle_preds))
                self.valid_shuffle_loss_secondary = tf.losses.mean_squared_error(
                    labels=self.y_valid_shuffle, predictions=self.valid_shuffle_preds)

            self.valid_shuffle_cart_error = grasp_metrics.cart_error(
                self.y_valid_shuffle, self.valid_shuffle_preds)
            if self.rotation_only is True or self.stacking_reward is True:
                self.valid_shuffle_cart_error = tf.zeros([1])
            else:
                self.valid_shuffle_cart_error = tf.reduce_mean(self.valid_shuffle_cart_error)
            if self.translation_only is True or self.stacking_reward is True:
                self.valid_shuffle_angle_error = tf.zeros([1])
            else:
                self.valid_shuffle_angle_error = grasp_metrics.angle_error(
                    self.y_valid_shuffle, self.valid_shuffle_preds)
                self.valid_shuffle_angle_error = tf.reduce_mean(self.valid_shuffle_angle_error)
            self.valid_shuffle_mae = tf.metrics.mean_absolute_error(
                self.y_valid_shuffle, self.valid_shuffle_preds)
            self.valid_shuffle_mae = tf.reduce_mean(self.valid_shuffle_mae)

        else:
            cast_type = tf.to_int32
            self.valid_shuffle_preds = tf.argmax(logits, axis=1)
            self.valid_shuffle_preds = cast_type(self.valid_shuffle_preds)
            self.valid_shuffle_acc = tf.equal(self.valid_shuffle_preds, self.y_valid_shuffle)
            self.valid_shuffle_acc = cast_type(self.valid_shuffle_acc)
            self.valid_shuffle_acc = tf.reduce_sum(self.valid_shuffle_acc)

    def connect_controller(self, controller_model, verbose=0):
        if self.fixed_arc is None:
            sample_arc = controller_model.sample_arc
            normal_arc, reduce_arc = sample_arc
            # self.print_arc = tf.Print([0], [normal_arc, reduce_arc], 'connect_controller(): [normal_arc, reduce_arc]: ', summarize=20)

            if verbose:
                normal_arc = tf.Print(normal_arc, [normal_arc, reduce_arc], 'connect_controller(): [normal_arc, reduce_arc]: ', summarize=20)
            self.normal_arc = normal_arc
            self.reduce_arc = reduce_arc
        else:
            fixed_arc = np.array([int(x)
                                  for x in self.fixed_arc.split(" ") if x])
            self.normal_arc = fixed_arc[:4 * self.num_cells]
            self.reduce_arc = fixed_arc[4 * self.num_cells:]

        self._build_train()
        self._build_valid()
        self._build_test()
