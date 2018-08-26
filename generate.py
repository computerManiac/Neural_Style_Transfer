import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import cv2
import skimage


style_image_path = "monet.jpg"
content_image_path = "louvre_small.jpg"
image_width = 400
image_height = 300
means = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
noise_ratio = 0.6
alpha = 100
beta = 5
vgg_model_dir = 'imagenet-vgg-verydeep-19.mat'
sess = tf.Session()

model = load_vgg_model(vgg_model_dir)
style_image = scipy.misc.imread(style_image_path)
style_image = cv2.resize(style_image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
content_image = scipy.misc.imread(content_image_path)
content_image = cv2.resize(content_image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)

def prepare_image(image):
    
    image = np.reshape(image, ((1,) + image.shape))
    image = image - means
    
    return image

def save_img(path, image):
    
    image = image + means
    image = np.clip(image[0],0,255).astype('uint8')
    scipy.misc.imsave(path,image)

def content_cost(a_C,a_G):
    
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_C_unrolled = tf.reshape(tf.transpose(a_C,perm=[0,3,1,2]),[n_C,n_W*n_H])
    a_G_unrolled = tf.reshape(tf.transpose(a_G,perm=[0,3,1,2]),[n_C,n_W*n_H])
    
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled))) * (1/(4*n_C*n_W*n_H))
    
    return J_content

def gram_matrix(A):
    
    GA = tf.matmul(A,tf.transpose(A))
    return GA

def layer_style_cost(a_S,a_G):
    
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_S = tf.reshape(tf.transpose(a_S,perm=[0,3,1,2]),[n_C,n_W*n_H])
    a_G = tf.reshape(tf.transpose(a_G,perm=[0,3,1,2]),[n_C,n_W*n_H])
    
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    
    J_style_layer = tf.reduce_sum(tf.square((tf.subtract(GS,GG)))) * (1/(4*np.square(n_H*n_W)*(n_C**2)))
    
    return J_style_layer

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

def style_cost(model):
    
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = layer_style_cost(a_S, a_G)
        
        J_style += coeff * J_style_layer
    return J_style

def total_cost(J_content, J_style, alpha = 15, beta = 25):
    return (alpha*J_content) + (beta*J_style)


#tf.reset_default_graph()


def noise_image(content_image, ratio = noise_ratio):
    noise_image = np.random.uniform(0, 255, (1, image_height, image_width, 3)).astype('float32')
    
    input_image = noise_image * ratio + content_image * (1 - ratio)
    
    return input_image

style_image = prepare_image(style_image)
content_image = prepare_image(content_image)
generated_image = noise_image(content_image)

sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_C = sess.run(out)
a_G = out
J_content = content_cost(a_C,a_G)

sess.run(model['input'].assign(style_image))
J_style = style_cost(model)

J = total_cost(J_content,J_style)

optimizer = tf.train.AdamOptimizer(1.0)
train_step = optimizer.minimize(J)

def model_nn(sess,input_image,num_iters=200):
    
    sess.run(tf.global_variables_initializer())
    
    sess.run(model['input'].assign(input_image))

    for i in range(num_iters):
        sess.run(train_step)
        
        generated_image = sess.run(model['input'])
        
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            save_img("output/" + str(i) + ".png", generated_image)
        
    save_img("output/" + "generated_image.png", generated_image)
    return generated_image

model_nn(sess, generated_image, num_iters = 10000)

sess.close()
