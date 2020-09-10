import numpy as np
import tensorflow as tf

NOISE_DIM = 96

def leaky_relu(x, alpha=0.01):
    """Compute the leaky ReLU activation function.
    
    Inputs:
    - x: TensorFlow Tensor with arbitrary shape
    - alpha: leak parameter for leaky ReLU
    
    Returns:
    TensorFlow Tensor with the same shape as x
    """
    # TODO: implement leaky ReLU
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return tf.maximum(x, alpha * x)
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
def sample_noise(batch_size, dim, seed=None):
    """Generate random uniform noise from -1 to 1.
    
    Inputs:
    - batch_size: integer giving the batch size of noise to generate
    - dim: integer giving the dimension of the noise to generate
    
    Returns:
    TensorFlow Tensor containing uniform noise in [-1, 1] with shape [batch_size, dim]
    """
    if seed is not None:
        tf.random.set_seed(seed)
    # TODO: sample and return noise
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return tf.random.uniform(shape=(batch_size, dim), minval=-1, maxval=1)
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
def discriminator(seed=None):
    """Compute discriminator score for a batch of input images.
    
    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
    
    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """
    if seed is not None:
        tf.random.set_seed(seed)

    model = None
    ##############################################################################
    # TODO: Implement architecture                                               #
    #                                                                            #
    # HINT: tf.keras.models.Sequential might be helpful.                         #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    #####################################################################
    # GIVEN ARCHITECTURE:                                               #
    #                                                                   #
    # - Fully connected layer with input size 784 and output size 256   #
    # - LeakyReLU with alpha 0.01                                       #
    # - Fully connected layer with output size 256                      #
    # - LeakyReLU with alpha 0.01                                       #
    # - Fully connected layer with output size 1                        #
    #####################################################################
    
    # NOTE: with "lazy-evalutaion-like" workflow, 
    # if I omit the `input_shape` or the Input layer,
    # Weights will not be created before this model is first called 
    
    layers = [
        tf.keras.Input((784,)),
        tf.keras.layers.Dense(256, activation=lambda x: leaky_relu(x, 0.01), use_bias=True), 
        tf.keras.layers.Dense(256, activation=lambda x: leaky_relu(x, 0.01), use_bias=True), 
        tf.keras.layers.Dense(1, use_bias=True)
    ]
    
    model = tf.keras.Sequential(layers)
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return model

def generator(noise_dim=NOISE_DIM, seed=None):
    """Generate images from a random noise vector.
    
    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
    
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """

    if seed is not None:
        tf.random.set_seed(seed)
    model = None
    ##############################################################################
    # TODO: Implement architecture                                               #
    #                                                                            #
    # HINT: tf.keras.models.Sequential might be helpful.                         #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    
    ###########################################################################################
    # GIVEN ARCHITECTURE:                                                                     #
    #                                                                                         #
    # - Fully connected layer with inupt size tf.shape(z)[1] (the number of noise dimensions) #
    #   and output size 1024                                                                  #
    # - `ReLU`                                                                                #
    # - Fully connected layer with output size 1024                                           #
    # - `ReLU`                                                                                #
    # - Fully connected layer with output size 784                                            #
    # - `TanH` (To restrict every element of the output to be in the range [-1,1])            #
    ###########################################################################################
    
    layers = [
        tf.keras.Input((noise_dim,)),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu, use_bias=True),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu, use_bias=True),
        tf.keras.layers.Dense(784, activation=tf.nn.tanh, use_bias=True)
    ]
    
    model = tf.keras.Sequential(layers)
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return model

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.
    
    Inputs:
    - logits_real: Tensor of shape (N, 1) giving scores for the real data.
    - logits_fake: Tensor of shape (N, 1) giving scores for the fake data.
    
    Returns:
    - loss: Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    loss = cross_entropy(y_true=tf.ones(logits_real.shape), y_pred=logits_real)
    loss += cross_entropy(y_true=tf.zeros(logits_fake.shape), y_pred=logits_fake)
    # Optional way to compute loss on logits_fake:
    # loss = cross_entropy(y_true=tf.ones(logits_real.shape), y_pred=logits_real)
    # loss += cross_entropy(y_true=tf.ones(logits_fake.shape), y_pred=-logits_fake)
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Optional ways to compute loss (G for generator and D for discriminator below):
    #
    # 1. Transfer output of G back to scores, then compute loss according to definition:
    # scores = tf.math.exp(logits_fake) / (tf.math.exp(logits_fake) + 1.0)
    # loss = - tf.math.reduce_mean(tf.math.log(scores))
    #
    # 2. Regard output of G as logits of scores (actually it is ! check out the notebook cell for details),
    #    then compute loss based on output of G
    #    NOTE: this method seems to achieve the highest precision
    # loss = - tf.math.reduce_mean(logits_fake) + tf.math.reduce_mean(tf.math.log(1.0 + tf.math.exp(logits_fake)))
    #
    # 3. Use the packed function tf.keras.losses.BinaryCrossentropy(from_logits=True)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss = cross_entropy(y_true=tf.ones(logits_fake.shape), y_pred=logits_fake)
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss


def get_solvers(learning_rate=1e-3, beta1=0.5):
    """Create solvers for GAN training.
    
    Inputs:
    - learning_rate: learning rate to use for both solvers
    - beta1: beta1 parameter for both solvers (first moment decay)
    
    Returns:
    - D_solver: instance of tf.optimizers.Adam with correct learning_rate and beta1
    - G_solver: instance of tf.optimizers.Adam with correct learning_rate and beta1
    """
    # TODO: create an AdamOptimizer for D_solver and G_solver
    D_solver = None
    G_solver = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    D_solver = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1, name='D_adam')
    G_solver = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1, name='G_adam')
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return D_solver, G_solver

def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: Tensor of shape (N, 1) giving scores for the real data.
    - scores_fake: Tensor of shape (N, 1) giving scores for the fake data.
    
    Outputs:
    - loss: A Tensor containing the loss.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    N, _ = scores_real.shape
    
    loss = tf.nn.l2_loss(scores_real - 1) / N
    loss += tf.nn.l2_loss(scores_fake) / N
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: Tensor of shape (N, 1) giving scores for the fake data.
    
    Outputs:
    - loss: A Tensor containing the loss.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    N, _ = scores_fake.shape
    
    loss = tf.nn.l2_loss(scores_fake - 1) / N
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

def dc_discriminator():
    """Compute discriminator score for a batch of input images.
    
    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
    
    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """
    model = None
    ##############################################################################
    # TODO: Implement architecture                                               #
    #                                                                            #
    # HINT: tf.keras.models.Sequential might be helpful.                         #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ##############################################################################
    # GIVEN ARCHITECTURE:                                                        #
    #                                                                            #
    # - Conv2D: 32 Filters, 5x5, Stride 1, padding 0                             #
    # - Leaky ReLU(alpha=0.01)                                                   #
    # - Max Pool 2x2, Stride 2                                                   #
    # - Conv2D: 64 Filters, 5x5, Stride 1, padding 0                             #
    # - Leaky ReLU(alpha=0.01)                                                   #
    # - Max Pool 2x2, Stride 2                                                   #
    # - Flatten                                                                  #
    # - Fully Connected with output size 4 x 4 x 64                              #
    # - Leaky ReLU(alpha=0.01)                                                   #
    # - Fully Connected with output size 1                                       #
    ##############################################################################
    
    layers = [
        tf.keras.Input((784,)),
        tf.keras.layers.Reshape((28, 28, 1)),
        
        tf.keras.layers.Conv2D(32, kernel_size=5, strides=1, padding='valid', activation=lambda x: leaky_relu(x, 0.01), use_bias=True),
        tf.keras.layers.MaxPool2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(64, kernel_size=5, strides=1, padding='valid', activation=lambda x: leaky_relu(x, 0.01), use_bias=True),
        tf.keras.layers.MaxPool2D((2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4*4*64, activation=lambda x: leaky_relu(x, 0.01), use_bias=True),
        tf.keras.layers.Dense(1, use_bias=True)
    ]
    
    model = tf.keras.Sequential(layers)
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return model


def dc_generator(noise_dim=NOISE_DIM):
    """Generate images from a random noise vector.
    
    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
    
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    model = tf.keras.models.Sequential()
    # TODO: implement architecture
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    ############################################################################
    # GIVEN ARCHITECTURE:                                                      #
    #                                                                          #
    # - Fully connected with output size 1024                                  #
    # - `ReLU`                                                                 #
    # - BatchNorm                                                              #
    # - Fully connected with output size 7 x 7 x 128                           #
    # - `ReLU`                                                                 #
    # - BatchNorm                                                              #
    # - Resize into Image Tensor of size 7, 7, 128                             #
    # - Conv2D^T (transpose): 64 filters of 4x4, stride 2                      #
    # - `ReLU`                                                                 #
    # - BatchNorm                                                              #
    # - Conv2d^T (transpose): 1 filter of 4x4, stride 2                        #
    # - `TanH`                                                                 #
    ############################################################################
    
    layers = [
        tf.keras.Input((noise_dim,)),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu, use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(7*7*128, activation=tf.nn.relu, use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Reshape((7, 7, 128)),
        tf.keras.layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation=tf.nn.tanh),
    ]
    
    model = tf.keras.Sequential(layers)
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return model


# a giant helper function
def run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss,\
              show_every=250, print_every=20, batch_size=128, num_epochs=10, noise_size=96):
    """Train a GAN for a certain number of epochs.
    
    Inputs:
    - D: Discriminator model
    - G: Generator model
    - D_solver: an Optimizer for Discriminator
    - G_solver: an Optimizer for Generator
    - generator_loss: Generator loss
    - discriminator_loss: Discriminator loss
    Returns:
        Nothing
    """
    mnist = MNIST(batch_size=batch_size, shuffle=True)
    
    iter_count = 0
    images = []
    for epoch in range(num_epochs):
        for (x, _) in mnist:
            with tf.GradientTape() as tape:
                real_data = x
                logits_real = D(preprocess_img(real_data))

                g_fake_seed = sample_noise(batch_size, noise_size)
                # fake_images = G(g_fake_seed)
                fake_images = G(g_fake_seed, training=True)                   # should I add `training=True` for DCGAN ?
                logits_fake = D(tf.reshape(fake_images, [batch_size, 784]))

                d_total_error = discriminator_loss(logits_real, logits_fake)
                d_gradients = tape.gradient(d_total_error, D.trainable_variables)      
                D_solver.apply_gradients(zip(d_gradients, D.trainable_variables))
            
            with tf.GradientTape() as tape:
                g_fake_seed = sample_noise(batch_size, noise_size)
                # fake_images = G(g_fake_seed)
                fake_images = G(g_fake_seed, training=True)                  # same question as above

                gen_logits_fake = D(tf.reshape(fake_images, [batch_size, 784]))
                g_error = generator_loss(gen_logits_fake)
                g_gradients = tape.gradient(g_error, G.trainable_variables)      
                G_solver.apply_gradients(zip(g_gradients, G.trainable_variables))

            if (iter_count % show_every == 0):
                print('Epoch: {}, Iter: {}, D: {:.4}, G:{:.4}'.format(epoch, iter_count,d_total_error,g_error))
                imgs_numpy = fake_images.cpu().numpy()
                images.append(imgs_numpy[0:16])
                
            iter_count += 1
    
    # random noise fed into our generator
    z = sample_noise(batch_size, noise_size)
    # generated images
    G_sample = G(z)
    
    return images, G_sample[:16]

class MNIST(object):
    def __init__(self, batch_size, shuffle=False):
        """
        Construct an iterator object over the MNIST data
        
        Inputs:
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        train, _ = tf.keras.datasets.mnist.load_data()
        X, y = train
        X = X.astype(np.float32)/255
        X = X.reshape((X.shape[0], -1))
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B)) 

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def count_params(model):
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(p.shape) for p in model.weights])
    return param_count