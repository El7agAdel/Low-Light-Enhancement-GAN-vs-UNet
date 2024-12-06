import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

class GAN:

    def __init__(self, in_path = "./", out_path = "./", num_epochs = 150, batch_size = 64, img_Width = 100, img_Height = 100, img_Depth = 1):
        # The batch size we'll use for training
        tf.test.gpu_device_name()

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.img_Width = img_Width
        self.img_Height = img_Height
        self.img_Depth = img_Depth

        # Size of the image required to train our model
        
        self.In_Imgs_Path = sorted(os.listdir(in_path))
        self.Out_Imgs_Path = sorted(os.listdir(out_path))
        self.x = []
        self.y = []


        self.MSE = tf.keras.losses.MeanSquaredError()
        self.Generator_Optimizer = tf.keras.optimizers.Adam( 0.0005 )
        self.Discriminator_Optimizer = tf.keras.optimizers.Adam( 0.0005 )
        self.Cross_Entropy = tf.keras.losses.BinaryCrossentropy()

        self.Generator_Model = self.Build_Generator(self.img_Width, self.img_Height, self.img_Depth)
        self.Discriminator_Model = self.Build_Discriminator(self.img_Width, self.img_Height, self.img_Depth)
        



    def Load_Dataset(self, ):
            
        for x_img, y_img in zip(self.In_Imgs_Path, self.Out_Imgs_Path):


            img_x = cv2.imread(x_img)
            img_y = cv2.imread(y_img)
            x_img_resized = cv2.resize(img_x, (self.img_Width, self.img_Height))
            y_img_resized = cv2.resize(img_y, (self.img_Width, self.img_Height))

            self.x.append( x_img_resized )
            self.y.append( y_img_resized )

        train_x, _, train_y, _ = train_test_split( np.array(self.x) , np.array(self.y) , test_size=0.1 )

        # Construct tf.data.Dataset object
        self.dataset = tf.data.Dataset.from_tensor_slices( ( train_x , train_y ) )
        self.dataset = self.dataset.batch( self.batch_size )



    def Gen_block(self, input, num_filters, kernel_size):
        Layer = tf.keras.layers.Conv2D( num_filters , kernel_size= kernel_size , strides=1 )( input )
        Layer = tf.keras.layers.LeakyReLU()( Layer )
        return Layer

    def Dis_block(self, num_filters, kernel_size, img_Width = None, img_Height = None, img_Depth = None):
        Layers = [
        tf.keras.layers.Conv2D( num_filters , kernel_size=kernel_size , strides=1 , activation='relu' , input_shape=( img_Width, img_Height, img_Depth ) ),
        tf.keras.layers.Conv2D( num_filters , kernel_size=kernel_size , strides=1, activation='relu'  ),
        tf.keras.layers.MaxPooling2D(),
        ]
        
        return Layers

    def Dis_Dense(self, ):

        Dense_Layer = [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense( 512, activation='relu'  )  ,
        tf.keras.layers.Dense( 128 , activation='relu' ) ,
        tf.keras.layers.Dense( 16 , activation='relu' ) ,
        tf.keras.layers.Dense( 1 , activation='sigmoid' ) 
        ]

        return Dense_Layer

    def Build_Generator(self, img_Width, img_Height, img_Depth):

        inputs = tf.keras.layers.Input( shape=( img_Width , img_Height , img_Depth ) )

        Layer1 = self.Gen_block(input = inputs, num_filters = 16 , kernel_size=( 5 , 5 ))
        Layer1 = self.Gen_block(input = Layer1, num_filters = 32 , kernel_size=( 3 , 3 ))
        Layer1 = self.Gen_block(input = Layer1, num_filters = 32 , kernel_size=( 3 , 3 ))

        Layer2 = self.Gen_block(input = Layer1, num_filters = 32 , kernel_size=( 5 , 5 ))
        Layer2 = self.Gen_block(input = Layer2, num_filters = 64 , kernel_size=( 3 , 3 ))
        Layer2 = self.Gen_block(input = Layer2, num_filters = 64 , kernel_size=( 3 , 3 ))

        Layer3 = self.Gen_block(input = Layer2, num_filters = 64 , kernel_size=( 5 , 5 ))
        Layer3 = self.Gen_block(input = Layer3, num_filters = 128 , kernel_size=( 3 , 3 ))
        Layer3 = self.Gen_block(input = Layer3, num_filters = 128 , kernel_size=( 3 , 3 ))


        bottleneck = tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='tanh' , padding='same' )( Layer3 )

        concat1 = tf.keras.layers.Concatenate()( [ bottleneck , Layer3 ] )
        Layer_Up3 = tf.keras.layers.Conv2DTranspose( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' )( concat1 )
        Layer_Up3 = tf.keras.layers.Conv2DTranspose( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' )( Layer_Up3 )
        Layer_Up3 = tf.keras.layers.Conv2DTranspose( 64 , kernel_size=( 5 , 5 ) , strides=1 , activation='relu' )( Layer_Up3 )

        concat2 = tf.keras.layers.Concatenate()( [ Layer_Up3 , Layer2 ] )
        Layer_Up2 = tf.keras.layers.Conv2DTranspose( 64 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' )( concat2 )
        Layer_Up2 = tf.keras.layers.Conv2DTranspose( 64 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' )( Layer_Up2 )
        Layer_Up2 = tf.keras.layers.Conv2DTranspose( 32 , kernel_size=( 5 , 5 ) , strides=1 , activation='relu' )( Layer_Up2 )

        concat3 = tf.keras.layers.Concatenate()( [ Layer_Up2 , Layer1 ] )
        Layer_Up1 = tf.keras.layers.Conv2DTranspose( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu')( concat3 )
        Layer_Up1 = tf.keras.layers.Conv2DTranspose( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu')( Layer_Up1 )
        Layer_Up1 = tf.keras.layers.Conv2DTranspose( 3 , kernel_size=( 5 , 5 ) , strides=1 , activation='relu')( Layer_Up1 )

        model = tf.keras.models.Model( inputs , Layer_Up1 )
        model.summary
        return model

    def Build_Discriminator(self, img_Width, img_Height, img_Depth):
        

        Layer1 = self.Dis_block(32, ( 7 , 7 ), img_Width, img_Height, img_Depth) 

        Layer2 = self.Dis_block(64, ( 5 , 5 )) 

        Layer3 = self.Dis_block(128, ( 3 , 3 )) 

        Layer4 = self.Dis_block(256, ( 3 , 3 )) 

        Dense_Layer = self.Dis_Dense()

        layers = Layer1 + Layer2 + Layer3 + Layer4 +Dense_Layer

        model = tf.keras.models.Sequential( layers )
        model.summary
        return model

    def Discriminator_Loss(self, real_output, fake_output):
        
        real_loss = self.Cross_Entropy(tf.ones_like(real_output) - tf.random.uniform( shape=real_output.shape , maxval=0.1 ) , real_output)
        fake_loss = self.Cross_Entropy(tf.zeros_like(fake_output) + tf.random.uniform( shape=fake_output.shape , maxval=0.1  ) , fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def Generator_Loss(self, fake_output , real_y):
        real_y = tf.cast( real_y , 'float32' )
        return self.MSE( fake_output , real_y )

    @tf.function
    def Train_Step(self, input_x , real_y ):
    
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate an image -> G( x )
            generated_images = self.Generator_Model( input_x , training=True)
            # Probability that the given image is real -> D( x )
            real_output = self.Discriminator_Model( real_y, training=True)
            # Probability that the given image is the one generated -> D( G( x ) )
            generated_output = self.Discriminator_Model(generated_images, training=True)
            
            # L2 Loss -> || y - G(x) ||^2
            gen_loss = self.Generator_Loss( generated_images , real_y )
            # Log loss for the Discriminator_Model
            disc_loss = self.Discriminator_Loss( real_output, generated_output )
        
        #tf.keras.backend.print_tensor( tf.keras.backend.mean( gen_loss ) )
        #tf.keras.backend.print_tensor( gen_loss + disc_loss )

        # Compute the gradients
        gradients_of_generator = gen_tape.gradient(gen_loss, self.Generator_Model.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.Discriminator_Model.trainable_variables)

        # Optimize with Adam
        self.Generator_Optimizer.apply_gradients(zip(gradients_of_generator, self.Generator_Model.trainable_variables))
        self.Discriminator_Optimizer.apply_gradients(zip(gradients_of_discriminator, self.Discriminator_Model.trainable_variables))



    def Run_Training(self, ):
        self.Load_Dataset()
        for epoch in range( self.num_epochs ):
            print( epoch )
            for ( x , y ) in self.dataset:
                # Here ( x , y ) represents a batch from our training dataset.
                print( x.shape )
                self.Train_Step( x , y )


tt = GAN( in_path = "./", out_path = "./", num_epochs = 150, batch_size = 64, img_Width = 100, img_Height = 100, img_Depth = 1)