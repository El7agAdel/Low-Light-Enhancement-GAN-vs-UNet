import tensorflow as tf
import random
import cv2
import os 
from matplotlib import pyplot as plt


class Unet:
   def __init__(self, in_path = "./", out_path = "./", num_epochs = 150, batch_size = 64, img_Width = 100, img_Height = 100, img_Depth = 1):
        # The batch size we'll use for training
        tf.test.gpu_device_name()

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.steps_per_epoch = (4250)//batch_size
        
        self.img_Width = img_Width
        self.img_Height = img_Height
        self.img_Depth = img_Depth

        # Size of the image required to train our model
        
        self.In_Imgs_Train_Path = in_path
        self.Out_Imgs_Train_Path = out_path

        self.In_Imgs = sorted(os.listdir(self.In_Imgs_Path))
        self.Out_Imgs = sorted(os.listdir(self.Out_Imgs_Path))


def Review_Training_data(self,):
    image_number = random.randint(0, len(self.In_Imgs))
    test_In_Imgs = cv2.imread(self.In_Imgs_Path + self.In_Imgs[image_number])
    test_Out_Imgs = cv2.imread(self.Out_Imgs_Path + self.Out_Imgs[image_number])

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(test_In_Imgs)
    plt.subplot(122)
    plt.imshow(test_Out_Imgs, cmap='gray')
    plt.show()



def conv_block(self,input, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(input)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x



def encoder_block(self,input, num_filters):
    x = self.conv_block(input, num_filters)
    p = tf.keras.layers.MaxPooling2D((2, 2))(x)
    return x, p  



def decoder_block(self,input, skip_features, num_filters):
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = self.conv_block(x, num_filters)
    return x



def U2NET(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS=0):
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    s1, p1 = self.encoder_block(inputs, 64)
    s2, p2 = self.encoder_block(p1, 128)
    s3, p3 = self.encoder_block(p2, 256)
    s4, p4 = self.encoder_block(p3, 512)

    b1 = self.conv_block(p4, 1024) #Bridge

    d1 = self.decoder_block(b1, s4, 512)
    d2 = self.decoder_block(d1, s3, 256)
    d3 = self.decoder_block(d2, s2, 128)
    d4 = self.decoder_block(d3, s1, 64)

    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)  #Binary (can be multiclass)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary
    return model



def my_image_mask_generator(self, image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)


def Run_Training(self,model, model_name):
        history = model.fit(my_generator, validation_data=validation_datagen, 
                            steps_per_epoch=self.steps_per_epoch, 
                            validation_steps=self.steps_per_epoch, epochs=500)

        model.save(f'./{model_name}.h5')




# seed=24
# batch_size= 64
# img_data_gen_args = dict(
#                      rescale=1/255,
#                      rotation_range=90,
#                      width_shift_range=0.3,
#                      height_shift_range=0.3,
#                      shear_range=0.5,
#                      zoom_range=0.3,
#                      horizontal_flip=True,
#                      vertical_flip=True,
#                      fill_mode='reflect')

# mask_data_gen_args = dict(
#                      rescale=1/255,
#                      rotation_range=90,
#                      width_shift_range=0.3,
#                      height_shift_range=0.3,
#                      shear_range=0.5,
#                      zoom_range=0.3,
#                      horizontal_flip=True,
#                      vertical_flip=True,
#                      fill_mode='reflect') 



# image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(**img_data_gen_args)
# mask_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(**mask_data_gen_args)



# image_generator = image_data_generator.flow_from_directory(
#     directory=image_directory,
#     target_size=(256, 256),
#     batch_size=batch_size,
#     class_mode=None,
#     seed=seed,
# )
# valid_img_generator = image_data_generator.flow_from_directory(
#     directory=image_test_directory,
#     target_size=(256, 256),
#     batch_size=batch_size,
#     class_mode=None,
#     seed=seed,
# )


# mask_generator = mask_data_generator.flow_from_directory(
#     directory=mask_directory,
#     target_size=(256, 256),
#     color_mode="grayscale",
#     batch_size=batch_size,
#     class_mode=None,
#     seed=seed,
# )
# valid_mask_generator = mask_data_generator.flow_from_directory(
#     directory=mask_test_directory,
#     target_size=(256, 256),
#     color_mode="grayscale",
#     batch_size=batch_size,
#     class_mode=None,
#     seed=seed,
# )

# my_generator = my_image_mask_generator(image_generator, mask_generator)
# validation_datagen = my_image_mask_generator(valid_img_generator, valid_mask_generator)


# x = image_generator.next()
# y = mask_generator.next()
# for i in range(0,1):
#     image = x[i]
#     mask = y[i]
#     plt.subplot(1,2,1)
#     plt.imshow(image)
#     plt.subplot(1,2,2)
#     plt.imshow(mask[:,:,0] , cmap='gray')
#     plt.show()



    

    