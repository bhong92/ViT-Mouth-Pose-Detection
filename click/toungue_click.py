import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.optimizers import RMSprop

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

from pynput.mouse import Button, Controller
from threading import Thread
from time import sleep
from imutils import face_utils

import dlib
import sys, time, os
from sklearn.model_selection import train_test_split

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
class T_click():

    def __init__(self):
        self.num_tongue_frames = 500
        self.mouse = Controller()
        self.tongue_out = 0

        self.camstart = False
        self.frame = None

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #@zhiqi need to make sure this file is in the click directory

        self.num_classes = 4
        self.input_shape = (32, 32, 3)
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.batch_size = 256
        self.num_epochs = 25
        self.image_size = 72  # We'll resize input images to this size
        self.patch_size = 6  # Size of the patches to be extract from the input images
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection_dim = 64
        self.num_heads = 4
        self.transformer_units = [
            self.projection_dim * 2,
            self.projection_dim,
        ]  # Size of the transformer layers
        self.transformer_layers = 8
        self.mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

        self.data_augmentation = keras.Sequential(
            [
                layers.Normalization(),
                layers.Resizing(self.image_size, self.image_size),
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(factor=0.02),
                layers.RandomZoom(
                    height_factor=0.2, width_factor=0.2
                ),
            ],
            name="data_augmentation",
        )
        t1 = Thread(target=self.t_click_detect_continuously, args=())
        t1.start()
    
    def get_data(self, vid):
        sub_folder = time.strftime('%Y-%m-%d_%H%M%S')
        if not os.path.exists('./dataset'):
            os.mkdir('./dataset')
        os.mkdir(f'./dataset/{sub_folder}')
        print('DATA CAPTURING PHASE')
        print('stick the tip of your tongue out and scan all corners of the screen until the next message')
        sys.stdout.write('Ready?'+' ')
        for i in range(3,0,-1):
            sys.stdout.write(str(i)+'... ')
            sys.stdout.flush()
            time.sleep(1)
        print('\n')

        x_data = np.array([])

        for i in range(self.num_tongue_frames):
            results, img = vid.read()
            if not results:
                continue
            img_name = f"./dataset/{sub_folder}/tongue_{i}.png"
            cv2.imwrite(img_name, img)
            faces = self.detector(img, 1)

            for (f, face) in enumerate(faces):
            
                landmarks = self.predictor(img, face)
                landmarks = face_utils.shape_to_np(landmarks)

                # 51 middle of upper lip
                # 57 middle of lower lip
                # 48 outter left corner of the lip
                # 54 outter right corner of the lip
                
                mouth_list = [landmarks[34], landmarks[9], landmarks[48], landmarks[54]] # sides of mouth, nose and chin for top and bottom
                mouth_list = np.array(mouth_list)
                buffer = 0
                min_x = np.min(mouth_list[:,0]) - buffer
                min_y = np.min(mouth_list[:,1]) - buffer
                max_x = np.max(mouth_list[:,0]) + buffer + 1
                max_y = np.max(mouth_list[:,1]) + buffer + 1


                im = img[ min_y:max_y, min_x:max_x]
                im = cv2.resize(im, (32, 32))
                r = im[:,:,0] #Slicing to get R data
                g = im[:,:,1] #Slicing to get G data
                b = im[:,:,2] #Slicing to get B data
                
                if not len(x_data):
                    x_data = np.array([[r] + [g] + [b]],np.uint8)
                    y_data = np.array([[0]])
                else:
                    curr = np.array([[r] + [g] + [b]],np.uint8)
                    x_data = np.append(x_data, curr, 0)
                    y_data = np.append(y_data, np.array([[0]]))

                print(f"{self.num_tongue_frames - i}: Keep the tip of your tongue out")

        print('now relax your face to a normal resting position and scan the screen until the next message')
        sys.stdout.write('Ready?'+' ')
        for i in range(3,0,-1):
            sys.stdout.write(str(i)+'... ')
            sys.stdout.flush()
            time.sleep(1)
        print('\n')

        for i in range(self.num_tongue_frames):

            # convert JS response to OpenCV Image
            results, img = vid.read()
            if not results:
                continue
            img_name = f"./dataset/{sub_folder}/relaxed_{i}.png"
            cv2.imwrite(img_name, img)
            faces = self.detector(img, 1)

            for (f, face) in enumerate(faces):
            
                landmarks = self.predictor(img, face)
                landmarks = face_utils.shape_to_np(landmarks)

                # 51 middle of upper lip
                # 57 middle of lower lip
                # 48 outter left corner of the lip
                # 54 outter right corner of the lip
                
                mouth_list = [landmarks[34], landmarks[9], landmarks[48], landmarks[54]] # sides of mouth, nose and chin for top and bottom
                mouth_list = np.array(mouth_list)
                buffer = 0
                min_x = np.min(mouth_list[:,0]) - buffer
                min_y = np.min(mouth_list[:,1]) - buffer
                max_x = np.max(mouth_list[:,0]) + buffer + 1
                max_y = np.max(mouth_list[:,1]) + buffer + 1

                im = img[ min_y:max_y, min_x:max_x]
                im = cv2.resize(im, (32, 32))
                r = im[:,:,0] #Slicing to get R data
                g = im[:,:,1] #Slicing to get G data
                b = im[:,:,2] #Slicing to get B data

                curr = np.array([[r] + [g] + [b]],np.uint8)
                x_data = np.append(x_data, curr, 0)
                y_data = np.append(y_data, np.array([[1]]))

                print(f'{self.num_tongue_frames-i}: Keep your face relaxed')

        print('Now make various expressions e.g. mouth open, puckered lips...')
        sys.stdout.write('Ready?'+' ')
        for i in range(3,0,-1):
            sys.stdout.write(str(i)+'... ')
            sys.stdout.flush()
            time.sleep(1)
        print('\n')

        for i in range(self.num_tongue_frames):
            results, img = vid.read()
            if not results:
                continue
            img_name = f"./dataset/{sub_folder}/smile_{i}.png"
            cv2.imwrite(img_name, img)
            faces = self.detector(img, 1)

            for (f, face) in enumerate(faces):

                landmarks = self.predictor(img, face)
                landmarks = face_utils.shape_to_np(landmarks)

                # 51 middle of upper lip
                # 57 middle of lower lip
                # 48 outter left corner of the lip
                # 54 outter right corner of the lip
                
                mouth_list = [landmarks[34], landmarks[9], landmarks[48], landmarks[54]] # sides of mouth, nose and chin for top and bottom
                mouth_list = np.array(mouth_list)
                buffer = 0
                min_x = np.min(mouth_list[:,0]) - buffer
                min_y = np.min(mouth_list[:,1]) - buffer
                max_x = np.max(mouth_list[:,0]) + buffer + 1
                max_y = np.max(mouth_list[:,1]) + buffer + 1

                im = img[ min_y:max_y, min_x:max_x]
                im = cv2.resize(im, (32, 32))
                r = im[:,:,0] #Slicing to get R data
                g = im[:,:,1] #Slicing to get G data
                b = im[:,:,2] #Slicing to get B data

                curr = np.array([[r] + [g] + [b]],np.uint8)
                x_data = np.append(x_data, curr, 0)
                y_data = np.append(y_data, np.array([[2]]))
                
                print(f'{self.num_tongue_frames-i}: Please keep making faces')

        x_data = np.transpose(x_data, (0, 2, 3, 1))
        print(f'data collection complete! X: {x_data.shape} y: {y_data.shape}')
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
        return x_train, x_test, y_train, y_test
    
    def mlp(self, x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x
    def make_model(self):
        inputs = layers.Input(shape=self.input_shape)
        # Augment data.
        augmented = self.data_augmentation(inputs)
        # Create patches.
        patches = Patches(self.patch_size)(augmented)
        # Encode patches.
        encoded_patches = PatchEncoder(self.num_patches, self.projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(self.transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = self.mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        features = self.mlp(representation, hidden_units=self.mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(self.num_classes)(features)
        # Create the Keras model.
        model = keras.Model(inputs=inputs, outputs=logits)
        return model

    def detect(self, model, img):
        faces = self.detector(img, 1)
        for (f, face) in enumerate(faces):
            
            landmarks = self.predictor(img, face)
            landmarks = face_utils.shape_to_np(landmarks)
            
            mouth_list = [landmarks[34], landmarks[9], landmarks[48], landmarks[54]] # sides of mouth, nose and chin for top and bottom
            mouth_list = np.array(mouth_list)
            buffer = 0
            min_x = np.min(mouth_list[:,0]) - buffer
            min_y = np.min(mouth_list[:,1]) - buffer
            max_x = np.max(mouth_list[:,0]) + buffer + 1
            max_y = np.max(mouth_list[:,1]) + buffer + 1

            im = img[ min_y:max_y, min_x:max_x]
            im = cv2.resize(im, (32, 32))

            r = im[:,:,0] #Slicing to get R data
            g = im[:,:,1] #Slicing to get G data
            b = im[:,:,2] #Slicing to get B data

            curr = np.array([[r] + [g] + [b]],np.uint8)
            curr = np.transpose(curr, (0, 2, 3, 1))

            mouth = model.predict(curr)
            mouth_idx = np.argmax(mouth)
            return mouth_idx

    def t_click_train(self):
        vid = cv2.VideoCapture(0)
        (x_train, x_test, y_train, y_test) = self.get_data(vid)
        self.data_augmentation.layers[0].adapt(x_train)

        model = self.make_model()
        optimizer = tfa.optimizers.AdamW(learning_rate=self.learning_rate, weight_decay=self.weight_decay)

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
        )

        model.fit(
            x=x_train,
            y=y_train,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            validation_split=0.1,
        )
        _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
        model.save('click/saved_t_click_model/')
        vid.release()
        cv2.destroyAllWindows()
        return model
    
    def t_click_detect_continuously(self):
        # vid = cv2.VideoCapture(0)
        model = tf.keras.models.load_model('click/saved_t_click_model/')
        print("model", model)
        last_three_list = [0,0,0]
        thresh_H = 0.6
        thresh_L = 0.4

        clicked_recent = False

        while True:
            if not self.camstart:
                sleep(0.001)
                continue
            # print("----------running tongue detecting---------")
            t_out = self.detect(model, self.frame)
            print("current t_out:",t_out)
            last_three_list.pop(0)
            last_three_list.append(t_out)

            avg = float(sum(filter(None, last_three_list))) / float(len(last_three_list))

            if avg == 0 and not clicked_recent:
                print('####################### CLICK ACTIONED #######################')
                clicked_recent = True
                # self.mouse.click(Button.left, 2) #this double clicks the mouse
                self.mouse.press(Button.left) #this was for click and drag - need the commented code below as well for that
            else:
                print('-------------------------- RELEASED --------------------------')
                clicked_recent = False
                self.mouse.release(Button.left)




if __name__ == '__main__':
    # T_click().t_click_train()
    T_click().t_click_detect_continuously()
    