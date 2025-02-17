import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
img_size = 64
import numpy as np
import time

# Cargar el modelo guardado en formato HDF5
model = tf.keras.models.load_model('malaria-cell-cnn-32_32_1.h5', compile=False)


# Imprimir el resumen de la arquitectura del modelo
#model.summary()

# testing some images
infected_cell = "../Malaria/testsamples/parasitized/p2.png"
#infected_cell = "../Malaria/testsamples/parasitized/p1.png"
uninfected_cell = "../Malaria/testsamples/uninfected/u1.png"

_, ax = plt.subplots(1, 2)
ax[0].imshow(plt.imread(uninfected_cell))
ax[0].title.set_text("Uninfected Cell")
ax[1].imshow(plt.imread(infected_cell))
ax[1].title.set_text("Parasitized Cell")
plt.show()

img_arr_uninfected = cv2.imread(uninfected_cell, cv2.IMREAD_GRAYSCALE)
img_arr_infected = cv2.imread(infected_cell, cv2.IMREAD_GRAYSCALE)
# resize the images to (70x70)
img_arr_uninfected = cv2.resize(img_arr_uninfected, (img_size, img_size))
img_arr_infected = cv2.resize(img_arr_infected, (img_size, img_size))
# scale to [0, 1]
img_arr_infected = img_arr_infected / 255
img_arr_uninfected = img_arr_uninfected / 255
# reshape to fit the neural network dimensions
# (changing shape from (70, 70) to (1, 70, 70, 1))
img_arr_infected = img_arr_infected.reshape(1, *img_arr_infected.shape)
img_arr_infected = np.expand_dims(img_arr_infected, axis=3)
img_arr_uninfected = img_arr_uninfected.reshape(1, *img_arr_uninfected.shape)
img_arr_uninfected = np.expand_dims(img_arr_uninfected, axis=3)

# perform inference
start_time_i = time.time()
infected_result = model.predict(img_arr_infected)[0][0]
end_time_i = time.time()
elapsed_time_i = end_time_i - start_time_i
start_time_u = time.time()
uninfected_result = model.predict(img_arr_uninfected)[0][0]
end_time_u = time.time()
elapsed_time_u = end_time_u - start_time_u
print(f"Infected: {infected_result}")
print(f"Uninfected: {uninfected_result}")

print("Elapsed time Uninfected: %s" %(elapsed_time_u))
print("Elapsed time Infected: %s" %(elapsed_time_i))
