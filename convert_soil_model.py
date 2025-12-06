import tensorflow as tf

# Step 1: Load the .h5 model
model = tf.keras.models.load_model("soil_model.h5")

# Step 2: Export as SavedModel for TFLite conversion
model.export("saved_model_format")

# Step 3: Convert from SavedModel directory
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_format")
tflite_model = converter.convert()

# Step 4: Save the .tflite model
with open("soil_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Successfully converted to TFLite!")
