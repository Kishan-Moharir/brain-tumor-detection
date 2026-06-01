import os
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model, load_model

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection using CNN",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------
# ADVANCED CSS UI
# ---------------------------------
st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(to right, #020617, #0f172a);
    color: white;
}

/* Hide Streamlit Menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main Title */
.main-title {
    font-size: 65px;
    font-weight: 800;
    text-align: center;
    color: #00E5FF;
    margin-top: 20px;
    text-shadow: 0px 0px 25px rgba(0,229,255,0.9);
}

/* Subtitle */
.sub-title {
    text-align: center;
    color: #cbd5e1;
    font-size: 22px;
    margin-bottom: 40px;
}

/* Upload Box */
.upload-box {
    background: rgba(255,255,255,0.05);
    border: 2px dashed #00E5FF;
    border-radius: 25px;
    padding: 30px;
    box-shadow: 0px 0px 20px rgba(0,229,255,0.2);
}

/* Cards */
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    box-shadow: 0px 0px 15px rgba(255,255,255,0.1);
}

/* Result Cards */
.success-card {
    background: linear-gradient(to right, #16a34a, #22c55e);
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: white;
    box-shadow: 0px 0px 20px rgba(34,197,94,0.5);
}

.error-card {
    background: linear-gradient(to right, #dc2626, #ef4444);
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: white;
    box-shadow: 0px 0px 20px rgba(239,68,68,0.5);
}

.warning-card {
    background: linear-gradient(to right, #f59e0b, #facc15);
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: black;
}

/* Confidence */
.confidence {
    margin-top: 15px;
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    color: #00E5FF;
}

/* Divider */
.divider {
    height: 2px;
    background: linear-gradient(to right, transparent, #00E5FF, transparent);
    margin-top: 30px;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# HERO SECTION
# ---------------------------------
st.markdown(
    '<div class="main-title">🧠 Brain Tumor Detection using CNN</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI Powered MRI Classification System using Deep Learning & Explainable AI</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ---------------------------------
# TOP CARDS
# ---------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="card">
    <h2>⚡ CNN Model</h2>
    <p>MobileNetV2</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
    <h2>🧠 Classes</h2>
    <p>5 Categories</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
    <h2>📊 Accuracy</h2>
    <p>Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card">
    <h2>🔥 Explainable AI</h2>
    <p>Grad-CAM</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------
# CONFIG
# ---------------------------------
IMG_SIZE = 224
DATASET_PATH = "dataset"
MODEL_PATH = "model.h5"

# ---------------------------------
# LOAD DATA
# ---------------------------------
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

class_names = list(train_data.class_indices.keys())

# ---------------------------------
# MODEL
# ---------------------------------
def build_model():

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)

    output = Dense(len(class_names), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ---------------------------------
# LOAD OR TRAIN MODEL
# ---------------------------------
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = build_model()
    model.fit(train_data, validation_data=val_data, epochs=5)
    model.save(MODEL_PATH)

# ---------------------------------
# GRAD-CAM
# ---------------------------------
def get_gradcam(img_array, model):

    last_conv_layer = None

    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            last_conv_layer = layer
            break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        class_idx = np.argmax(predictions[0])

        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap

# ---------------------------------
# UPLOAD SECTION
# ---------------------------------
st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Upload Brain MRI Image",
    type=["jpg","png","jpeg"]
)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------
# PREDICTION
# ---------------------------------
if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    img = cv2.imdecode(file_bytes, 1)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼 Uploaded MRI Scan")
        st.image(img, use_column_width=True)

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    prediction = model.predict(img_array)[0]

    class_index = np.argmax(prediction)

    class_name = class_names[class_index]

    confidence = np.max(prediction) * 100

    with col2:

        st.subheader("📊 Prediction Result")

        if class_name == "notbrain":

            st.markdown(
                """
                <div class="warning-card">
                ❌ Invalid Image <br>
                Please Upload Brain MRI
                </div>
                """,
                unsafe_allow_html=True
            )

            st.stop()

        elif class_name == "notumor":

            st.markdown(
                """
                <div class="success-card">
                ✅ No Tumor Detected
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="error-card">
                ⚠️ Brain Tumor Detected <br>
                Type: {class_name.capitalize()}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            f"""
            <div class="confidence">
            Confidence Score: {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------------------------
    # GRAD CAM
    # ---------------------------------
    heatmap = get_gradcam(img_array, model)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.subheader("🔥 Explainable AI - Tumor Highlight")

    st.image(
        superimposed_img.astype(np.uint8),
        use_column_width=True
    )

# ---------------------------------
# HOW IT WORKS
# ---------------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.subheader("⚙️ How The System Works")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="card">
    <h3>1️⃣ Upload</h3>
    <p>Upload MRI Scan</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
    <h3>2️⃣ Preprocess</h3>
    <p>Resize & Normalize</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
    <h3>3️⃣ CNN Analysis</h3>
    <p>Feature Extraction</p>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="card">
    <h3>4️⃣ Prediction</h3>
    <p>Tumor Classification</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown("""
<center>

### 🧠 Brain Tumor Detection using CNN

Developed using TensorFlow, Streamlit, MobileNetV2 & Explainable AI

</center>
""", unsafe_allow_html=True)