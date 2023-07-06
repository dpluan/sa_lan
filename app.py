# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper
import pafy
import yt_dlp as youtube_dl

# Setting page layout
st.set_page_config(
    page_title="Ứng dụng YOLOv8 để phát hiện các  tàu, sà lan chở hàng quá tải",
    page_icon="DPL",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("ỨNG DỤNG YOLOv8 ĐỂ PHÁT HIỆN CÁC TÀU, SÀ LAN CHỞ HÀNG QUÁ TẢI")

# Sidebar
st.sidebar.header("Chọn độ tin cậy của mô hình")

# Model Options
#model_type = st.sidebar.radio(
    #"Chọn công việc", ['Nhận dạng', 'Phân vùng'])

confidence = float(st.sidebar.slider(
    "Độ tin cậy", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation

model_type = 'Nhận dạng'
if model_type == 'Nhận dạng':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Phân vùng':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Không thể nạp mô hình. Hãy kiểm tra lại đường dẫn chứa tệp weight của mô hình: {model_path}")
    st.error(ex)

st.sidebar.header("Lựa chọn hình ảnh/ Video")
source_radio = st.sidebar.radio(
    "Chọn nguồn dữ liệu", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Chọn một hình ảnh ...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Hình ảnh trước khi nhận dạng",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Hình ảnh trước khi nhận dạng",
                         use_column_width=True)
        except Exception as ex:
            st.error("Có lỗi xảy ra khi mở hình ảnh.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Hình ảnh sau khi được nhận dạng',
                     use_column_width=True)
        else:
            if st.sidebar.button('Nhận dạng'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Hình ảnh sau khi được nhận dạng',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("Chưa có hình ảnh nào được nạp lên!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Vui lòng chọn đúng nguồn dữ liệu!")
