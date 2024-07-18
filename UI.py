import streamlit as st
import tempfile
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

DEMO_VIDEO = 'file for normal\car1.mp4'

st.title('The One App for Dtr')

st.markdown(
    """

    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;

    }
    </style>

    
    """,
    unsafe_allow_html=True,

)

st.sidebar.title('Implemented Models')
st.sidebar.subheader('parameters')

@st.cache_resource()
def image_resize(image, width=None, height=None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    
    if width is None:
        r = width / float(h)
        dim = (int(w * r),height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    return resized

app_mode = st.sidebar.selectbox('Choose the App model',
['Run on Video(Object Detection)'])


if app_mode == 'Run on Video(Object Detection)':
    
    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")

    if record:
        st.checkbox("Recording", value=True)

    st.markdown(
        """

        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 350px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 350px;
            margin-left: -350px;

        }
        </style>

        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Output")

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    ##We get out input video here
    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture("file for normal\car1.mp4")
            tfflie.name = DEMO_VIDEO
        
    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    #Recording PArt
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input Video')

    st.sidebar.video(tfflie.name)

    fps = 0
    i = 0
    deft = 0

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown("0")
    
    with kpi2:
        st.markdown("**Default**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    model = YOLO("YOLO weights/yolov8n.pt")

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane",
                  "bus", "train", "truck", "boat", "traffic light", 
                  "fire hydrant", "stop sign", "parking meter", "bench", 
                  "bird", "cat", "dog", "horse", "sheep", "cow", 
                  "elephant", "bear", "zebra", "giraffe", "backpack", 
                  "umbrella", "handbag", "tie", "suitcase", "frisbee", 
                  "skis", "snowboard", "sports ball", "kite", 
                  "baseball bat", "baseball glove", "skateboard", 
                  "surfboard", "tennis racket", "bottle", "wine glass", 
                  "cup", "fork", "knife", "spoon", "bowl", "banana", 
                  "apple", "sandwich", "orange", "broccoli", "carrot", 
                  "hot dog", "pizza", "donut", "cake", "chair", "sofa", 
                  "potted plant", "bed", "dining table", "toilet", 
                  "tv monitor", "laptop", "mouse", "remote", "keyboard", 
                  "cell phone", "microwave", "oven", "toaster", "sink", 
                  "refrigerator", "book", "clock", "vase", "scissors", 
                  "teddy bear", "hair drier", "toothbrush"
                  ]

    prevTime = 0
    while vid.isOpened():
        i +=1
        success, frame = vid.read()
        results = model(frame, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:

                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                w, h = x2-x1, y2-y1
                cv2.rectangle(frame,(x1, y1, w, h), (255, 0, 0))
                # Confidence
                conf = math.ceil((box.conf[0]*100))/100
                print(conf)
                # Class Name
                cls = int(box.cls[0])

                cvzone.putTextRect(frame, f'{classNames[cls]}' f'{conf}', (max(0, x1),max(35, y1)), scale=1, thickness=2)

        # Display the processed frame
        stframe.image(frame, channels='RGB', use_column_width=True)

        # FPS Calculation
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        if record:
            out.write(frame)

        kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
        kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{deft}</h1>", unsafe_allow_html=True)
        kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)


        #Dashboard
        frame = cv2.resize(frame, (0,0), fx = 0.8, fy = 0.8)
        frame = image_resize(image = frame, width = 720, height=1280)
        stframe.image(frame,channels = 'BGR',use_column_width=True)

    st.text('Video Processed')
    output_video = open('output1.mp4','rb')
    out_bytes = output_video.read()
    st.video(out_bytes)
