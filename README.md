# Military Soldier Safety and Weapon Detection using YOLO and Computer Vision

**Team:** AKSHAYA S V, AISHWARIYA A, MIDHUN M P, SURABI P S

---

## 📋 Project Description

This repository implements a real-time detection system designed to enhance military soldier safety by identifying soldiers, weapons, vehicles, and other relevant objects in live video feeds or images. Leveraging the power of YOLO (You Only Look Once) object detection and OpenCV, this solution provides on-the-fly inference suitable for deployment in edge devices or as part of a desktop/web application.

Key features:

* 🔍 **Multi-class Detection:** Identifies soldiers, weapons, military vehicles, civilians, and more.
* 🚀 **Real-time Inference:** Processes webcam or video streams at interactive frame rates.
* 📊 **Streamlit Interface:** Offers a simple web UI for uploading media, adjusting confidence thresholds, and visualizing results.
* ⚙️ **Easy Integration:** Modular code for training custom models, running batch inference, or embedding in larger systems.

## 🛠️ Requirements

The primary dependencies are listed in `requirements.txt`. To install, run:

```bash
pip install -r requirements.txt
```

**requirements.txt**:

```text
torch>=1.12.0
torchvision>=0.13.0
opencv-python
streamlit
pyyaml
numpy
matplotlib
```

## 📂 Repository Structure

```
├── data/
│   ├── images/          # Train/val/test image folders
│   └── labels/          # Corresponding YOLO-format label files
├── runs/                # Training & inference outputs
├── train.py             # Script to train the YOLO model
├── detect.py            # Inference script for images, video, or webcam
├── app.py               # Streamlit-based UI for uploads & visualization
├── data.yaml            # Dataset configuration (paths & class names)
├── requirements.txt     # Python package dependencies
└── README.md            # Project overview and usage
```
## 📈 Dataset
https://drive.google.com/file/d/1COKHeY4TYfcz-QjBx2qbg5p5h0eVPLPH/view

## ⚙️ How It Works

1. **Data Preparation:**

   * Place your images under `data/images/{train,val,test}`.
   * Store corresponding YOLO-format `.txt` labels in `data/labels/{train,val,test}`.
2. **Configuration:**

   * Edit `data.yaml` to point to your train/val/test directories and define `nc` (number of classes) and `names` (class labels).
3. **Training:**

   * Execute `python train.py` to start training on YOLOv5 (or v8) with your dataset. Outputs (checkpoints, logs) are saved under `runs/train/military_yolo/`.
4. **Inference:**

   * For images/videos:

     ```bash
     python detect.py --source path/to/image_or_video --weights runs/train/military_yolo/weights/best.pt --conf 0.25
     ```
   * For webcam:

     ```bash
     python detect.py --source 0 --weights runs/train/military_yolo/weights/best.pt --conf 0.25
     ```
5. **Streamlit App (Optional):**

   * Launch the web interface:

     ```bash
     streamlit run app.py
     ```
   * Use the sidebar to upload media, adjust the confidence threshold, and view annotated results in real time.

## ▶️ Typical Usage

```bash
# Clone repository
git clone https://github.com/<your-username>/military-safety-yolo.git
cd military-safety-yolo

# Setup virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train your model
python train.py

# Run detection on an image\python detect.py --source data/images/test/sample.jpg

# Launch Streamlit UI
streamlit run app.py
```

## 📈 Results & Output

* Trained model checkpoints and logs: `runs/train/military_yolo/`
* Inference outputs (annotated images/videos) are displayed live or saved as specified in the scripts.

## 🤝 Contributing

Feel free to open issues or submit pull requests for enhancements, bug fixes, or additional features.

---

*Prepared by Team AKSHAYA S V, AISHWARIYA A, MIDHUN M P, SURABI P S*
