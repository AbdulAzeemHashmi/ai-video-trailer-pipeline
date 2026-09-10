<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&pause=1000&color=8E44AD&center=true&vCenter=true&width=800&lines=AI+Video+Trailer+Pipeline;AI+Powered+Video+Analysis+Pipeline;YOLO11+plus+BLIP+plus+PyTorch+plus+Librosa;From+Raw+Footage+to+a+Scored+Trailer" alt="Typing SVG" />

![GitHub last commit](https://img.shields.io/github/last-commit/AbdulAzeemHashmi/ai-video-trailer-pipeline?color=8E44AD&style=for-the-badge)
![GitHub stars](https://img.shields.io/github/stars/AbdulAzeemHashmi/ai-video-trailer-pipeline?color=yellow&style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/AbdulAzeemHashmi/ai-video-trailer-pipeline?color=orange&style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

</div>

# AI Video Trailer Pipeline

Welcome to the **AI Video Trailer Pipeline** repository. This project was originally built for an Artificial Intelligence Open Ended Lab.

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/213910845-af37a709-8995-40d6-be59-724526e3c3d7.gif" width="450">
</div>

This repository holds a complete, end to end AI video processing pipeline built in a single Jupyter Notebook. Starting from one raw input video, the pipeline detects people in every frame, extracts visual, object, and audio features, trains a small machine learning model to score how impactful each segment is, selects the strongest clips, applies a stylized visual theme, generates AI written captions, and finally stitches everything into a finished trailer with a supporting evaluation graph.

---

## What This Pipeline Actually Does

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="400">
</div>

The notebook loads a single video called `original.mp4` and pushes it through six connected tasks, producing four output videos and one evaluation chart.

```
original.mp4
     │
     ▼
YOLO11 Person Detection
     │
     ├──▶ Task 1: ROI blur and bounding box video
     │
     ▼
Feature Extraction (visual std, object count, MFCC audio score)
     │
     ▼
Logistic Regression Impact Scoring Model
     │
     ▼
Top Segment Selection
     │
     ├──▶ Task 3: Raw top clips trailer
     │
     ▼
Creepy Visual Theme (darkened bodies, glowing red eyes, desaturation)
     │
     ├──▶ Task 4: Stylized visuals only trailer
     │
     ▼
BLIP Captioning plus Keyword Injection
     │
     ├──▶ Task 5: Final trailer with captions
     │
     ▼
Task 6: Scene Impact Score Timeline graph
```

---

## Repository Contents

| File | Description |
|---|---|
| `AI_OEL.ipynb` | Main Jupyter Notebook containing the full pipeline, from model loading to final export |
| `Task1_ROI_Processing.mp4` | Every frame with a detected person blurred inside a green bounding box |
| `Task3_Raw_Selection.mp4` | The five highest scoring raw segments, joined together with no styling |
| `Task4_Creepy_Visuals.mp4` | The same top segments with darkened bodies, glowing red eyes, and desaturation applied |
| `Task5_Final_Trailer.mp4` | The finished trailer with BLIP generated, keyword modified captions overlaid |
| `Task6_Evaluation_Graph.png` | Line chart of the impact score for every segment against the median high impact threshold |

---

## Tech Stack and Models Used

<p>
<img src="https://img.shields.io/badge/YOLO11-Ultralytics-00FFFF?style=flat-square">
<img src="https://img.shields.io/badge/BLIP-Image%20Captioning-FF69B4?style=flat-square">
<img src="https://img.shields.io/badge/MoviePy-Video%20Editing-1E90FF?style=flat-square">
<img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=flat-square&logo=opencv&logoColor=white">
<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
<img src="https://img.shields.io/badge/Scikit--Learn-Logistic%20Regression-F7931E?style=flat-square&logo=scikitlearn&logoColor=white">
<img src="https://img.shields.io/badge/Librosa-Audio%20Features-8A2BE2?style=flat-square">
<img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat-square">
</p>

* **YOLO11, Ultralytics:** Detects people in each frame in real time and supplies the object count feature used for scoring
* **BLIP, Bootstrapping Language Image Pre training:** Generates a natural language caption for the first frame of every selected clip
* **MoviePy and OpenCV:** Handle resizing, frame level image transforms, sub clipping, and final video export
* **Librosa:** Extracts MFCC audio features from each segment so sound intensity feeds into the impact score
* **Scikit Learn:** A Logistic Regression classifier with L2 regularization learns which segments count as high impact
* **PyTorch:** The deep learning backend powering YOLO11 and BLIP, running on GPU when available
* **Matplotlib:** Plots the final Scene Impact Score Timeline

---

## Lab Tasks Overview

<details>
<summary><b>Task 1, ROI Processing, click to expand</b></summary>
<br>
For every frame, YOLO11 locates each person, converts that region of interest to grayscale, applies a Gaussian blur, then draws a green bounding box around it. The result is exported as an independent video showing the raw detection and processing step in action.
</details>

<details>
<summary><b>Task 2, Feature Extraction and Model Training, click to expand</b></summary>
<br>
The video is split into two second segments. For each segment, three features are computed: visual intensity from the pixel standard deviation, object count from YOLO11, and an audio score from the mean MFCC of the segment's audio. A weighted combination of these features produces a raw score, which is thresholded at the median to create binary labels. A Logistic Regression model with L2 regularization is then trained on these features and labels.
</details>

<details>
<summary><b>Task 3, Raw Segment Selection, click to expand</b></summary>
<br>
The trained model's raw scores are used to rank all segments. The top five highest scoring segments are selected in their original chronological order and joined into an unedited trailer, so the selection logic can be checked before any stylization is added.
</details>

<details>
<summary><b>Task 4, Creepy Theme Visuals, click to expand</b></summary>
<br>
Each of the top segments is reprocessed frame by frame. Detected people are darkened to thirty percent brightness, two glowing red circles are drawn at eye level to simulate glowing eyes, and the entire frame is partially desaturated for an eerie, muted look.
</details>

<details>
<summary><b>Task 5, Caption Generation and Final Trailer, click to expand</b></summary>
<br>
BLIP generates a natural language caption from the first frame of each stylized clip. Selected keywords in the caption are swapped for more atmospheric alternatives, for example replacing common nouns with phrases that fit the creepy theme. The captions are overlaid as red text at the bottom of each clip before the segments are joined into the final trailer.
</details>

<details>
<summary><b>Task 6, Scene Impact Score Evaluation, click to expand</b></summary>
<br>
The raw impact score for every segment is plotted as a timeline, with a dashed horizontal line marking the median high impact threshold. This graph, saved as <code>Task6_Evaluation_Graph.png</code>, gives a clear visual summary of how the model ranked the video over time.
</details>

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257467-871d32b7-e401-42e8-a166-fcfd7baa4c6b.gif" width="120">
</div>

---

## How to Run

**Step 1, clone this repository**

```bash
git clone https://github.com/AbdulAzeemHashmi/ai-video-trailer-pipeline.git
cd ai-video-trailer-pipeline
```

**Step 2, install dependencies**

```bash
pip install ultralytics torch torchvision moviepy opencv-python matplotlib transformers scikit-learn librosa
```

**Step 3, add your input video**

Place a video file named `original.mp4` in the same folder as the notebook. This is the raw footage the entire pipeline runs on.

**Step 4, open the notebook**

```bash
jupyter notebook AI_OEL.ipynb
```

**Step 5, run all cells in sequence**

The pipeline will automatically download the YOLO11 and BLIP model weights on the first run, then generate all four output videos and the evaluation graph in order.

---

## Output Summary

Once the notebook finishes, you should see the following new files in your folder.

- `Task1_ROI_Processing.mp4`
- `Task3_Raw_Selection.mp4`
- `Task4_Creepy_Visuals.mp4`
- `Task5_Final_Trailer.mp4`
- `Task6_Evaluation_Graph.png`

---

<div align="center">

## Author

**Abdul Azeem Hashmi** ([@AbdulAzeemHashmi](https://github.com/AbdulAzeemHashmi/))

<img src="https://img.shields.io/badge/Course-Artificial%20Intelligence-8E44AD?style=for-the-badge">

<br><br>

<img src="https://user-images.githubusercontent.com/74038190/212257468-1e9a91f1-b626-4baa-b15d-5c385dfa7ed2.gif" width="100">

If you found this project helpful, consider giving it a star.

</div>
