# CNN-Attention-Pedal-Generation

This project generates expressive piano sustain pedal (CC64) from MIDI using a CNN–Attention model. It captures both local and long-range musical patterns, and converts predictions into realistic pedal events, enhancing the musicality of AI-generated performances.

\---

## 🎧 Overview

The sustain pedal is essential for expressive piano performance, yet most AI-generated MIDI lacks pedal information, resulting in mechanical and less realistic sound.

This project proposes a CNN–Attention model to automatically generate sustain pedal (CC64) signals from MIDI.  
It combines local temporal modeling (CNN) with long-range dependency modeling (self-attention), and includes a post-processing pipeline to produce realistic pedal events.

\---

## 🚀 Features

* 🎹 Automatic sustain pedal (CC64) generation from MIDI
* 🧠 CNN + Multi-head Attention architecture
* ⏱ Frame-level prediction with event-level refinement
* 🔧 Post-processing (hysteresis, debouncing, repedal merging)
* 📊 Standardized evaluation (F1, MAE, onset/offset metrics)
* 🎼 Demo results on real and AI-generated MIDI

\---

## 🧱 Repository Structure

CNN-Attention-Pedal-Generation/
├── demo/  
├── scripts/  
├── src/  
├── assets/  
├── checkpoints/  
├── data/

\---

## ⚙️ Installation

git clone https://github.com/your-username/CNN-Attention-Pedal-Generation.git
cd CNN-Attention-Pedal-Generation

pip install -r requirements.txt

\---

## 📂 Dataset

This project uses the MAESTRO v3.0.0 dataset.

Download from:
https://magenta.tensorflow.org/datasets/maestro

Then preprocess:

python scripts/build\_dataset.py

\---

## 🏋️ Training

python scripts/train.py

\---

## 🎹 Inference

python scripts/infer.py --input input.mid --output output.mid

\---

## 📊 Evaluation

python scripts/evaluate.py

Optional:

python scripts/tune\_postprocess.py

\---

## 🎼 Demo

Examples are provided in the demo/ directory with input MIDI, output MIDI, and visualizations.

\---

## 🧠 Method

* Input: Piano roll from MIDI
* Model: CNN + Attention
* Output: Frame-level pedal probabilities
* Post-processing: hysteresis, debouncing, repedal merging

\---

## 📈 Results

* Frame-level F1: 0.802
* Onset F1: 0.657

\---

## 📜 License

MIT License

