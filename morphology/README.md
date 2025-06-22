# Morphology Module – AaaS MVP

This folder contains the foundation for developing the **Sperm Morphology 
Analysis** module as part of the Andrology-as-a-Service (AaaS) platform 
MVP.

## 📁 Folder Structure



## 🧪 Current Dataset (SCIAN)

- Categories:
  - Normal
  - Tapered
  - Pyriform
  - Amorphous
  - Duplicate (duplicated_XXX.jpg)

Use only `normal` and the 3 abnormal categories to build binary 
(normal/abnormal) and multiclass classifiers.

---

## 🔧 Suggested Next Steps

1. **Data Preprocessing**
   - Resize to consistent shape (e.g., 100×100 px)
   - Normalize pixel values
   - Augmentations (optional)

2. **Model Development**
   - Baseline: CNN with Keras or PyTorch
   - Bonus: Visualize feature maps

3. **Evaluation**
   - Accuracy, Precision, Recall, F1
   - Per-class metrics

4. **Output**
   - Classification report
   - Confusion matrix
   - Bonus: Grad-CAM or similar visualization

---

## 📌 Notes

- Avoid using the `duplicated` folder from SCIAN — it's meant for testing 
duplication, not for model training.
- Start with 200–500 images (small prototype), then scale.
- Prepare for possible extension to real-time microscope integration in 
the future.



