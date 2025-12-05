# Clinovia EF Microservice

The **Clinovia EF Microservice** provides fully isolated, production-grade inference for predicting **Ejection Fraction (EF)** from echocardiogram videos using a 3D-CNN PyTorch model.

This service:
- Downloads the EF model from S3 at startup  
- Runs inference using PyTorch  
- Returns EF %, severity classification, and metadata  
- Is deployable to **Railway**, **Docker**, or any container platform  

---

## 🚀 Features

- FastAPI-based asynchronous API
- Automatic model download from AWS S3
- Supports MP4/MOV/DICOM videos (depending on OpenCV codecs)
- Stateless: no storage other than temporary files
- Designed as a **standalone microservice** for EF prediction

---


---

## 🔧 Environment Variables

The following are required (Railway → Variables tab):

| Variable | Description |
|---------|-------------|
| `AWS_ACCESS_KEY_ID` | S3 access key |
| `AWS_SECRET_ACCESS_KEY` | S3 secret key |
| `AWS_REGION` | Example: `us-east-1` |
| `CLINOVIA_S3_BUCKET` | S3 bucket name (default: `clinovia.ai`) |
| `EF_MODEL_KEY` | Path inside bucket to model file |
| `LOCAL_MODEL_PATH` | Path where model is stored inside container (default: `/tmp/ef_model.pth`) |

Example `.env` for local testing:



