# Deployment

This app is a Streamlit deployment for the saved dropout-risk model artifacts.

## Run Locally

```bash
streamlit run app.py
```

## Run With Docker

Build the image:

```bash
docker build -t dropout-risk-app .
```

Run the container:

```bash
docker run -p 8501:8501 dropout-risk-app
```

Open:

```text
http://localhost:8501
```

## Important

The app uses the saved files in `model_artifacts/`:

- `calibrated_pipeline.joblib`
- `final_pipeline.joblib`
- `metadata.joblib`

If the notebook model is retrained or thresholds change, rerun the artifact-saving cell in the notebook before rebuilding the Docker image.

## Streamlit Community Cloud

For Streamlit Community Cloud, push these files to GitHub:

- `app.py`
- `requirements.txt`
- `model_artifacts/`

Then set the app entry point to:

```text
app.py
```

Docker is not required for Streamlit Community Cloud.
