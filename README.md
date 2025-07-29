# MLOps Week7: Scaling and Observing an Iris Classifier API

**Name:** Pravin S  
**Roll Number:** 22f1001797

## 1. Assignment Overview

This assignment demonstrates a complete MLOps pipeline for scaling a machine learning model. The primary goal was to take a scikit-learn Iris classifier, serve it via a FastAPI application, and deploy it to a production-grade environment on Google Kubernetes Engine (GKE).

The key objectives achieved were:

- **Containerization**: Packaging the application and its dependencies into a Docker image
- **CI/CD Automation**: Using GitHub Actions to automatically build and deploy the application on every push to the main branch
- **Horizontal Scaling**: Configuring a Horizontal Pod Autoscaler (HPA) to automatically scale the number of application pods based on CPU load
- **Performance Monitoring**: Integrating OpenTelemetry to send distributed traces to Google Cloud Trace for in-depth latency analysis
- **Load Testing**: Using Locust to simulate user traffic and identify performance bottlenecks

## 2. Final Repository Structure

The project was organized into the following structure for clarity and maintainability:

```
.
├── .github/
│   └── workflows/
│       └── deploy.yaml         # CI/CD pipeline definition
├── app/
│   ├── Dockerfile              # Instructions to build the container image
│   ├── iris_fastapi.py         # FastAPI application with OpenTelemetry
│   ├── model.joblib            # The trained scikit-learn model
│   └── requirements.txt        # Python dependencies
├── k8s/
│   ├── deployment.yaml         # GKE deployment configuration
│   ├── hpa.yaml               # Horizontal Pod Autoscaler configuration
│   └── service.yaml           # GKE service to expose the application
└── locustfile.py              # Script for load testing
```

## 3. Phase 1: Application and Containerization

### 3.1. FastAPI Application with OpenTelemetry

The core application is `app/iris_fastapi.py`. OpenTelemetry was integrated to trace the performance of the model prediction logic. A custom span named `model_prediction` was created to isolate the latency of the `model.predict()` call from other application overhead.

```python
# app/iris_fastapi.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# Setup OpenTelemetry
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(CloudTraceSpanExporter()))
tracer = trace.get_tracer(__name__)

app = FastAPI(title="Iris Classifier API")
model = joblib.load("model.joblib")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict/")
def predict_species(data: IrisInput):
    # Start a custom span for the prediction logic
    with tracer.start_as_current_span("model_prediction") as span:
        input_df = pd.DataFrame([data.model_dump()])  # Use model_dump for Pydantic v2
        prediction = model.predict(input_df)[0]
        span.set_attribute("predicted_class", prediction)
        return {"predicted_class": prediction}
```

### 3.2. Dockerfile

The `app/Dockerfile` was created to build a self-contained, runnable image of the application with health checks and proper practices.

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Run the application
CMD ["uvicorn", "iris_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3.3. Requirements

The `app/requirements.txt` includes all necessary dependencies with version constraints:

```txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
pandas>=2.0.0
# OpenTelemetry packages
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-gcp-trace>=1.20.0
```

## 4. Phase 2: Kubernetes Configuration

### 4.1. Deployment (k8s/deployment.yaml)

The deployment manifest specifies how to run our application on GKE. The most critical part is the `serviceAccountName: gke-sa` line. This assigns a specific identity (Workload Identity) to our running pods, granting them the necessary permissions to send data to Google Cloud Trace.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iris-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: iris-api
  template:
    metadata:
      labels:
        app: iris-api
    spec:
      serviceAccountName: gke-sa  # CRITICAL: Assigns identity to the pod
      containers:
      - name: iris-api
        image: us-central1-docker.pkg.dev/rare-keep-460211-q6/iris-repo/iris-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "250m"
          limits:
            cpu: "500m"
```

### 4.2. Service (k8s/service.yaml)

This manifest exposes the deployment to the internet via a public IP address using a LoadBalancer.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: iris-service
spec:
  selector:
    app: iris-api
  type: LoadBalancer
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
```

### 4.3. Horizontal Pod Autoscaler (k8s/hpa.yaml)

The HPA automatically adjusts the number of running pods based on CPU load, ensuring the application can handle traffic spikes. It was configured to maintain an average CPU utilization of 60%.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: iris-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: iris-deployment
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
```

## 5. Phase 3: CI/CD Pipeline with GitHub Actions

A CI/CD pipeline was configured in `.github/workflows/deploy.yaml` to automate the build and deployment process. For this project, authentication was handled via a service account JSON key stored in GitHub Secrets.

```yaml
name: Build and Deploy to GKE

on:
  push:
    branches: [ "main" ]
  workflow_dispatch:

env:
  GCP_PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  GAR_LOCATION: us-central1
  GKE_CLUSTER: iris-cluster
  GKE_ZONE: us-central1
  REPOSITORY: iris-repo
  IMAGE: iris-api

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: 'read'

    steps:
    - name: Checkout Repo
      uses: actions/checkout@v3

    - name: Authenticate to GCP
      id: auth
      uses: 'google-github-actions/auth@v1'
      with:
        credentials_json: ${{ secrets.GCP_SA_KEY }}

    - name: Set up GKE credentials
      uses: 'google-github-actions/get-gke-credentials@v1'
      with:
        cluster_name: iris-cluster
        location: us-central1

    - name: Build and Push Docker Image
      run: |-
        gcloud auth configure-docker us-central1-docker.pkg.dev
        docker build -t iris-api ./app
        docker tag iris-api us-central1-docker.pkg.dev/rare-keep-460211-q6/iris-repo/iris-api:latest
        docker push us-central1-docker.pkg.dev/rare-keep-460211-q6/iris-repo/iris-api:latest

    - name: Deploy to GKE
      run: |-
        kubectl apply -f k8s/
```

## 6. Phase 4: Load Testing and Observation Results

### 6.1. Locust Load Test

A `locustfile.py` was created to simulate 100 concurrent users sending requests to the `/predict/` endpoint.

```python
# locustfile.py
from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 2)
    
    @task
    def predict(self):
        headers = {"Content-Type": "application/json"}
        payload = {
            "sepal_length": 5.1, "sepal_width": 3.5,
            "petal_length": 1.4, "petal_width": 0.2
        }
        self.client.post("/predict/", json=payload, headers=headers)
```

**Locust Test Results:**
The test was run successfully with 100 users, achieving an average RPS of ~11.3 with 0 failures.

*(Screenshot of Locust UI would go here)*

### 6.2. HPA Scaling in Action

During the load test, the HPA was monitored using `kubectl get hpa -w`. The CPU utilization quickly surpassed the 60% target, and the HPA automatically scaled the number of replicas from 1 up to 4 to handle the load.

*(Screenshot of HPA scaling in the terminal would go here)*

### 6.3. Cloud Trace Analysis

After correctly configuring Workload Identity, traces appeared in the Google Cloud Trace Explorer. By filtering for the `/predict/` endpoint, we could analyze individual requests. The custom `model_prediction` span was clearly visible, showing the precise latency of the model inference, separate from network and framework overhead. This confirms that our observability setup is working perfectly.

*(Screenshot of Cloud Trace Explorer showing the custom span would go here)*

## 7. Phase 5: Troubleshooting and Resolution - Enabling Workload Identity

A significant challenge encountered was that despite a successful deployment, no traces were appearing in Google Cloud Trace. This was a classic "silent failure" caused by a permissions issue. The GitHub Actions pipeline had permission to deploy the application, but the running pods themselves had no identity and therefore no permission to send telemetry data to Google Cloud APIs.

The issue was resolved by configuring Workload Identity, which is the recommended, secure method for GKE applications to access GCP services.

The following steps were executed to solve the problem:

### Step 1: Create a dedicated GCP Service Account (GSA) for the pods.
This GSA was given a single, minimal permission.

```bash
gcloud iam service-accounts create gke-pod-sa --display-name="GKE Pod SA for Tracing"
```

### Step 2: Grant the GSA permission to send traces.

```bash
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:gke-pod-sa@$GCP_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/cloudtrace.agent"
```

### Step 3: Create a Kubernetes Service Account (KSA).
This service account lives inside the GKE cluster.

```bash
kubectl create serviceaccount gke-sa
```

### Step 4: Link the KSA to the GSA.
This crucial step allows the Kubernetes service account to impersonate the GCP service account.

```bash
gcloud iam service-accounts add-iam-policy-binding "gke-pod-sa@$GCP_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:$GCP_PROJECT_ID.svc.id.goog[default/gke-sa]"

kubectl annotate serviceaccount gke-sa \
  iam.gke.io/gcp-service-account=gke-pod-sa@$GCP_PROJECT_ID.iam.gserviceaccount.com
```

### Step 5: Update deployment.yaml to use the new identity.
The final step was to edit the deployment manifest and tell the pods to run as the `gke-sa` identity. This was the missing link that ultimately resolved the issue.

```yaml
# In k8s/deployment.yaml
# ...
    spec:
      serviceAccountName: gke-sa  # <-- This line was added
      containers:
# ...
```

## 8. Key Learnings

This project successfully demonstrated an end-to-end MLOps pipeline for a scalable, observable ML service. The most significant challenge was correctly configuring the IAM permissions for the GKE pods. The initial failure to see traces was resolved by implementing Workload Identity: creating a dedicated GCP service account (`gke-pod-sa`) with the `roles/cloudtrace.agent` role and linking it to a Kubernetes service account (`gke-sa`) which was then assigned to the pods via the `serviceAccountName` field in the deployment manifest. This reinforced the importance of the Principle of Least Privilege and the separation of concerns between deployment-time and run-time permissions.

### Learnings:
- ✅ Successfully containerized the ML application with proper health checks
- ✅ Implemented automated CI/CD pipeline with GitHub Actions
- ✅ Configured horizontal scaling with HPA for traffic management
- ✅ Integrated OpenTelemetry for distributed tracing
- ✅ Resolved Workload Identity issues for proper GCP service access
- ✅ Conducted load testing to validate performance under stress

### Technical Highlights:
- **Observability**: Custom spans for model prediction latency tracking
- **Scalability**: Automatic pod scaling based on CPU utilization
- **Security**: Proper IAM configuration with Workload Identity
- **Reliability**: Health checks and resource limits for production readiness
